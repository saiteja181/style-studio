"""Style Studio API - customer consultation + hairstyle preview generation."""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from backend.face_analysis import analyze_face
from backend.customer_analysis import analyze_customer, AnalysisError
from backend.style_matcher import recommend_styles
from backend.input_pipeline import prepare_upload, PreflightError
from backend.inpaint import (
    generate_preview_inpaint,
    generate_preview_auto,
    generate_preview_expert,
    generate_preview_erase_then_inpaint,
    InpaintError,
)

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_PATH = PROJECT_ROOT / "catalogue" / "styles.json"
REFERENCES_DIR = PROJECT_ROOT / "catalogue" / "references"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOADS_DIR = PROJECT_ROOT / "tests" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Style Studio API",
    version="0.2.0",
    description="Indian hairstyle consultation + preview generation for salons.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve reference images so the UI can show them.
if REFERENCES_DIR.exists():
    app.mount("/references", StaticFiles(directory=str(REFERENCES_DIR)), name="references")

# Serve generated/uploaded images.
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Serve the frontend static page (if it exists).
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="app")


class HealthResponse(BaseModel):
    status: str
    version: str
    catalogue_styles: int
    replicate_configured: bool
    anthropic_configured: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=app.version,
        catalogue_styles=_load_catalogue_count(),
        replicate_configured=bool(os.getenv("REPLICATE_API_TOKEN")),
        anthropic_configured=bool(os.getenv("ANTHROPIC_API_KEY")),
    )


@app.post("/consult")
async def consult(
    image: UploadFile = File(...),
    gender: Optional[str] = Form(None),
    occasion: Optional[str] = Form(None),
    use_vision: bool = Form(True),
    top_n: int = Form(5),
) -> dict:
    """Full consultation: customer analysis + ranked style recommendations.

    Returns:
      {
        "profile": {face_shape, jawline, skin_tone_bucket, hair_color_descriptor,
                    hair_texture, hairline_shape, estimated_gender, ...},
        "recommendations": [
          {style_id, style_name, suit_score, reasoning, reference_url, ...}
        ]
      }
    """
    _validate_image_upload(image)

    saved_path = await _save_upload(image)

    try:
        profile = analyze_customer(
            selfie_path=saved_path,
            use_vision_lm=use_vision and bool(os.getenv("ANTHROPIC_API_KEY")),
            gender_hint=gender if gender in ("male", "female") else None,
        )
    except AnalysisError as e:
        raise HTTPException(status_code=422, detail=str(e))

    recs = recommend_styles(
        profile=profile,
        top_n=top_n,
        occasion=occasion,
        gender_filter=gender if gender in ("male", "female") else None,
    )

    rec_dicts = []
    for r in recs:
        meta = r.style_metadata
        ref = meta.get("reference_image_path")
        rec_dicts.append({
            "style_id": r.style_id,
            "style_name": r.style_name,
            "suit_score": r.suit_score,
            "reasoning": r.reasoning,
            "length": meta.get("length"),
            "occasion": meta.get("occasion", []),
            "cultural": meta.get("cultural", []),
            "style_traits": meta.get("style_traits", []),
            "reference_url": f"/references/{ref}" if ref else None,
            "has_reference": bool(ref),
        })

    return {
        "profile": profile.to_dict(),
        "recommendations": rec_dicts,
        "uploaded_image_url": f"/uploads/{saved_path.name}",
    }


@app.post("/generate-batch")
async def generate_batch(
    image: UploadFile = File(...),
    style_ids: str = Form(...),     # comma-separated list
    mode: str = Form("transform"),
    seed: Optional[int] = Form(42),
) -> dict:
    """Generate multiple style previews in parallel (one FLUX call per style,
    fired concurrently). Returns a dict of {style_id: result_or_error}.

    Salon use case: customer picks their top 3 styles from the recommendations,
    sees all 3 previews ready in ~60-90s instead of waiting 3-5 min sequentially.
    """
    import asyncio
    _validate_image_upload(image)
    saved_path = await _save_upload(image)
    ids = [s.strip() for s in style_ids.split(",") if s.strip()]

    if not os.getenv("REPLICATE_API_TOKEN"):
        raise HTTPException(status_code=503,
                            detail="REPLICATE_API_TOKEN not configured")

    def _gen_one(sid: str) -> dict:
        try:
            if mode == "transform":
                r = generate_preview_erase_then_inpaint(
                    selfie_path=saved_path, style_id=sid, seed=seed)
            elif mode == "expert":
                r = generate_preview_expert(
                    selfie_path=saved_path, style_id=sid, seed=seed)
            else:
                r = generate_preview_inpaint(
                    selfie_path=saved_path, style_id=sid, seed=seed)
            return {"status": "ok", **r.to_dict()}
        except Exception as e:
            return {"status": "error", "style_id": sid, "detail": str(e)}

    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        *[loop.run_in_executor(None, _gen_one, sid) for sid in ids],
        return_exceptions=False,
    )
    return {"results": {ids[i]: results[i] for i in range(len(ids))}}


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    style_id: str = Form(...),
    mode: str = Form("expert"),
    seed: Optional[int] = Form(42),
) -> dict:
    """Generate a hairstyle preview for the uploaded photo + style.

    mode: "expert" (Claude consult + FLUX, best quality, requires ANTHROPIC_API_KEY)
          | "auto" (Qwen2-VL caption + FLUX)
          | "manual" (uses catalogue prompt_template + FLUX)
    """
    _validate_image_upload(image)
    saved_path = await _save_upload(image)

    if not os.getenv("REPLICATE_API_TOKEN"):
        raise HTTPException(status_code=503,
                            detail="REPLICATE_API_TOKEN not configured")

    try:
        if mode == "transform":
            # Erase existing hair, then inpaint the new style on bald canvas.
            # Best for DRAMATIC transformations (straight->curly, short->long, etc.).
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise HTTPException(status_code=503,
                                    detail="ANTHROPIC_API_KEY required for transform mode")
            result = generate_preview_erase_then_inpaint(
                selfie_path=saved_path, style_id=style_id, seed=seed,
            )
        elif mode == "expert":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise HTTPException(status_code=503,
                                    detail="ANTHROPIC_API_KEY required for expert mode")
            result = generate_preview_expert(
                selfie_path=saved_path, style_id=style_id, seed=seed,
                validate=bool(os.getenv("ANTHROPIC_API_KEY")), max_retries=1,
            )
        elif mode == "auto":
            result = generate_preview_auto(
                selfie_path=saved_path, style_id=style_id, seed=seed,
                validate=bool(os.getenv("ANTHROPIC_API_KEY")), max_retries=1,
            )
        elif mode == "manual":
            result = generate_preview_inpaint(
                selfie_path=saved_path, style_id=style_id, seed=seed,
                validate=bool(os.getenv("ANTHROPIC_API_KEY")), max_retries=1,
            )
        else:
            raise HTTPException(status_code=400,
                                detail=f"Unknown mode: {mode}")
    except InpaintError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result.to_dict()


@app.get("/catalogue")
def catalogue() -> dict:
    """Return the full style catalogue with reference image URLs filled in."""
    styles = _load_catalogue()
    for s in styles:
        ref = s.get("reference_image_path")
        if ref:
            s["reference_url"] = f"/references/{ref}"
    return {"styles": styles}


@app.get("/catalogue/{style_id}")
def get_style(style_id: str) -> dict:
    for style in _load_catalogue():
        if style["id"] == style_id:
            ref = style.get("reference_image_path")
            if ref:
                style["reference_url"] = f"/references/{ref}"
            return style
    raise HTTPException(status_code=404, detail=f"Style not found: {style_id}")


# Legacy face-only analysis for backwards compat.
@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> dict:
    _validate_image_upload(image)
    contents = await image.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unreadable image: {e}")

    result = analyze_face(np.array(pil))
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected. Try a clearer front-facing photo.",
        )
    return result.to_dict()


# ---- helpers ----

def _validate_image_upload(image: UploadFile) -> None:
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg",
                                  "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content-type: {image.content_type}",
        )


async def _save_upload(image: UploadFile) -> Path:
    """Normalise an upload (EXIF transpose + resize + face pre-flight) and
    return the path of the cleaned JPEG.  Raises HTTPException 422 if the
    photo is unusable so staff can retake before we spend Replicate budget."""
    contents = await image.read()
    try:
        saved_path, report = prepare_upload(
            raw_bytes=contents, target_dir=UPLOADS_DIR, filename_hint="selfie",
        )
    except PreflightError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "code": e.report.code,
                "message": e.report.message,
                **e.report.to_dict(),
            },
        ) from e
    if report.warnings:
        logger.info("upload warnings: %s", report.warnings)
    return saved_path


def _load_catalogue() -> list[dict]:
    if not CATALOGUE_PATH.exists():
        return []
    with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_catalogue_count() -> int:
    return len(_load_catalogue())
