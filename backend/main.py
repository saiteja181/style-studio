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
from backend.input_pipeline import prepare_upload, PreflightError, PreflightReport
from backend.kontext_engine import generate_preview, GenerationError, StyleNotFoundError
from backend.retention import lifespan_with_sweeper

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
    version="0.3.1",
    description="Indian hairstyle consultation + preview generation for salons.",
    lifespan=lambda app: lifespan_with_sweeper(UPLOADS_DIR, app),
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
    engine: str
    catalogue_styles: int
    replicate_configured: bool
    anthropic_configured: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=app.version,
        engine="flux-kontext-pro",
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

    saved_path, report = await _save_upload(image)

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

    from backend.skin_palette import recommend_palette
    palette = recommend_palette(profile.skin_tone_bucket)

    return {
        "profile": profile.to_dict(),
        "recommendations": rec_dicts,
        "palette": palette,
        "uploaded_image_url": f"/uploads/{saved_path.name}",
    }


@app.post("/generate-batch")
async def generate_batch(
    image: UploadFile = File(...),
    style_ids: str = Form(...),     # comma-separated list
    seed: Optional[int] = Form(42),
) -> dict:
    """Generate multiple previews in parallel.  Returns
    {style_id: PreviewResult.to_dict() | {"error": "..."}}.
    """
    import asyncio
    _validate_image_upload(image)
    saved_path, report = await _save_upload(image)
    head_covering_type = (
        report.head_covering.get("covering_type")
        if report.head_covering.get("detected") else None
    )

    profile = analyze_customer(
        selfie_path=saved_path,
        use_vision_lm=bool(os.getenv("ANTHROPIC_API_KEY")),
    ).to_dict()

    ids = [s.strip() for s in style_ids.split(",") if s.strip()]
    loop = asyncio.get_running_loop()

    def _gen_one(sid: str) -> dict:
        try:
            r = generate_preview(
                source_path=saved_path, style_id=sid,
                customer_profile=profile, seed=seed if seed is not None else 42,
                head_covering_type=head_covering_type,
            )
            return r.to_dict()
        except Exception as e:
            return {"error": str(e), "style_id": sid}

    results = await asyncio.gather(
        *[loop.run_in_executor(None, _gen_one, sid) for sid in ids],
        return_exceptions=False,
    )
    return {"results": {ids[i]: results[i] for i in range(len(ids))}}


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    style_id: str = Form(...),
    seed: Optional[int] = Form(42),
) -> dict:
    """Generate a single hairstyle preview using FLUX Kontext.

    Returns PreviewResult.to_dict() shape (see backend.kontext_engine).
    """
    _validate_image_upload(image)
    saved_path, report = await _save_upload(image)
    head_covering_type = (
        report.head_covering.get("covering_type")
        if report.head_covering.get("detected") else None
    )

    profile = analyze_customer(
        selfie_path=saved_path,
        use_vision_lm=bool(os.getenv("ANTHROPIC_API_KEY")),
    ).to_dict()

    try:
        result = generate_preview(
            source_path=saved_path,
            style_id=style_id,
            customer_profile=profile,
            seed=seed if seed is not None else 42,
            head_covering_type=head_covering_type,
        )
    except StyleNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return result.to_dict()


@app.post("/generate-beard")
async def generate_beard(
    image: UploadFile = File(...),
    beard_style_id: str = Form(...),
    seed: Optional[int] = Form(42),
) -> dict:
    """Generate a beard-only preview using FLUX Kontext.

    Returns PreviewResult.to_dict() shape, same as /generate.
    """
    from backend.beard_engine import generate_beard_preview, BeardStyleNotFoundError

    _validate_image_upload(image)
    saved_path, report = await _save_upload(image)
    head_covering_type = (
        report.head_covering.get("covering_type")
        if report.head_covering.get("detected") else None
    )

    profile = analyze_customer(
        selfie_path=saved_path,
        use_vision_lm=bool(os.getenv("ANTHROPIC_API_KEY")),
    ).to_dict()

    try:
        result = generate_beard_preview(
            source_path=saved_path,
            beard_style_id=beard_style_id,
            customer_profile=profile,
            seed=seed if seed is not None else 42,
            head_covering_type=head_covering_type,
        )
    except BeardStyleNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))
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


async def _save_upload(image: UploadFile) -> tuple[Path, PreflightReport]:
    """Normalise an upload (EXIF transpose + resize + face pre-flight) and
    return the path of the cleaned JPEG plus the PreflightReport.  Raises
    HTTPException 422 if the photo is unusable so staff can retake before we
    spend Replicate budget.

    The report carries the head_covering detection result so downstream
    engines can shrink the face polygon when a covering is present.
    """
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
    return saved_path, report


def _load_catalogue() -> list[dict]:
    if not CATALOGUE_PATH.exists():
        return []
    with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_catalogue_count() -> int:
    return len(_load_catalogue())
