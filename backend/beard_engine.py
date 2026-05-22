"""Beard preview generation using FLUX Kontext.

Mirrors backend.kontext_engine but with two key differences:
  1. Catalogue source is catalogue/beards.json (not styles.json).
  2. face_composite is called with mode="beard" so Kontext owns the lower
     face (mouth + jaw + cheeks) while eyes/nose stay byte-perfect from
     source.

Cost: ~$0.04 per preview, same as the hair engine.  Validator is currently
not wired in for beard previews (no reference photos in the beard catalogue
yet) - this can be added in a follow-up if quality regressions appear.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from backend.kontext_engine import (
    PreviewResult, GenerationError, _call_kontext,
)

logger = logging.getLogger(__name__)

BEARDS_PATH = Path(__file__).resolve().parent.parent / "catalogue" / "beards.json"

_BEARDS_CACHE: Optional[list[dict]] = None


class BeardStyleNotFoundError(GenerationError):
    """Raised when beard_style_id is not present in the beard catalogue."""


def generate_beard_preview(
    source_path: Path,
    beard_style_id: str,
    customer_profile: dict,
    seed: int = 42,
) -> PreviewResult:
    """Run a beard-only preview through FLUX Kontext.

    No retry loop and no validator: beard catalogue currently has no
    reference photos, so the validator branch would be skipped anyway.
    """
    style = _load_beard_style(beard_style_id)
    if style is None:
        raise BeardStyleNotFoundError(f"Unknown beard style: {beard_style_id}")

    from backend.face_composite import paste_source_face

    uploads_dir = Path(
        os.getenv("STYLE_STUDIO_UPLOADS_DIR")
        or (Path(__file__).resolve().parent.parent / "tests" / "uploads")
    )

    base = style.get("prompt_template") or style.get("name", "")
    prompt = (
        f"Change ONLY the facial hair to: {base}. "
        "Keep the hair on top of the head, eyes, expression, eyebrows, "
        "glasses, clothing, hands, and background exactly identical to the "
        "original photo.  Modify only the jawline / cheek / chin / moustache "
        "area.  Photoreal, same ambient indoor lighting as the source, no "
        "studio lighting, no halo."
    )

    started = time.time()
    raw_url = _call_kontext(source_path, prompt, seed)
    composited = paste_source_face(
        source_path=source_path,
        kontext_output_url_or_path=raw_url,
        output_dir=uploads_dir,
        mode="beard",
    )
    final_image_url = f"/uploads/{composited.name}"
    elapsed_ms = int((time.time() - started) * 1000)

    return PreviewResult(
        image_url=final_image_url,
        style_id=beard_style_id,
        style_name=style.get("name", beard_style_id),
        prompt=prompt,
        seed=seed,
        validator_verdict="skipped_no_reference",
        retries=0,
        elapsed_ms=elapsed_ms,
    )


def _load_beard_style(beard_style_id: str) -> Optional[dict]:
    global _BEARDS_CACHE
    if _BEARDS_CACHE is None:
        if not BEARDS_PATH.exists():
            return None
        with BEARDS_PATH.open("r", encoding="utf-8") as f:
            _BEARDS_CACHE = json.load(f)
    for b in _BEARDS_CACHE:
        if b.get("id") == beard_style_id:
            return b
    return None
