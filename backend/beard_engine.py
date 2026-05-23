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
    CostLedger, COST_USD_KONTEXT,
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
    max_retries: int = 1,
    head_covering_type: Optional[str] = None,
    cost_ledger: Optional[CostLedger] = None,
) -> PreviewResult:
    """Run a beard-only preview through FLUX Kontext.

    No retry loop and no validator: beard catalogue currently has no
    reference photos, so the validator branch would be skipped anyway.

    Args:
        head_covering_type: optional covering label from SP 1.7's detector
            (turban / hijab / ghoonghat / cap_hat / other).  Threaded into
            face_composite so the upper polygon is shrunk and fabric does
            not bleed back over the Kontext output.
        cost_ledger: optional shared CostLedger so beard previews respect
            the same per-customer budget as hair previews.  If None,
            a fresh ledger is created (default cap from env).
    """
    style = _load_beard_style(beard_style_id)
    if style is None:
        raise BeardStyleNotFoundError(f"Unknown beard style: {beard_style_id}")

    from backend.face_composite import paste_source_face

    uploads_dir = Path(
        os.getenv("STYLE_STUDIO_UPLOADS_DIR")
        or (Path(__file__).resolve().parent.parent / "tests" / "uploads")
    )
    if cost_ledger is None:
        cost_ledger = CostLedger()

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
    attempt_idx = -1
    verdict = "skipped_no_anthropic_key"
    final_image_url = None

    for attempt_idx in range(max_retries + 1):
        attempt_seed = seed if attempt_idx == 0 else seed + 1000 + attempt_idx
        # Same label as the hair pipeline so a shared ledger doesn't
        # produce two separate breakdown rows for what is really the
        # same Replicate model at the same price.
        cost_ledger.check_and_charge("kontext", COST_USD_KONTEXT)
        raw_url = _call_kontext(source_path, prompt, attempt_seed)
        composited = paste_source_face(
            source_path=source_path,
            kontext_output_url_or_path=raw_url,
            output_dir=uploads_dir,
            mode="beard",
            head_covering_type=head_covering_type,
        )
        final_image_url = f"/uploads/{composited.name}"

        # Beard catalogue has no reference photos today, so the validator
        # branch never fires in production.  When references are added in
        # a future sub-project, _validate_beard can be wired in here.
        if not os.getenv("ANTHROPIC_API_KEY"):
            verdict = "skipped_no_anthropic_key"
            break
        verdict = "skipped_no_reference"
        break

    retries = max(0, min(attempt_idx, max_retries))
    elapsed_ms = int((time.time() - started) * 1000)

    return PreviewResult(
        image_url=final_image_url,
        style_id=beard_style_id,
        style_name=style.get("name", beard_style_id),
        prompt=prompt,
        seed=seed,
        validator_verdict=verdict,
        retries=retries,
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
