"""Core generation engine: FLUX Kontext Pro via Replicate.

This module is the ONLY place that imports `replicate`.  Public surface is
`generate_preview()` (added in Task 5) and the `PreviewResult` dataclass.
Failures raise `GenerationError`; callers map that to HTTP 502.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import replicate

logger = logging.getLogger(__name__)

KONTEXT_MODEL = "black-forest-labs/flux-kontext-pro"


@dataclass
class PreviewResult:
    image_url: str           # served path /uploads/<file>.png
    style_id: str
    style_name: str
    prompt: str
    seed: int
    validator_verdict: str   # "pass" | "fail" | "uncertain" | "skipped"
    retries: int
    elapsed_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


class GenerationError(RuntimeError):
    """Raised when the Kontext call cannot produce any image at all."""


class StyleNotFoundError(GenerationError):
    """Raised when style_id is not present in the catalogue."""


def _call_kontext(
    source_path: Path,
    prompt: str,
    seed: int,
    safety_tolerance: int = 2,
    style: Optional[dict] = None,
) -> str:
    """Single Replicate call.  Returns the output URL string.

    Raises GenerationError on any failure (network, API rejection, missing
    URL in the response).
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise GenerationError("REPLICATE_API_TOKEN not set in environment")

    try:
        with Path(source_path).open("rb") as img_f:
            output = replicate.run(
                KONTEXT_MODEL,
                input={
                    "prompt": prompt,
                    "input_image": img_f,
                    "aspect_ratio": "match_input_image",
                    "output_format": "png",
                    "safety_tolerance": safety_tolerance,
                    # Per-style override added in sub-project 8: short male
                    # cuts (pompadour / korean fringe / textured crop / buzz /
                    # classic side part) set upsampling=False because the
                    # upsampler was inventing a dramatic forelock-strand across
                    # the face on those styles.
                    "prompt_upsampling": (
                        style.get("upsampling", True) if style is not None else True
                    ),
                    "seed": seed,
                },
            )
    except Exception as e:
        raise GenerationError(f"Kontext call failed: {e}") from e

    url = _extract_first_url(output)
    if not url:
        raise GenerationError(f"Kontext returned no URL: {output!r}")
    return url


def _extract_first_url(output) -> Optional[str]:
    """Replicate may return a string, list of strings, or an object with .url."""
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str):
            return first
        url = getattr(first, "url", None)
        if isinstance(url, str):
            return url
    url = getattr(output, "url", None)
    if isinstance(url, str):
        return url
    return None


import json
import time

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "catalogue" / "styles.json"
REFERENCES_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "references"


def generate_preview(
    source_path: Path,
    style_id: str,
    customer_profile: dict,
    seed: int = 42,
    max_retries: int = 1,
    head_covering_type: Optional[str] = None,
) -> PreviewResult:
    """Run a full preview: build prompt -> Kontext -> face composite ->
    validate -> retry on fail.

    Raises GenerationError if every attempt fails to produce an image.

    Args:
        head_covering_type: optional covering label from SP 1.7's detector
            (turban / hijab / ghoonghat / cap_hat / other).  Threaded into
            face_composite so the upper polygon is shrunk and fabric does
            not bleed back over the Kontext output.
    """
    style = _load_style(style_id)
    if style is None:
        raise StyleNotFoundError(f"Unknown style: {style_id}")
    ref_path = _resolve_reference_path(style)

    from backend.prompt_builder import build_edit_prompt
    from backend.face_composite import paste_source_face

    uploads_dir = Path(
        os.getenv("STYLE_STUDIO_UPLOADS_DIR")
        or (Path(__file__).resolve().parent.parent / "tests" / "uploads")
    )

    attempt_idx = -1  # bound even when max_retries < 0 and the loop never runs
    started = time.time()
    verdict = "skipped"
    retries = 0
    final_image_url = None
    final_prompt = ""

    for attempt_idx in range(max_retries + 1):
        attempt_seed = seed if attempt_idx == 0 else seed + 1000 + attempt_idx
        final_prompt = build_edit_prompt(
            style=style, customer_profile=customer_profile,
            source_path=source_path, reference_path=ref_path,
        )

        raw_url = _call_kontext(source_path, final_prompt, attempt_seed, style=style)
        composited = paste_source_face(
            source_path=source_path,
            kontext_output_url_or_path=raw_url,
            output_dir=uploads_dir,
            head_covering_type=head_covering_type,
        )
        final_image_url = f"/uploads/{composited.name}"

        if not os.getenv("ANTHROPIC_API_KEY"):
            verdict = "skipped_no_anthropic_key"
            break
        if ref_path is None:
            verdict = "skipped_no_reference"
            break
        verdict = _validate(source_path, ref_path, composited)
        logger.info("validator attempt %d: %s", attempt_idx + 1, verdict)
        if verdict in ("pass", "uncertain"):
            # 'uncertain' counts as ship - validator parse error shouldn't
            # burn a second Kontext call.
            break

    retries = max(0, min(attempt_idx, max(0, max_retries)))
    elapsed_ms = int((time.time() - started) * 1000)
    return PreviewResult(
        image_url=final_image_url,
        style_id=style_id,
        style_name=style.get("name", style_id),
        prompt=final_prompt,
        seed=seed,
        validator_verdict=verdict,
        retries=retries,
        elapsed_ms=elapsed_ms,
    )


def _validate(
    source_path: Path, reference_path: Path, composited_path: Path,
) -> str:
    try:
        from backend.output_validator import validate_generation
        verdict_dict = validate_generation(
            source_path=source_path, reference_path=reference_path,
            generated_url=composited_path.as_uri(),
        )
        return verdict_dict.get("verdict", "uncertain")
    except Exception as e:
        logger.warning("validator unavailable: %s", e)
        return "uncertain"


_CATALOGUE_CACHE: Optional[list[dict]] = None


def _load_style(style_id: str) -> Optional[dict]:
    """Return the catalogue entry for style_id, or None if not found.
    The full catalogue is parsed once and cached at module scope."""
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is None:
        if not CATALOGUE_PATH.exists():
            return None
        with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
            _CATALOGUE_CACHE = json.load(f)
    for s in _CATALOGUE_CACHE:
        if s.get("id") == style_id:
            return s
    return None


def _resolve_reference_path(style: dict) -> Optional[Path]:
    ref = style.get("reference_image_path")
    if not ref:
        return None
    p = Path(ref)
    if not p.is_absolute():
        p = REFERENCES_DIR / p
    return p if p.exists() else None
