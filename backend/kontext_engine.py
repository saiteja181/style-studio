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


def _call_kontext(
    source_path: Path,
    prompt: str,
    seed: int,
    safety_tolerance: int = 2,
) -> str:
    """Single Replicate call.  Returns the output URL string.

    Raises GenerationError on any failure (network, API rejection, missing
    URL in the response).
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise GenerationError("REPLICATE_API_TOKEN not set in environment")
    try:
        import replicate
    except ImportError as e:
        raise GenerationError("replicate SDK not installed") from e

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
                    "prompt_upsampling": False,
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
