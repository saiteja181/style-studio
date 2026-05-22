"""Expert hair stylist consult: given a source photo and a reference style
photo, produce an EDIT INSTRUCTION for FLUX Kontext that adapts the style
to this specific customer's face shape and hairline."""
from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "consults_cache"

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 500
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are an expert hair stylist writing an EDIT INSTRUCTION for an AI image editor (FLUX Kontext).  The editor will modify the customer's photo to apply the target hairstyle while keeping the rest of the photo intact.

You will see TWO images:
1. SOURCE: the customer's current photo.
2. REFERENCE: the target hairstyle the customer wants.

Write a single-paragraph edit instruction that:
- Describes the target hairstyle in anatomically specific terms: approximate length in cm, hair direction (forward/back/up/parted), fade location and height for cuts that have one, fringe extent for styles with bangs, exposed-temple vs covered-temple decisions.
- Names the styling explicitly (e.g. "pompadour", "korean fringe", "textured crop"), but does not rely on the name alone - describe what it looks like.
- References the customer's face shape, jawline, or hairline visible in SOURCE when it helps adapt the cut (e.g. "the customer has a square jaw; soften the sides slightly").
- Ends with this exact sentence: "Keep face, eyes, expression, beard, eyebrows, glasses, clothing, hands, and background identical to source."

Output ONLY the edit instruction.  No preamble, no markdown, no quotes, no list bullets.  One paragraph, 80-160 words."""


class ConsultError(RuntimeError):
    pass


def consult_for_style(
    source_image_path: Path,
    reference_image_path: Path,
    use_cache: bool = True,
    model: str = DEFAULT_MODEL,
) -> str:
    if not source_image_path.exists():
        raise ConsultError(f"Source not found: {source_image_path}")
    if not reference_image_path.exists():
        raise ConsultError(f"Reference not found: {reference_image_path}")

    cache_key = _cache_key(source_image_path, reference_image_path, model)
    cache_file = CACHE_DIR / f"{cache_key}.txt"
    if use_cache and cache_file.exists():
        logger.info("consult cache hit: %s + %s",
                    source_image_path.name, reference_image_path.name)
        return cache_file.read_text(encoding="utf-8").strip()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConsultError("ANTHROPIC_API_KEY is not set in environment")

    try:
        import anthropic as _provider
    except ImportError as e:
        raise ConsultError("provider client library not installed") from e

    client = _provider.Anthropic()  # reads ANTHROPIC_API_KEY from env

    source_b64 = base64.standard_b64encode(source_image_path.read_bytes()).decode("ascii")
    ref_b64 = base64.standard_b64encode(reference_image_path.read_bytes()).decode("ascii")
    source_mime = _mime_type(source_image_path)
    ref_mime = _mime_type(reference_image_path)

    logger.info("consult source=%s ref=%s model=%s",
                source_image_path.name, reference_image_path.name, model)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text":
                        "SOURCE (the customer's current photo):"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": source_mime, "data": source_b64,
                    }},
                    {"type": "text", "text": "REFERENCE (the target hairstyle):"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": ref_mime, "data": ref_b64,
                    }},
                    {"type": "text", "text":
                        "Write the edit instruction for the hairstyle change. "
                        "Single paragraph, 80-160 words."},
                ],
            }],
        )
    except Exception as e:
        raise ConsultError(f"Vision call failed: {e}") from e

    if not response.content or not getattr(response.content[0], "text", None):
        raise ConsultError(f"empty response: {response!r}")

    prompt = response.content[0].text.strip().strip('"').strip("'").strip()

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(prompt, encoding="utf-8")
        logger.info("cached consult to %s", cache_file)

    return prompt


def _cache_key(source_path: Path, reference_path: Path, model: str) -> str:
    h = hashlib.sha1()
    h.update(PROMPT_VERSION.encode("utf-8"))
    h.update(b"|")
    h.update(model.encode("utf-8"))
    h.update(b"|")
    h.update(source_path.read_bytes())
    h.update(b"|")
    h.update(reference_path.read_bytes())
    return f"{source_path.stem}__{reference_path.stem}_{h.hexdigest()[:12]}"


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/jpeg"
