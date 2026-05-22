"""Auto-caption a reference hairstyle photo via Qwen2-VL (vision language model).

Salons add styles by dropping a reference photo into the catalogue. This
module replaces hand-written prompts with model-generated descriptions of
each reference photo. Captions are cached on disk so we only pay once per
reference image per model version.

Why Qwen2-VL over Florence-2:
  Florence-2 gives generic captions ("short spiky cut with side part") for
  every hairstyle. Qwen2-VL accepts a guided prompt so we can ask it
  specifically about length, texture, parting, side detail, and fade pattern.
  Much more useful for downstream inpaint.

Cost: ~$0.01 per Qwen2-VL call, paid once per reference image then cached.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "captions_cache"

CAPTION_MODEL = (
    "lucataco/qwen2-vl-7b-instruct:"
    "bf57361c75677fc33d480d0c5f02926e621b2caa2000347cb74aeae9d2ca07ee"
)
MODEL_TAG = "qwen2vl-7b"   # used in cache key so switching models invalidates cache

# Guided prompt - extracts hair-specific details, not "this is a man with hair".
HAIR_DESCRIPTION_PROMPT = (
    "Look at the person's HAIR ONLY. Describe the hairstyle in 2 to 3 sentences "
    "with these concrete details:\n"
    "1. Length on top (very short, short, medium, long) and approximate centimeters.\n"
    "2. Length on the sides (shaved, very short, medium) and whether there is a fade "
    "or taper pattern.\n"
    "3. Texture (straight, wavy, curly, coiled).\n"
    "4. Parting if any (side part, middle part, no part).\n"
    "5. Top styling (slicked back, swept forward, pompadour height, spiky, flat).\n"
    "6. Hair color (jet black, dark brown, brown, etc).\n"
    "7. Hairline shape if distinctive.\n"
    "Do NOT describe the face, clothes, background, expression, age, or beard. "
    "Only the hair on the head."
)

DEFAULT_MAX_TOKENS = 256


class CaptionError(RuntimeError):
    """Raised for any failure during VLM captioning."""


def describe_hairstyle(
    reference_image_path: Path,
    use_cache: bool = True,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Get a hair-focused description of the reference photo.

    Args:
        reference_image_path: path to a hairstyle reference image.
        use_cache: when True (default), reads/writes a per-image cache file
            so we only pay Replicate once per unique (image, model) pair.
        max_tokens: token budget for the VLM response (default 256).

    Raises:
        CaptionError: any failure.
    """
    if not reference_image_path.exists():
        raise CaptionError(f"Reference image not found: {reference_image_path}")

    cache_key = _cache_key(reference_image_path)
    cache_file = CACHE_DIR / f"{cache_key}.txt"

    if use_cache and cache_file.exists():
        logger.info("caption cache hit for %s", reference_image_path.name)
        return cache_file.read_text(encoding="utf-8").strip()

    if not os.getenv("REPLICATE_API_TOKEN"):
        raise CaptionError("REPLICATE_API_TOKEN is not set in .env")

    try:
        import replicate
    except ImportError as e:
        raise CaptionError("`replicate` package not installed") from e

    logger.info("Qwen2-VL captioning %s", reference_image_path.name)
    try:
        with reference_image_path.open("rb") as f:
            output = replicate.run(CAPTION_MODEL, input={
                "media": f,
                "prompt": HAIR_DESCRIPTION_PROMPT,
                "max_new_tokens": max_tokens,
            })
    except Exception as e:
        raise CaptionError(f"Qwen2-VL call failed: {e}") from e

    caption = _extract_caption_text(output)
    if not caption:
        raise CaptionError(f"could not extract caption from output: {output!r}")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(caption, encoding="utf-8")
        logger.info("cached caption to %s", cache_file)

    return caption


def describe_hair_only(reference_image_path: Path, use_cache: bool = True) -> str:
    """Hair-focused caption suitable for use directly as an inpaint prompt.

    Qwen2-VL already returns a hair-focused description because of our guided
    prompt, so this is currently a thin wrapper. Kept as a separate function
    so callers don't depend on the captioner-specific behaviour.
    """
    return describe_hairstyle(reference_image_path, use_cache=use_cache)


# ---- helpers ----

def _cache_key(image_path: Path) -> str:
    """Cache key includes image content hash AND model tag so re-saved files
    or model swaps invalidate the cache automatically."""
    h = hashlib.sha1()
    h.update(MODEL_TAG.encode("utf-8"))
    h.update(b"|")
    h.update(image_path.read_bytes())
    return f"{image_path.stem}_{MODEL_TAG}_{h.hexdigest()[:10]}"


def _extract_caption_text(output) -> Optional[str]:
    """Qwen2-VL on Replicate returns either a single string OR a list of token
    strings (streaming). Concatenate everything cleanly."""
    if output is None:
        return None
    if isinstance(output, str):
        return output.strip() or None
    if isinstance(output, list):
        # Most common: list of string chunks. Concatenate.
        chunks = []
        for item in output:
            if isinstance(item, str):
                chunks.append(item)
            else:
                # FileOutput-like with .url? Skip.
                pass
        joined = "".join(chunks).strip()
        return joined or None
    if isinstance(output, dict):
        for key in ("output", "text", "caption", "result", "response"):
            if key in output:
                inner = output[key]
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()
                if isinstance(inner, list):
                    chunks = [x for x in inner if isinstance(x, str)]
                    joined = "".join(chunks).strip()
                    if joined:
                        return joined
        # Last resort: stringify any long-enough string value.
        for v in output.values():
            if isinstance(v, str) and len(v.strip()) > 20:
                return v.strip()
    return None
