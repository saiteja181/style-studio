"""Vision-LM consult: writes adapted inpaint prompts from a source + reference pair.

Reads provider credentials from environment variables only. No keys embedded.
"""
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

SYSTEM_PROMPT = """You are a hair stylist who knows that customers want to see "themselves with a new haircut", NOT a magazine model. The output should look like the SAME casual photo of the same person, just with different hair - not a glamour shot, not a barbershop showroom photo, not a professional portrait.

You will see TWO images:
1. CUSTOMER: the salon customer (often a casual photo - imperfect lighting, everyday quality).
2. REFERENCE: the target hairstyle (often a polished pro photo - this is just the style reference, NOT the look-and-feel target).

Your job: write a prompt for an image inpainting model (FLUX Fill Pro) that will repaint ONLY the hair region of the customer's photo with the target style, while keeping the photo's overall feel (casual, natural, same lighting, same softness) intact.

Write a SHORT prompt (2-4 sentences, no bullet points, no markdown) that:
- Describes the target hairstyle structurally (length on top, length on sides, parting, texture, fade/taper if any) BUT in modest, understated terms.
- Explicitly says "natural everyday look, matches the casual lighting of the source photo, not a studio shoot, not glossy, not polished, not magazine-quality."
- Says hair color stays as the customer's natural deep black/dark hair, no highlights.
- Includes blending language: "natural hairline transition, soft realistic hair texture matching the source photo's grain and softness, looks like a casual snapshot not a barbershop ad."

Critical: do NOT describe the customer's clothes, background, expression, age, body, beard, or anything other than the hair on the head. Avoid words like: glossy, voluminous, sleek, structured, polished, professional, magazine, barbershop showroom. Use words like: natural, subtle, everyday, casual, soft, modest, realistic.

Return ONLY the prompt text. No preamble, no quotation marks, no labels."""


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
                        "CUSTOMER (apply the target hairstyle to this person):"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": source_mime, "data": source_b64,
                    }},
                    {"type": "text", "text": "REFERENCE (the target hairstyle):"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": ref_mime, "data": ref_b64,
                    }},
                    {"type": "text", "text":
                        "Write the inpaint prompt for the hair region only. "
                        "Single paragraph, 3-5 sentences."},
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
