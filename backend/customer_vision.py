"""Optional vision-LM probe to refine the customer profile.

Augments backend.customer_analysis with features that need true visual
understanding: hair texture (straight/wavy/curly/coiled), hairline shape,
and gender if not provided by the caller.

Costs ~$0.01 per call. Cached per photo hash.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "customer_cache"

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 400

VISION_SYSTEM = """You are an experienced hair stylist analyzing a customer photo before consultation.

Look at the photo and report ONLY structural/visual facts about hair and gender. Do not comment on attractiveness, ethnicity, age, or personality.

Return a strict JSON object with exactly these keys:
{
  "gender": "male" | "female" | "unknown",
  "hair_texture": "straight" | "wavy" | "curly" | "coiled" | "unknown",
  "hairline_shape": "rounded" | "m-shape" | "widows-peak" | "square" | "receding" | "unknown",
  "current_hair_length": "very short" | "short" | "medium" | "long" | "very long",
  "current_hair_color": short description (e.g. "deep black", "dark brown with grey at temples"),
  "summary": one sentence factual description of the hair currently visible
}

Output ONLY the JSON object. No preamble, no markdown code fences, no labels."""


class CustomerVisionError(RuntimeError):
    pass


def probe_customer_features(
    selfie_path: Path,
    use_cache: bool = True,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Return a dict with gender / hair_texture / hairline_shape / etc.

    Falls back to {"gender": "unknown", ...} on parse failure.

    Raises CustomerVisionError on transport-level failure.
    """
    if not selfie_path.exists():
        raise CustomerVisionError(f"Selfie not found: {selfie_path}")

    cache_key = _cache_key(selfie_path, model)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise CustomerVisionError("ANTHROPIC_API_KEY is not set")

    try:
        import anthropic as _provider
    except ImportError as e:
        raise CustomerVisionError("anthropic package not installed") from e

    client = _provider.Anthropic()

    img_b64 = base64.standard_b64encode(selfie_path.read_bytes()).decode("ascii")
    mime = _mime(selfie_path)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=VISION_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this customer's hair and gender:"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": mime, "data": img_b64,
                    }},
                ],
            }],
        )
    except Exception as e:
        raise CustomerVisionError(f"Vision call failed: {e}") from e

    if not response.content or not getattr(response.content[0], "text", None):
        raise CustomerVisionError("empty response from vision")

    raw = response.content[0].text.strip()
    # Strip code fences if model returned any.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Customer vision returned non-JSON: %s", raw[:200])
        parsed = {
            "gender": "unknown", "hair_texture": "unknown",
            "hairline_shape": "unknown", "current_hair_length": "unknown",
            "current_hair_color": "unknown", "summary": raw[:200],
        }

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    return parsed


def _cache_key(image_path: Path, model: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"|")
    h.update(image_path.read_bytes())
    return f"{image_path.stem}_{h.hexdigest()[:12]}"


def _mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"): return "image/jpeg"
    if suffix == ".png": return "image/png"
    if suffix == ".webp": return "image/webp"
    return "image/jpeg"
