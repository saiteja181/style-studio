"""Detect head coverings (turban, hijab, ghoonghat, cap) in a customer photo.

Why this exists: FLUX Kontext correctly REMOVES head coverings when asked
to render a new hairstyle - this is the right behaviour for a hairstyle
preview, but for a Sikh customer wearing a dastar, a Muslim woman wearing
a hijab, or a Hindu woman wearing a ghoonghat, the removal can be
religiously offensive if shown without confirmation.

This module surfaces a warning to salon staff so they can confirm with
the customer before generating.  It is a SOFT gate: detection adds a
warning to PreflightReport.warnings but does not block the upload.

Cost: ~$0.005 per uncached call via Claude Haiku 4.5 vision.  Skipped
entirely when ANTHROPIC_API_KEY is not set.  Cached per image-bytes hash.
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

CACHE_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "head_covering_cache"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 150

SYSTEM_PROMPT = """You are a vision classifier.  Given a customer's photo, your job is to detect any head covering visible on the head.

Head coverings include:
- "turban" (Sikh dastar, Sufi pagdi, decorative turban) - a wrapped cloth on the head
- "hijab" (Muslim headscarf covering the hair and neck)
- "ghoonghat" (Hindu veil pulled over the hair from a sari pallu)
- "cap_hat" (any baseball cap, beanie, fedora, religious skullcap, etc.)
- "other" (anything else covering the hair - hair-band, bandana, etc.)
- "none" (no covering; hair is visible)

Return ONLY a strict JSON object:
{
  "detected": true | false,
  "covering_type": "turban" | "hijab" | "ghoonghat" | "cap_hat" | "other" | "none",
  "confidence": "high" | "medium" | "low" | "none"
}

Rules:
- If hair is fully visible and no covering is on the head, return detected=false, covering_type="none", confidence="none".
- If a covering is visible but partially obscured or ambiguous, use confidence="low" or "medium".
- "turban" is reserved for wrapped-cloth structures.  A baseball cap is NOT a turban.
- Output ONLY the JSON.  No preamble, no markdown."""

# Soft-warning copy per covering type.  These are shown to salon staff in
# the PreflightReport warnings list.
WARNING_COPY = {
    "turban": (
        "A turban (dastar) is visible in this photo.  For Sikh customers, "
        "removing the dastar is religiously sensitive - please confirm with "
        "the customer that they want a preview showing them without it before "
        "generating."
    ),
    "hijab": (
        "A hijab is visible in this photo.  For practising Muslim customers, "
        "showing a preview without the hijab may not be appropriate - please "
        "confirm with the customer before generating."
    ),
    "ghoonghat": (
        "A ghoonghat (veil) is visible in this photo.  Confirm with the "
        "customer that they want a preview showing them without it."
    ),
    "cap_hat": (
        "A cap or hat is visible in this photo.  The preview will show the "
        "customer without it - confirm this is acceptable."
    ),
    "other": (
        "A head covering is visible in this photo.  The preview will show "
        "the customer without it - confirm this is acceptable."
    ),
}


def detect_head_covering(image_path: Path, use_cache: bool = True) -> dict:
    """Detect head coverings in image_path.  Always returns a dict with the
    keys: detected, covering_type, confidence, message.

    When ANTHROPIC_API_KEY is not set, returns a no-op dict (no detection,
    no call, no message).  Salon staff sees no warning.
    """
    no_op = {
        "detected": False, "covering_type": "none",
        "confidence": "none", "message": "",
    }
    if not os.getenv("ANTHROPIC_API_KEY"):
        return no_op

    if not Path(image_path).exists():
        logger.warning("head_covering: image not found at %s", image_path)
        return no_op

    cache_key = _cache_key(image_path)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        import anthropic
    except ImportError:
        logger.warning("head_covering: anthropic SDK not installed; skipping")
        return no_op

    try:
        img_bytes = Path(image_path).read_bytes()
        img_b64 = base64.standard_b64encode(img_bytes).decode("ascii")
        mime = _mime(Path(image_path))

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify the head covering in this photo:"},
                    {"type": "image", "source": {"type": "base64",
                        "media_type": mime, "data": img_b64}},
                    {"type": "text", "text": "Return ONLY the JSON object."},
                ],
            }],
        )
    except Exception as e:
        logger.warning("head_covering: Anthropic call failed: %s", e)
        return no_op

    if not response.content or not getattr(response.content[0], "text", None):
        return no_op
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("head_covering: non-JSON response: %s", raw[:120])
        return no_op

    covering_type = parsed.get("covering_type", "none")
    detected = bool(parsed.get("detected", False))
    confidence = parsed.get("confidence", "none")

    if detected and covering_type in WARNING_COPY:
        message = WARNING_COPY[covering_type]
    else:
        message = ""

    result = {
        "detected": detected,
        "covering_type": covering_type,
        "confidence": confidence,
        "message": message,
    }

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("head_covering: cache write failed: %s", e)
    return result


def _cache_key(image_path: Path) -> str:
    h = hashlib.sha1()
    h.update(image_path.read_bytes())
    return f"hc_{h.hexdigest()[:14]}"


def _mime(path: Path) -> str:
    s = path.suffix.lower()
    if s in (".jpg", ".jpeg"): return "image/jpeg"
    if s == ".png": return "image/png"
    if s == ".webp": return "image/webp"
    return "image/jpeg"
