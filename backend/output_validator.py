"""Post-generation quality validator using Claude vision.

After FLUX produces a hairstyle preview, ask Claude to compare the source
photo, the reference style, and the generated output. Return a pass/fail
verdict + reasoning. If FAIL, the caller retries with a different seed.

This is what makes the system trustable for production salon use: the
customer never sees a preview that doesn't actually show the requested
style.

Cost: ~$0.005 per validation call. Retries cost a full generation ($0.10)
but only fire when needed (typically 10-20% of generations).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "validation_cache"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"   # cheap + fast for QC
MAX_TOKENS = 200

VALIDATOR_SYSTEM_VERSION = "v1"

VALIDATOR_SYSTEM = """You are a strict QA reviewer for an AI hairstyle preview tool.

You will see THREE images:
1. SOURCE: the customer's original photo.
2. REFERENCE: the target hairstyle the customer wants.
3. GENERATED: an AI-produced preview showing the customer with the target style.

Your job is to decide: does the GENERATED image actually show the customer with the target hairstyle, while keeping their face identity intact?

Criteria:
- FACE IDENTITY: the GENERATED face must clearly be the same person as SOURCE (same eye spacing, nose shape, jaw, chin, expression).
- STYLE TRANSFER: the GENERATED hair must visibly resemble the target style from REFERENCE in its key structural features (length, texture, parting, fade, volume direction).
- IT IS OK IF the GENERATED hair is a slightly modest version of the reference; not OK if it just looks like the customer's original hair barely changed.
- SCENE PRESERVATION: clothes and background should still match SOURCE (not REFERENCE).

Return ONLY a strict JSON object:
{
  "verdict": "pass" | "fail",
  "identity_match": "strong" | "weak" | "lost",
  "style_match": "strong" | "modest" | "missing",
  "scene_preserved": true | false,
  "one_line_reason": "short sentence on what's right/wrong"
}

PASS only if identity_match is "strong" AND style_match is at least "modest" AND scene_preserved is true. Otherwise FAIL.

Output ONLY the JSON. No preamble, no markdown."""


class ValidationError(RuntimeError):
    pass


def validate_generation(
    source_path: Path,
    reference_path: Path,
    generated_url: str,
    use_cache: bool = True,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Compare source/reference/generated and return pass/fail verdict.

    Returns dict with keys: verdict, identity_match, style_match,
    scene_preserved, one_line_reason.
    """
    cache_key = _cache_key(source_path, reference_path, generated_url, model)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValidationError("ANTHROPIC_API_KEY not set")

    try:
        import anthropic as _provider
    except ImportError as e:
        raise ValidationError("anthropic package not installed") from e

    # Download generated image to bytes for base64 encoding
    try:
        with urllib.request.urlopen(generated_url, timeout=30) as resp:
            gen_bytes = resp.read()
    except Exception as e:
        raise ValidationError(f"could not download generated image: {e}") from e

    client = _provider.Anthropic()

    src_b64 = base64.standard_b64encode(source_path.read_bytes()).decode("ascii")
    ref_b64 = base64.standard_b64encode(reference_path.read_bytes()).decode("ascii")
    gen_b64 = base64.standard_b64encode(gen_bytes).decode("ascii")
    src_mime = _mime(source_path)
    ref_mime = _mime(reference_path)
    gen_mime = "image/png" if generated_url.lower().endswith(".png") else "image/jpeg"

    logger.info("validator: source=%s ref=%s", source_path.name, reference_path.name)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=VALIDATOR_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "SOURCE (the customer):"},
                    {"type": "image", "source": {"type": "base64",
                        "media_type": src_mime, "data": src_b64}},
                    {"type": "text", "text": "REFERENCE (target hairstyle):"},
                    {"type": "image", "source": {"type": "base64",
                        "media_type": ref_mime, "data": ref_b64}},
                    {"type": "text", "text": "GENERATED (the AI preview):"},
                    {"type": "image", "source": {"type": "base64",
                        "media_type": gen_mime, "data": gen_b64}},
                    {"type": "text", "text": "Return ONLY the JSON verdict."},
                ],
            }],
        )
    except Exception as e:
        raise ValidationError(f"validation call failed: {e}") from e

    if not response.content or not getattr(response.content[0], "text", None):
        raise ValidationError("empty validator response")

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("validator returned non-JSON: %s", raw[:200])
        # IMPORTANT: do NOT fail-open here.  Returning "pass" on a parser
        # error silently ships bad previews to the customer.  Mark uncertain
        # so the caller can decide (retry once, or surface a warning to the
        # salon staff).  Skip cache so a flaky single call doesn't poison
        # future lookups.
        return {
            "verdict": "uncertain",
            "identity_match": "unknown", "style_match": "unknown",
            "scene_preserved": True,
            "one_line_reason": f"could not parse validator output: {raw[:100]}",
        }

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    return parsed


def _cache_key(source: Path, reference: Path, gen_url: str, model: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"|")
    h.update(VALIDATOR_SYSTEM_VERSION.encode("utf-8"))
    h.update(b"|")
    h.update(source.read_bytes())
    h.update(b"|")
    h.update(reference.read_bytes())
    h.update(b"|")
    h.update(gen_url.encode("utf-8"))
    return f"validate_{h.hexdigest()[:14]}"


def _mime(path: Path) -> str:
    s = path.suffix.lower()
    if s in (".jpg", ".jpeg"): return "image/jpeg"
    if s == ".png": return "image/png"
    if s == ".webp": return "image/webp"
    return "image/jpeg"
