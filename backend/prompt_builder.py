"""Build the natural-language edit prompt sent to FLUX Kontext.

Layers (lightest to heaviest):
  1. Base description - catalogue `prompt_template` if present, else built
     from style name + traits + length + cultural + gender.
  2. Optional Anthropic expert rewrite when ANTHROPIC_API_KEY is set AND the
     style has a reference photo on disk.  Reuses backend.expert_consult.
  3. Customer hair-colour hex anchor (no bleach / no colour drift).
  4. Texture contrast clause when source texture disagrees with target.
  5. Per-style negative-feature clause (P1.1: stronger anti-forelock for
     short male cuts where Kontext keeps drawing a strand across the eye).
  6. Kontext "Change ONLY the hairstyle to:" wrapper + identity-preservation
     clause.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Forelock-suppression membership is now CATALOG-DRIVEN via the
# per-style `suppress_forelock: true` field in catalogue/styles.json.
# Editors can flag a new style without code changes.  This fixes the
# overfitting risk where adding `mens_caesar` would have silently
# missed the suppression unless the developer remembered to edit two
# files (here AND the catalog upsampling flag).


def build_edit_prompt(
    style: dict,
    customer_profile: dict,
    source_path: Path,
    reference_path: Optional[Path],
    glasses_detected: bool = False,
) -> str:
    """Compose the full Kontext prompt.  See module docstring for layers.

    glasses_detected: when True, prepend a stronger preservation clause
    so Kontext does not remove the customer's glasses when restyling
    the hair around them.  Detected by input_pipeline._detect_glasses.
    """
    base = style.get("prompt_template")
    if not base:
        base = _default_from_style(style)

    if (os.getenv("ANTHROPIC_API_KEY") and reference_path
            and Path(reference_path).exists()):
        try:
            from backend import expert_consult
            base = expert_consult.consult_for_style(
                source_image_path=source_path,
                reference_image_path=reference_path,
            )
        except Exception as e:
            logger.info("expert_consult unavailable, using catalogue base: %s", e)

    colour = _colour_clause(customer_profile)
    texture = _texture_contrast_clause(style, customer_profile)
    forelock = _forelock_clause(style)
    glasses = (
        " The customer is wearing GLASSES; preserve the glasses exactly "
        "as in the source, do not remove them, do not change frame shape "
        "or colour."
    ) if glasses_detected else ""
    return (
        f"Change ONLY the hairstyle to: {base}.{colour}{texture}{forelock}{glasses} "
        "This is a complete hairstyle change. The new hair must look "
        "visibly different from the source hair in shape, length, or styling "
        "- do not preserve the original silhouette. "
        "Avoid asymmetric forelock locks, do not draw a single dramatic "
        "strand falling across the face, no stylegan2 watermark, no extra "
        "hair lock beyond what the style describes. "
        "Keep the face, eyes, expression, beard, eyebrows, glasses, "
        "clothing, hands, and background exactly identical to the original "
        "photo - do not change anything below the eyebrows. Photoreal, same "
        "ambient indoor lighting as the source, no studio lighting, no halo."
    )


def _forelock_clause(style: dict) -> str:
    """Stronger per-style anti-forelock clause for styles where Kontext
    has been observed to insert an unwanted dramatic forelock falling
    across the eye, despite the global anti-forelock language in the
    base prompt.  Triggered by catalog field `suppress_forelock: true`
    so new styles inherit the fix without code edits."""
    if not style.get("suppress_forelock"):
        return ""
    return (
        " The forehead is FULLY VISIBLE and CLEAR. The hair stays "
        "ABOVE the eyebrows on all sides. No hair lock, no fringe, "
        "no strand, no curl crosses the eye, brow, or cheek. The "
        "hairline ends cleanly at the temples."
    )


def _default_from_style(style: dict) -> str:
    name = style.get("name") or "hairstyle"
    traits = style.get("style_traits") or []
    length = style.get("length") or ""
    cultural = style.get("cultural") or []
    gender = style.get("gender") or ""

    parts = [f"A {name}"]
    if length:
        parts.append(f"{length} length")
    if traits:
        parts.append(", ".join(traits[:6]))
    if cultural:
        parts.append(f"{', '.join(cultural[:3])} style")
    if gender:
        parts.append(f"suited to {gender} customer")
    return ", ".join(parts)


def _colour_clause(customer_profile: dict) -> str:
    rgb = customer_profile.get("hair_color_rgb")
    if not rgb or len(rgb) != 3:
        return ""
    try:
        r, g, b = (int(c) for c in rgb)
    except (TypeError, ValueError):
        return ""
    hex_code = f"#{r:02x}{g:02x}{b:02x}"
    return (
        f" Keep the hair colour the customer's natural shade ({hex_code}); "
        "no bleach, no highlights, no colour drift."
    )


def _texture_contrast_clause(style: dict, customer_profile: dict) -> str:
    src = (customer_profile.get("hair_texture") or "").lower()
    if not src or src == "unknown":
        return ""
    compat = [t.lower() for t in style.get("compat_texture", [])]
    traits = [t.lower() for t in style.get("style_traits", [])]
    if src in compat:
        return ""
    target = next((t for t in traits
                   if t in ("straight", "wavy", "curly", "coiled")), None)
    if not target:
        return ""
    return (
        f" The new hair texture is visibly {target}, clearly different from "
        f"the customer's source {src} texture - do not retain the original "
        "hair shape."
    )
