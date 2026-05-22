"""Rule-based + heuristic style recommendation engine.

Given a CustomerProfile and the catalogue, returns a ranked list of styles
with a numeric suit-score and a one-sentence reasoning for each.

The rules are codified from standard hairstylist suitability guidance,
adapted for Indian salon context (most customers: deep-black wavy-to-coarse
hair, oval/round face dominant, mix of traditional and modern preferences).

Scoring (0-115, clamped to 0-100):
  face_shape match (0-40)
  jawline match    (0-20)
  hairline match   (-8 to 15)
  texture compat   (0-15)
  occasion match   (0-10)
  length compat    (0-10)
  popularity boost (0-5)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from backend.customer_analysis import CustomerProfile

logger = logging.getLogger(__name__)

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "catalogue" / "styles.json"

# ---- Stylist domain knowledge ----

# Face shape preferences for styles. Each style declares which face shapes
# it suits well. We boost / penalize based on overlap.
FACE_SHAPE_PRIORITIES = {
    "oval":   ["oval", "any"],
    "round":  ["round", "oval"],
    "square": ["square", "oval"],
    "heart":  ["heart", "oval"],
    "long":   ["long", "oval"],
    "diamond": ["diamond", "oval"],
}

# For a soft jawline, layered/textured styles flatter more. For sharp,
# bold and structured cuts work. For weak/rounded, height on top draws up.
JAWLINE_LIKES = {
    "sharp":   ["structured", "fade", "blunt", "side-part", "undercut"],
    "soft":    ["textured", "layered", "wavy", "fringe", "swept"],
    "square":  ["soft layers", "side-swept", "wavy", "fringe"],
    "rounded": ["height on top", "pompadour", "swept up", "long swept-back"],
}

# Avoid lists per face shape (negative scoring).
FACE_SHAPE_AVOID = {
    "round":   ["bowl-cut", "buzz all-over", "blunt-fringe-across-forehead"],
    "square":  ["sharp-undercut-hard-line", "blunt-fringe"],
    "heart":   ["tall-pompadour", "slick-back-no-volume"],
    "long":    ["tall-pompadour", "ponytail-bare-forehead"],
    "diamond": ["super-short-crop"],
    "oval":    [],
    "round_oblong": [],
}

# Hair texture compatibility - styles that need straight hair won't suit
# coiled customers without major work.
TEXTURE_COMPAT = {
    "straight": {"straight", "wavy", "any"},
    "wavy":     {"wavy", "straight", "any"},
    "curly":    {"curly", "wavy", "coiled", "any"},
    "coiled":   {"coiled", "curly", "any"},
    "unknown":  {"straight", "wavy", "curly", "coiled", "any"},
}


@dataclass
class StyleRecommendation:
    style_id: str
    style_name: str
    suit_score: int
    reasoning: str
    style_metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def recommend_styles(
    profile: CustomerProfile,
    top_n: int = 5,
    occasion: Optional[str] = None,
    gender_filter: Optional[str] = None,
) -> list[StyleRecommendation]:
    """Return top-N styles ranked by suit-score.

    Args:
        profile: CustomerProfile from backend.customer_analysis.analyze_customer.
        top_n: how many to return.
        occasion: optional filter ("daily" / "bridal" / "party" / "professional").
        gender_filter: overrides profile.estimated_gender. If both None and
            estimated_gender is "unknown", returns mixed.
    """
    catalogue = _load_catalogue()
    gender = (gender_filter or profile.estimated_gender or "unknown").lower()

    candidates = []
    for style in catalogue:
        # Gender filter
        style_gender = (style.get("gender") or "unisex").lower()
        if gender != "unknown" and style_gender not in (gender, "unisex"):
            continue

        # Occasion filter (if requested)
        if occasion:
            occasions = [o.lower() for o in style.get("occasion", [])]
            if occasion.lower() not in occasions:
                continue

        score, reasoning = _score_style(style, profile)
        candidates.append(StyleRecommendation(
            style_id=style["id"],
            style_name=style["name"],
            suit_score=score,
            reasoning=reasoning,
            style_metadata=style,
        ))

    candidates.sort(key=lambda r: r.suit_score, reverse=True)
    return candidates[:top_n]


# ---- scoring ----

def _score_style(style: dict, profile: CustomerProfile) -> tuple[int, str]:
    score = 0
    reasons_pro: list = []     # positive sentences
    reasons_con: list = []     # caveats

    suits = [s.lower() for s in style.get("suits_face", [])]
    fs = profile.face_shape
    style_traits = [t.lower() for t in style.get("style_traits", [])]
    traits_text = " ".join(style_traits)

    # Face shape match (0-40)
    if fs in suits:
        score += 40
        reasons_pro.append(f"flatters your {fs} face shape")
    elif fs == "oval":
        score += 30
        reasons_pro.append("works naturally with your oval face")
    elif "any" in suits or not suits:
        score += 25
    else:
        score += 10
        reasons_con.append(f"not the most flattering for {fs} faces")

    # Face shape penalties
    avoids = [a.lower() for a in style.get("avoid_for", [])]
    if fs in avoids:
        score -= 25
        reasons_con.append(f"can emphasize {fs}-face proportions in the wrong way")

    # Jawline match (0-20) with proper reasoning sentences
    jaw_score, jaw_reason = _jawline_fit(profile.jawline, style_traits)
    score += jaw_score
    if jaw_reason:
        reasons_pro.append(jaw_reason)

    # Hairline match (-8 to +15) with reasoning
    hairline_score, hairline_reason = _hairline_fit(profile.hairline_shape, style_traits)
    score += hairline_score
    if hairline_score < 0:
        reasons_con.append(hairline_reason)
    elif hairline_reason:
        reasons_pro.append(hairline_reason)

    # Texture compatibility (0-15)
    style_textures = set(t.lower() for t in style.get("compat_texture", ["any"]))
    customer_compatible = TEXTURE_COMPAT.get(profile.hair_texture, {"any"})
    if "any" in style_textures or style_textures & customer_compatible:
        score += 15
    else:
        score += 3
        reasons_con.append(f"may need extra styling for your {profile.hair_texture} hair")

    # Occasion/length default mid
    score += 8

    # Popularity (0-5)
    score += int(style.get("popularity_boost", 3))

    # Quality tier - prefers styles that reliably produce good transformations
    # on face-only crops (the typical salon kiosk input). Styles whose key
    # feature is behind the head get penalized.
    tier = (style.get("quality_tier") or "medium").lower()
    if tier == "high":
        score += 12
    elif tier == "limited":
        score -= 15
        reasons_con.append("style feature is mostly behind the head, "
                           "harder to show in a face-only preview")

    score = max(0, min(100, score))

    reasoning = _compose_reasoning(style.get("name", "this style"),
                                   reasons_pro, reasons_con, style_traits)
    return score, reasoning


def _jawline_fit(jawline: str, style_traits: list) -> tuple[int, str]:
    """Return (score_contribution, optional_reasoning_sentence)."""
    rules = {
        "sharp": {
            "good_traits": {"structured", "fade", "blunt", "side-part", "undercut", "professional"},
            "good_msg": "the structured shape complements your sharp jawline",
            "bad_msg": "this style is softer than your jawline deserves",
        },
        "soft": {
            "good_traits": {"textured", "layered", "wavy", "fringe", "swept", "soft"},
            "good_msg": "the soft texture frames your jawline naturally",
            "bad_msg": "the hard lines may compete with your softer jawline",
        },
        "square": {
            "good_traits": {"soft", "layered", "swept", "wavy", "fringe", "textured"},
            "good_msg": "the soft layers balance your square jawline",
            "bad_msg": "the angular cut emphasizes the square jaw",
        },
        "rounded": {
            "good_traits": {"height on top", "pompadour", "swept up", "long swept-back",
                            "voluminous", "structured"},
            "good_msg": "the height on top draws the eye upward and slims your rounded jawline",
            "bad_msg": "this style doesn't add the vertical lift your rounded jawline needs",
        },
    }
    rule = rules.get(jawline)
    if not rule:
        return 10, ""
    overlap = sum(1 for t in style_traits if t in rule["good_traits"])
    if overlap >= 2:
        return 20, rule["good_msg"]
    if overlap == 1:
        return 14, rule["good_msg"]
    return 6, ""


def _hairline_fit(hairline: str, style_traits: list) -> tuple[int, str]:
    """Return (score_contribution, optional_reasoning_sentence) for hairline fit.

    Hairlines are particularly important for male customers: an M-shape
    receding hairline looks best with forward-falling fringes that cover
    the recession, while pompadours and slick-backs expose it.  A widow's
    peak frames nicely with side- or centre-parted styles but disappears
    under a full blunt fringe.
    """
    rules = {
        "rounded": {
            "good_traits": set(),   # neutral - works with most styles
            "bad_traits": set(),
            "good_msg": "",
            "bad_msg": "",
        },
        "m-shape": {
            "good_traits": {"fringe", "forward", "swept down", "curtain bangs",
                            "textured", "crop", "korean fringe", "soft"},
            "bad_traits": {"slick-back", "pompadour", "swept up",
                           "exposed hairline", "undercut"},
            "good_msg": "the forward-falling shape masks an M-shape recession naturally",
            "bad_msg": "this style exposes the hairline; an M-shape recession would be visible",
        },
        "widows-peak": {
            "good_traits": {"side-part", "center-part", "swept", "frame", "soft",
                            "layered", "side-swept"},
            "bad_traits": {"blunt fringe", "full fringe", "closed forehead",
                           "blunt-fringe-across-forehead"},
            "good_msg": "the parting frames your widow's peak instead of hiding it",
            "bad_msg": "a blunt fringe flattens the widow's peak's natural character",
        },
        "square": {
            "good_traits": {"soft", "layered", "side-swept", "fringe", "wavy",
                            "textured"},
            "bad_traits": {"harsh", "geometric", "blunt-fringe-across-forehead"},
            "good_msg": "soft layers complement your square hairline without harsh contrast",
            "bad_msg": "a geometric shape exaggerates the square hairline",
        },
    }
    rule = rules.get((hairline or "").lower())
    if rule is None:
        # Unknown / unspecified hairline -> neutral midpoint
        return 8, ""
    # If any "bad" trait overlaps, surface as caveat
    if any(t in rule["bad_traits"] for t in style_traits):
        return -8, rule["bad_msg"]
    overlap = sum(1 for t in style_traits if t in rule["good_traits"])
    if overlap >= 2:
        return 15, rule["good_msg"]
    if overlap == 1:
        return 10, rule["good_msg"]
    return 5, ""   # neutral, no message


def _compose_reasoning(style_name: str, pros: list, cons: list,
                       traits: list) -> str:
    """Produce a real-stylist sentence (not template debris)."""
    if pros:
        head = f"{style_name} {pros[0]}"
        if len(pros) >= 2:
            head += f", and {pros[1]}"
        if cons:
            return head + f" - note: {cons[0]}."
        return head + "."

    # No pros - lead with the signature trait or a generic line
    if traits:
        line = f"{style_name} brings {traits[0]} energy to the cut"
        if cons:
            line += f", but {cons[0]}"
        return line + "."

    return f"{style_name} - a safe modern choice."


def _load_catalogue() -> list[dict]:
    if not CATALOGUE_PATH.exists():
        return []
    with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)
