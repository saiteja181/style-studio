"""Skin-tone-conditioned hair-colour palette suggestion.

Recommends 4 hair-colour options for the customer's detected skin-tone
bucket.  Suggestions are not personalised to face shape or current hair
colour - they're skin-tone-driven only.  Salon stylist can refine on top.

Lookup table built from standard hair-colour theory:
  - Warm skin (wheat, medium, dusky) -> warm browns/chestnuts/coppers
  - Cool/fair skin -> ash browns, soft blondes
  - Dark skin -> rich dark browns, blacks, deep wine/burgundy highlights

India tilt: 'Natural dark brown' or 'Natural black' is included as the
default-safe option in every bucket, since most Indian customers prefer
to stay close to their natural shade.
"""
from __future__ import annotations

from typing import Optional


# Per-bucket lookup.  Order matters: first entry is the safe default
# (closest to natural shade for that skin tone).
PALETTES: dict[str, list[dict]] = {
    "fair": [
        {
            "name": "Natural dark brown",
            "hex": "#3C2A20",
            "sub_tone": "warm brown",
            "why": "Safe default; close to most fair-skin Indian customers' natural shade.",
        },
        {
            "name": "Ash brown",
            "hex": "#5A4A42",
            "sub_tone": "cool brown",
            "why": "Cool tone pairs with fair skin's pink/yellow undertones without overpowering.",
        },
        {
            "name": "Honey blonde",
            "hex": "#A6803C",
            "sub_tone": "warm blonde",
            "why": "Adds warmth and dimension; works best with fair skin that already has gold undertones.",
        },
        {
            "name": "Caramel highlights",
            "hex": "#8B5A30",
            "sub_tone": "warm highlight",
            "why": "Soft contrast against fair skin; lifts the face without a dramatic base change.",
        },
    ],
    "wheat": [
        {
            "name": "Natural dark brown",
            "hex": "#3A2818",
            "sub_tone": "warm brown",
            "why": "Safe default; matches most wheat-skin Indian customers' natural shade.",
        },
        {
            "name": "Warm chestnut",
            "hex": "#5C3A1E",
            "sub_tone": "warm brown",
            "why": "Pairs with the warm undertones in wheat skin; brightens the face without contrast.",
        },
        {
            "name": "Mahogany",
            "hex": "#4A1F1C",
            "sub_tone": "warm red-brown",
            "why": "Adds richness and a subtle red shimmer that flatters wheat undertones in daylight.",
        },
        {
            "name": "Caramel highlights",
            "hex": "#8B5A30",
            "sub_tone": "warm highlight",
            "why": "Lifts a dark base with face-framing highlights without a full colour change.",
        },
    ],
    "medium": [
        {
            "name": "Natural dark brown",
            "hex": "#2F1F14",
            "sub_tone": "warm brown",
            "why": "Safe default; matches most medium-skin Indian customers' natural shade.",
        },
        {
            "name": "Warm chestnut",
            "hex": "#5C3A1E",
            "sub_tone": "warm brown",
            "why": "Warms up medium skin; the universal salon recommendation for this bucket.",
        },
        {
            "name": "Copper highlights",
            "hex": "#9A5028",
            "sub_tone": "warm copper",
            "why": "Bold contrast that flatters medium skin's golden undertones; popular for parties.",
        },
        {
            "name": "Cocoa brown",
            "hex": "#3D2616",
            "sub_tone": "neutral brown",
            "why": "Slightly cooler than chestnut; reads professional and modern in office light.",
        },
    ],
    "dusky": [
        {
            "name": "Natural black",
            "hex": "#1A1010",
            "sub_tone": "neutral black",
            "why": "Safe default; matches most dusky-skin Indian customers' natural shade.",
        },
        {
            "name": "Warm chestnut highlights",
            "hex": "#5C3A1E",
            "sub_tone": "warm highlight",
            "why": "Subtle warmth around the face; doesn't strip the rich base colour.",
        },
        {
            "name": "Burgundy / wine",
            "hex": "#3E1218",
            "sub_tone": "cool deep red",
            "why": "Dramatic depth that flatters dusky skin; popular for festive occasions and weddings.",
        },
        {
            "name": "Plum brown",
            "hex": "#3A1A26",
            "sub_tone": "cool red-brown",
            "why": "Modern alternative to plain black; reads luxurious in indoor lighting.",
        },
    ],
    "dark": [
        {
            "name": "Natural black",
            "hex": "#0E0808",
            "sub_tone": "neutral black",
            "why": "Safe default; matches most dark-skin Indian customers' natural shade.",
        },
        {
            "name": "Warm dark brown",
            "hex": "#2A1810",
            "sub_tone": "warm brown",
            "why": "Slight warmth shift without losing depth; very natural and salon-grade.",
        },
        {
            "name": "Burgundy",
            "hex": "#2E0810",
            "sub_tone": "deep red",
            "why": "Bold, flattering contrast against dark skin; eye-catching for events.",
        },
        {
            "name": "Auburn highlights",
            "hex": "#7A2E18",
            "sub_tone": "warm red highlight",
            "why": "Face-framing auburn streaks that pop against dark skin; festive and modern.",
        },
    ],
}

# When skin_tone_bucket is unknown/missing, fall back to medium - the
# centre-of-distribution for Indian customers and the safest default.
_FALLBACK_BUCKET = "medium"


def recommend_palette(skin_tone_bucket: Optional[str]) -> list[dict]:
    """Return 4 hair-colour suggestions for the customer's skin tone.

    Args:
        skin_tone_bucket: one of fair/wheat/medium/dusky/dark, or any other
            string (treated as unknown).

    Returns:
        A list of 4 dicts, each with name, hex, sub_tone, why.
        Always returns 4 entries; never empty.
    """
    bucket = (skin_tone_bucket or "").strip().lower()
    if bucket not in PALETTES:
        bucket = _FALLBACK_BUCKET
    return [dict(entry) for entry in PALETTES[bucket]]
