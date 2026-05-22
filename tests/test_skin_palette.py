"""Tests for backend.skin_palette."""
from __future__ import annotations

import pytest


def test_palette_returns_four_entries_for_each_supported_bucket():
    from backend.skin_palette import recommend_palette, PALETTES
    for bucket in ("fair", "wheat", "medium", "dusky", "dark"):
        p = recommend_palette(bucket)
        assert len(p) == 4, f"{bucket} returned {len(p)} entries"
        for entry in p:
            assert set(entry.keys()) == {"name", "hex", "sub_tone", "why"}, entry
            assert entry["hex"].startswith("#") and len(entry["hex"]) == 7, entry["hex"]


def test_palette_first_entry_is_a_natural_default():
    """The first entry per bucket must be the natural-shade safe default
    (matches Indian salon convention of starting with the customer's own
    family colour)."""
    from backend.skin_palette import recommend_palette
    for bucket in ("fair", "wheat", "medium"):
        first = recommend_palette(bucket)[0]
        assert "Natural dark brown" in first["name"]
    for bucket in ("dusky", "dark"):
        first = recommend_palette(bucket)[0]
        assert "Natural black" in first["name"]


def test_palette_falls_back_to_medium_for_unknown_bucket():
    from backend.skin_palette import recommend_palette, PALETTES
    assert recommend_palette("unknown_bucket") == PALETTES["medium"]
    assert recommend_palette(None) == PALETTES["medium"]
    assert recommend_palette("") == PALETTES["medium"]
    assert recommend_palette("FAIR") == PALETTES["fair"]   # case-insensitive


def test_palette_returns_independent_copies():
    """Caller must not be able to mutate the module's table."""
    from backend.skin_palette import recommend_palette, PALETTES
    p = recommend_palette("medium")
    p[0]["name"] = "MUTATED"
    assert PALETTES["medium"][0]["name"] != "MUTATED"
