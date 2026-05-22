"""Tests for backend.style_matcher hairline weighting and overall recommend()."""
from __future__ import annotations

import pytest

from backend.customer_analysis import CustomerProfile


def _profile(**overrides) -> CustomerProfile:
    """Build a CustomerProfile with sensible defaults for tests."""
    defaults = dict(
        face_shape="oval",
        jawline="soft",
        skin_tone_bucket="medium",
        skin_rgb=(180, 140, 110),
        hair_color_rgb=(40, 30, 25),
        hair_color_descriptor="dark brown",
        hair_texture="straight",
        hairline_shape="rounded",
        estimated_gender="male",
        landmark_metrics={},
        notes=[],
    )
    defaults.update(overrides)
    return CustomerProfile(**defaults)


def test_hairline_fit_m_shape_prefers_forward_fringe():
    from backend.style_matcher import _hairline_fit

    score, msg = _hairline_fit(
        "m-shape",
        ["fringe", "forward", "textured", "modern"],
    )
    assert score >= 10, f"expected bonus for M-shape + fringe, got {score}"
    assert "M-shape" in msg or "recession" in msg


def test_hairline_fit_m_shape_penalises_slick_back():
    from backend.style_matcher import _hairline_fit
    score, msg = _hairline_fit(
        "m-shape",
        ["slick-back", "pompadour", "swept up"],
    )
    assert score < 0, f"expected penalty for M-shape + slick-back, got {score}"
    assert "exposes" in msg or "visible" in msg.lower()


def test_hairline_fit_widows_peak_frames_with_side_part():
    from backend.style_matcher import _hairline_fit
    score, msg = _hairline_fit(
        "widows-peak",
        ["side-part", "structured", "sleek"],
    )
    assert score >= 10
    assert "widow" in msg.lower() or "framing" in msg.lower() or "frame" in msg.lower()


def test_hairline_fit_rounded_is_neutral():
    from backend.style_matcher import _hairline_fit
    score, _ = _hairline_fit("rounded", ["fade", "structured"])
    # Rounded hairline -> neutral, no bonus, no penalty
    assert 0 <= score <= 8


def test_hairline_fit_unknown_returns_neutral():
    from backend.style_matcher import _hairline_fit
    score, msg = _hairline_fit("unknown", ["fade", "structured"])
    assert score == 8
    assert msg == ""


def test_recommend_styles_prefers_fringe_for_m_shape_hairline():
    """End-to-end: an M-shape-hairline customer should see fringe-style
    cuts ranked above pompadour/slick-back."""
    from backend.style_matcher import recommend_styles

    profile = _profile(hairline_shape="m-shape", estimated_gender="male")
    recs = recommend_styles(profile=profile, top_n=10, gender_filter="male")
    ids = [r.style_id for r in recs]

    # mens_korean_fringe should be in the top half; mens_pompadour should
    # NOT lead.  Both must be in the catalogue (verified in sub-project 1.5).
    if "mens_korean_fringe" in ids and "mens_pompadour" in ids:
        fringe_rank = ids.index("mens_korean_fringe")
        pompadour_rank = ids.index("mens_pompadour")
        assert fringe_rank < pompadour_rank, (
            f"M-shape customer should see fringe ranked above pompadour; "
            f"got fringe@{fringe_rank} vs pompadour@{pompadour_rank}"
        )
