"""Tests for backend.input_pipeline._resize_for_flux.

Regression coverage for the aspect-ratio distortion bug found in
sub-project 1.5's Opus code review: 500x750 portraits were being upscaled
into 768x768 squares because the MIN_EDGE floor was applied per-dimension
instead of preserving aspect ratio.
"""
from __future__ import annotations

import pytest
from PIL import Image


def _solid_image(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), (128, 128, 128))


def test_resize_preserves_aspect_ratio_for_small_portrait():
    """A 500x750 portrait (Pexels-typical) must keep its 2:3 aspect ratio
    after upscaling to the FLUX-friendly minimum edge size.

    Before the Opus-review fix: 500x750 -> 768x768 (face stretched ~16% wide).
    After: 500x750 -> 768x1152 (true 2:3 preserved).
    """
    from backend.input_pipeline import _resize_for_flux

    src = _solid_image(500, 750)
    out = _resize_for_flux(src)
    w, h = out.size

    src_ratio = 500 / 750
    out_ratio = w / h
    assert abs(out_ratio - src_ratio) < 0.04, (
        f"aspect ratio drifted from {src_ratio:.3f} to {out_ratio:.3f} "
        f"({w}x{h}); face geometry would be visibly distorted"
    )


def test_resize_preserves_aspect_ratio_for_small_landscape():
    """A small 750x500 landscape (mirror of the portrait case) must keep
    its 3:2 ratio after upscaling."""
    from backend.input_pipeline import _resize_for_flux

    src = _solid_image(750, 500)
    out = _resize_for_flux(src)
    w, h = out.size

    src_ratio = 750 / 500
    out_ratio = w / h
    assert abs(out_ratio - src_ratio) < 0.04, (
        f"aspect ratio drifted from {src_ratio:.3f} to {out_ratio:.3f} "
        f"({w}x{h})"
    )


def test_resize_preserves_aspect_ratio_for_large_portrait():
    """A 4032x3024 phone shot (4:3 landscape) must downscale to fit the
    LONG_EDGE_TARGET while keeping its 4:3 ratio."""
    from backend.input_pipeline import _resize_for_flux

    src = _solid_image(4032, 3024)
    out = _resize_for_flux(src)
    w, h = out.size

    src_ratio = 4032 / 3024
    out_ratio = w / h
    assert abs(out_ratio - src_ratio) < 0.04, (
        f"aspect ratio drifted from {src_ratio:.3f} to {out_ratio:.3f} "
        f"({w}x{h})"
    )
    assert max(w, h) <= 1536 + 16, f"long edge {max(w, h)} > LONG_EDGE_TARGET"


def test_resize_short_edge_meets_min_edge_for_small_portrait():
    """After fix, a small portrait's short edge should at minimum reach
    MIN_EDGE so FLUX has enough resolution to work with."""
    from backend.input_pipeline import _resize_for_flux, MIN_EDGE

    src = _solid_image(500, 750)
    out = _resize_for_flux(src)
    w, h = out.size

    assert min(w, h) >= MIN_EDGE - 16, (
        f"short edge {min(w, h)} dropped below MIN_EDGE={MIN_EDGE}"
    )


def test_resize_returns_multiples_of_32():
    """Both dimensions must be multiples of DIM_MULTIPLE (32) so FLUX
    accepts them without internal padding."""
    from backend.input_pipeline import _resize_for_flux, DIM_MULTIPLE

    for src_dims in [(500, 750), (750, 500), (1024, 1024), (4032, 3024)]:
        src = _solid_image(*src_dims)
        out = _resize_for_flux(src)
        w, h = out.size
        assert w % DIM_MULTIPLE == 0, f"width {w} not divisible by {DIM_MULTIPLE} for source {src_dims}"
        assert h % DIM_MULTIPLE == 0, f"height {h} not divisible by {DIM_MULTIPLE} for source {src_dims}"
