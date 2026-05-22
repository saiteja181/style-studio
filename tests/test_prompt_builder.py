"""Tests for backend.prompt_builder."""
from __future__ import annotations


def test_default_from_style_includes_name_length_traits_culture_gender():
    from backend.prompt_builder import _default_from_style
    style = {
        "id": "mens_pompadour",
        "name": "Modern Pompadour with Skin Fade",
        "gender": "male",
        "length": "medium",
        "style_traits": ["pompadour", "fade", "structured", "height on top"],
        "cultural": ["modern"],
    }
    p = _default_from_style(style)
    assert "Modern Pompadour" in p
    assert "medium length" in p
    assert "pompadour" in p
    assert "modern" in p
    assert "male" in p


def test_default_from_style_handles_missing_fields():
    from backend.prompt_builder import _default_from_style
    style = {"id": "mystery", "name": "Mystery Cut"}
    p = _default_from_style(style)
    assert "Mystery Cut" in p
    assert isinstance(p, str)


def test_colour_clause_uses_hex_when_rgb_supplied():
    from backend.prompt_builder import _colour_clause
    c = _colour_clause({"hair_color_rgb": (50, 40, 38)})
    assert "#322826" in c
    assert "natural" in c.lower()


def test_colour_clause_blank_when_missing():
    from backend.prompt_builder import _colour_clause
    assert _colour_clause({}) == ""
    assert _colour_clause({"hair_color_rgb": None}) == ""


def test_texture_contrast_fires_when_source_disagrees_with_target():
    from backend.prompt_builder import _texture_contrast_clause
    style = {
        "compat_texture": ["straight"],
        "style_traits": ["straight", "sleek", "modern"],
    }
    profile = {"hair_texture": "curly"}
    c = _texture_contrast_clause(style, profile)
    assert "straight" in c
    assert "curly" in c


def test_texture_contrast_quiet_when_unknown():
    from backend.prompt_builder import _texture_contrast_clause
    style = {"compat_texture": ["straight"], "style_traits": ["straight"]}
    profile = {"hair_texture": "unknown"}
    assert _texture_contrast_clause(style, profile) == ""


def test_build_edit_prompt_includes_imperative_clause():
    """The output of build_edit_prompt must contain the imperative clause that
    pushes Kontext to commit to a real hair change instead of editing
    conservatively.  This was the lever that unblocked men's style
    differentiation in sub-project 1.5."""
    from backend.prompt_builder import build_edit_prompt
    from pathlib import Path

    style = {"name": "Test Cut", "prompt_template": "a short crop"}
    profile = {"hair_color_rgb": (40, 30, 25), "hair_texture": "unknown"}
    out = build_edit_prompt(
        style=style, customer_profile=profile,
        source_path=Path("/tmp/nope.jpg"),  # not opened; expert_consult skipped
        reference_path=None,
    )
    assert "visibly different from the source" in out, (
        f"missing imperative clause; got: {out!r}"
    )
    assert "Keep the face" in out, "identity-preservation clause must remain"
    assert "Change ONLY the hairstyle to:" in out, "Kontext wrapper must remain"
