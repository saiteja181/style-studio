"""Tests for backend.beard_engine."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

SOURCE_MAN = PROJECT_ROOT / "tests" / "selfies" / "young_indian_man.jpg"


def test_load_beard_style_returns_dict_for_known_id():
    from backend.beard_engine import _load_beard_style
    s = _load_beard_style("clean_shaven")
    assert s is not None
    assert s["id"] == "clean_shaven"
    assert s["gender"] == "male"
    assert "prompt_template" in s


def test_load_beard_style_returns_none_for_unknown_id():
    from backend.beard_engine import _load_beard_style
    assert _load_beard_style("does_not_exist_xyz") is None


def test_unknown_beard_style_raises_beard_style_not_found():
    from backend.beard_engine import (
        generate_beard_preview, BeardStyleNotFoundError,
    )
    with pytest.raises(BeardStyleNotFoundError):
        generate_beard_preview(
            source_path=SOURCE_MAN,
            beard_style_id="does_not_exist_xyz",
            customer_profile={"hair_color_rgb": (40, 30, 25), "hair_texture": "unknown"},
            seed=42,
        )


@pytest.mark.skipif(
    not os.getenv("REPLICATE_API_TOKEN"),
    reason="REPLICATE_API_TOKEN not set; skipping live Replicate test",
)
def test_generate_beard_preview_smoke(monkeypatch, tmp_path):
    """Live: clean_shaven on the young Indian man source.  Cost ~$0.04."""
    from backend.beard_engine import generate_beard_preview
    from backend.kontext_engine import PreviewResult
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))

    profile = {"hair_color_rgb": (40, 30, 25), "hair_texture": "unknown"}
    result = generate_beard_preview(
        source_path=SOURCE_MAN, beard_style_id="clean_shaven",
        customer_profile=profile, seed=42,
    )
    assert isinstance(result, PreviewResult)
    assert result.style_id == "clean_shaven"
    assert result.image_url.startswith(("/uploads/", "http"))
    assert result.validator_verdict == "skipped_no_reference"
    assert result.elapsed_ms > 0


def test_beard_style_not_found_subclasses_generation_error():
    from backend.beard_engine import BeardStyleNotFoundError
    from backend.kontext_engine import GenerationError
    assert issubclass(BeardStyleNotFoundError, GenerationError)
