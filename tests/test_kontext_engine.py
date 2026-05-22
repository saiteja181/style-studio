"""Smoke tests for backend.kontext_engine.  These hit the real Replicate API
and cost ~$0.04 per run; skipped automatically when REPLICATE_API_TOKEN is
not configured so CI can still pass without billing keys."""
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

SOURCE_MAN = PROJECT_ROOT / "tests" / "selfies" / "test_random_indian_man.jpg"


@pytest.mark.skipif(
    not os.getenv("REPLICATE_API_TOKEN"),
    reason="REPLICATE_API_TOKEN not set; skipping live Replicate test",
)
def test_call_kontext_returns_url():
    from backend.kontext_engine import _call_kontext

    url = _call_kontext(
        source_path=SOURCE_MAN,
        prompt=("Change ONLY the hairstyle to: a short textured crop with a "
                "fade. Keep the face exactly identical."),
        seed=42,
    )
    assert isinstance(url, str)
    assert url.startswith(("http://", "https://"))
    assert ".png" in url.lower() or ".jpg" in url.lower() or "replicate" in url


def test_preview_result_dataclass_has_required_fields():
    from backend.kontext_engine import PreviewResult
    r = PreviewResult(
        image_url="/uploads/foo.png", style_id="x", style_name="X",
        prompt="p", seed=42, validator_verdict="skipped",
        retries=0, elapsed_ms=1234,
    )
    d = r.to_dict()
    for k in ("image_url", "style_id", "style_name", "prompt", "seed",
              "validator_verdict", "retries", "elapsed_ms"):
        assert k in d
