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


@pytest.mark.skipif(
    not os.getenv("REPLICATE_API_TOKEN"),
    reason="REPLICATE_API_TOKEN not set; skipping live test",
)
def test_generate_preview_end_to_end(tmp_path, monkeypatch):
    """Live end-to-end on the Indian-male source + textured_crop style.
    Cost: ~$0.04 (Kontext) + ~$0.006 (validator, if Anthropic configured)."""
    from backend.kontext_engine import generate_preview, PreviewResult
    from backend.customer_analysis import analyze_customer

    # Use a private uploads dir so the test doesn't pollute /uploads/.
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))

    profile = analyze_customer(
        selfie_path=SOURCE_MAN, use_vision_lm=False,
    ).to_dict()

    result = generate_preview(
        source_path=SOURCE_MAN,
        style_id="mens_textured_crop",
        customer_profile=profile,
        seed=42,
        max_retries=1,
    )
    assert isinstance(result, PreviewResult)
    assert result.image_url.startswith(("/uploads/", "http"))
    assert result.style_id == "mens_textured_crop"
    assert result.retries in (0, 1)
    assert result.validator_verdict in ("pass", "fail", "uncertain", "skipped")
    assert result.elapsed_ms > 0


@pytest.mark.skipif(
    not os.getenv("REPLICATE_API_TOKEN"),
    reason="REPLICATE_API_TOKEN not set; skipping live test",
)
def test_generate_route_returns_preview(tmp_path, monkeypatch):
    """The /generate route accepts no `mode` parameter and returns a
    PreviewResult dict shape."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))
    from backend.main import app
    client = TestClient(app)

    with SOURCE_MAN.open("rb") as f:
        resp = client.post(
            "/generate",
            files={"image": ("man.jpg", f, "image/jpeg")},
            data={"style_id": "mens_textured_crop", "seed": "42"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for k in ("image_url", "style_id", "validator_verdict", "elapsed_ms"):
        assert k in body, f"missing key {k} in {body!r}"
    assert body["style_id"] == "mens_textured_crop"
