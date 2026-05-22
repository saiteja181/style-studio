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
    assert result.validator_verdict in (
        "pass", "fail", "uncertain", "skipped",
        "skipped_no_anthropic_key", "skipped_no_reference",
    )
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


def test_generate_route_returns_404_for_unknown_style(tmp_path, monkeypatch):
    """Unknown style_id must return 404 (client error), not 502 (server error).
    This is a $0 test - the catalogue lookup fails before any Replicate call."""
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)

    with SOURCE_MAN.open("rb") as f:
        resp = client.post(
            "/generate",
            files={"image": ("man.jpg", f, "image/jpeg")},
            data={"style_id": "does_not_exist_xyz", "seed": "42"},
        )
    assert resp.status_code == 404, (
        f"expected 404 for unknown style, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "Unknown style" in body.get("detail", ""), body


def test_retries_counter_capped_at_max_retries(monkeypatch, tmp_path):
    """When the validator says 'fail' on every attempt, the result's retries
    field must equal max_retries (not max_retries + 1).  Regression for the
    off-by-one observed in sub-project 1's final code review."""
    import backend.kontext_engine as ke
    import backend.face_composite as fc

    # Force the Anthropic + reference-path branch so the validator runs.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-for-test")
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))

    # Stub out the actual Replicate + composite + validator calls.
    monkeypatch.setattr(
        ke, "_call_kontext",
        lambda source_path, prompt, seed, style=None: "https://example.test/fake.png",
    )
    fake_png = tmp_path / "fake_output.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header bytes
    monkeypatch.setattr(
        fc, "paste_source_face",
        lambda source_path, kontext_output_url_or_path, output_dir, **kw: fake_png,
    )
    monkeypatch.setattr(
        ke, "_validate",
        lambda source_path, reference_path, composited_path: "fail",
    )

    # Pick a real style id that has a reference photo so the validator branch fires.
    profile = {"hair_color_rgb": (40, 30, 25), "hair_texture": "unknown"}
    result = ke.generate_preview(
        source_path=SOURCE_MAN,
        style_id="mens_pompadour",   # has reference_image_path in catalogue
        customer_profile=profile,
        seed=42,
        max_retries=1,
    )
    assert result.retries == 1, (
        f"expected retries=1 after two failing attempts with max_retries=1, "
        f"got {result.retries}"
    )
    assert result.validator_verdict == "fail"


def test_generate_preview_negative_max_retries_does_not_crash(monkeypatch, tmp_path):
    """max_retries=-1 must not raise UnboundLocalError.  Negative values are
    nonsensical but a defensive caller should get retries=0, not a crash."""
    import backend.kontext_engine as ke

    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))

    # The loop body never executes when max_retries=-1.  We don't need any
    # stubs because _call_kontext is never reached.
    profile = {"hair_color_rgb": (40, 30, 25), "hair_texture": "unknown"}
    # Picking a style with no reference so the validator branch is also a
    # no-op; though it doesn't matter since the loop never runs.
    result = ke.generate_preview(
        source_path=SOURCE_MAN,
        style_id="mens_classic_side_part",
        customer_profile=profile,
        seed=42,
        max_retries=-1,
    )
    assert result.retries == 0


def test_style_not_found_error_is_generation_error():
    """StyleNotFoundError must remain a subclass of GenerationError so callers
    using `except GenerationError` continue to catch it.  Insurance against a
    refactor breaking the exception hierarchy."""
    from backend.kontext_engine import StyleNotFoundError, GenerationError
    assert issubclass(StyleNotFoundError, GenerationError)


def test_call_kontext_reads_per_style_upsampling_override(monkeypatch):
    """When a style declares upsampling=false, _call_kontext must send
    prompt_upsampling=False to Replicate.  Defaults to True otherwise."""
    import backend.kontext_engine as ke

    captured = {}

    def fake_run(model_ref, input=None):
        captured["payload"] = input
        return "https://example.test/fake.png"

    monkeypatch.setenv("REPLICATE_API_TOKEN", "fake-token")
    monkeypatch.setattr(ke, "replicate", type("R", (), {"run": staticmethod(fake_run)})())

    # Style with upsampling explicitly off
    ke._call_kontext(
        source_path=SOURCE_MAN, prompt="test prompt", seed=42,
        style={"upsampling": False},
    )
    assert captured["payload"]["prompt_upsampling"] is False

    # Style without upsampling key -> default True
    captured.clear()
    ke._call_kontext(
        source_path=SOURCE_MAN, prompt="test prompt", seed=42,
        style={"name": "no override"},
    )
    assert captured["payload"]["prompt_upsampling"] is True

    # No style argument -> default True
    captured.clear()
    ke._call_kontext(
        source_path=SOURCE_MAN, prompt="test prompt", seed=42,
    )
    assert captured["payload"]["prompt_upsampling"] is True
