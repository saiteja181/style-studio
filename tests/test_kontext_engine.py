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
    import backend.face_swap as fs

    # Force the Anthropic + reference-path branch so the validator runs.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-for-test")
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))

    # Stub out the actual Replicate + face-swap + validator calls.
    monkeypatch.setattr(
        ke, "_call_kontext",
        lambda source_path, prompt, seed, style=None: "https://example.test/fake.png",
    )
    fake_png = tmp_path / "fake_output.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header bytes
    monkeypatch.setattr(
        fs, "swap_face",
        lambda identity_path, target_url_or_path, output_dir, **kw: fake_png,
    )
    # _validate now returns the full verdict dict, not just the string
    # (post-Phase 1 best-of-N change).
    monkeypatch.setattr(
        ke, "_validate",
        lambda source_path, reference_path, composited_path: {
            "verdict": "fail",
            "identity_match": "weak",
            "style_match": "modest",
            "scene_preserved": True,
            "composite_clean": "obvious_artifacts",
            "one_line_reason": "stubbed fail",
        },
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


def test_generate_preview_negative_max_retries_raises_generation_error(
    monkeypatch, tmp_path,
):
    """max_retries=-1 means zero attempts run.  Post-Phase 1 best-of-N
    surfaces this as a GenerationError instead of silently returning a
    PreviewResult with no image - which had been a latent bug (caller
    would crash trying to use the empty image_url)."""
    import backend.kontext_engine as ke
    from backend.kontext_engine import GenerationError

    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))

    profile = {"hair_color_rgb": (40, 30, 25), "hair_texture": "unknown"}
    with pytest.raises(GenerationError):
        ke.generate_preview(
            source_path=SOURCE_MAN,
            style_id="mens_classic_side_part",
            customer_profile=profile,
            seed=42,
            max_retries=-1,
        )


def test_style_not_found_error_is_generation_error():
    """StyleNotFoundError must remain a subclass of GenerationError so callers
    using `except GenerationError` continue to catch it.  Insurance against a
    refactor breaking the exception hierarchy."""
    from backend.kontext_engine import StyleNotFoundError, GenerationError
    assert issubclass(StyleNotFoundError, GenerationError)


# ------------------- Phase 2 / Phase 4: CostLedger tests --------------------


def test_cost_ledger_basic_charge_and_refund():
    """check_and_charge increments spent, refund decrements, breakdown
    tracks per-label totals."""
    from backend.kontext_engine import CostLedger
    ledger = CostLedger(cap_usd=1.00)
    ledger.check_and_charge("kontext", 0.04)
    ledger.check_and_charge("face_swap", 0.01)
    assert abs(ledger.spent_usd - 0.05) < 1e-9
    assert ledger.breakdown == {"kontext": 0.04, "face_swap": 0.01}
    ledger.refund("face_swap", 0.01)
    assert abs(ledger.spent_usd - 0.04) < 1e-9
    assert "face_swap" not in ledger.breakdown
    assert ledger.breakdown == {"kontext": 0.04}


def test_cost_ledger_refund_does_not_go_negative():
    """Double-refund or over-refund must floor at 0, not produce a
    negative spent_usd (which would silently grant unlimited budget)."""
    from backend.kontext_engine import CostLedger
    ledger = CostLedger(cap_usd=1.00)
    ledger.check_and_charge("kontext", 0.04)
    ledger.refund("kontext", 0.04)
    ledger.refund("kontext", 0.04)  # second refund: no charge to refund
    assert ledger.spent_usd == 0.0


def test_cost_ledger_cap_exceeded_raises():
    """check_and_charge must raise CostCapExceeded BEFORE incrementing
    so the ledger doesn't reserve a budget we can't honor."""
    from backend.kontext_engine import CostLedger, CostCapExceeded
    ledger = CostLedger(cap_usd=0.05)
    ledger.check_and_charge("a", 0.04)
    with pytest.raises(CostCapExceeded) as exc:
        ledger.check_and_charge("b", 0.02)  # would push to 0.06, over 0.05
    # Spent did not advance because the exception was raised before increment.
    assert abs(ledger.spent_usd - 0.04) < 1e-9
    # The error message should name the call type so operators can debug.
    assert "b" in str(exc.value)


def test_cost_ledger_reads_env_per_instance(monkeypatch):
    """Earlier the default cap was captured at module load.  After the
    review fix, env changes between instantiations take effect on the
    next CostLedger() construction."""
    from backend.kontext_engine import CostLedger
    monkeypatch.setenv("STYLE_STUDIO_CUSTOMER_COST_CAP_USD", "0.10")
    l1 = CostLedger()
    assert abs(l1.cap_usd - 0.10) < 1e-9
    monkeypatch.setenv("STYLE_STUDIO_CUSTOMER_COST_CAP_USD", "0.50")
    l2 = CostLedger()
    assert abs(l2.cap_usd - 0.50) < 1e-9


def test_cost_cap_exceeded_extends_generation_error():
    """CostCapExceeded MUST stay a GenerationError subclass so existing
    `except GenerationError` callers still catch it, but HTTP handlers
    map it separately to 429."""
    from backend.kontext_engine import CostCapExceeded, GenerationError
    assert issubclass(CostCapExceeded, GenerationError)


def test_preview_cache_hit_skips_kontext_call(monkeypatch, tmp_path):
    """Phase 5.1 regression: identical (source, style, seed) calls must
    serve from the preview cache without re-billing Replicate or
    Anthropic.  Stubs out the heavy calls, asserts they're not invoked
    on the second generate_preview() call."""
    import backend.kontext_engine as ke
    import backend.face_swap as fs

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-for-test")
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))
    # Redirect the preview cache to a temp dir so this test is isolated
    # from anything left over from production runs.
    monkeypatch.setattr(ke, "PREVIEW_CACHE_DIR", tmp_path / "pcache")

    kontext_calls = {"n": 0}
    swap_calls = {"n": 0}
    validate_calls = {"n": 0}

    def _fake_kontext(*args, **kwargs):
        kontext_calls["n"] += 1
        return "https://example.test/fake.png"

    fake_png = tmp_path / "fake.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    def _fake_swap(*args, **kwargs):
        swap_calls["n"] += 1
        return fake_png

    def _fake_validate(*args, **kwargs):
        validate_calls["n"] += 1
        return {
            "verdict": "pass",
            "identity_match": "strong",
            "style_match": "strong",
            "scene_preserved": True,
            "composite_clean": "clean",
            "one_line_reason": "ok",
        }

    monkeypatch.setattr(ke, "_call_kontext", _fake_kontext)
    monkeypatch.setattr(fs, "swap_face", _fake_swap)
    monkeypatch.setattr(ke, "_validate", _fake_validate)

    profile = {"hair_color_rgb": (40, 30, 25), "hair_texture": "unknown"}
    r1 = ke.generate_preview(
        source_path=SOURCE_MAN, style_id="mens_pompadour",
        customer_profile=profile, seed=42, max_retries=0,
    )
    # First call: 1 kontext + 1 swap + 1 validate
    assert kontext_calls["n"] == 1
    assert swap_calls["n"] == 1
    assert validate_calls["n"] == 1
    assert r1.validator_verdict == "pass"

    # Second call identical inputs: cache hit, no model calls
    r2 = ke.generate_preview(
        source_path=SOURCE_MAN, style_id="mens_pompadour",
        customer_profile=profile, seed=42, max_retries=0,
    )
    assert kontext_calls["n"] == 1, "cache hit should not call Kontext"
    assert swap_calls["n"] == 1, "cache hit should not call face-swap"
    assert validate_calls["n"] == 1, "cache hit should not call validator"
    assert r2.validator_verdict == "cached"


def test_preview_cache_key_changes_with_inputs():
    """Cache key must depend on EVERY input that affects the output -
    source bytes, style, seed, head_covering, glasses.  Changing any
    one must invalidate the cache so customers don't get stale results
    after a glasses-detected re-run, etc."""
    from backend.kontext_engine import _preview_cache_key
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"A" * 64)
        src1 = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"B" * 64)
        src2 = Path(f.name)

    base = _preview_cache_key(src1, "mens_pompadour", 42, None, False)
    assert _preview_cache_key(src2, "mens_pompadour", 42, None, False) != base, \
        "different source bytes must produce different cache key"
    assert _preview_cache_key(src1, "mens_buzz_cut", 42, None, False) != base, \
        "different style must produce different cache key"
    assert _preview_cache_key(src1, "mens_pompadour", 99, None, False) != base, \
        "different seed must produce different cache key"
    assert _preview_cache_key(src1, "mens_pompadour", 42, "turban", False) != base, \
        "different head_covering must produce different cache key"
    assert _preview_cache_key(src1, "mens_pompadour", 42, None, True) != base, \
        "different glasses flag must produce different cache key"
    # Identical inputs must produce identical keys (regression for
    # accidental seeding with time/random).
    assert _preview_cache_key(src1, "mens_pompadour", 42, None, False) == base


def test_generate_preview_returns_best_so_far_when_cap_hits_mid_loop(
    monkeypatch, tmp_path,
):
    """If we have a viable best from earlier attempts and the cap is hit
    on a later attempt, ship the best instead of throwing the work away.
    Regression for the reviewer-flagged 'spent $0.50, got nothing' case."""
    import backend.kontext_engine as ke
    import backend.face_swap as fs

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-for-test")
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))

    monkeypatch.setattr(
        ke, "_call_kontext",
        lambda source_path, prompt, seed, style=None: "https://example.test/fake.png",
    )
    fake_png = tmp_path / "fake.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(
        fs, "swap_face",
        lambda identity_path, target_url_or_path, output_dir, **kw: fake_png,
    )
    monkeypatch.setattr(
        ke, "_validate",
        lambda source_path, reference_path, composited_path: {
            "verdict": "fail",
            "identity_match": "weak",
            "style_match": "modest",
            "scene_preserved": True,
            "composite_clean": "minor_artifacts",
            "one_line_reason": "stub",
        },
    )

    # Cap is tight: enough for ONE full attempt (kontext + face-swap +
    # validator = 0.04 + 0.01 + 0.015 = 0.065), but not two.
    ledger = ke.CostLedger(cap_usd=0.065)

    profile = {"hair_color_rgb": (40, 30, 25), "hair_texture": "unknown"}
    result = ke.generate_preview(
        source_path=SOURCE_MAN,
        style_id="mens_pompadour",
        customer_profile=profile,
        seed=42,
        max_retries=2,         # would normally do 3 attempts
        cost_ledger=ledger,
    )
    # Got a result back instead of raising; this is the best-so-far ship.
    assert result.image_url is not None
    assert result.validator_verdict == "fail"


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
