"""Tests for backend.head_covering.detect_head_covering."""
from __future__ import annotations

from pathlib import Path
from PIL import Image


def _write_blank(path: Path, color=(128, 128, 128), size=(800, 600)) -> Path:
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


def test_returns_noop_when_anthropic_key_missing(tmp_path, monkeypatch):
    """Without ANTHROPIC_API_KEY set, no call is made and we return the
    no-op result."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from backend.head_covering import detect_head_covering

    img = _write_blank(tmp_path / "blank.jpg")
    result = detect_head_covering(img, use_cache=False)
    assert result == {
        "detected": False, "covering_type": "none",
        "confidence": "none", "message": "",
    }


def test_returns_noop_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    from backend.head_covering import detect_head_covering
    result = detect_head_covering(tmp_path / "missing.jpg", use_cache=False)
    assert result["detected"] is False
    assert result["covering_type"] == "none"


def test_cached_result_skips_api(tmp_path, monkeypatch):
    """A cached result must be returned without calling the API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-cache-test")
    import backend.head_covering as hc

    img = _write_blank(tmp_path / "cached.jpg")
    key = hc._cache_key(img)
    cache_dir = hc.CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        '{"detected": true, "covering_type": "turban", "confidence": "high", '
        '"message": "test turban warning"}',
        encoding="utf-8",
    )
    try:
        result = hc.detect_head_covering(img, use_cache=True)
        assert result["detected"] is True
        assert result["covering_type"] == "turban"
        assert "turban" in result["message"]
    finally:
        (cache_dir / f"{key}.json").unlink(missing_ok=True)


def test_warning_copy_covers_all_documented_types():
    """Every covering_type listed in the SYSTEM_PROMPT should have a
    salon-friendly warning message in WARNING_COPY."""
    from backend.head_covering import WARNING_COPY
    for t in ("turban", "hijab", "ghoonghat", "cap_hat", "other"):
        assert t in WARNING_COPY
        assert len(WARNING_COPY[t]) > 20   # not a placeholder


def test_turban_warning_mentions_sikh_sensitivity():
    """Turban warning must explicitly call out Sikh religious context so
    salon staff know to confirm with the customer."""
    from backend.head_covering import WARNING_COPY
    assert "Sikh" in WARNING_COPY["turban"] or "dastar" in WARNING_COPY["turban"]
