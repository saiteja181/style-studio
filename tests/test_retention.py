"""Tests for backend.retention sweeper."""
from __future__ import annotations

import time
from pathlib import Path


def _touch(p: Path, mtime: float) -> None:
    p.write_bytes(b"x")
    import os
    os.utime(p, (mtime, mtime))


def test_sweep_deletes_files_older_than_ttl(tmp_path):
    from backend.retention import sweep_once
    now = time.time()
    # Two old files (60 min ago), one fresh (1 min ago)
    _touch(tmp_path / "old_a.jpg", now - 60 * 60)
    _touch(tmp_path / "old_b.jpg", now - 60 * 60)
    _touch(tmp_path / "fresh.jpg", now - 1 * 60)

    deleted = sweep_once(tmp_path, ttl_minutes=30, now=now)
    assert deleted == 2
    assert not (tmp_path / "old_a.jpg").exists()
    assert not (tmp_path / "old_b.jpg").exists()
    assert (tmp_path / "fresh.jpg").exists()


def test_sweep_ttl_zero_disables(tmp_path):
    """STYLE_STUDIO_UPLOAD_TTL_MIN=0 must be a clean disable, not 'delete
    everything'."""
    from backend.retention import sweep_once
    _touch(tmp_path / "stale.jpg", time.time() - 99999)
    deleted = sweep_once(tmp_path, ttl_minutes=0)
    assert deleted == 0
    assert (tmp_path / "stale.jpg").exists()


def test_sweep_does_not_recurse(tmp_path):
    """Sweeper handles files only - never enters subdirectories."""
    from backend.retention import sweep_once
    (tmp_path / "subdir").mkdir()
    nested = tmp_path / "subdir" / "old.jpg"
    _touch(nested, time.time() - 60 * 60)
    deleted = sweep_once(tmp_path, ttl_minutes=30)
    assert deleted == 0
    assert nested.exists()


def test_sweep_missing_directory_is_zero(tmp_path):
    from backend.retention import sweep_once
    missing = tmp_path / "does_not_exist"
    deleted = sweep_once(missing, ttl_minutes=30)
    assert deleted == 0


def test_ttl_minutes_env_var_override(monkeypatch):
    from backend.retention import _ttl_minutes
    monkeypatch.setenv("STYLE_STUDIO_UPLOAD_TTL_MIN", "5")
    assert _ttl_minutes() == 5
    monkeypatch.setenv("STYLE_STUDIO_UPLOAD_TTL_MIN", "0")
    assert _ttl_minutes() == 0
    monkeypatch.setenv("STYLE_STUDIO_UPLOAD_TTL_MIN", "garbage")
    assert _ttl_minutes() == 30   # falls back to default
    monkeypatch.delenv("STYLE_STUDIO_UPLOAD_TTL_MIN", raising=False)
    assert _ttl_minutes() == 30


def test_lifespan_smoke(tmp_path, monkeypatch):
    """Lifespan handler should start the sweep task and cancel cleanly on
    shutdown.  This is a smoke test - just ensure no exceptions."""
    import asyncio
    monkeypatch.setenv("STYLE_STUDIO_UPLOAD_TTL_MIN", "30")
    from backend.retention import lifespan_with_sweeper

    async def _run():
        async with lifespan_with_sweeper(tmp_path):
            # mock FastAPI yielding control to its request loop
            await asyncio.sleep(0.01)
        # exit -> cancels background task; no errors

    asyncio.run(_run())
