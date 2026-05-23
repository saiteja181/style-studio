"""DPDPA-compliant upload retention.

Indian Digital Personal Data Protection Act (DPDPA, 2025) requires
data-minimisation for personal data.  Customer selfies count as personal
data.  Once a salon staffer has shown the preview to a customer and the
customer has left, retaining the source photo serves no purpose.

This module sweeps the uploads directory periodically, deleting files
older than STYLE_STUDIO_UPLOAD_TTL_MIN (default 30) minutes.  The sweeper
runs as a FastAPI lifespan background task at SWEEP_INTERVAL_S seconds.

Set STYLE_STUDIO_UPLOAD_TTL_MIN=0 to disable the sweeper (useful for
local development).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TTL_MIN = 30
SWEEP_INTERVAL_S = 5 * 60   # every 5 minutes


def _ttl_minutes() -> int:
    """Read TTL from env var, falling back to default.  Zero disables."""
    raw = os.getenv("STYLE_STUDIO_UPLOAD_TTL_MIN")
    if raw is None or raw.strip() == "":
        return DEFAULT_TTL_MIN
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("invalid STYLE_STUDIO_UPLOAD_TTL_MIN=%r, using default %d",
                       raw, DEFAULT_TTL_MIN)
        return DEFAULT_TTL_MIN


def sweep_once(directory: Path, ttl_minutes: int, now: float = None) -> int:
    """Delete files in directory older than ttl_minutes.  Returns count deleted.

    ttl_minutes=0 disables (returns 0 without scanning).  Used directly by
    tests and indirectly by the background loop.

    Files only -- never recurses into subdirectories, never deletes the
    directory itself.
    """
    if ttl_minutes <= 0 or not directory.exists():
        return 0
    if now is None:
        now = time.time()
    cutoff = now - ttl_minutes * 60
    deleted = 0
    for child in directory.iterdir():
        if not child.is_file():
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            try:
                child.unlink()
                deleted += 1
            except OSError as e:
                logger.warning("retention: could not delete %s: %s", child, e)
    return deleted


async def _sweep_loop(
    directory: Path, interval_s: int = SWEEP_INTERVAL_S,
    extra_dirs: tuple = (),
) -> None:
    """Forever: sweep, sleep, repeat.  Logs one line per sweep cycle.

    extra_dirs: additional directories that also contain customer-face
    PNGs and need the same data-minimisation TTL applied.  Currently
    used for the preview cache (catalogue/preview_cache) added in
    Phase 5.1 - those PNGs contain composited customer faces and must
    follow the same DPDPA retention rules as uploads/.
    """
    while True:
        try:
            ttl = _ttl_minutes()
            if ttl > 0:
                for d in (directory, *extra_dirs):
                    deleted = sweep_once(d, ttl)
                    if deleted:
                        logger.info(
                            "retention: deleted %d expired files from %s",
                            deleted, d,
                        )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("retention sweep raised: %s", e)
        await asyncio.sleep(interval_s)


@asynccontextmanager
async def lifespan_with_sweeper(directory: Path, app=None, extra_dirs: tuple = ()):
    """FastAPI lifespan handler that runs the sweep loop in the background.
    Logs the TTL setting on startup so operators see retention behaviour.
    Yields control to FastAPI's request loop; cancels the sweep task on
    shutdown.

    Usage in main.py:
        app = FastAPI(..., lifespan=lambda app: lifespan_with_sweeper(
            UPLOADS_DIR, app, extra_dirs=(PREVIEW_CACHE_DIR,)))
    """
    ttl = _ttl_minutes()
    all_dirs = (directory, *extra_dirs)
    if ttl == 0:
        logger.info("retention: DISABLED (STYLE_STUDIO_UPLOAD_TTL_MIN=0)")
        task = None
    else:
        logger.info("retention: sweeping %s every %ds, TTL=%dm",
                    [str(p) for p in all_dirs], SWEEP_INTERVAL_S, ttl)
        task = asyncio.create_task(_sweep_loop(directory, extra_dirs=extra_dirs))
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
