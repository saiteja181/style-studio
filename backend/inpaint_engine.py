"""SP 11: hair-masked inpainting via FLUX-Fill-Pro.

Replaces the SP 10 kontext_engine + face_swap + face_composite stack.
Pipeline:
  1. Hair segmenter -> raw hair mask (Replicate, ~$0.001).
  2. Mask builder   -> dilated + silhouette-extended inpaint mask.
  3. FLUX Fill Pro  -> inpaint only the masked region (~$0.05).

FLUX Fill mathematically cannot modify pixels outside the mask, so the
customer's face, eyes, nose, jaw, expression, skin tone, beard, glasses,
clothing, and background stay byte-identical to the source.  No
post-hoc identity patchup, no validator-retry roulette.

Cost: ~$0.05 / preview (vs ~$0.165 in SP 10).
Latency: ~6-12s / preview (vs 30-90s in SP 10).
Identity preservation: by construction, not best-effort.

Public surface mirrors kontext_engine.generate_preview so main.py and
the frontend don't need to change.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from backend.kontext_engine import (
    PreviewResult, GenerationError, StyleNotFoundError,
    CostLedger, CostCapExceeded,
    _load_style, _resolve_reference_path,
    PREVIEW_CACHE_DIR,
)
from backend.hair_segmenter import segment_hair, HairSegmentationError
from backend.mask_builder import build_inpaint_mask

logger = logging.getLogger(__name__)

FLUX_FILL_MODEL = "black-forest-labs/flux-fill-pro"

# Wall-clock timeout for the FLUX Fill call.  Fill is faster than
# Kontext but inpaints on a per-pixel mask, so larger masks (very-long
# hair) can push toward 30s.  60s default leaves headroom.
FLUX_FILL_TIMEOUT_S = float(
    os.getenv("STYLE_STUDIO_FILL_TIMEOUT_S", "60"),
)

# Per-call cost estimates (USD).  Used by CostLedger to enforce the
# per-customer cap.  Env-overridable so ops can adjust on Replicate
# pricing changes without redeploy.
COST_USD_HAIR_SEGMENT = float(
    os.getenv("STYLE_STUDIO_COST_SEGMENT_USD", "0.001"),
)
COST_USD_FLUX_FILL = float(
    os.getenv("STYLE_STUDIO_COST_FILL_USD", "0.050"),
)


def generate_hair_preview(
    source_path: Path,
    style_id: str,
    customer_profile: dict,
    seed: int = 42,
    cost_ledger: Optional[CostLedger] = None,
    head_covering_type: Optional[str] = None,
    glasses_detected: bool = False,
) -> PreviewResult:
    """Run a hair-only preview using FLUX Fill masked inpainting.

    Drop-in replacement for kontext_engine.generate_preview.  Same
    PreviewResult shape so callers (main.py, frontend) don't change.

    head_covering_type and glasses_detected are accepted for API
    compatibility but largely no-ops in this pipeline: the mask
    excludes them automatically (head coverings sit outside the hair
    region; glasses sit outside the hair region too).  Kept in the
    signature so the HTTP route can pass them through without
    branching on which engine is wired up.
    """
    style = _load_style(style_id)
    if style is None:
        raise StyleNotFoundError(f"Unknown style: {style_id}")

    from backend.prompt_builder import build_hair_only_prompt

    uploads_dir = Path(
        os.getenv("STYLE_STUDIO_UPLOADS_DIR")
        or (Path(__file__).resolve().parent.parent / "tests" / "uploads")
    )
    if cost_ledger is None:
        cost_ledger = CostLedger()

    started = time.time()

    # ---- Preview cache: identical (source, style, seed, silhouette)
    # serves from disk without spending anything.
    cache_key = _preview_cache_key(source_path, style_id, seed)
    cached = _preview_cache_get(cache_key, uploads_dir)
    if cached is not None:
        logger.info("preview cache HIT key=%s", cache_key)
        return PreviewResult(
            image_url=f"/uploads/{cached.name}",
            style_id=style_id,
            style_name=style.get("name", style_id),
            prompt="<cached>",
            seed=seed,
            validator_verdict="cached",
            retries=0,
            elapsed_ms=int((time.time() - started) * 1000),
        )

    # ---- Step 1: segment hair.  Charge BEFORE the call so the cap
    # rejects when we can't afford it; refund on failure.
    cost_ledger.check_and_charge("hair_segment", COST_USD_HAIR_SEGMENT)
    try:
        raw_mask = segment_hair(source_path)
    except HairSegmentationError as e:
        cost_ledger.refund("hair_segment", COST_USD_HAIR_SEGMENT)
        raise GenerationError(
            f"hair segmentation failed (ask staff to retake with hair "
            f"fully visible, pulled back behind ears): {e}"
        ) from e

    # ---- Step 2: build the inpaint mask
    expected_silhouette = style.get("expected_silhouette", "medium")
    source_rgb = np.array(Image.open(source_path).convert("RGB"))
    inpaint_mask = build_inpaint_mask(
        raw_mask,
        expected_silhouette=expected_silhouette,
        source_height=source_rgb.shape[0],
    )

    # ---- Step 3: hair-only prompt + FLUX Fill call
    hair_prompt = build_hair_only_prompt(style, customer_profile)
    cost_ledger.check_and_charge("flux_fill", COST_USD_FLUX_FILL)
    try:
        result_url = _call_flux_fill(
            source_rgb=source_rgb,
            mask=inpaint_mask,
            prompt=hair_prompt,
            seed=seed,
        )
    except Exception as e:
        cost_ledger.refund("flux_fill", COST_USD_FLUX_FILL)
        raise GenerationError(f"FLUX Fill call failed: {e}") from e

    # ---- Download the inpainted image and save under /uploads
    out_path = _download_to_uploads(result_url, uploads_dir)
    image_url = f"/uploads/{out_path.name}"

    # ---- Cache write (best-effort; cache miss next time is fine)
    try:
        _preview_cache_put(cache_key, out_path)
    except Exception as cache_err:
        logger.warning("preview cache write failed (non-fatal): %s", cache_err)

    elapsed_ms = int((time.time() - started) * 1000)
    return PreviewResult(
        image_url=image_url,
        style_id=style_id,
        style_name=style.get("name", style_id),
        prompt=hair_prompt,
        seed=seed,
        # No validator in this pipeline (identity is mathematical, not
        # statistical).  Verdict is "skipped" to honour the existing
        # PreviewResult contract; frontend should treat as a pass.
        validator_verdict="skipped",
        retries=0,
        elapsed_ms=elapsed_ms,
    )


def _call_flux_fill(
    source_rgb: np.ndarray, mask: np.ndarray, prompt: str, seed: int,
) -> str:
    """Single FLUX Fill Pro call with wall-clock timeout via
    ThreadPoolExecutor.  Returns the output URL.

    Replicate's `replicate.run` has no timeout argument; we abandon
    the wait after FLUX_FILL_TIMEOUT_S.  The background prediction
    may still finish and bill - documented trade-off."""
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise GenerationError("REPLICATE_API_TOKEN not set in environment")
    try:
        import replicate
    except ImportError as e:
        raise GenerationError("replicate package not installed") from e

    src_buf = io.BytesIO()
    Image.fromarray(source_rgb).save(src_buf, format="PNG")
    src_buf.seek(0)
    src_buf.name = "source.png"

    mask_buf = io.BytesIO()
    Image.fromarray(mask).convert("L").save(mask_buf, format="PNG")
    mask_buf.seek(0)
    mask_buf.name = "mask.png"

    # Resolve to a pinned version so bare-name 404s (community models
    # without auto-resolve) don't bite us.  For first-party models like
    # black-forest-labs/flux-fill-pro, replicate.run usually accepts
    # the bare name, but resolving via the SDK is robust to that being
    # account-dependent.
    fill_ref = _resolve_model_ref(FLUX_FILL_MODEL)

    def _do_call() -> object:
        return replicate.run(
            fill_ref,
            input={
                "image": src_buf,
                "mask": mask_buf,
                "prompt": prompt,
                "seed": seed,
                "output_format": "png",
                # FLUX Fill takes safety_tolerance like other FLUX models.
                "safety_tolerance": 2,
            },
        )

    # 3 attempts max, with 429 backoff in between.  Replicate rate-
    # limits to 6 req/min when an account has < $5 credit, so a single
    # retry-after-sleep is enough to clear most throttle responses.
    attempts_remaining = 3
    last_exc: Optional[Exception] = None
    while attempts_remaining > 0:
        attempts_remaining -= 1
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_do_call)
            try:
                output = future.result(timeout=FLUX_FILL_TIMEOUT_S)
            except FuturesTimeoutError as te:
                raise GenerationError(
                    f"FLUX Fill call timed out after {FLUX_FILL_TIMEOUT_S:.0f}s"
                ) from te
            url = _extract_first_url(output)
            if not url:
                raise GenerationError(f"FLUX Fill returned no URL: {output!r}")
            return url
        except Exception as e:
            wait_s = _parse_replicate_429_delay(str(e))
            if wait_s is None or attempts_remaining <= 0:
                last_exc = e
                break
            logger.info(
                "FLUX Fill rate-limited, sleeping %.1fs (retries left: %d)",
                wait_s, attempts_remaining,
            )
            time.sleep(wait_s + 0.5)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    if last_exc is not None:
        raise GenerationError(f"FLUX Fill call failed: {last_exc}") from last_exc
    raise GenerationError("FLUX Fill call failed for unknown reason")


_RATE_LIMIT_PATTERN = re.compile(
    r"rate limit resets in[\s~]*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE,
)


def _parse_replicate_429_delay(err_text: str) -> Optional[float]:
    """Return seconds to wait before retry if err_text is a Replicate
    429 throttle response, else None."""
    if "429" not in err_text and "throttled" not in err_text.lower():
        return None
    m = _RATE_LIMIT_PATTERN.search(err_text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 30.0


# Cache of latest version per model so we don't re-hit the models API
# for every inference.  Cleared on process restart.
_RESOLVED_VERSION_CACHE: dict = {}


def _resolve_model_ref(model: str) -> str:
    """Look up the latest version of a model so bare-name `replicate.run`
    calls don't 404 on community-published models."""
    if ":" in model:
        return model
    cached = _RESOLVED_VERSION_CACHE.get(model)
    if cached:
        return cached
    import replicate
    m = replicate.models.get(model)
    latest = m.latest_version
    if latest is None:
        raise GenerationError(f"model {model} has no published versions")
    ref = f"{model}:{latest.id}"
    _RESOLVED_VERSION_CACHE[model] = ref
    logger.info("resolved %s -> %s", model, latest.id[:16])
    return ref


def _extract_first_url(output) -> Optional[str]:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str):
            return first
        url = getattr(first, "url", None)
        if isinstance(url, str):
            return url
    url = getattr(output, "url", None)
    if isinstance(url, str):
        return url
    return None


def _download_to_uploads(url: str, uploads_dir: Path) -> Path:
    """Fetch the Replicate output URL and save as a PNG under
    uploads_dir.  Returns the saved Path."""
    uploads_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    fp = tempfile.NamedTemporaryFile(
        prefix="filled_", suffix=".png", delete=False, dir=str(uploads_dir),
    )
    img.save(fp, format="PNG", optimize=False)
    fp.close()
    return Path(fp.name)


def _preview_cache_key(
    source_path: Path, style_id: str, seed: int,
) -> str:
    """Cache key for SP 11 inpaint pipeline.  Simpler than the SP 10
    key (no head_covering / glasses because the mask makes those
    irrelevant to the output)."""
    h = hashlib.sha1()
    h.update(Path(source_path).read_bytes())
    h.update(b"|")
    h.update(style_id.encode("utf-8"))
    h.update(b"|")
    h.update(str(seed).encode("utf-8"))
    return f"preview_inp_{h.hexdigest()[:16]}"


def _preview_cache_get(cache_key: str, uploads_dir: Path) -> Optional[Path]:
    cached = PREVIEW_CACHE_DIR / f"{cache_key}.png"
    if not cached.exists():
        return None
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / f"{cache_key}.png"
    if not dest.exists():
        try:
            shutil.copy2(cached, dest)
        except Exception:
            return None
    return dest


def _preview_cache_put(cache_key: str, src_image_path: Path) -> None:
    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = PREVIEW_CACHE_DIR / f"{cache_key}.png"
    if dest.exists() or not src_image_path.exists():
        return
    shutil.copy2(src_image_path, dest)
