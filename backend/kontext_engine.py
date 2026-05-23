"""Core generation engine: FLUX Kontext Pro via Replicate.

This module is the ONLY place that imports `replicate`.  Public surface is
`generate_preview()` (added in Task 5) and the `PreviewResult` dataclass.
Failures raise `GenerationError`; callers map that to HTTP 502.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import replicate

logger = logging.getLogger(__name__)

KONTEXT_MODEL = "black-forest-labs/flux-kontext-pro"

# Phase 5.1: cache final composited preview by (source_bytes_hash,
# style_id, seed).  Re-clicking "try this style again" or hitting
# /generate twice with the same input should serve from disk instead
# of burning another ~$0.065 in Replicate + Anthropic spend.
PREVIEW_CACHE_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "preview_cache"

# Phase 2.1: per-call wall-clock timeout on Replicate Kontext calls.
# Default 90s catches the "Replicate is taking forever" case without
# burning customer wait time.  Tunable via env var so a slower style or
# resolution can be accommodated without code change.
KONTEXT_TIMEOUT_S = float(os.getenv("STYLE_STUDIO_KONTEXT_TIMEOUT_S", "90"))

# Phase 2.2: per-call cost estimates in USD.  Env-overridable so ops
# can adjust when Replicate or Anthropic change pricing without a
# redeploy.  Per-validator-model costs live in COST_USD_VALIDATOR_BY_MODEL
# so the ledger charges the actual model used, not a hardcoded one.
COST_USD_KONTEXT = float(os.getenv("STYLE_STUDIO_COST_KONTEXT_USD", "0.040"))
COST_USD_FACESWAP = float(os.getenv("STYLE_STUDIO_COST_FACESWAP_USD", "0.010"))
COST_USD_VALIDATOR_BY_MODEL = {
    "claude-haiku-4-5-20251001": float(os.getenv("STYLE_STUDIO_COST_HAIKU_USD", "0.005")),
    "claude-haiku-4-5": float(os.getenv("STYLE_STUDIO_COST_HAIKU_USD", "0.005")),
    "claude-sonnet-4-6": float(os.getenv("STYLE_STUDIO_COST_SONNET_USD", "0.015")),
    "claude-sonnet-4-5": float(os.getenv("STYLE_STUDIO_COST_SONNET_USD", "0.015")),
    "claude-sonnet-4-5-20250929": float(os.getenv("STYLE_STUDIO_COST_SONNET_USD", "0.015")),
}
# Default validator-cost when the model isn't in the map (conservative
# over-estimate so the cap won't accidentally over-spend).
COST_USD_VALIDATOR_DEFAULT = float(os.getenv("STYLE_STUDIO_COST_VALIDATOR_DEFAULT_USD", "0.015"))


def _validator_cost_usd() -> float:
    """Look up the actual validator model's cost from the env-aware map.
    Imported lazily because output_validator imports kontext_engine
    indirectly (via beard_engine in some paths)."""
    try:
        from backend.output_validator import DEFAULT_MODEL as v_model
    except ImportError:
        return COST_USD_VALIDATOR_DEFAULT
    return COST_USD_VALIDATOR_BY_MODEL.get(v_model, COST_USD_VALIDATOR_DEFAULT)


@dataclass
class PreviewResult:
    image_url: str           # served path /uploads/<file>.png
    style_id: str
    style_name: str
    prompt: str
    seed: int
    validator_verdict: str   # "pass" | "fail" | "uncertain" | "skipped"
    retries: int
    elapsed_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


class GenerationError(RuntimeError):
    """Raised when the Kontext call cannot produce any image at all."""


class StyleNotFoundError(GenerationError):
    """Raised when style_id is not present in the catalogue."""


class CostCapExceeded(GenerationError):
    """Raised when a per-customer cost cap would be exceeded by another call."""


def _default_cost_cap_usd() -> float:
    """Read STYLE_STUDIO_CUSTOMER_COST_CAP_USD per instance, not at
    module-load time, so env var changes (tests, hot reload) take
    effect on the next CostLedger() construction."""
    return float(os.getenv("STYLE_STUDIO_CUSTOMER_COST_CAP_USD", "0.50"))


@dataclass
class CostLedger:
    """Tracks Replicate + Anthropic spend per generate_preview session
    (or across multiple previews for one customer) and refuses to
    proceed when the next call would exceed `cap_usd`.

    Default cap (USD 0.50) covers a typical 3-attempt preview with
    headroom; tighten or widen via the env var on the operator side.

    Semantics is RESERVE-AND-REFUND: charges happen before the call
    starts (so we never start a call we can't afford) but free or
    fallback paths (validator cache hits, face-swap-failed-fell-back-
    to-paste) refund what they didn't spend.  The cap is therefore
    "max we could spend", not "actual spend so far"; final spent_usd
    after a session represents real spend."""
    cap_usd: float = field(default_factory=_default_cost_cap_usd)
    spent_usd: float = 0.0
    breakdown: dict = field(default_factory=dict)

    def check_and_charge(self, label: str, cost_usd: float) -> None:
        """Reserve `cost_usd` against the ledger under `label`.
        Raises CostCapExceeded BEFORE incrementing if this reservation
        would push us over the cap."""
        if self.spent_usd + cost_usd > self.cap_usd:
            raise CostCapExceeded(
                f"per-customer cost cap ${self.cap_usd:.2f} would be exceeded "
                f"by another '{label}' call (${cost_usd:.3f}); already spent "
                f"${self.spent_usd:.3f}. Breakdown: {self.breakdown}"
            )
        self.spent_usd += cost_usd
        self.breakdown[label] = self.breakdown.get(label, 0.0) + cost_usd

    def refund(self, label: str, cost_usd: float) -> None:
        """Refund a previously-reserved charge that didn't actually
        bill (cache hit, fallback path).  Floors at 0 to defend
        against double-refunds."""
        self.spent_usd = max(0.0, self.spent_usd - cost_usd)
        current = self.breakdown.get(label, 0.0)
        if current <= cost_usd:
            self.breakdown.pop(label, None)
        else:
            self.breakdown[label] = current - cost_usd


def _call_kontext(
    source_path: Path,
    prompt: str,
    seed: int,
    safety_tolerance: int = 2,
    style: Optional[dict] = None,
    timeout_s: float = KONTEXT_TIMEOUT_S,
) -> str:
    """Single Replicate call with a wall-clock timeout.  Returns the
    output URL string.

    The Replicate SDK's `replicate.run` is a blocking poll that has no
    timeout argument; we wrap it in a ThreadPoolExecutor and abandon
    the wait after `timeout_s`.  Caveat: the underlying prediction may
    still complete in the background and bill you for it - we just
    stop waiting.  For full cancellation we'd need to use the
    predictions API and call .cancel(), which is a bigger refactor.

    Raises GenerationError on any failure (network, API rejection,
    missing URL, timeout).
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise GenerationError("REPLICATE_API_TOKEN not set in environment")

    def _do_call() -> object:
        with Path(source_path).open("rb") as img_f:
            return replicate.run(
                KONTEXT_MODEL,
                input={
                    "prompt": prompt,
                    "input_image": img_f,
                    "aspect_ratio": "match_input_image",
                    "output_format": "png",
                    "safety_tolerance": safety_tolerance,
                    # Per-style override added in sub-project 8: short male
                    # cuts (pompadour / korean fringe / textured crop / buzz /
                    # classic side part) set upsampling=False because the
                    # upsampler was inventing a dramatic forelock-strand across
                    # the face on those styles.
                    "prompt_upsampling": (
                        style.get("upsampling", True) if style is not None else True
                    ),
                    "seed": seed,
                },
            )

    # Manual executor lifecycle so we don't block in __exit__ waiting
    # for the abandoned Replicate poll - that would defeat the timeout.
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_do_call)
        try:
            output = future.result(timeout=timeout_s)
        except FuturesTimeoutError as te:
            raise GenerationError(
                f"Kontext call timed out after {timeout_s:.0f}s"
            ) from te
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"Kontext call failed: {e}") from e
    finally:
        # cancel_futures=True (3.9+) cancels the pending poll if not started.
        # wait=False means we don't block on the abandoned background thread.
        executor.shutdown(wait=False, cancel_futures=True)

    url = _extract_first_url(output)
    if not url:
        raise GenerationError(f"Kontext returned no URL: {output!r}")
    return url


def _extract_first_url(output) -> Optional[str]:
    """Replicate may return a string, list of strings, or an object with .url."""
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


import json
import time

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "catalogue" / "styles.json"
REFERENCES_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "references"


def generate_preview(
    source_path: Path,
    style_id: str,
    customer_profile: dict,
    seed: int = 42,
    max_retries: int = 2,
    head_covering_type: Optional[str] = None,
    cost_ledger: Optional[CostLedger] = None,
    glasses_detected: bool = False,
) -> PreviewResult:
    """DEPRECATED as of SP 11: use inpaint_engine.generate_hair_preview.

    The new pipeline replaces Kontext + face-swap + validator-retry
    with FLUX-Fill-Pro masked inpainting.  Identity is guaranteed by
    construction (FLUX Fill cannot modify pixels outside the mask) so
    the best-of-N retry roulette this function performs is unnecessary.
    main.py routes /generate to inpaint_engine, not here.

    This function is kept for backwards compat with existing tests and
    for callers that want the Kontext baseline for comparison.

    Original docstring follows:

    Run a full preview: build prompt -> Kontext -> face composite ->
    validate.  Generates up to (max_retries + 1) attempts with different
    seeds and returns the BEST-scoring one (best-of-N).  Default is
    3 attempts: covers the natural variability in Kontext + face-swap
    while keeping cost bounded.

    Raises GenerationError if every attempt fails to produce an image.

    Args:
        head_covering_type: optional covering label from SP 1.7's detector
            (turban / hijab / ghoonghat / cap_hat / other).  Threaded into
            face_composite so the upper polygon is shrunk and fabric does
            not bleed back over the Kontext output.
    """
    style = _load_style(style_id)
    if style is None:
        raise StyleNotFoundError(f"Unknown style: {style_id}")
    ref_path = _resolve_reference_path(style)

    from backend.prompt_builder import build_edit_prompt
    from backend.face_composite import paste_source_face
    from backend.face_swap import swap_face, FaceSwapError

    uploads_dir = Path(
        os.getenv("STYLE_STUDIO_UPLOADS_DIR")
        or (Path(__file__).resolve().parent.parent / "tests" / "uploads")
    )

    # Cost cap: every Kontext / face-swap / validator call is charged
    # against this ledger.  Pass in an existing ledger to track spend
    # across multiple previews for the same customer; default creates
    # a fresh one with the env-configured cap.
    if cost_ledger is None:
        cost_ledger = CostLedger()

    started = time.time()

    # Phase 5.1: try the preview cache first.  Cache key includes the
    # exact source bytes + style + seed + head_covering + glasses so
    # any meaningful input change invalidates.  Cache hit costs $0
    # and bypasses Kontext + face-swap + validator entirely.
    cache_key = _preview_cache_key(
        source_path, style_id, seed, head_covering_type, glasses_detected,
    )
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

    best: Optional[dict] = None
    last_identity_error: Optional[Exception] = None
    attempt_idx = -1

    for attempt_idx in range(max_retries + 1):
        attempt_seed = seed if attempt_idx == 0 else seed + 1000 + attempt_idx
        attempt_prompt = build_edit_prompt(
            style=style, customer_profile=customer_profile,
            source_path=source_path, reference_path=ref_path,
            glasses_detected=glasses_detected,
        )

        # Reserve cap BEFORE making each call so we never start what we
        # can't afford to finish.  If reservation fails mid-loop AND we
        # already have a viable `best`, ship that instead of erroring -
        # the customer-visible behavior is "we ran what we could" not
        # "we threw away the work we paid for".
        try:
            cost_ledger.check_and_charge("kontext", COST_USD_KONTEXT)
        except CostCapExceeded:
            if best is not None:
                logger.info("cap reached mid-loop; returning best-so-far")
                break
            raise
        raw_url = _call_kontext(source_path, attempt_prompt, attempt_seed, style=style)

        # Identity step: try Replicate face-swap first (kills strand bleed +
        # lighting halo), fall back to the local polygon paste when the
        # swap call errors so the customer never sees a hard failure.
        # Broad exception catch on the fallback because PIL can raise
        # UnidentifiedImageError (OSError subclass), TypeError on weird
        # array shapes, ValueError from cv2 - all "this attempt failed,
        # try next" rather than "crash the whole call".
        try:
            cost_ledger.check_and_charge("face_swap", COST_USD_FACESWAP)
        except CostCapExceeded:
            if best is not None:
                break
            raise
        try:
            composited = swap_face(
                identity_path=source_path,
                target_url_or_path=raw_url,
                output_dir=uploads_dir,
            )
            logger.info("identity step: face-swap succeeded")
        except FaceSwapError as e:
            # Face-swap failed, so the reservation didn't bill.  Refund
            # the face_swap cost and try the local polygon paste (free).
            cost_ledger.refund("face_swap", COST_USD_FACESWAP)
            logger.warning(
                "face-swap failed, falling back to polygon paste: %s", e,
            )
            try:
                composited = paste_source_face(
                    source_path=source_path,
                    kontext_output_url_or_path=raw_url,
                    output_dir=uploads_dir,
                    head_covering_type=head_covering_type,
                )
            except Exception as inner:
                # This attempt's identity step totally failed; remember
                # the exception so we can chain it to the final raise
                # (preserves the stack trace customers/operators need
                # for debugging) and try the next attempt.
                last_identity_error = inner
                logger.warning(
                    "attempt %d failed in identity step: face-swap=%s, paste=%s",
                    attempt_idx + 1, e, inner,
                )
                continue

        attempt_image_url = f"/uploads/{composited.name}"

        # Validate this attempt (or mark skipped if validator unavailable).
        if not os.getenv("ANTHROPIC_API_KEY"):
            verdict_dict = {"verdict": "skipped_no_anthropic_key"}
        elif ref_path is None:
            verdict_dict = {"verdict": "skipped_no_reference"}
        else:
            v_cost = _validator_cost_usd()
            try:
                cost_ledger.check_and_charge("validator", v_cost)
            except CostCapExceeded:
                if best is not None:
                    break
                raise
            verdict_dict = _validate(source_path, ref_path, composited)
            # If the validator hit its on-disk cache, no API call was
            # made; refund the reservation so the ledger reflects reality.
            if verdict_dict.get("_from_cache"):
                cost_ledger.refund("validator", v_cost)
        verdict = verdict_dict.get("verdict", "uncertain")
        score = _score_verdict(verdict_dict)
        logger.info(
            "attempt %d: verdict=%s score=%d", attempt_idx + 1, verdict, score,
        )

        attempt = {
            "image_url": attempt_image_url,
            "verdict_dict": verdict_dict,
            "verdict": verdict,
            "score": score,
            "prompt": attempt_prompt,
            "seed": attempt_seed,
            "attempt_idx": attempt_idx,
        }
        if best is None or score > best["score"]:
            best = attempt
        # Short-circuit: a clean pass with strong style+identity is
        # already the maximum possible score; running more attempts
        # cannot find a better candidate.
        if (verdict == "pass"
                and verdict_dict.get("composite_clean") == "clean"
                and verdict_dict.get("identity_match") == "strong"
                and verdict_dict.get("style_match") == "strong"):
            logger.info("clean pass on attempt %d; stopping early", attempt_idx + 1)
            break

    if best is None:
        if last_identity_error is not None:
            raise GenerationError(
                f"all {max_retries + 1} attempts failed in the identity step"
            ) from last_identity_error
        raise GenerationError(
            f"all {max_retries + 1} attempts failed in the identity step"
        )

    # retries here means "attempts beyond the first", not "how many
    # retries it took to find the best".  attempts_run is also exposed
    # in best so callers needing per-attempt accounting can use it.
    retries = max(0, min(attempt_idx, max(0, max_retries)))
    elapsed_ms = int((time.time() - started) * 1000)

    # Phase 5.1: persist the best attempt to the preview cache so the
    # next identical request is free.  Only cache passing/uncertain
    # outputs AND when the composite is clean - we never want to ship
    # the same fail-with-clean-composite output twice, because a
    # different seed might give a real pass next time.
    best_verdict_dict = best["verdict_dict"]
    cache_eligible = (
        best["verdict"] in ("pass", "uncertain")
        and best_verdict_dict.get("composite_clean") in (
            "clean", "minor_artifacts", "unknown",
        )
    )
    if cache_eligible:
        try:
            _preview_cache_put(
                cache_key,
                src_image_path=uploads_dir / Path(best["image_url"]).name,
            )
        except Exception as e:
            logger.warning("preview cache write failed (non-fatal): %s", e)

    return PreviewResult(
        image_url=best["image_url"],
        style_id=style_id,
        style_name=style.get("name", style_id),
        prompt=best["prompt"],
        seed=best["seed"],
        validator_verdict=best["verdict"],
        retries=retries,
        elapsed_ms=elapsed_ms,
    )


# Phase 5.1: preview cache helpers.  Keyed by sha1 of the canonical
# (source bytes, style id, seed, head_covering, glasses) so identical
# repeat requests serve from disk.
def _preview_cache_key(
    source_path: Path, style_id: str, seed: int,
    head_covering_type: Optional[str], glasses_detected: bool,
) -> str:
    h = hashlib.sha1()
    h.update(Path(source_path).read_bytes())
    h.update(b"|")
    h.update(style_id.encode("utf-8"))
    h.update(b"|")
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    h.update((head_covering_type or "").encode("utf-8"))
    h.update(b"|")
    h.update(b"1" if glasses_detected else b"0")
    return f"preview_{h.hexdigest()[:16]}"


def _preview_cache_get(cache_key: str, uploads_dir: Path) -> Optional[Path]:
    """Return a Path inside uploads_dir if cached output exists.
    The PNG is copied (not symlinked) so /uploads serving stays valid."""
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
    """Store the best preview in the cache for future hits.  Copies
    rather than moves so the original in /uploads remains servable."""
    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = PREVIEW_CACHE_DIR / f"{cache_key}.png"
    if dest.exists():
        return
    if not src_image_path.exists():
        return
    shutil.copy2(src_image_path, dest)


# Best-of-N scoring: higher is better.  Built to mirror the validator's
# pass criterion ladder so the same logic is in one place.
_VERDICT_BASE_SCORE = {
    "pass": 1000,
    "uncertain": 500,
    "fail": 0,
    "skipped_no_anthropic_key": 200,  # ship without validation, no info
    "skipped_no_reference": 200,
}
_IDENTITY_SCORE = {"strong": 100, "weak": 30, "lost": 0, "unknown": 50}
_CLEAN_SCORE = {"clean": 100, "minor_artifacts": 50, "obvious_artifacts": 0,
                "unknown": 25}
_STYLE_SCORE = {"strong": 60, "modest": 30, "missing": 0, "unknown": 20}


def _score_verdict(v: dict) -> int:
    """Rank a validator verdict for best-of-N selection.  Higher score
    means a better candidate to show the customer."""
    score = _VERDICT_BASE_SCORE.get(v.get("verdict", ""), 0)
    score += _IDENTITY_SCORE.get(v.get("identity_match", ""), 0)
    score += _CLEAN_SCORE.get(v.get("composite_clean", ""), 0)
    score += _STYLE_SCORE.get(v.get("style_match", ""), 0)
    if v.get("scene_preserved") is True:
        score += 30
    return score


def _validate(
    source_path: Path, reference_path: Path, composited_path: Path,
) -> dict:
    """Run the validator and return the full verdict dict (not just the
    pass/fail string) so best-of-N can score on sub-criteria."""
    try:
        from backend.output_validator import validate_generation
        return validate_generation(
            source_path=source_path, reference_path=reference_path,
            generated_url=composited_path.as_uri(),
        )
    except Exception as e:
        logger.warning("validator unavailable: %s", e)
        return {
            "verdict": "uncertain",
            "identity_match": "unknown", "style_match": "unknown",
            "scene_preserved": None, "composite_clean": "unknown",
            "one_line_reason": f"validator unavailable: {e}",
        }


_CATALOGUE_CACHE: Optional[list[dict]] = None


def _load_style(style_id: str) -> Optional[dict]:
    """Return the catalogue entry for style_id, or None if not found.
    The full catalogue is parsed once and cached at module scope."""
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is None:
        if not CATALOGUE_PATH.exists():
            return None
        with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
            _CATALOGUE_CACHE = json.load(f)
    for s in _CATALOGUE_CACHE:
        if s.get("id") == style_id:
            return s
    return None


def _resolve_reference_path(style: dict) -> Optional[Path]:
    ref = style.get("reference_image_path")
    if not ref:
        return None
    p = Path(ref)
    if not p.is_absolute():
        p = REFERENCES_DIR / p
    return p if p.exists() else None
