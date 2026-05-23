"""Replicate face-swap: the production identity step.

After Kontext renders the styled image (right hair, body, background, but
wrong face), we swap the customer's face onto it using a hosted face-swap
model.  This replaces the polygon-paste approach in face_composite.py for
the production pipeline; face_composite stays as a fallback when the swap
call fails (Replicate down, no face detected in the Kontext output, etc).

Why face-swap beats polygon paste:
  - The swap model only writes the central face region, so source hair
    strands inside the polygon are NOT pasted back on top of Kontext's
    new hairstyle.  This kills the strand-bleed artifact.
  - The swap model blends the face into the target's lighting, so the
    polygon-edge halo and Lab-mismatch halo go away.
  - 512+ px output vs cdingram/face-swap's 128 px crop, so no
    "plastic skin" texture loss.

Cost: ~$0.01 per call on Replicate.

Failure modes (caller must handle):
  - REPLICATE_API_TOKEN unset -> raises FaceSwapError.
  - Network / model timeout -> raises FaceSwapError.
  - Model returns target unchanged (no face locked) -> we cannot
    detect this here without re-checking; the post-generation
    validator catches it downstream and triggers retry.
"""
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)

# Primary face-swap model.  Default is cdingram/face-swap (inswapper_128),
# the same model the team's earlier experiments validated on this corpus.
# Override with STYLE_STUDIO_FACESWAP_MODEL env var to upgrade to a
# higher-res model (e.g. "omniedgeio/face-swap" or
# "xiankgx/face-swap:cff87316e31787df12002c9e20a78a017a36cb31fde9862d8dedd15ab29b7288")
# once you have an account with access.
PRIMARY_MODEL = os.getenv(
    "STYLE_STUDIO_FACESWAP_MODEL",
    "cdingram/face-swap:"
    "d1d6ea8c8be89d664a07a457526f7128109dee7030fdac424788d762c71ed111",
)

# Fallback identical to primary for now; when STYLE_STUDIO_FACESWAP_MODEL
# is set to a higher-res model, this acts as a safety net.
FALLBACK_MODEL = (
    "cdingram/face-swap:"
    "d1d6ea8c8be89d664a07a457526f7128109dee7030fdac424788d762c71ed111"
)

DEFAULT_TIMEOUT_S = 60.0


class FaceSwapError(RuntimeError):
    """Raised when the Replicate face-swap call cannot produce an image."""


def swap_face(
    identity_path: Path,
    target_url_or_path: Union[str, Path],
    output_dir: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Path:
    """Swap the face from `identity_path` onto the image at `target_url_or_path`.

    Args:
        identity_path: customer's pre-flight-normalised photo (the face
            we want to PUT IN).
        target_url_or_path: Replicate URL or local Path of the Kontext-
            generated image (the body / hair / background we want to
            keep, with the wrong face).
        output_dir: where to save the swapped PNG.
        timeout_s: per-call timeout.  Replicate calls that hang past
            this raise FaceSwapError so the caller can fall back.

    Returns:
        Path to the swapped PNG saved in `output_dir`.

    Raises:
        FaceSwapError: REPLICATE_API_TOKEN unset, network error, model
            failure, or no URL in the response.
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise FaceSwapError("REPLICATE_API_TOKEN not set in environment")

    try:
        import replicate
    except ImportError as e:
        raise FaceSwapError("replicate package not installed") from e

    target_bytes = _load_target_bytes(target_url_or_path, timeout_s=timeout_s)
    target_buf = io.BytesIO(target_bytes)
    target_buf.name = "target.png"  # replicate inspects .name for content type

    # Try the configured models in order.  In the default config PRIMARY
    # and FALLBACK are the same model, so the loop is effectively one
    # try; when the operator overrides STYLE_STUDIO_FACESWAP_MODEL to a
    # higher-quality 512+ model, the loop gives us a safety net to the
    # known-working cdingram model.
    models = (PRIMARY_MODEL,) if PRIMARY_MODEL == FALLBACK_MODEL \
        else (PRIMARY_MODEL, FALLBACK_MODEL)
    last_exc: Optional[Exception] = None
    for model in models:
        attempts_remaining = 3
        while attempts_remaining > 0:
            attempts_remaining -= 1
            try:
                with Path(identity_path).open("rb") as id_f:
                    target_buf.seek(0)
                    output = replicate.run(
                        model,
                        input=_build_input_payload(model, id_f, target_buf),
                    )
                url = _extract_first_url(output)
                if not url:
                    raise FaceSwapError(
                        f"face-swap {_model_name(model)} returned no URL: {output!r}"
                    )
                return _download_and_save(url, output_dir, timeout_s=timeout_s)
            except FaceSwapError as e:
                # Our own error - no point retrying the same call
                last_exc = e
                break
            except Exception as e:
                # Rate-limit: respect the reset window Replicate gives us
                # ("Your rate limit resets in ~5s.").  Retry up to 3 times
                # then fall through to next model.  Any non-429 exception
                # is treated as permanent for this model and we fall
                # through immediately - no point hammering a model that
                # 404'd or returned an SDK error.
                wait_s = _parse_replicate_429_delay(str(e))
                if wait_s is None:
                    logger.warning(
                        "face-swap %s failed (non-retryable): %s",
                        _model_name(model), e,
                    )
                    last_exc = e
                    break
                if attempts_remaining <= 0:
                    logger.warning(
                        "face-swap %s exhausted rate-limit retries: %s",
                        _model_name(model), e,
                    )
                    last_exc = e
                    break
                logger.info(
                    "face-swap %s rate-limited, sleeping %.1fs (attempts left: %d)",
                    _model_name(model), wait_s, attempts_remaining,
                )
                time.sleep(wait_s + 0.5)
                # loop again

    raise FaceSwapError(
        f"all face-swap models failed; last error: {last_exc}"
    ) from last_exc


_RATE_LIMIT_PATTERN = re.compile(
    r"rate limit resets in[\s~]*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE,
)


def _parse_replicate_429_delay(err_text: str) -> Optional[float]:
    """If err_text is a Replicate 429 response, return the seconds to wait
    before retry.  Returns None for any other error so the caller doesn't
    retry indefinitely on permanent failures."""
    if "429" not in err_text and "throttled" not in err_text.lower():
        return None
    m = _RATE_LIMIT_PATTERN.search(err_text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 30.0  # safe default if Replicate's message format changes


def _build_input_payload(model: str, identity_file, target_file) -> dict:
    """Each face-swap model has its own input key names.  Switch on the
    model id."""
    name = model.split(":", 1)[0]
    if name in ("easel/advanced-face-swap", "easel-ai/advanced-face-swap",
                "codeplugtech/face-swap"):
        return {
            "swap_image": identity_file,
            "target_image": target_file,
        }
    if name == "cdingram/face-swap":
        return {
            "swap_image": identity_file,
            "input_image": target_file,
        }
    # Default schema used by inswapper-style models on Replicate.
    return {"swap_image": identity_file, "input_image": target_file}


def _load_target_bytes(
    src: Union[str, Path], timeout_s: float,
) -> bytes:
    """Read the Kontext output to bytes.  Accepts a Replicate URL or a
    local file Path."""
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.exists():
            return p.read_bytes()
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=timeout_s) as resp:
            return resp.read()
    raise FaceSwapError(f"target image not found: {src}")


def _extract_first_url(output) -> Optional[str]:
    """Replicate returns string, list[str], or object with .url depending
    on model + SDK version.  Normalise."""
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


def _download_and_save(
    url: str, output_dir: Path, timeout_s: float,
) -> Path:
    """Download the swap result and save as a PNG in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        raw = resp.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    fp = tempfile.NamedTemporaryFile(
        prefix="swapped_", suffix=".png", delete=False, dir=str(output_dir),
    )
    img.save(fp, format="PNG", optimize=False)
    fp.close()
    return Path(fp.name)


def _model_name(model_ref: str) -> str:
    return model_ref.split(":", 1)[0]
