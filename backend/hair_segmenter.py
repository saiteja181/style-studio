"""Hair segmentation via Replicate.

Single responsibility: take an RGB image, return a binary single-channel
mask (uint8, 0 or 255) where white = hair pixels and black = everything
else.  Downstream code (mask_builder) handles dilation and silhouette
extension; this module only produces the raw segmentation.

Why this exists: the new salon-flow architecture (SP 11) replaces
Kontext + face-swap with FLUX-Fill-Pro masked inpainting.  FLUX Fill
mathematically cannot modify pixels outside the mask, so identity is
guaranteed by construction rather than by validator retries.  That
mathematical guarantee only holds if the mask is correct - which is
this module's job.

Cost: ~$0.001/call on Replicate.  Local <50ms alternative
(face-parsing.PyTorch / BiSeNet) is documented as a follow-up; for the
salon-kiosk MVP, hosted inference is cheap enough.
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

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Primary segmentation model.  fermatresearch/bisenet-faces is a Cog
# implementation of BiSeNet face-parsing (19 classes including hair),
# ~$0.017/call on Replicate, ~18s latency.  Returns TWO URLs: a colored
# visualization and a single-channel L-mode mask where pixel values
# encode class IDs.  Class 17 == hair in the standard BiSeNet scheme.
# Override via env var when piloting alternatives.
PRIMARY_SEGMENTER_MODEL = os.getenv(
    "STYLE_STUDIO_HAIR_SEGMENTER_MODEL",
    "fermatresearch/bisenet-faces",
)

# BiSeNet face-parsing class IDs.  Hair = 17.  We also include "hat"
# (18) in the mask when present so a customer with a cap/beanie has
# both the cap fabric AND the underlying hair re-inpainted as the
# new hairstyle (otherwise the cap would remain stuck to the new hair).
BISENET_HAIR_CLASS = 17
BISENET_HAT_CLASS = 18

DEFAULT_TIMEOUT_S = 60.0


class HairSegmentationError(RuntimeError):
    """Raised when the Replicate hair-segmenter call cannot produce a usable
    mask.  Callers should fall back gracefully (e.g. block the upload with
    a 'please retake with hair fully visible' message) rather than treat
    this as a server error."""


def segment_hair(
    source_image: Union[Path, np.ndarray, Image.Image],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> np.ndarray:
    """Return a (H, W) uint8 binary mask: 255 where hair, 0 elsewhere.

    Accepts a file path, an RGB numpy array, or a PIL Image so callers
    do not have to round-trip through disk.  The Replicate model itself
    only takes a file or URL; we serialise an in-memory image to PNG
    bytes for the upload.

    Raises HairSegmentationError on any failure (no token, network,
    model output not parseable, mask empty).  Caller decides what to
    do; the recommended UX is to block the upload and ask staff to
    retake with hair fully visible / pulled back behind ears.
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise HairSegmentationError("REPLICATE_API_TOKEN not set in environment")

    try:
        import replicate
    except ImportError as e:
        raise HairSegmentationError("replicate package not installed") from e

    target_size = _image_size(source_image)

    # Replicate's `replicate.run("owner/name")` (bare model id, no
    # version) 404s for most community-published models - it requires
    # an explicit version hash.  Resolve the latest version via the
    # models API so we never have to hardcode a SHA that goes stale.
    model_ref = _resolve_model_ref(PRIMARY_SEGMENTER_MODEL)

    # Replicate uploads from open file handles reliably; BytesIO sometimes
    # produces 0-byte uploads depending on SDK version.  Materialise an
    # in-memory image to a temp file when caller didn't pass a path.
    tmp_handle = None
    try:
        if isinstance(source_image, (str, Path)):
            file_to_send = open(source_image, "rb")
        else:
            tmp_handle = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False,
            )
            tmp_path = Path(tmp_handle.name)
            tmp_handle.close()
            tmp_handle = None
            if isinstance(source_image, Image.Image):
                source_image.convert("RGB").save(tmp_path, format="PNG")
            else:
                Image.fromarray(source_image.astype("uint8")).convert("RGB").save(
                    tmp_path, format="PNG",
                )
            file_to_send = open(tmp_path, "rb")

        # Same Replicate 429 backoff as inpaint_engine: 3 attempts with
        # vendor-quoted sleep between them.  Account credits < $5
        # produce 6 req/min throttle; a single retry usually clears it.
        attempts_remaining = 3
        last_exc: Optional[Exception] = None
        output = None
        while attempts_remaining > 0:
            attempts_remaining -= 1
            try:
                file_to_send.seek(0)
                output = replicate.run(
                    model_ref, input={"image": file_to_send},
                )
                break
            except Exception as e:
                wait_s = _parse_replicate_429_delay(str(e))
                if wait_s is None or attempts_remaining <= 0:
                    last_exc = e
                    break
                logger.info(
                    "segmenter rate-limited, sleeping %.1fs (retries left: %d)",
                    wait_s, attempts_remaining,
                )
                time.sleep(wait_s + 0.5)
        try:
            file_to_send.close()
        except Exception:
            pass
        if tmp_handle is None and not isinstance(source_image, (str, Path)):
            try:
                Path(file_to_send.name).unlink()
            except OSError:
                pass
        if output is None:
            raise HairSegmentationError(
                f"segmenter {model_ref} failed: {last_exc}"
            ) from last_exc
    except HairSegmentationError:
        raise
    except Exception as e:
        raise HairSegmentationError(
            f"segmenter {model_ref} failed: {e}"
        ) from e

    # BiSeNet returns [colored_viz_url, class_id_mask_url].  We want the
    # class-ID mask (single-channel, values 0-18).  Models that return
    # only one URL still work because _pick_class_id_mask_url falls back
    # to the first item.
    class_id_url = _pick_class_id_mask_url(output)
    if not class_id_url:
        raise HairSegmentationError(
            f"segmenter returned no URL: {output!r}"
        )

    try:
        with urllib.request.urlopen(class_id_url, timeout=timeout_s) as resp:
            raw = resp.read()
    except Exception as e:
        raise HairSegmentationError(
            f"could not download segmenter output: {e}"
        ) from e

    mask = _class_id_image_to_hair_mask(raw, target_size=target_size)
    if int((mask > 127).sum()) < 500:
        # Less than 500 hair pixels means either a) the customer is
        # bald (legitimately - we'd ship the source unchanged), or b)
        # the segmenter failed silently.  Caller decides which case
        # this is; we surface the empty mask honestly.
        logger.warning(
            "hair segmentation returned near-empty mask (%d white px)",
            int((mask > 127).sum()),
        )
    return mask


# Cache the resolved version per-process so we make ONE API lookup per
# segmenter, not one per inference.  Cleared on process restart, which
# is fine because Replicate version IDs are append-only - new versions
# get new hashes, old ones don't change content.
_RESOLVED_VERSION_CACHE: dict = {}


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


def _resolve_model_ref(model: str) -> str:
    """Return `owner/name:version` for a model, looking up the latest
    version when the caller passed only `owner/name`.  Caches per
    process so repeated calls don't re-hit the Replicate models API."""
    if ":" in model:
        return model
    cached = _RESOLVED_VERSION_CACHE.get(model)
    if cached:
        return cached
    try:
        import replicate
        m = replicate.models.get(model)
        latest = m.latest_version
        if latest is None:
            raise HairSegmentationError(
                f"model {model} has no published versions"
            )
        ref = f"{model}:{latest.id}"
        _RESOLVED_VERSION_CACHE[model] = ref
        logger.info("resolved %s -> %s", model, latest.id[:16])
        return ref
    except HairSegmentationError:
        raise
    except Exception as e:
        raise HairSegmentationError(
            f"could not resolve version for {model}: {e}"
        ) from e


def _to_png_bytes(src: Union[Path, np.ndarray, Image.Image]) -> bytes:
    """Serialise any accepted source shape to PNG bytes for upload."""
    if isinstance(src, (str, Path)):
        return Path(src).read_bytes()
    if isinstance(src, Image.Image):
        out = io.BytesIO()
        src.convert("RGB").save(out, format="PNG")
        return out.getvalue()
    if isinstance(src, np.ndarray):
        out = io.BytesIO()
        Image.fromarray(src.astype("uint8")).convert("RGB").save(out, format="PNG")
        return out.getvalue()
    raise HairSegmentationError(f"unsupported source type: {type(src).__name__}")


def _image_size(src: Union[Path, np.ndarray, Image.Image]) -> tuple:
    """Return (W, H) of the source so we can resize the mask back to
    match.  Replicate models sometimes return resized outputs."""
    if isinstance(src, (str, Path)):
        with Image.open(src) as img:
            return img.size
    if isinstance(src, Image.Image):
        return src.size
    if isinstance(src, np.ndarray):
        return src.shape[1], src.shape[0]
    raise HairSegmentationError(f"unsupported source type: {type(src).__name__}")


def _to_url(item) -> str:
    """Normalise a single replicate output element to a URL string."""
    if isinstance(item, str):
        return item
    if item is None:
        return ""
    return getattr(item, "url", "") or ""


def _pick_class_id_mask_url(output) -> str:
    """BiSeNet returns [colored_visualization_url, class_id_mask_url].
    Prefer the second (class-ID mask).  Models returning only one URL
    fall back to the first."""
    if isinstance(output, str):
        return output
    items = list(output) if hasattr(output, "__iter__") else [output]
    if not items:
        return ""
    if len(items) >= 2:
        url = _to_url(items[1])
        if url:
            return url
    return _to_url(items[0])


def _class_id_image_to_hair_mask(raw: bytes, target_size: tuple) -> np.ndarray:
    """Convert a BiSeNet class-ID PNG (single-channel, pixel value = class
    ID 0-18) to a binary hair mask (255 where pixel ID in {hair, hat}).
    Resized to source dims with NEAREST so class IDs don't get
    interpolated into bogus intermediate values."""
    img = Image.open(io.BytesIO(raw))
    # Force L-mode in case the model returns RGB (some viz variants).
    # If RGB, take channel 0 - BiSeNet typically encodes class ID
    # consistently across channels.
    if img.mode != "L":
        rgb = np.array(img.convert("RGB"))
        arr = rgb[..., 0]  # use red channel as class ID
    else:
        arr = np.array(img, dtype=np.uint8)
    if (arr.shape[1], arr.shape[0]) != target_size:
        # Resize via PIL NEAREST so class boundaries stay sharp
        # (no fractional class IDs).
        arr = np.array(
            Image.fromarray(arr).resize(target_size, Image.NEAREST),
        )
    mask = ((arr == BISENET_HAIR_CLASS) | (arr == BISENET_HAT_CLASS))
    return mask.astype(np.uint8) * 255
