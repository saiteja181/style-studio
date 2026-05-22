"""Post-processing upscaler to sharpen FLUX inpaint outputs.

FLUX produces plausible hair but with soft micro-detail. Running the output
through clarity-upscaler restores individual hair strands, skin pores,
fabric weave - the small-grain texture that makes a photo look real instead
of AI-smoothed.

Cost: ~$0.02-0.04 per upscale call (Replicate clarity-upscaler).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

UPSCALER_MODEL = (
    "philz1337x/clarity-upscaler:"
    "dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"
)

# Tuning for "restore detail, do NOT reinvent content"
DEFAULT_SCALE = 1.5
DEFAULT_CREATIVITY = 0.3      # low = faithful to input
DEFAULT_RESEMBLANCE = 0.8     # high = preserve original structure
DEFAULT_SHARPEN = 3.0         # moderate sharpening for hair strands
DEFAULT_STEPS = 18
DEFAULT_DYNAMIC = 5           # mild HDR


class EnhanceError(RuntimeError):
    pass


def enhance_image(
    image_url_or_path,
    scale: float = DEFAULT_SCALE,
    creativity: float = DEFAULT_CREATIVITY,
    resemblance: float = DEFAULT_RESEMBLANCE,
    sharpen: float = DEFAULT_SHARPEN,
    steps: int = DEFAULT_STEPS,
    dynamic: float = DEFAULT_DYNAMIC,
    seed: int = 1337,
    preserve_mask_path=None,
) -> str:
    """Upscale + sharpen an image. Returns the URL of the enhanced output.

    Accepts either a URL (passed through to Replicate) or a local Path.

    Args:
        preserve_mask_path: optional Path to a mask image where BLACK pixels
            are preserved (not regenerated) and WHITE pixels are enhanced. Use
            this to keep the face identical while sharpening only the hair.
            NOTE: clarity-upscaler expects white = upscale, black = preserve,
            which matches our hair mask convention exactly.
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise EnhanceError("REPLICATE_API_TOKEN not set")

    try:
        import replicate
    except ImportError as e:
        raise EnhanceError("`replicate` package not installed") from e

    input_image = image_url_or_path
    file_handles = []
    if hasattr(input_image, "open"):
        fh = input_image.open("rb")
        file_handles.append(fh)
        input_image = fh

    payload = {
        "image": input_image,
        "scale_factor": scale,
        "creativity": creativity,
        "resemblance": resemblance,
        "sharpen": sharpen,
        "num_inference_steps": steps,
        "dynamic": dynamic,
        "seed": seed,
        "output_format": "png",
    }

    if preserve_mask_path is not None and hasattr(preserve_mask_path, "exists") \
            and preserve_mask_path.exists():
        # clarity-upscaler convention: WHITE = preserve, BLACK = upscale.
        # Our hair mask has WHITE = hair (the region we WANT upscaled).
        # So invert before sending: white-out the face/clothes/background
        # (preserve them) and black-out the hair (let the upscaler re-render
        # only the hair with extra detail).
        import cv2 as _cv2
        import tempfile as _tempfile
        from pathlib import Path as _Path
        src = _cv2.imread(str(preserve_mask_path), _cv2.IMREAD_GRAYSCALE)
        if src is None:
            raise EnhanceError(f"could not read mask {preserve_mask_path}")
        inverted = 255 - src
        tmp = _tempfile.NamedTemporaryFile(suffix="_inv_mask.png", delete=False)
        tmp.close()
        inverted_path = _Path(tmp.name)
        _cv2.imwrite(str(inverted_path), inverted)
        mask_fh = inverted_path.open("rb")
        file_handles.append(mask_fh)
        payload["mask"] = mask_fh
        logger.info("enhance: scale=%s sharpen=%s creativity=%s (face-preserved via inverted mask)",
                    scale, sharpen, creativity)
    else:
        logger.info("enhance: scale=%s sharpen=%s creativity=%s",
                    scale, sharpen, creativity)

    try:
        output = replicate.run(UPSCALER_MODEL, input=payload)
    except Exception as e:
        raise EnhanceError(f"upscaler failed: {e}") from e
    finally:
        for fh in file_handles:
            try: fh.close()
            except: pass

    url = _extract_first_url(output)
    if not url:
        raise EnhanceError(f"no URL in upscaler output: {output!r}")
    return url


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
