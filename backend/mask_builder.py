"""Turn a raw hair mask into an inpaint-ready mask.

Three responsibilities:
  1. Dilate around the hairline by ~10px so the inpainter has room to
     redraw the hair edge without leaving a visible seam against the
     old hair boundary.
  2. Close small holes (eg. hair gaps between curls) so the inpainter
     sees one solid region instead of speckle.
  3. For length-increasing styles (buzz -> long), extend the mask
     downward over the shoulder / background area so the inpainter
     has somewhere to draw the new long hair.  Per-style hint comes
     from the catalogue's `expected_silhouette` field.

Why this matters: FLUX Fill literally cannot modify pixels outside
the mask.  If the mask doesn't include where the new hairstyle will
SIT (eg. a long braid falling past shoulders), the new hair gets
clipped off and the customer sees a partial style.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Silhouette categories with how much to extend the mask DOWNWARD past
# the bottom of the source hair (as a fraction of image height).  These
# are not face-shape dependent - they're "how much vertical space does
# this style occupy below where the current hair ends".  Tuned for the
# salon-portrait crop (face fills ~25-50% of frame height).
SILHOUETTE_DOWN_EXTENSION = {
    "short":     0.00,   # buzz / crop / fade - never extends past existing hair
    "medium":    0.10,   # bob / pompadour - small downward growth at most
    "long":      0.25,   # shoulder-length / mid-back updos / juda
    "very_long": 0.50,   # full braid / very long flowing hair past shoulders
}

# Default dilation around the hair edge before inpainting.  ~10px on a
# 1024-1536 long-edge image is the "give FLUX room to redraw the edge
# without a hard seam" sweet spot.  Scaled by image height so smaller
# previews don't get over-dilated.
HAIRLINE_DILATION_FRAC_OF_H = 0.008      # ~12px on 1536-tall image
HOLE_CLOSE_FRAC_OF_H = 0.012             # ~18px - bridges small hair gaps


def build_inpaint_mask(
    raw_hair_mask: np.ndarray,
    expected_silhouette: str = "medium",
    source_height: Optional[int] = None,
) -> np.ndarray:
    """Return an (H, W) uint8 0/255 mask ready for FLUX Fill.

    expected_silhouette: one of SILHOUETTE_DOWN_EXTENSION keys.  Unknown
    values fall back to "medium" rather than raising - new catalog
    entries shouldn't break production if someone forgets to tag them.
    """
    if raw_hair_mask.ndim == 3:
        # Defensive: accept HxWx1 or HxWx3 by collapsing to single channel.
        raw_hair_mask = raw_hair_mask[..., 0]
    if raw_hair_mask.dtype != np.uint8:
        raw_hair_mask = (raw_hair_mask > 0).astype(np.uint8) * 255

    h = raw_hair_mask.shape[0]
    base_h = source_height if source_height else h

    mask = raw_hair_mask.copy()

    # 1. Close small holes inside the hair region (gaps between curls,
    # speckle from segmenter false negatives).  MORPH_CLOSE = dilate then
    # erode, so it fills holes smaller than the kernel without growing
    # the outer boundary.
    close_k = max(3, int(base_h * HOLE_CLOSE_FRAC_OF_H) | 1)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_k, close_k),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # 2. Dilate the outer boundary so FLUX Fill has room to redraw the
    # hairline.  Without this the new hair stops exactly where the old
    # hair used to be, leaving a visible discontinuity.
    dilate_k = max(3, int(base_h * HAIRLINE_DILATION_FRAC_OF_H) | 1)
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_k, dilate_k),
    )
    mask = cv2.dilate(mask, dilate_kernel)

    # 3. Length-increasing extension.  Find the bottom-most row of the
    # current mask, then extend downward by silhouette_frac * height.
    # The extended region is a vertical strip the same width as the
    # current mask's bottom, so the inpainter draws hair that flows
    # naturally from where the existing hair ends.
    silhouette_frac = SILHOUETTE_DOWN_EXTENSION.get(
        expected_silhouette, SILHOUETTE_DOWN_EXTENSION["medium"],
    )
    if silhouette_frac > 0:
        mask = _extend_downward(mask, silhouette_frac, base_h)

    return mask


def _extend_downward(
    mask: np.ndarray, frac: float, base_h: int,
) -> np.ndarray:
    """Add a downward extension to the mask for long-hair styles.

    Approach: for each column that has any white pixel in the existing
    mask, paint white from the bottom-most white row down by
    `frac * base_h` additional rows (clipped to image bounds).  This
    grows a "hair shadow" downward in the right columns rather than
    smearing the whole bottom of the image white.
    """
    h, w = mask.shape
    extra = int(round(frac * base_h))
    if extra <= 0:
        return mask
    mask_bool = mask > 127
    if not mask_bool.any():
        return mask
    # For each column, find the bottom-most white row.  -1 means "no
    # mask in this column" so we skip it.
    rows = np.arange(h)[:, None]
    rows_in_mask = np.where(mask_bool, rows, -1)
    col_bottom = rows_in_mask.max(axis=0)        # shape (w,)
    valid_cols = col_bottom >= 0
    if not valid_cols.any():
        return mask
    # Build a mask of "below the column bottom, within `extra` rows"
    # for every valid column - vectorised, no Python loop.
    row_indices = np.arange(h)[:, None]          # (h, 1)
    in_extension = (
        (row_indices > col_bottom[None, :])
        & (row_indices <= (col_bottom + extra)[None, :])
        & valid_cols[None, :]
    )
    out = mask.copy()
    out[in_extension] = 255
    return out


def save_mask_preview(
    source_rgb: np.ndarray, mask: np.ndarray, out_path,
    alpha: float = 0.45,
) -> None:
    """Debug helper: overlay the mask on the source for visual inspection.
    Red tint where the mask is white, untouched elsewhere.  Used by
    tests/dev tooling, not by production code paths."""
    overlay = source_rgb.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    mask_3 = (mask > 127).astype(np.uint8)[..., None]
    overlay = (
        overlay * (1 - alpha * mask_3) + red * (alpha * mask_3)
    ).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)
