"""Refined hair-region mask for FLUX Fill Pro.

The U-band geometric mask (inpaint._build_local_hair_mask) is a rectangle
wrapping the head from one ear, over the top, to the other.  It is robust
but coarse:
  - paints background pixels above the head (halo around the silhouette)
  - paints forehead skin as hair when the style is forehead-baring
  - never reaches stray hair sticking past the rectangle (long flyaways)

This module refines the U-band by intersecting it with MediaPipe's selfie
segmentation (person-vs-background, ~30 ms locally) and subtracting the face
region derived from landmarks.  Result:
  - background pixels are dropped from the mask (no halo)
  - the boundary follows the person's actual silhouette, not a polygon
  - the face polygon (eyes, nose, mouth, lower cheeks) is preserved

Cost: 0 API calls, ~50 ms extra CPU on a 1024 px image.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

_mp_selfie = mp.solutions.selfie_segmentation

# Face-protection polygon: landmark indices around the eye + nose + mouth zone
# that should NEVER be repainted as hair, even if the U-band overlaps them.
# Source: MediaPipe Face Mesh canonical indices.
FACE_PROTECTION_INDICES = [
    234,  # left ear-side
    93, 132, 58, 172, 136, 150, 149, 176, 148,
    152,  # chin
    377, 400, 378, 379, 365, 397, 288, 361, 323,
    454,  # right ear-side
    # back across upper face just under hairline
    356, 389, 251, 284, 332, 297, 338,
    10,   # mid-forehead
    109, 67, 103, 54, 21, 162, 127,
]
# How much above the face arc to lift the protection (negative => higher).
# Positive value bites INTO the forehead; negative value leaves more forehead
# free for the new hairline to land on.
FACE_PROTECTION_LIFT_RATIO = -0.04


def refine_with_selfie_segmentation(
    image_rgb: np.ndarray,
    u_band_mask: np.ndarray,
    landmarks_xy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Intersect U-band mask with selfie-segmentation person mask, subtract the
    face polygon, and return the refined hair mask (uint8 0/255).

    Args:
        image_rgb: original photo, HxWx3 uint8.
        u_band_mask: the coarse U-band mask from inpaint._build_local_hair_mask,
            HxW uint8.  Feather is fine - we'll re-feather at the end.
        landmarks_xy: optional Nx2 array of facial landmarks in pixel coords.
            If supplied, the face protection polygon is drawn from them; if
            not, only the selfie-seg intersection is applied.

    Returns:
        HxW uint8 mask with white (255) = inpaint, black (0) = preserve.
    """
    h, w = image_rgb.shape[:2]

    # 1. Selfie-segmentation person mask.  model_selection=1 is the "landscape"
    # model tuned for close-range portrait selfies.
    with _mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        seg_result = seg.process(image_rgb)

    if seg_result.segmentation_mask is None:
        logger.info("hair_mask: selfie-seg failed, returning U-band as-is")
        return u_band_mask

    # The mask is float 0..1.  Threshold at 0.5; dilate slightly so we don't
    # eat hair pixels at the silhouette edge.
    person = (seg_result.segmentation_mask >= 0.5).astype(np.uint8) * 255
    person = cv2.dilate(person, _ellipse_kernel(11))

    # 2. Threshold the U-band so we work with a hard mask, then AND with person.
    u_hard = (u_band_mask >= 128).astype(np.uint8) * 255
    refined = cv2.bitwise_and(u_hard, person)

    # 3. Subtract the face polygon so eyes/nose/mouth stay preserved.
    if landmarks_xy is not None and len(landmarks_xy) > max(FACE_PROTECTION_INDICES):
        face_poly = _build_face_protection_polygon(landmarks_xy, w, h)
        if face_poly is not None:
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_mask, [face_poly], 255)
            refined = cv2.bitwise_and(refined, cv2.bitwise_not(face_mask))

    # 4. Re-feather the boundary for soft blend.
    refined = cv2.GaussianBlur(refined, (21, 21), 0)

    if (refined > 16).sum() < 500:
        # Refinement collapsed the mask - selfie-seg probably failed.  Fall
        # back to U-band so we never ship an empty mask to FLUX.
        logger.warning("hair_mask: refined mask is empty, using U-band")
        return u_band_mask

    return refined


def _build_face_protection_polygon(
    landmarks_xy: np.ndarray, w: int, h: int,
) -> Optional[np.ndarray]:
    """Polygon around the face zone that must stay un-inpainted."""
    try:
        pts = landmarks_xy[FACE_PROTECTION_INDICES].copy()
    except IndexError:
        return None

    if FACE_PROTECTION_LIFT_RATIO != 0.0:
        # Find face height and lift the top portion of the polygon up so the
        # protected region doesn't extend too high into the hairline area.
        forehead_y = float(pts[:, 1].min())
        chin_y = float(pts[:, 1].max())
        face_h = max(1.0, chin_y - forehead_y)
        top_band_threshold = forehead_y + 0.30 * face_h
        is_upper = pts[:, 1] <= top_band_threshold
        pts[is_upper, 1] += face_h * FACE_PROTECTION_LIFT_RATIO

    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts.astype(np.int32)


def _ellipse_kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
