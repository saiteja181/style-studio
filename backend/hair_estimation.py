"""Hairline curve extraction from MediaPipe Face Mesh landmarks.

This is a fast, dependency-free proxy for hair segmentation. It returns the
curve along the upper face boundary, optionally offset upward to approximate
the actual hairline (which typically sits 1-3 cm above the forehead-skin
boundary in adults with full hair).

For pixel-precise hair masks we will later add a Replicate face-parsing
backend, but this module is enough to (a) generate a ControlNet mask region
for "where hair should be", and (b) constrain generation to not migrate the
hairline downward into the forehead.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# MediaPipe Face Mesh indices that trace the upper boundary of the face.
# Ordered left-ear -> top-of-head -> right-ear so the resulting polyline
# reads left-to-right across the forehead.
UPPER_FACE_ARC_INDICES = [
    234, 127, 162, 21, 54, 103, 67, 109,
    10,
    338, 297, 332, 284, 251, 389, 356, 454,
]

# Outer face oval landmarks for computing face height (used to offset the
# arc upward into the approximate hairline region).
LM_FOREHEAD_TOP = 10
LM_CHIN_BOTTOM = 152


@dataclass
class HairlineEstimate:
    """Approximate hairline curve in image pixel coordinates."""

    points: list[list[float]]  # [[x, y], ...] ordered left->right
    method: str                # "face_arc_offset"
    offset_ratio: float        # fraction of face height that arc was lifted

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_hairline(
    landmarks_xy: np.ndarray,
    offset_ratio: float = 0.08,
) -> HairlineEstimate:
    """Derive an approximate hairline polyline from face mesh landmarks.

    Args:
        landmarks_xy: (468, 2) array of MediaPipe Face Mesh points in pixel coords.
        offset_ratio: how much to shift the arc upward, as a fraction of total
            face height (forehead_top -> chin). 0.08 (~8%) approximates a
            typical adult hairline relative to the visible forehead boundary.

    Returns:
        HairlineEstimate with points ordered left -> top -> right.
    """
    if landmarks_xy.shape[0] < 468:
        raise ValueError(
            f"Expected at least 468 face mesh landmarks, got {landmarks_xy.shape[0]}"
        )

    arc = landmarks_xy[UPPER_FACE_ARC_INDICES].copy()  # (N, 2)
    face_height = float(
        abs(landmarks_xy[LM_CHIN_BOTTOM][1] - landmarks_xy[LM_FOREHEAD_TOP][1])
    )
    offset_px = face_height * offset_ratio

    # Shift only the y-coord upward (y grows downward in image space).
    arc[:, 1] -= offset_px

    # Clamp to non-negative (don't go off the top of the image).
    arc[:, 1] = np.clip(arc[:, 1], 0, None)

    return HairlineEstimate(
        points=arc.tolist(),
        method="face_arc_offset",
        offset_ratio=offset_ratio,
    )


def build_hair_region_mask(
    image_shape: tuple[int, int],
    hairline: HairlineEstimate,
    landmarks_xy: np.ndarray,
    extend_above_ratio: float = 0.6,
) -> np.ndarray:
    """Build a binary mask covering the approximate hair region above the hairline.

    The mask is True for pixels considered part of "hair-zone" (above the hairline
    curve, bounded on left/right by the face arc edges, and extending upward by
    `extend_above_ratio` of face height).

    Args:
        image_shape: (height, width) of the source image.
        hairline: result of estimate_hairline().
        landmarks_xy: full mesh landmarks (used to compute face height).
        extend_above_ratio: how far above the hairline to extend the mask,
            as a fraction of face height. 0.6 covers most volume hairstyles.

    Returns:
        (H, W) uint8 array with 255 inside the hair region, 0 outside.
    """
    import cv2

    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array(hairline.points, dtype=np.float32)  # (N, 2)
    face_height = float(
        abs(landmarks_xy[LM_CHIN_BOTTOM][1] - landmarks_xy[LM_FOREHEAD_TOP][1])
    )
    upper_lift = face_height * extend_above_ratio

    # Build polygon: hairline arc + upper boundary (lifted) closing back.
    upper = pts.copy()
    upper[:, 1] -= upper_lift
    upper[:, 1] = np.clip(upper[:, 1], 0, None)

    polygon = np.concatenate([pts, upper[::-1]], axis=0)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask
