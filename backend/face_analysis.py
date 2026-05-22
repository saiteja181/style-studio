"""Face detection, landmarking, and face-shape classification via MediaPipe."""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

_mp_face_mesh = mp.solutions.face_mesh

# MediaPipe Face Mesh landmark indices we use for the face-shape heuristic.
# Reference: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LM_FOREHEAD_TOP = 10
LM_CHIN_BOTTOM = 152
LM_LEFT_CHEEK = 234
LM_RIGHT_CHEEK = 454
LM_LEFT_FOREHEAD = 103
LM_RIGHT_FOREHEAD = 332
LM_LEFT_JAW = 172
LM_RIGHT_JAW = 397


@dataclass
class FaceAnalysisResult:
    face_detected: bool
    face_shape: Optional[str]
    landmark_count: int
    image_width: int
    image_height: int
    metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_face(image_rgb: np.ndarray) -> Optional[FaceAnalysisResult]:
    """Run MediaPipe Face Mesh on an RGB image. Returns None if no face found."""
    h, w = image_rgb.shape[:2]

    with _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(image_rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

    shape, metrics = classify_face_shape(pts)

    return FaceAnalysisResult(
        face_detected=True,
        face_shape=shape,
        landmark_count=len(pts),
        image_width=w,
        image_height=h,
        metrics=metrics,
    )


def classify_face_shape(pts: np.ndarray) -> tuple[str, dict]:
    """Approximate face shape from landmark ratios. Heuristic v1.

    Returns (shape, metrics_dict). metrics_dict is exposed so the caller can
    inspect or tune later. Categories: oval, round, heart, square, long.
    """
    forehead_top = pts[LM_FOREHEAD_TOP]
    chin = pts[LM_CHIN_BOTTOM]
    left_cheek = pts[LM_LEFT_CHEEK]
    right_cheek = pts[LM_RIGHT_CHEEK]
    left_forehead = pts[LM_LEFT_FOREHEAD]
    right_forehead = pts[LM_RIGHT_FOREHEAD]
    left_jaw = pts[LM_LEFT_JAW]
    right_jaw = pts[LM_RIGHT_JAW]

    face_length = float(abs(chin[1] - forehead_top[1]))
    cheek_width = float(abs(right_cheek[0] - left_cheek[0]))
    forehead_width = float(abs(right_forehead[0] - left_forehead[0]))
    jaw_width = float(abs(right_jaw[0] - left_jaw[0]))

    length_to_width = face_length / cheek_width if cheek_width else 1.0
    forehead_to_jaw = forehead_width / jaw_width if jaw_width else 1.0
    cheek_to_jaw = cheek_width / jaw_width if jaw_width else 1.0

    metrics = {
        "face_length": round(face_length, 1),
        "cheek_width": round(cheek_width, 1),
        "forehead_width": round(forehead_width, 1),
        "jaw_width": round(jaw_width, 1),
        "length_to_width": round(length_to_width, 3),
        "forehead_to_jaw": round(forehead_to_jaw, 3),
        "cheek_to_jaw": round(cheek_to_jaw, 3),
    }

    # Heuristic decision tree. Tuned for typical front-facing selfies.
    if length_to_width >= 1.50:
        shape = "long"
    elif length_to_width <= 1.10 and forehead_to_jaw < 1.20:
        shape = "round"
    elif forehead_to_jaw >= 1.25 and length_to_width < 1.45:
        shape = "heart"
    elif abs(forehead_width - jaw_width) / max(forehead_width, jaw_width) < 0.08 \
            and length_to_width < 1.35:
        shape = "square"
    else:
        shape = "oval"

    return shape, metrics


def draw_landmarks_overlay(
    image_rgb: np.ndarray, result: FaceAnalysisResult, alpha: float = 0.5
) -> np.ndarray:
    """Return a copy of the image with landmarks + face-shape label drawn.

    Used for visual QA — confirm at a glance that detection looked right.
    Returns an RGB array (callers convert to BGR if writing via cv2.imwrite).
    """
    h, w = image_rgb.shape[:2]
    overlay = image_rgb.copy()

    with _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as fm:
        fm_result = fm.process(image_rgb)

    if fm_result.multi_face_landmarks:
        for lm in fm_result.multi_face_landmarks[0].landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)

    blended = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

    label = f"shape: {result.face_shape}  ({result.metrics['length_to_width']} L/W)"
    cv2.rectangle(blended, (10, 10), (10 + 8 * len(label), 38), (0, 0, 0), -1)
    cv2.putText(
        blended, label, (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return blended
