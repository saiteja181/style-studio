"""Customer analysis - extract everything the style recommender needs.

Given a customer's photo, return a structured profile:
  - face_shape (oval / round / heart / square / long / diamond)
  - jawline (sharp / soft / square / rounded)
  - skin_tone bucket (fair / wheat / medium / dusky / dark)
  - skin_rgb (mean RGB sampled from forehead)
  - hair_color_rgb (mean RGB sampled from above hairline)
  - hair_texture (straight / wavy / curly / coiled) - via vision LM
  - hairline_shape (rounded / m-shape / widow's-peak / square)
  - estimated_gender (male / female / unknown) - heuristic
  - landmark_metrics (raw ratios for downstream reasoning)

Local features come from MediaPipe Face Mesh + pixel sampling. Nuanced
features (hair texture, hairline shape) optionally come from a vision LM
call via backend.expert_consult.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for measurements
LM_FOREHEAD_TOP = 10
LM_CHIN_BOTTOM = 152
LM_LEFT_CHEEK = 234
LM_RIGHT_CHEEK = 454
LM_LEFT_FOREHEAD = 103
LM_RIGHT_FOREHEAD = 332
LM_LEFT_JAW_CORNER = 172
LM_RIGHT_JAW_CORNER = 397
LM_LEFT_JAW_MID = 136
LM_RIGHT_JAW_MID = 365
LM_NOSE_TIP = 1
LM_BETWEEN_EYES = 168

# Skin sampling: small patch on the forehead (above brows, below hairline)
LM_SKIN_PATCH_CENTER = 151   # mid-forehead

# Hair color sampling indices: not landmarks - we sample above the face arc
# in pixel space.

FACE_SHAPES = ("oval", "round", "heart", "square", "long", "diamond")
JAWLINES = ("sharp", "soft", "square", "rounded")
SKIN_TONES = ("fair", "wheat", "medium", "dusky", "dark")
HAIR_TEXTURES = ("straight", "wavy", "curly", "coiled")
HAIRLINE_SHAPES = ("rounded", "m-shape", "widows-peak", "square")
GENDERS = ("male", "female", "unknown")


@dataclass
class CustomerProfile:
    face_shape: str
    jawline: str
    skin_tone_bucket: str
    skin_rgb: tuple
    hair_color_rgb: tuple
    hair_color_descriptor: str
    hair_texture: str            # may be "unknown" if vision LM not called
    hairline_shape: str          # may be "unknown"
    estimated_gender: str        # "male" / "female" / "unknown"
    landmark_metrics: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class AnalysisError(RuntimeError):
    pass


def analyze_customer(
    selfie_path: Path,
    use_vision_lm: bool = False,
    gender_hint: Optional[str] = None,
) -> CustomerProfile:
    """Full customer profile from a single selfie.

    Args:
        selfie_path: customer photo.
        use_vision_lm: when True, call the vision LM (Claude via
            backend.expert_consult.consult_for_style style) to refine
            hair_texture, hairline_shape, estimated_gender. Costs ~$0.01
            per call, cached per photo.
        gender_hint: if known ("male"/"female"), skips gender estimation.

    Raises AnalysisError on any failure.
    """
    if not selfie_path.exists():
        raise AnalysisError(f"Selfie not found: {selfie_path}")

    try:
        pil = Image.open(selfie_path).convert("RGB")
    except Exception as e:
        raise AnalysisError(f"Could not open selfie: {e}") from e

    image_rgb = np.array(pil)
    h, w = image_rgb.shape[:2]

    with _mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(image_rgb)

    if not result.multi_face_landmarks:
        raise AnalysisError("No face detected in customer photo.")

    landmarks = result.multi_face_landmarks[0].landmark
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

    # Local feature extraction
    metrics = _compute_metrics(pts)
    face_shape = _classify_face_shape(metrics)
    jawline = _classify_jawline(metrics)
    skin_rgb = _sample_skin(image_rgb, pts)
    skin_bucket = _bucket_skin_tone(skin_rgb)
    hair_rgb = _sample_hair_color(image_rgb, pts)
    hair_color_desc = _describe_hair_color(hair_rgb)
    gender = gender_hint or "unknown"

    # Optional vision LM nuance (texture, hairline shape, gender if not hinted)
    hair_texture = "unknown"
    hairline_shape = "unknown"
    notes: list = []
    if use_vision_lm:
        try:
            from backend.customer_vision import probe_customer_features
            vision = probe_customer_features(selfie_path)
            hair_texture = vision.get("hair_texture", "unknown")
            hairline_shape = vision.get("hairline_shape", "unknown")
            if gender == "unknown":
                gender = vision.get("gender", "unknown")
            notes.append(f"vision-LM: {vision.get('summary', '').strip()}")
        except Exception as e:
            logger.warning("vision LM probe failed: %s", e)
            notes.append(f"vision-LM unavailable ({e})")

    return CustomerProfile(
        face_shape=face_shape,
        jawline=jawline,
        skin_tone_bucket=skin_bucket,
        skin_rgb=tuple(int(c) for c in skin_rgb),
        hair_color_rgb=tuple(int(c) for c in hair_rgb),
        hair_color_descriptor=hair_color_desc,
        hair_texture=hair_texture,
        hairline_shape=hairline_shape,
        estimated_gender=gender,
        landmark_metrics=metrics,
        notes=notes,
    )


# ---- local classifiers ----

def _compute_metrics(pts: np.ndarray) -> dict:
    """Geometric ratios used by the face-shape and jawline classifiers."""
    forehead_top = pts[LM_FOREHEAD_TOP]
    chin = pts[LM_CHIN_BOTTOM]
    left_cheek = pts[LM_LEFT_CHEEK]
    right_cheek = pts[LM_RIGHT_CHEEK]
    left_forehead = pts[LM_LEFT_FOREHEAD]
    right_forehead = pts[LM_RIGHT_FOREHEAD]
    left_jaw = pts[LM_LEFT_JAW_CORNER]
    right_jaw = pts[LM_RIGHT_JAW_CORNER]

    face_length = float(abs(chin[1] - forehead_top[1]))
    cheek_width = float(abs(right_cheek[0] - left_cheek[0]))
    forehead_width = float(abs(right_forehead[0] - left_forehead[0]))
    jaw_width = float(abs(right_jaw[0] - left_jaw[0]))

    # Jaw angle: how sharp the corner is - lower angle = sharper
    jaw_angle_left = _angle_at(pts[LM_LEFT_JAW_CORNER],
                               pts[LM_LEFT_CHEEK],
                               pts[LM_LEFT_JAW_MID])
    jaw_angle_right = _angle_at(pts[LM_RIGHT_JAW_CORNER],
                                pts[LM_RIGHT_CHEEK],
                                pts[LM_RIGHT_JAW_MID])
    jaw_angle = (jaw_angle_left + jaw_angle_right) / 2

    return {
        "face_length": round(face_length, 1),
        "cheek_width": round(cheek_width, 1),
        "forehead_width": round(forehead_width, 1),
        "jaw_width": round(jaw_width, 1),
        "length_to_width": round(face_length / cheek_width if cheek_width else 1.0, 3),
        "forehead_to_jaw": round(forehead_width / jaw_width if jaw_width else 1.0, 3),
        "cheek_to_jaw": round(cheek_width / jaw_width if jaw_width else 1.0, 3),
        "forehead_to_cheek": round(forehead_width / cheek_width if cheek_width else 1.0, 3),
        "jaw_angle_deg": round(jaw_angle, 1),
    }


def _classify_face_shape(m: dict) -> str:
    """Heuristic face shape from metrics. Tuned for adult faces."""
    lw = m["length_to_width"]
    fj = m["forehead_to_jaw"]
    cj = m["cheek_to_jaw"]
    fc = m["forehead_to_cheek"]

    if lw >= 1.45:
        return "long"
    if lw <= 1.05 and abs(fj - 1.0) < 0.12 and cj >= 1.10:
        return "round"
    if fj >= 1.25 and lw < 1.45:
        return "heart"
    # Diamond: widest at cheeks, narrow forehead and jaw
    if cj >= 1.18 and fc < 0.92 and fj < 1.15:
        return "diamond"
    # Square: forehead ~= jaw ~= cheeks, jaw angle small (sharp)
    if abs(fj - 1.0) < 0.08 and lw < 1.30 and m["jaw_angle_deg"] < 115:
        return "square"
    return "oval"


def _classify_jawline(m: dict) -> str:
    """Sharp/soft/square/rounded from jaw angle + ratios."""
    angle = m["jaw_angle_deg"]
    fj = m["forehead_to_jaw"]
    if angle < 105:
        return "sharp"
    if angle < 120:
        return "square" if abs(fj - 1.0) < 0.10 else "sharp"
    if angle < 140:
        return "soft"
    return "rounded"


def _sample_skin(image_rgb: np.ndarray, pts: np.ndarray, patch_radius: int = 15) -> np.ndarray:
    """Average RGB in a small patch on the forehead (above brows, below hairline)."""
    h, w = image_rgb.shape[:2]
    cx, cy = pts[LM_SKIN_PATCH_CENTER]
    cx, cy = int(cx), int(cy)
    x0 = max(0, cx - patch_radius)
    x1 = min(w, cx + patch_radius)
    y0 = max(0, cy - patch_radius)
    y1 = min(h, cy + patch_radius)
    patch = image_rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([128, 128, 128])
    return patch.reshape(-1, 3).mean(axis=0)


def _bucket_skin_tone(rgb: np.ndarray) -> str:
    """Map mean RGB to an Indian-context skin tone bucket.

    Uses luminance + R-G-B balance. Not a fairness scale; just a descriptor
    for matching hair color recommendations.
    """
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b  # perceptual luminance
    if lum >= 200:
        return "fair"
    if lum >= 165:
        return "wheat"
    if lum >= 130:
        return "medium"
    if lum >= 95:
        return "dusky"
    return "dark"


def _sample_hair_color(
    image_rgb: np.ndarray, pts: np.ndarray, patch_radius: int = 18,
) -> np.ndarray:
    """Sample hair colour from MULTIPLE patches across the actual hair zone.

    A single patch above the forehead fails on:
      - grey/silver hair where shadow patches dominate the mean
      - patchy lighting where one bright spot biases the colour
      - hairlines with a widow's peak / receding edge

    Strategy here:
      - Sample 7 patches: centre, ~1/3 left+right, ~2/3 left+right, and a
        higher patch on each side closer to where hair texture lives.
      - Reject any patch whose mean luminance is near-black (likely shadow
        UNDER the hair, not the hair itself) UNLESS all patches qualify as
        shadow (in which case the person has very dark hair, keep them).
      - Take the median of the surviving patches per channel - median is far
        more robust to one bright/dark outlier than mean.
    """
    h, w = image_rgb.shape[:2]
    forehead_y = float(pts[LM_FOREHEAD_TOP][1])
    chin_y = float(pts[LM_CHIN_BOTTOM][1])
    left_x = float(pts[LM_LEFT_CHEEK][0])
    right_x = float(pts[LM_RIGHT_CHEEK][0])
    face_h = max(1.0, abs(chin_y - forehead_y))
    face_w = max(1.0, abs(right_x - left_x))

    # Patch centres in (x, y) - all above the hairline by 5-25% of face height.
    centres: list[tuple[int, int]] = []
    for dy_frac, dx_frac in [
        (0.10, 0.0),   # mid-forehead
        (0.18, -0.25), # mid-left
        (0.18,  0.25), # mid-right
        (0.25, -0.40), # outer-left
        (0.25,  0.40), # outer-right
        (0.35,  0.0),  # well above mid-forehead
        (0.32, -0.15),
        (0.32,  0.15),
    ]:
        cx = int(pts[LM_FOREHEAD_TOP][0] + face_w * dx_frac)
        cy = int(forehead_y - face_h * dy_frac)
        centres.append((cx, cy))

    samples: list[np.ndarray] = []
    for cx, cy in centres:
        x0 = max(0, cx - patch_radius)
        x1 = min(w, cx + patch_radius)
        y0 = max(0, cy - patch_radius)
        y1 = min(h, cy + patch_radius)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        patch = image_rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        samples.append(patch)

    if not samples:
        return np.array([20, 20, 20], dtype=np.float32)

    all_pixels = np.concatenate(samples, axis=0)

    # Drop pixels that are almost black (likely shadow rather than hair).  If
    # the threshold removes everything (a person with truly very dark hair),
    # keep all pixels so we still return a sensible colour.
    luminance = all_pixels.mean(axis=1)
    kept = all_pixels[luminance > 25]
    if kept.shape[0] < 50:
        kept = all_pixels

    return np.median(kept, axis=0)


def _describe_hair_color(rgb: np.ndarray) -> str:
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    lum = (r + g + b) / 3
    if lum < 35:
        return "jet black"
    if lum < 70:
        return "deep black"
    if lum < 110:
        return "dark brown"
    if lum < 150:
        return "medium brown"
    if lum < 190:
        return "light brown"
    return "grey or blonde"


def _angle_at(vertex: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Angle at vertex between vectors vertex->a and vertex->b, in degrees."""
    va = a - vertex
    vb = b - vertex
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 180.0
    cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
