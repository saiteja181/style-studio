"""Paste the source customer's face polygon onto a Kontext-generated image.

This is what makes the "identity" guarantee hold despite Kontext regenerating
the whole image: we composite source pixels back over a polygon covering
eyes, nose, mouth, cheeks, and jaw.  The forehead, hairline, and ears stay
as Kontext output so the new hairstyle can render freely.
"""
from __future__ import annotations

import io
import logging
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Union

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_mp_face_mesh = mp.solutions.face_mesh

# Face polygon indices, in counter-clockwise order, covering eyes/nose/mouth/
# cheeks/jaw but EXCLUDING forehead/hairline/ears.  Traverses jawline from one
# ear to the other (along the chin), then back across the eyebrow line.
# Source: MediaPipe Face Mesh canonical 478-point map.
FACE_POLYGON_INDICES = [
    # right side jaw (from ear down to chin)
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
    # chin
    152,
    # left side jaw (chin back up to ear)
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
    # along left brow to mid-forehead at brow line, then across right brow
    356, 389, 251, 284, 332, 297,
    9,    # between-brows, mid line
    67, 109, 103, 54, 21, 162, 127,
]

# Upper-face polygon: covers eyebrows + eyes + nose + upper cheeks but EXCLUDES
# mouth, jaw, lower cheeks, chin.  Used by beard-preview mode so Kontext can
# redraw the lower face (beard area) while identity around the eyes is locked.
# Traverses: right ear -> up the right side -> across the eyebrow ridge ->
# down the left side to left ear -> across the lower-cheek / philtrum level
# (dipping to ~y_norm=0.32, well below the nose tip so feathering doesn't
# bleed into the nose pixel) -> back to right ear.
# The bottom boundary intentionally extends into the philtrum zone so the
# nose tip (lm1) sits at least 18 px inside the polygon; with feather_px=18
# the Gaussian blur does not soften the nose pixel at all.
UPPER_FACE_POLYGON_INDICES = [
    # right ear, up right temple to brow line
    454, 356, 389, 251, 284, 332, 297,
    # across mid-forehead
    9,
    # down the left brow and side to left ear
    67, 109, 103, 54, 21, 162, 127,
    # bottom boundary: left outer cheek going DOWN to philtrum level then across
    58, 215,       # left lower cheek (~y_norm 0.32)
    92, 165,       # crossing toward nose underside
    186, 39, 37,   # left philtrum / below-nostril
    267, 269,      # right philtrum / below-nostril
    322, 410, 416, # right lower cheek crossing outward
    288, 367, 365, # right outer cheek ascending
    361, 323,      # back up to right ear level
    # closes to 454 (right ear, first vertex)
]

# Median Lab-distance threshold below which a pixel counts as "skin-similar".
# Calibrated so dark hair / turban (Lab L < 30 or large chroma offset from
# skin) is rejected, but the customer's actual skin under typical salon
# lighting (within ~20 Lab units of the cheek median) is preserved.
SKIN_LAB_DISTANCE = 28.0

# MediaPipe Face Mesh landmark indices for skin-sample patches.  Two on each
# cheek + two on the forehead.  These sit firmly on skin even with eyebrows,
# beards, and head coverings present.
SKIN_SAMPLE_INDICES = [
    50, 280,        # high cheekbones (left/right)
    205, 425,       # mid cheeks (left/right)
    151,            # mid forehead (often under turban; weight lower)
]


def _sample_skin_lab(rgb_image: np.ndarray, landmarks_xy: np.ndarray) -> np.ndarray:
    """Median Lab colour from skin patches at SKIN_SAMPLE_INDICES.

    Returns a (3,) float Lab vector.  Falls back to a generic Indian-skin Lab
    colour (L=140, a=130, b=140) when no usable patches are found - this
    keeps the filter conservative and never throws.
    """
    h, w = rgb_image.shape[:2]
    lab_full = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    patch_lab_means = []
    radius = 8
    for idx in SKIN_SAMPLE_INDICES:
        if idx >= len(landmarks_xy):
            continue
        cx, cy = int(landmarks_xy[idx][0]), int(landmarks_xy[idx][1])
        x0, x1 = max(0, cx - radius), min(w, cx + radius)
        y0, y1 = max(0, cy - radius), min(h, cy + radius)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        patch = lab_full[y0:y1, x0:x1].reshape(-1, 3)
        # Drop near-black and near-white pixels - typical lighting reflections
        # / heavy shadow inside the patch don't represent median skin colour.
        l = patch[:, 0]
        keep = (l > 30) & (l < 240)
        if keep.sum() >= 16:
            patch_lab_means.append(patch[keep].mean(axis=0))
    if not patch_lab_means:
        return np.array([140.0, 130.0, 140.0], dtype=np.float32)
    return np.median(np.stack(patch_lab_means, axis=0), axis=0)


def _build_skin_only_mask(
    rgb_image: np.ndarray,
    geometric_mask: np.ndarray,
    landmarks_xy: np.ndarray,
) -> np.ndarray:
    """Return a uint8 mask that is the geometric face polygon AND-ed with
    a skin-similarity filter.  Pixels that are inside the polygon but whose
    Lab colour is far from the customer's sampled skin median (dark fabric,
    hair, background) are EXCLUDED from preservation.

    Small interior holes (eyes, mouth, nostrils, lips) are CLOSED via
    morphology before the final smoothing so the skin filter only ever
    removes the BIG non-skin regions (turban fabric, hijab edges, hair
    bleeding through the temples).  Without this closing pass, the binary
    Lab mask drops alpha at the nose tip / cheek where adjacent
    mouth/eye pixels are non-skin and a Gaussian blur drags neighbouring
    skin pixels down with them.
    """
    skin_lab = _sample_skin_lab(rgb_image, landmarks_xy)
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    # Per-pixel Lab distance from the sampled skin median:
    delta = lab - skin_lab.reshape(1, 1, 3)
    dist = np.sqrt((delta * delta).sum(axis=2))
    skin_similar = (dist < SKIN_LAB_DISTANCE).astype(np.uint8) * 255
    # Morphological closing fills the small interior holes (eyes, lips,
    # nostrils) so they stay inside the preserved region.  The kernel must
    # be larger than the biggest feature hole (~25 px on a 1024-tall image
    # for the open mouth on a smile photo) but smaller than the smallest
    # non-skin region we want to KEEP excluded (turban temple band ~80 px).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    skin_filled = cv2.morphologyEx(skin_similar, cv2.MORPH_CLOSE, kernel)
    # AND with the existing geometric mask:
    combined = cv2.bitwise_and(geometric_mask, skin_filled)
    # Smooth the resulting mask - the per-pixel filter is binary which can
    # create stippled edges.  A small Gaussian blur recovers a feathered look.
    return cv2.GaussianBlur(combined, (9, 9), 0)


def paste_source_face(
    source_path: Path,
    kontext_output_url_or_path: Union[str, Path],
    output_dir: Path,
    feather_px: int = 18,
    mode: str = "hair",
    head_covering_type: Optional[str] = None,
) -> Path:
    """Composite the customer's face polygon (from MediaPipe) onto a Kontext
    output image.

    Args:
        source_path: pre-flight-normalised customer photo.
        kontext_output_url_or_path: Replicate URL or local Path of the
            Kontext-generated image.
        output_dir: where to write the composited PNG.
        feather_px: Gaussian blur radius for the polygon edge, in pixels.
            ~18 gives a soft seam between the source face and the new hair.
        mode: "hair" (default) preserves the full face polygon (eyes, nose,
            mouth, cheeks, jaw); the new hairstyle can render freely above
            the brow line.  "beard" preserves only the upper face (eyes,
            nose, eyebrows, upper cheeks); the new beard can render freely
            on the jaw and lower cheeks.
        head_covering_type: when SP 1.7's detector identifies a head covering
            (turban / hijab / ghoonghat / cap_hat / other), the upper face
            polygon is shrunk by 12% of face height so covering fabric stays
            outside the preserved region.  None (default) leaves the polygon
            at full extent.

    Returns:
        Path to the composited PNG.

    Raises:
        FileNotFoundError: if source_path is missing.
        PIL.UnidentifiedImageError: if either image fails to decode
            (subclass of OSError).
    """
    source_rgb = np.array(Image.open(source_path).convert("RGB"))
    kontext_rgb = _load_rgb(kontext_output_url_or_path)

    # Match dimensions to source (Kontext may return a slightly different size
    # depending on the input aspect ratio).
    h, w = source_rgb.shape[:2]
    if kontext_rgb.shape[:2] != (h, w):
        kontext_rgb = cv2.resize(
            kontext_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4,
        )

    polygon = UPPER_FACE_POLYGON_INDICES if mode == "beard" else FACE_POLYGON_INDICES
    face_alpha = _build_face_alpha(
        source_rgb, feather_px=feather_px, polygon_indices=polygon,
    )
    if face_alpha is None:
        logger.warning(
            "face_composite: no face detected in source; returning raw Kontext"
        )
        return _save_png(kontext_rgb, output_dir, prefix="kontext_only_")

    # When the source has a head covering, shrink the upper boundary of the
    # preserved face region so any covering fabric inside the geometric
    # polygon (turban temples, hijab edges, cap brim) does NOT get composited
    # back over the Kontext output.  Combined with the skin-only filter,
    # this gives belt-and-braces protection against fabric-bleed.
    if head_covering_type in ("turban", "hijab", "ghoonghat", "cap_hat", "other") \
            and face_alpha is not None:
        h_img, w_img = face_alpha.shape
        # Find the topmost row where the mask is non-zero
        rows_with_mask = np.where(face_alpha.max(axis=1) > 16)[0]
        if rows_with_mask.size > 0:
            top_y = int(rows_with_mask[0])
            bottom_y = int(rows_with_mask[-1])
            face_h = max(1, bottom_y - top_y)
            shrink_px = int(face_h * 0.12)
            # Zero out the top `shrink_px` rows of the mask:
            cutoff = min(h_img, top_y + shrink_px)
            face_alpha[top_y:cutoff, :] = 0
            # Re-feather the new top edge:
            face_alpha = cv2.GaussianBlur(face_alpha, (15, 15), 0)

    alpha = (face_alpha.astype(np.float32) / 255.0)[..., None]
    composed = (
        kontext_rgb.astype(np.float32) * (1.0 - alpha)
        + source_rgb.astype(np.float32) * alpha
    )
    composed = np.clip(composed, 0, 255).astype(np.uint8)
    return _save_png(composed, output_dir, prefix="composed_")


def _build_face_alpha(
    image_rgb: np.ndarray, feather_px: int,
    polygon_indices: list = None,
) -> Optional[np.ndarray]:
    """Build a feathered alpha mask covering the face polygon, then AND it
    with a skin-similarity filter so non-skin pixels inside the polygon
    (turban fabric, dark hair, background bleed) are excluded.

    Returns None if MediaPipe finds no face in the image.
    """
    h, w = image_rgb.shape[:2]
    with _mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(image_rgb)
    if not result.multi_face_landmarks:
        return None
    landmarks = result.multi_face_landmarks[0].landmark
    poly_indices = polygon_indices or FACE_POLYGON_INDICES
    if len(landmarks) <= max(poly_indices):
        return None

    landmarks_xy = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
    poly = landmarks_xy[poly_indices].astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    geometric_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(geometric_mask, [poly], 255)
    if feather_px > 0:
        k = max(3, feather_px * 2 + 1)
        geometric_mask = cv2.GaussianBlur(geometric_mask, (k, k), 0)

    return _build_skin_only_mask(image_rgb, geometric_mask, landmarks_xy)


def _load_rgb(src: Union[str, Path]) -> np.ndarray:
    """Load an RGB image from a local path or http(s) URL."""
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=60) as resp:
            return np.array(Image.open(io.BytesIO(resp.read())).convert("RGB"))
    raise FileNotFoundError(f"image source not found: {src}")


def _save_png(rgb: np.ndarray, output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp = tempfile.NamedTemporaryFile(
        prefix=prefix, suffix=".png", delete=False, dir=str(output_dir),
    )
    Image.fromarray(rgb).save(fp, format="PNG", optimize=False)
    fp.close()
    return Path(fp.name)
