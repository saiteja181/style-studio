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


def paste_source_face(
    source_path: Path,
    kontext_output_url_or_path: Union[str, Path],
    output_dir: Path,
    feather_px: int = 18,
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

    face_alpha = _build_face_alpha(source_rgb, feather_px=feather_px)
    if face_alpha is None:
        logger.warning(
            "face_composite: no face detected in source; returning raw Kontext"
        )
        return _save_png(kontext_rgb, output_dir, prefix="kontext_only_")

    alpha = (face_alpha.astype(np.float32) / 255.0)[..., None]
    composed = (
        kontext_rgb.astype(np.float32) * (1.0 - alpha)
        + source_rgb.astype(np.float32) * alpha
    )
    composed = np.clip(composed, 0, 255).astype(np.uint8)
    return _save_png(composed, output_dir, prefix="composed_")


def _build_face_alpha(
    image_rgb: np.ndarray, feather_px: int,
) -> Optional[np.ndarray]:
    """Build a feathered alpha mask covering the face polygon.

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
    if len(landmarks) <= max(FACE_POLYGON_INDICES):
        return None

    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
    poly = pts[FACE_POLYGON_INDICES].astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    if feather_px > 0:
        k = max(3, feather_px * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


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
