"""Post-FLUX boundary colour match + clean composite.

Why this exists:
  - FLUX Fill Pro inpaints the masked region, but the model often picks a
    luminance/colour for the new hair that subtly disagrees with the source
    photo's ambient lighting (warmer hair under cooler salon lights, etc.).
    A 1-2 point shift in LAB means is enough for a stylist to notice.
  - VAE round-trip can also drift pixels OUTSIDE the mask by a few values.
    The customer expects "same photo of me, different hair" to mean exactly
    that.

What it does:
  1. Loads source + generated + mask, ensures all three are the same size.
  2. Builds a thin ring just inside and just outside the mask boundary.
  3. Measures the mean LAB shift in that ring between source and generated.
  4. Applies the negated shift to the generated image so its hair matches
     the source's ambient colour.
  5. Composites byte-preserved source pixels back outside the mask using a
     soft alpha derived from the feathered mask.

Total cost: 0 API calls, ~80 ms of OpenCV.
"""
from __future__ import annotations

import io
import logging
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_mp_face_mesh = mp.solutions.face_mesh

# Forehead + cheek landmark indices for skin-anchored colour sampling.
SKIN_SAMPLE_INDICES = [
    10,   # mid-forehead
    109, 338,         # forehead L+R of mid
    151,              # upper forehead patch
    50, 280,          # high cheek L+R
    205, 425,         # mid cheek L+R
]

# Ring half-thickness in pixels for sampling boundary colour.  ~3% of face
# width is a reasonable default; we cap it so a tight crop doesn't sample
# pixels that have already been heavily edited.
RING_PX = 24
# Per-channel shift cap so a freak outlier (e.g. a stray bright pixel in
# the ring) can't tint the whole hair.
MAX_SHIFT_PER_CHANNEL = 18.0


def harmonise_with_source(
    source_path: Path,
    generated_url_or_path,
    mask_path: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """Return a PNG path with FLUX hair colour-shifted to match source lighting
    and source pixels composited back outside the mask.

    Args:
        source_path: the EXIF-normalised customer photo we fed to FLUX.
        generated_url_or_path: FLUX output (Replicate URL or local Path).
        mask_path: the binary mask used for inpainting (white=inpaint).
        output_dir: where to write the harmonised PNG.  Defaults to source's
            parent directory.

    Raises:
        FileNotFoundError: if source or mask is missing.
        ValueError: if any image fails to decode or dimensions don't match.
    """
    source = _load_rgb(source_path)
    generated = _load_rgb(generated_url_or_path)
    mask = _load_gray(mask_path)

    h, w = source.shape[:2]
    if generated.shape[:2] != (h, w):
        generated = cv2.resize(generated, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_bin = (mask >= 128).astype(np.uint8) * 255

    # Sample SKIN patches on the source as the "this is what ambient light
    # looks like on this person" reference.  Falls back to the boundary ring
    # when face landmarks aren't available (synthetic test images, etc.).
    skin_mask = _build_skin_sample_mask(source)
    inside_ring, _ = _build_boundary_rings(mask_bin, RING_PX)

    if skin_mask is not None and skin_mask.sum() > 1000 and inside_ring.sum() > 1000:
        shifted = _lab_shift(source, generated, inside_ring, skin_mask)
    else:
        _, outside_ring = _build_boundary_rings(mask_bin, RING_PX)
        if inside_ring.sum() < 1000 or outside_ring.sum() < 1000:
            logger.info("colour_match: no skin sample + boundary rings thin, skipping LAB shift")
            shifted = generated
        else:
            shifted = _lab_shift(source, generated, inside_ring, outside_ring)

    soft_alpha = mask.astype(np.float32) / 255.0
    soft_alpha = soft_alpha[..., None]
    blended = (
        source.astype(np.float32) * (1.0 - soft_alpha)
        + shifted.astype(np.float32) * soft_alpha
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out_dir = output_dir or source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = tempfile.NamedTemporaryFile(
        prefix="harmonised_", suffix=".png", delete=False, dir=str(out_dir),
    )
    Image.fromarray(blended).save(fp, format="PNG", optimize=False)
    fp.close()
    return Path(fp.name)


def _lab_shift(
    source_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    inside_ring: np.ndarray,
    outside_ring: np.ndarray,
) -> np.ndarray:
    """Shift the generated image's LAB so its boundary matches the source's."""
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gen_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    src_mean = _ring_mean(src_lab, outside_ring)
    gen_mean = _ring_mean(gen_lab, inside_ring)
    shift = src_mean - gen_mean
    shift = np.clip(shift, -MAX_SHIFT_PER_CHANNEL, MAX_SHIFT_PER_CHANNEL)
    logger.info("colour_match: LAB shift L=%.2f a=%.2f b=%.2f",
                float(shift[0]), float(shift[1]), float(shift[2]))

    out_lab = gen_lab + shift.reshape(1, 1, 3)
    out_lab[..., 0] = np.clip(out_lab[..., 0], 0, 255)
    out_lab[..., 1] = np.clip(out_lab[..., 1], 0, 255)
    out_lab[..., 2] = np.clip(out_lab[..., 2], 0, 255)
    return cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _ring_mean(lab_img: np.ndarray, ring_mask: np.ndarray) -> np.ndarray:
    """Mean per-channel LAB inside ring_mask (uint8, 0/255)."""
    sel = ring_mask > 0
    if not sel.any():
        return np.zeros(3, dtype=np.float32)
    return lab_img[sel].mean(axis=0)


def _build_skin_sample_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Disk patches at forehead + cheek landmarks - the right "this is the
    person's ambient lighting" reference for LAB matching.  Returns None when
    face detection fails; caller falls back to the outside-mask boundary ring.
    """
    h, w = rgb.shape[:2]
    with _mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lms = result.multi_face_landmarks[0].landmark
    pts = np.array([(lm.x * w, lm.y * h) for lm in lms])
    if len(pts) <= max(SKIN_SAMPLE_INDICES):
        return None

    # Patch radius scales with face size so we always grab enough pixels.
    face_h = max(40.0, float(abs(pts[152][1] - pts[10][1])))
    radius = max(8, int(face_h * 0.05))

    mask = np.zeros((h, w), dtype=np.uint8)
    for idx in SKIN_SAMPLE_INDICES:
        cx, cy = int(pts[idx][0]), int(pts[idx][1])
        cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
    return mask


def _build_boundary_rings(
    mask_bin: np.ndarray, ring_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inside_ring, outside_ring) - thin bands hugging the mask edge.

    inside_ring   = (mask) AND NOT (eroded mask)   -> pixels just inside
    outside_ring  = (dilated mask) AND NOT (mask)  -> pixels just outside
    """
    k = max(3, ring_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask_bin, kernel)
    dilated = cv2.dilate(mask_bin, kernel)
    inside_ring = cv2.bitwise_and(mask_bin, cv2.bitwise_not(eroded))
    outside_ring = cv2.bitwise_and(dilated, cv2.bitwise_not(mask_bin))
    return inside_ring, outside_ring


def _load_rgb(src) -> np.ndarray:
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
        if str(src).startswith(("http://", "https://")):
            return _download_rgb(str(src))
        raise FileNotFoundError(f"image not found: {src}")
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        return _download_rgb(src)
    raise ValueError(f"unsupported image source type: {type(src)}")


def _download_rgb(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def _load_gray(path: Path) -> np.ndarray:
    pil = Image.open(Path(path)).convert("L")
    return np.array(pil)
