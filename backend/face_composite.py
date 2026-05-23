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
from typing import Optional, Tuple, Union

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


def _sample_skin_lab_patches(
    rgb_image: np.ndarray, landmarks_xy: np.ndarray,
) -> np.ndarray:
    """Return an (N, 3) array of Lab patch means - one per usable skin
    landmark.  Each patch's mean is its own anchor for the nearest-patch
    distance calculation in _build_skin_only_mask.  Drops patches whose
    pixels are predominantly out-of-range (specular blow-out or deep shadow)
    so they don't pollute the patch set.

    Returns empty array if no patches are usable - caller falls back to
    unfiltered geometric mask.
    """
    h, w = rgb_image.shape[:2]
    lab_full = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    patches = []
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
        l = patch[:, 0]
        # Drop highlights and deep shadows within the patch - they don't
        # represent the lighting-anchored skin colour we want.
        keep = (l > 30) & (l < 240)
        if keep.sum() >= 16:
            patches.append(patch[keep].mean(axis=0))
    if not patches:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(patches, axis=0).astype(np.float32)


def _build_skin_only_mask(
    rgb_image: np.ndarray,
    geometric_mask: np.ndarray,
    landmarks_xy: np.ndarray,
) -> np.ndarray:
    """Return geometric_mask AND-ed with a skin-similarity mask.

    "Skin-similar" means: the pixel's Lab is close to AT LEAST ONE of the
    sampled skin patches.  Using nearest-patch distance instead of distance
    to the median anchor handles bi-modal lighting where one cheek is in
    highlight and the other in shadow - the patch nearest each lighting
    condition wins.

    Interior face features (eye whites, lips, nostrils, eyebrows) that are
    naturally darker than skin get filled in by MORPH_CLOSE with a kernel
    sized to image height (so the same code works at 768px test fixtures
    and 1536px production output).
    """
    skin_lab_patches = _sample_skin_lab_patches(rgb_image, landmarks_xy)
    if skin_lab_patches.shape[0] == 0:
        # Couldn't sample any usable skin - fall back to the unfiltered
        # geometric mask rather than mutilating identity.
        return geometric_mask

    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    # For each pixel, find the minimum Lab distance to ANY of the patches.
    # We iterate one patch at a time so peak memory stays at one (h, w)
    # distance array (vs. an (h*w, K, 3) tensor that would be 1.8 GB on a
    # 4480x6720 fixture).
    min_dist = np.full((h, w), np.inf, dtype=np.float32)
    for patch in skin_lab_patches:
        delta = lab - patch.reshape(1, 1, 3)
        dist = np.sqrt((delta * delta).sum(axis=2))              # (h, w)
        np.minimum(min_dist, dist, out=min_dist)

    skin_similar = (min_dist < SKIN_LAB_DISTANCE).astype(np.uint8) * 255

    # Fill interior face holes (eyes, lips, nostrils) so they're preserved
    # even though their Lab differs from skin.  Kernel size scales with
    # image height so the same constant works across 768px tests and
    # 1536px production images.  The 4% factor is calibrated empirically:
    # smaller and a 1536-tall mouth (~50 px) isn't fully closed; larger
    # (>=5%) and a narrow turban-fabric band gets bridged into the skin
    # region on 750-tall test fixtures.
    kernel_size = max(15, int(h * 0.04) // 2 * 2 + 1)  # odd >= 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    skin_similar = cv2.morphologyEx(skin_similar, cv2.MORPH_CLOSE, kernel)

    combined = cv2.bitwise_and(geometric_mask, skin_similar)
    return cv2.GaussianBlur(combined, (9, 9), 0)


# MediaPipe landmarks that sit ON the brow line of the face polygon's top
# edge (refine_landmarks=True canonical map).  These are the canonical
# "preserved row" landmarks - they live inside the polygon at the brow
# line, so the head-covering shrink must never push past them or the
# brow row gets re-rendered by Kontext.
BROW_LANDMARK_INDICES = [332, 297, 9, 67, 109, 103, 54]

# Anchor landmarks for the source->Kontext similarity transform.  Eye centres
# + chin gives a stable triangle that captures rotation, scale, and
# translation without being thrown off by mouth or expression changes.
ALIGNMENT_ANCHOR_INDICES = [33, 263, 152]  # left eye, right eye, chin


def _detect_face_landmarks(rgb_image: np.ndarray) -> Optional[np.ndarray]:
    """Return MediaPipe FaceMesh landmarks as an (N, 2) int-pixel array, or
    None if no face is found.  Wraps the boilerplate context manager so
    callers do not duplicate it."""
    h, w = rgb_image.shape[:2]
    with _mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(rgb_image)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    return np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)


def _alignment_transform(
    source_landmarks: np.ndarray, kontext_landmarks: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute a 2x3 similarity (rotation+scale+translation) matrix that
    maps source-image coordinates onto Kontext-image coordinates, using
    eye centres + chin as anchors.

    Returns None if cv2 cannot fit a transform, or if the fit is wildly
    out of range (>25 % scale change or >20 degrees rotation - those
    almost certainly mean MediaPipe locked onto a different feature on
    one side, and applying the transform would smear the face)."""
    src_pts = source_landmarks[ALIGNMENT_ANCHOR_INDICES].astype(np.float32)
    dst_pts = kontext_landmarks[ALIGNMENT_ANCHOR_INDICES].astype(np.float32)
    M, _inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        return None
    scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
    if not 0.80 <= scale <= 1.25:
        return None
    rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    if abs(rot_deg) > 20.0:
        return None
    return M


def _exclude_hair_bleed(
    source_rgb: np.ndarray, face_alpha: np.ndarray,
) -> np.ndarray:
    """Zero alpha for in-polygon pixels that belong to a HAIR connected
    component that ALSO touches outside the polygon.  Targets source hair
    strands that fall forward over forehead/cheek - they're connected to
    the head hair outside the polygon, so the component spans both.

    "Hair" = dark AND low-chroma (grayscale).  This separator is what lets
    the filter run safely on dark-skinned faces in moody lighting: shadow
    skin stays DARK but keeps its reddish chroma (a > 130 in OpenCV Lab),
    while hair is dark AND near-neutral (a, b within 25 of 128).  Eyebrows,
    eyelashes, lips, nostrils stay protected: they're isolated dark
    components fully surrounded by skin, with no path to outside-polygon
    hair via the hair connected-component.

    Run only when there is NO head covering.  With a head covering the
    skin filter already handles fabric exclusion via Lab distance, and
    flood-filling from the turban fabric would zero too much."""
    lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].astype(np.int32)
    a = lab[:, :, 1].astype(np.int32)
    b = lab[:, :, 2].astype(np.int32)
    chroma_sq = (a - 128) ** 2 + (b - 128) ** 2
    hair_pixel = ((l < 60) & (chroma_sq < 25 * 25)).astype(np.uint8)
    # Close gaps in the hair mask: curly hair has reflective highlights
    # that interrupt the dark-pixel connectivity, so a forehead strand
    # ends up as a separate component from the head hair behind it.
    # MORPH_CLOSE with a kernel scaled to image height (~0.5% of h)
    # bridges those mm-scale gaps without merging eyebrows into hair.
    h_img = source_rgb.shape[0]
    close_k = max(7, (int(h_img * 0.005) // 2) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    hair_pixel = cv2.morphologyEx(hair_pixel, cv2.MORPH_CLOSE, kernel)
    if int(hair_pixel.sum()) < 50:
        return face_alpha
    num_labels, labels = cv2.connectedComponents(hair_pixel, connectivity=8)
    if num_labels <= 1:
        return face_alpha

    inside_poly = face_alpha > 16
    # Per-component outside-polygon footprint.  A component must have
    # at least 200 pixels OUTSIDE the polygon (not just bordering it)
    # to count as hair-from-outside; this rules out eyebrow components
    # whose outer tip just brushes the polygon edge.
    outside_poly = ~inside_poly
    outside_hair_labels = labels[hair_pixel.astype(bool) & outside_poly]
    if outside_hair_labels.size == 0:
        return face_alpha
    counts = np.bincount(outside_hair_labels, minlength=num_labels)
    bleed_labels = np.where(counts >= 200)[0]
    bleed_labels = bleed_labels[bleed_labels != 0]
    if bleed_labels.size == 0:
        return face_alpha
    bleed_mask = np.isin(labels, bleed_labels) & inside_poly
    if not bleed_mask.any():
        return face_alpha

    # Erode the bleed_mask very slightly so the alpha-edge transition at
    # the polygon boundary is unaffected (we only want to zero pixels
    # CLEARLY inside the polygon, not the natural feathered edge).
    new_alpha = face_alpha.copy()
    new_alpha[bleed_mask] = 0
    return cv2.GaussianBlur(new_alpha, (9, 9), 0)


def _match_face_lab_mean(
    source_rgb: np.ndarray,
    kontext_rgb: np.ndarray,
    face_alpha: np.ndarray,
    max_l_shift: float = 25.0,
    max_ab_shift: float = 8.0,
) -> np.ndarray:
    """Shift source RGB so the mean Lab of pixels inside the face mask
    matches the mean Lab of the SAME mask region on the Kontext output.
    Eliminates the 'pasted face has different lighting than the rest of
    the picture' halo.

    The shift is capped so we never drift the source face so far that
    identity colour (skin tone, eye colour) is lost.  Cap is in OpenCV
    Lab units: L is 0-255 (~2.55 per L* unit), a and b are 0-255 offset
    from 128.  A 25-unit L shift is roughly 10 L* - enough to align
    indoor vs daylight white balance, not enough to whitewash the skin.
    """
    binary = face_alpha > 64
    if int(binary.sum()) < 100:
        return source_rgb
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    kontext_lab = cv2.cvtColor(kontext_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean = source_lab[binary].mean(axis=0)
    kn_mean = kontext_lab[binary].mean(axis=0)
    delta = kn_mean - src_mean
    delta[0] = float(np.clip(delta[0], -max_l_shift, max_l_shift))
    delta[1] = float(np.clip(delta[1], -max_ab_shift, max_ab_shift))
    delta[2] = float(np.clip(delta[2], -max_ab_shift, max_ab_shift))
    shifted = np.clip(source_lab + delta, 0, 255).astype(np.uint8)
    return cv2.cvtColor(shifted, cv2.COLOR_LAB2RGB)


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
    # Skin-similarity filter is only useful when the source has a head
    # covering (turban/hijab/cap) whose fabric falls inside the geometric
    # polygon.  On normal portraits the filter over-rejects legitimate
    # face skin under uneven lighting / 3-4 profile poses, causing Kontext
    # to bleed through and produce face drift.  So gate the filter on
    # head_covering_type detection.
    head_covering_detected = head_covering_type in (
        "turban", "hijab", "ghoonghat", "cap_hat", "other",
    )
    alpha_result = _build_face_alpha(
        source_rgb, feather_px=feather_px, polygon_indices=polygon,
        apply_skin_filter=head_covering_detected,
    )
    if alpha_result is None:
        logger.warning(
            "face_composite: no face detected in source; returning raw Kontext"
        )
        return _save_png(kontext_rgb, output_dir, prefix="kontext_only_")
    face_alpha = alpha_result[0]
    source_landmarks = alpha_result[1]

    # NOTE: _exclude_hair_bleed was tried here in earlier iterations but it
    # mis-classifies dark-skin-in-shadow as "hair" (skin shadow is dark AND
    # low-chroma, same signature as actual hair).  On dark-skinned customers
    # in moody lighting it zeroes most of the face polygon, producing
    # mottled outputs.  The Replicate face-swap path in
    # kontext_engine.generate_preview is the right fix for source hair
    # strands; the polygon paste remains a last-resort fallback that
    # cannot solve strand bleed without a proper segmentation model.

    # When the source has a head covering, shrink the upper boundary of the
    # preserved face region so any covering fabric inside the geometric
    # polygon (turban temples, hijab edges, cap brim) does NOT get composited
    # back over the Kontext output.  Combined with the skin-only filter,
    # this gives belt-and-braces protection against fabric-bleed.
    if head_covering_detected:
        h_img, w_img = face_alpha.shape
        rows_with_mask = np.where(face_alpha.max(axis=1) > 16)[0]
        if rows_with_mask.size > 0:
            top_y = int(rows_with_mask[0])
            bottom_y = int(rows_with_mask[-1])
            face_h = max(1, bottom_y - top_y)
            shrink_px = int(face_h * 0.12)
            # Cap the shrink well above the topmost eyebrow landmark so the
            # 15x15 Gaussian blur applied to the shrunk mask does not bleed
            # the zero band down into the eyebrow row.  Kernel half-width is
            # ~7 px; 12 px safety leaves a comfortable buffer.  Without this
            # cap, low-set brows on a tall-turban shot lose their eyebrow
            # row to the shrink and Kontext's eyebrow shape leaks back.
            brow_y_top = int(source_landmarks[BROW_LANDMARK_INDICES, 1].min())
            shrink_floor = max(0, brow_y_top - 12)
            mask_bool = face_alpha > 16
            row_indices = np.arange(h_img)[:, None]
            inf_for_empty = np.where(mask_bool, row_indices, h_img)
            col_top = inf_for_empty.min(axis=0)
            shrink_top_plus_px = np.minimum(col_top + shrink_px, shrink_floor)
            in_shrink_band = (
                (row_indices >= col_top[None, :])
                & (row_indices < shrink_top_plus_px[None, :])
            )
            shrunk = face_alpha.copy()
            shrunk[in_shrink_band] = 0
            face_alpha = cv2.GaussianBlur(shrunk, (15, 15), 0)

    # Pose-aware alignment + Lab colour match: only run when Kontext actually
    # produced a face we can detect.  This guards the unit-test path (where
    # the synthetic Kontext fill is a solid colour with no face) and the
    # rare production case where Kontext returns garbage.
    kontext_landmarks = _detect_face_landmarks(kontext_rgb)
    if kontext_landmarks is not None and len(kontext_landmarks) > max(ALIGNMENT_ANCHOR_INDICES):
        M = _alignment_transform(source_landmarks, kontext_landmarks)
        if M is not None:
            h_img, w_img = source_rgb.shape[:2]
            source_rgb = cv2.warpAffine(
                source_rgb, M, (w_img, h_img),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
            )
            face_alpha = cv2.warpAffine(
                face_alpha, M, (w_img, h_img),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        # Colour-match the source to Kontext WITHIN the face mask.  Runs
        # whether or not the alignment transform succeeded, because even
        # without a warp the lighting mismatch dominates the seam.
        source_rgb = _match_face_lab_mean(source_rgb, kontext_rgb, face_alpha)

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
    apply_skin_filter: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build a feathered alpha mask covering the face polygon, plus return
    the (N, 2) landmarks array for downstream use (brow-aware shrink,
    pose alignment).

    When apply_skin_filter=True, additionally AND the polygon with a
    Lab-space skin-similarity filter so non-skin pixels inside the polygon
    (turban fabric, dark hair, background bleed) get excluded.  This costs
    identity preservation under uneven lighting and 3-4 profile poses, so
    callers should only set True when the source actually has a head
    covering that needs filtering out.

    Returns (alpha, landmarks) tuple, or None if MediaPipe finds no face.
    """
    h, w = image_rgb.shape[:2]
    landmarks_xy = _detect_face_landmarks(image_rgb)
    if landmarks_xy is None:
        return None
    poly_indices = polygon_indices or FACE_POLYGON_INDICES
    if len(landmarks_xy) <= max(poly_indices):
        return None

    poly = landmarks_xy[poly_indices].astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    geometric_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(geometric_mask, [poly], 255)
    if feather_px > 0:
        k = max(3, feather_px * 2 + 1)
        geometric_mask = cv2.GaussianBlur(geometric_mask, (k, k), 0)

    if not apply_skin_filter:
        return geometric_mask, landmarks_xy
    return _build_skin_only_mask(image_rgb, geometric_mask, landmarks_xy), landmarks_xy


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
