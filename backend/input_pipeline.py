"""Input normalisation + pre-flight gate for salon live-capture photos.

What this fixes:
  - Phone cameras embed orientation in EXIF; FastAPI's upload bytes are not
    auto-rotated. MediaPipe then sees a sideways image and finds no face.
  - 12 MP phone shots inflate FLUX cost and blur our mask resolution. FLUX
    Fill Pro hits its sweet-spot around 1024-1536 long-edge with both dims
    as multiples of 32.
  - Salon lighting + cape + bad framing produce inputs the AI pipeline cannot
    rescue. A 30 ms local pre-flight tells staff "retake" BEFORE we spend
    $0.05 on a doomed FLUX call.

Contract: handed raw upload bytes, returns a normalised JPEG on disk plus a
PreflightReport. If the report's status is "block", do not call Replicate -
return the report to the staff and ask them to retake.
"""
from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Long-edge cap.  FLUX Fill Pro is happiest in the 1024-1536 range; bigger
# wastes API budget without quality gain because the model decodes through a
# fixed-size latent anyway.
LONG_EDGE_TARGET = 1536
MIN_EDGE = 768
# FLUX expects dimensions divisible by this.
DIM_MULTIPLE = 32

# Pre-flight thresholds.  Tightened in Phase 0 vs the original demo
# defaults; widened after a review pointed out the 0.25-0.55 band
# blocks legitimate phone-at-chest-level salon shots that produce 18-22%
# face fractions.  Strict bounds catch the cases that the downstream
# face-swap and validator cannot save, but should not catch every
# real-world salon shot.
MIN_FACE_FRACTION = 0.18     # phone-at-chest-level produces ~0.18-0.30; below this is "no head visible"
MAX_FACE_FRACTION = 0.70     # cropped face takes can hit 0.6+; only block clearly-too-tight passport crops
MIN_BLUR_SCORE = 80.0        # Laplacian variance; below this is camera-shake or OOF
MAX_HEAD_ROLL_DEG = 15.0     # in-plane tilt (15 deg is a "head slightly tilted" not 3/4)
MAX_HEAD_YAW_DEG = 25.0      # left-right turn; 25 deg is the standard "near-frontal" boundary in face-recognition
                              # literature (Yale Face DB / FERET protocols).  Above 25 deg face-swap quality
                              # degrades sharply.  Uncalibrated camera adds ~5 deg error on either side.
HAIR_FOREHEAD_FRACTION_MAX = 0.40  # max % of forehead-band that can be dark/hair (per-customer relative)

_mp_face_detection = mp.solutions.face_detection
_mp_face_mesh = mp.solutions.face_mesh


def _default_head_covering() -> dict:
    return {"detected": False, "covering_type": "none",
            "confidence": "none", "message": ""}


@dataclass
class PreflightReport:
    status: str                   # "ok" | "warn" | "block"
    code: str                     # machine-readable reason
    message: str                  # human-readable instruction for salon staff
    face_fraction: Optional[float] = None
    blur_score: Optional[float] = None
    original_size: tuple = ()     # (w, h) as uploaded
    normalised_size: tuple = ()   # (w, h) after EXIF + resize
    warnings: list = field(default_factory=list)
    head_covering: dict = field(default_factory=_default_head_covering)
    # Phase 4.2: detected glasses on the customer.  Threaded into the
    # prompt so Kontext is told to preserve them; otherwise it tends to
    # remove glasses when restyling the hair around them.
    glasses_detected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class PreflightError(RuntimeError):
    """Raised when the input cannot proceed to generation."""

    def __init__(self, report: PreflightReport):
        super().__init__(report.message)
        self.report = report


def prepare_upload(
    raw_bytes: bytes,
    target_dir: Path,
    filename_hint: str = "selfie",
) -> tuple[Path, PreflightReport]:
    """Normalise + pre-flight an uploaded image.

    Steps (in order, each one cheap and local):
      1. Decode bytes, apply EXIF transpose so the image is upright.
      2. Resize so long-edge <= LONG_EDGE_TARGET, dims rounded to multiple of 32.
      3. Detect a face with MediaPipe; reject if none, too small, or too large.
      4. Score blur via Laplacian variance; warn (not block) if soft.
      5. Save normalised JPEG to target_dir.

    Returns (saved_path, report).  Raises PreflightError if status == "block".
    """
    try:
        pil = Image.open(io.BytesIO(raw_bytes))
    except Exception as e:
        report = PreflightReport(
            status="block", code="UNREADABLE",
            message=f"Could not decode image: {e}",
        )
        raise PreflightError(report) from e

    original_size = pil.size
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    pil = _resize_for_flux(pil)
    normalised_size = pil.size

    rgb = np.array(pil)
    face_fraction = _face_fraction(rgb)
    blur_score = _blur_score(rgb)

    warnings: list[str] = []

    if face_fraction is None:
        report = PreflightReport(
            status="block", code="NO_FACE",
            message=("No face detected. Ask the customer to look at the "
                     "camera with the face centred in the frame."),
            original_size=original_size, normalised_size=normalised_size,
            blur_score=blur_score,
        )
        raise PreflightError(report)

    if face_fraction < MIN_FACE_FRACTION:
        report = PreflightReport(
            status="block", code="FACE_TOO_FAR",
            message=("Customer is too far from the camera. Move closer so "
                     "the face fills about a third of the frame."),
            face_fraction=face_fraction, blur_score=blur_score,
            original_size=original_size, normalised_size=normalised_size,
        )
        raise PreflightError(report)

    if face_fraction > MAX_FACE_FRACTION:
        # We tried auto-padding this case and FLUX paints flat colour into
        # the synthetic background area - worse than refusing.  A tight face
        # crop genuinely doesn't have enough head visible for the hair
        # pipeline to do useful work, so block with an actionable message.
        report = PreflightReport(
            status="block", code="FACE_TOO_CLOSE",
            message=(
                "Photo is too tightly cropped on the face - we need to see "
                "the whole head (forehead to back of neck) to preview hair "
                "changes. Take the photo from about chest level so the "
                "customer's full head and shoulders are visible."
            ),
            face_fraction=face_fraction, blur_score=blur_score,
            original_size=original_size, normalised_size=normalised_size,
        )
        raise PreflightError(report)

    # Multi-face block: salon flow expects one customer in frame.  Background
    # people or group shots produce ambiguous face-swap targets.
    n_faces = _count_faces(rgb)
    if n_faces > 1:
        report = PreflightReport(
            status="block", code="MULTIPLE_FACES",
            message=(
                f"Detected {n_faces} faces in the photo. Take a single-person "
                f"photo with no one else in the frame behind the customer."
            ),
            face_fraction=face_fraction, blur_score=blur_score,
            original_size=original_size, normalised_size=normalised_size,
        )
        raise PreflightError(report)

    # Pose block: 3/4 profile, big head-tilt, or strong yaw confuses both
    # the polygon paste fallback and the Replicate face-swap model.  Salon
    # staff is asked to retake straight-on.
    pose = _head_pose_degrees(rgb)
    if pose is not None:
        roll, yaw = pose
        if abs(roll) > MAX_HEAD_ROLL_DEG or abs(yaw) > MAX_HEAD_YAW_DEG:
            report = PreflightReport(
                status="block", code="POSE_NOT_FRONTAL",
                message=(
                    f"Customer's head is angled (roll {roll:+.0f} deg, "
                    f"yaw {yaw:+.0f} deg). Ask them to look straight at the "
                    f"camera with head upright, not tilted or turned."
                ),
                face_fraction=face_fraction, blur_score=blur_score,
                original_size=original_size, normalised_size=normalised_size,
            )
            raise PreflightError(report)

    # Hair-on-forehead block: if a substantial dark mass occupies the
    # forehead band (above-brows to brow line), source hair strands will
    # bleed into the face polygon during paste fallback AND confuse the
    # prompt's hair-style instructions.  Ask staff to pull hair back.
    hair_frac = _hair_on_forehead_fraction(rgb)
    if hair_frac is not None and hair_frac > HAIR_FOREHEAD_FRACTION_MAX:
        report = PreflightReport(
            status="block", code="HAIR_ON_FOREHEAD",
            message=(
                f"Customer's hair is falling on the forehead "
                f"(forehead {int(hair_frac * 100)}% covered). Pull hair "
                f"back behind the ears so the forehead and hairline are "
                f"clear, then retake."
            ),
            face_fraction=face_fraction, blur_score=blur_score,
            original_size=original_size, normalised_size=normalised_size,
        )
        raise PreflightError(report)

    if blur_score < MIN_BLUR_SCORE:
        warnings.append(
            f"Photo looks soft (blur score {blur_score:.0f} < {MIN_BLUR_SCORE:.0f}). "
            "Hold the camera steady and try again if the preview is unclear.")

    glasses_detected = _detect_glasses(rgb)
    saved_path = _save_jpeg(pil, target_dir, filename_hint)

    # Head-covering detection (turban / hijab / cap / ghoonghat).  Soft warning
    # only - the salon staff confirms with the customer before generating.
    # Skipped when ANTHROPIC_API_KEY is unset; costs ~$0.005 otherwise.
    hc_result = _default_head_covering()
    try:
        from backend.head_covering import detect_head_covering
        hc = detect_head_covering(saved_path)
        if isinstance(hc, dict):
            hc_result = hc
            if hc.get("detected") and hc.get("message"):
                warnings.append(hc["message"])
    except Exception as e:
        logger.info("head-covering detection skipped: %s", e)

    status = "warn" if warnings else "ok"
    code = "SOFT_FOCUS" if warnings else "OK"
    message = ("Input looks good." if not warnings
               else " ".join(warnings))

    report = PreflightReport(
        status=status, code=code, message=message,
        face_fraction=face_fraction, blur_score=blur_score,
        original_size=original_size, normalised_size=normalised_size,
        warnings=warnings,
        head_covering=hc_result,
        glasses_detected=glasses_detected,
    )
    logger.info("preflight: %s face=%.2f blur=%.0f size=%s",
                code, face_fraction, blur_score, normalised_size)
    return saved_path, report


def _resize_for_flux(pil: Image.Image) -> Image.Image:
    """Resize source to FLUX-friendly dimensions while PRESERVING aspect ratio.

    Bug fixed here (Opus code review of sub-project 1.5):
    the previous logic applied `max(MIN_EDGE, int(w * scale))` per-dimension,
    which silently SQUARED a 500x750 portrait into 768x768 (face stretched
    ~16% wider than reality).  WhatsApp-compressed selfies from Indian phones
    routinely arrive at sub-MIN_EDGE dimensions and would hit this.

    Correct flow:
      1. If short edge is below MIN_EDGE, upscale BOTH dimensions by the
         same factor so aspect ratio is preserved.
      2. If long edge exceeds LONG_EDGE_TARGET, downscale BOTH dimensions
         by the same factor.
      3. Round each dimension to nearest multiple of DIM_MULTIPLE.
    """
    src_w, src_h = pil.size
    w, h = float(src_w), float(src_h)

    short_edge = min(w, h)
    if short_edge < MIN_EDGE:
        scale_up = MIN_EDGE / short_edge
        w *= scale_up
        h *= scale_up

    long_edge = max(w, h)
    if long_edge > LONG_EDGE_TARGET:
        scale_down = LONG_EDGE_TARGET / long_edge
        w *= scale_down
        h *= scale_down

    new_w = ((int(round(w)) + DIM_MULTIPLE // 2) // DIM_MULTIPLE) * DIM_MULTIPLE
    new_h = ((int(round(h)) + DIM_MULTIPLE // 2) // DIM_MULTIPLE) * DIM_MULTIPLE
    new_w = max(DIM_MULTIPLE, new_w)
    new_h = max(DIM_MULTIPLE, new_h)

    if (new_w, new_h) == (src_w, src_h):
        return pil
    return pil.resize((new_w, new_h), Image.LANCZOS)


def _face_fraction(rgb: np.ndarray) -> Optional[float]:
    """Return the face bbox height / frame height, or None if no face."""
    h, _ = rgb.shape[:2]
    with _mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5,
    ) as det:
        result = det.process(rgb)
    if not result.detections:
        return None
    bbox = result.detections[0].location_data.relative_bounding_box
    return float(bbox.height)


def _count_faces(rgb: np.ndarray) -> int:
    """Count detected faces.  Salon-flow expects exactly 1.  Group shots
    or background people in frame produce ambiguous swaps so we block."""
    with _mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.4,
    ) as det:
        result = det.process(rgb)
    return 0 if not result.detections else len(result.detections)


def _head_pose_degrees(rgb: np.ndarray) -> Optional[tuple]:
    """Estimate (roll, yaw) in degrees via OpenCV solvePnP with the
    canonical 6-point MediaPipe model.  Roll = in-plane tilt, yaw =
    left-right turn.

    This replaces an earlier nose-to-eye asymmetry heuristic that
    had a hand-tuned 60-degree scale factor calibrated on one corpus
    and would mis-classify on faces of different widths or at
    different camera distances.  solvePnP is camera-model-agnostic
    when we don't have a calibrated camera (we use a generic
    focal-length-equals-image-width approximation that's standard
    practice for uncalibrated head pose).

    Returns None if no face is found.  Pitch is computed but not
    returned because customers nodding slightly is fine and
    constraining it would block too many real uploads.
    """
    h, w = rgb.shape[:2]
    with _mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark

    # Canonical 6-point head model in millimetres (standard PnP reference,
    # works across face sizes because solvePnP is metric-invariant up to
    # scale).  Points: nose tip, chin, left eye outer, right eye outer,
    # left mouth, right mouth.
    model_pts = np.array([
        (0.0,    0.0,    0.0),     # nose tip
        (0.0,   -63.6,  -12.5),    # chin
        (-43.3,  32.7,  -26.0),    # left eye outer corner
        (43.3,   32.7,  -26.0),    # right eye outer corner
        (-28.9, -28.9,  -24.1),    # left mouth corner
        (28.9,  -28.9,  -24.1),    # right mouth corner
    ], dtype=np.float64)
    image_pts = np.array([
        (lm[1].x   * w, lm[1].y   * h),
        (lm[152].x * w, lm[152].y * h),
        (lm[33].x  * w, lm[33].y  * h),
        (lm[263].x * w, lm[263].y * h),
        (lm[61].x  * w, lm[61].y  * h),
        (lm[291].x * w, lm[291].y * h),
    ], dtype=np.float64)

    # Uncalibrated camera assumption: focal length ~ image width, principal
    # point at image centre.  This is the textbook fallback when no
    # calibration is available; absolute angles drift a few degrees on
    # extreme aspect ratios but RELATIVE pose (which is what we gate on)
    # stays stable across face sizes / distances.
    focal = float(w)
    centre = (w * 0.5, h * 0.5)
    cam_mtx = np.array([
        [focal, 0.0,   centre[0]],
        [0.0,   focal, centre[1]],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _tvec = cv2.solvePnP(
        model_pts, image_pts, cam_mtx, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rot_mat, _jac = cv2.Rodrigues(rvec)
    # Decompose rotation to Euler (yaw=Y, pitch=X, roll=Z) using the same
    # convention as cv2.decomposeProjectionMatrix.
    sy = float(np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2))
    if sy > 1e-6:
        roll = float(np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0])))
        yaw = float(np.degrees(np.arctan2(-rot_mat[2, 0], sy)))
    else:
        roll = float(np.degrees(np.arctan2(-rot_mat[0, 1], rot_mat[1, 1])))
        yaw = float(np.degrees(np.arctan2(-rot_mat[2, 0], sy)))
    return roll, yaw


def _detect_glasses(rgb: np.ndarray) -> bool:
    """Heuristic glasses detector: look for a dark, low-chroma horizontal
    band across the eye region.  Glasses frames are dark and saturate
    very little (gray/black), while skin has positive a* (reddish).

    This is INTENTIONALLY a simple heuristic, not a full classifier.
    False positives (heavy eye-bag shadows) are fine because the only
    downstream effect is adding "preserve glasses" to the prompt; if
    Kontext sees no glasses in the source, the clause is a no-op.
    False negatives (very light wire frames, transparent acetate) are
    unavoidable without an ML detector.
    """
    h, w = rgb.shape[:2]
    with _mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(rgb)
    if not result.multi_face_landmarks:
        return False
    lm = result.multi_face_landmarks[0].landmark
    # Eye region: from a bit above the brow line to just below the eye
    # bottoms (landmarks 33/263 = eye outers; 168 = top of nose bridge;
    # 145/374 = eye bottoms).  This box covers where glasses would sit.
    eye_top_y = int(min(lm[33].y, lm[263].y, lm[168].y) * h) - int(h * 0.02)
    eye_bot_y = int(max(lm[145].y, lm[374].y) * h) + int(h * 0.005)
    eye_left_x = int(min(lm[127].x, lm[356].x) * w)
    eye_right_x = int(max(lm[127].x, lm[356].x) * w)
    if eye_bot_y - eye_top_y < 5 or eye_right_x - eye_left_x < 5:
        return False
    band = rgb[max(0, eye_top_y):min(h, eye_bot_y),
               max(0, eye_left_x):min(w, eye_right_x)]
    if band.size == 0:
        return False

    # Skin anchor from the high cheeks - same per-customer normalisation
    # approach as _hair_on_forehead_fraction so dark-skinned customers
    # don't false-positive.
    cheek_pixels = []
    for ci in (50, 280):
        cy = int(lm[ci].y * h)
        cx = int(lm[ci].x * w)
        r = 6
        patch = rgb[max(0, cy - r):min(h, cy + r),
                    max(0, cx - r):min(w, cx + r)]
        if patch.size > 0:
            cheek_pixels.append(patch.reshape(-1, 3))
    if not cheek_pixels:
        return False
    cheek_arr = np.concatenate(cheek_pixels, axis=0)
    cheek_lab = cv2.cvtColor(
        cheek_arr.reshape(1, -1, 3), cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3).astype(np.float32)
    cheek_l_mean = float(cheek_lab[:, 0].mean())
    cheek_chroma_mean = float(np.sqrt(
        (cheek_lab[:, 1] - 128) ** 2 + (cheek_lab[:, 2] - 128) ** 2
    ).mean())

    lab = cv2.cvtColor(band, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    chroma = np.sqrt((lab[:, :, 1] - 128) ** 2 + (lab[:, :, 2] - 128) ** 2)
    glasses_pixel = (l < cheek_l_mean - 35) & (chroma < cheek_chroma_mean - 6)
    glasses_frac = float(glasses_pixel.mean())
    # 8% of the eye band being dark+low-chroma is the empirical threshold
    # where a glasses frame is the most parsimonious explanation; below
    # that, eyebrows + lashes can account for the dark pixels.
    return glasses_frac > 0.08


def _hair_on_forehead_fraction(rgb: np.ndarray) -> Optional[float]:
    """Fraction of the forehead band (from polygon-top to brow line)
    occupied by hair-like pixels - RELATIVE TO the customer's own cheek
    skin tone.

    Absolute "dark" thresholds reject dark-skinned faces in shadow
    (their actual skin reads as dark + low-chroma, same as hair would).
    Instead, sample the customer's cheek as a skin anchor, then call a
    forehead pixel "hair" only if it's BOTH darker than the cheek anchor
    AND has lower chroma (saturation) than the cheek - the chroma drop
    is what reliably separates hair from skin regardless of overall
    luminance.

    Returns None if no face is detected.
    """
    h, w = rgb.shape[:2]
    with _mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    # Forehead band: from above-brows (landmark 10) down to between-brows
    # (landmark 9), bounded horizontally by the temples (127, 356).
    band_top_y = int(lm[10].y * h)
    band_bot_y = int(lm[9].y * h)
    if band_bot_y - band_top_y < 10:
        return 0.0
    band_left_x = int(min(lm[127].x, lm[356].x) * w)
    band_right_x = int(max(lm[127].x, lm[356].x) * w)
    band = rgb[band_top_y:band_bot_y, band_left_x:band_right_x]
    if band.size == 0:
        return 0.0

    # Sample the customer's cheek as a per-image skin anchor.  Use a small
    # patch around landmarks 50 (left cheek) and 280 (right cheek) - these
    # sit on the highest cheekbone area and are typically lit skin even
    # under uneven indoor lighting.
    cheek_pixels = []
    for cheek_idx in (50, 280):
        cy = int(lm[cheek_idx].y * h)
        cx = int(lm[cheek_idx].x * w)
        r = 6
        patch = rgb[max(0, cy - r):min(h, cy + r),
                    max(0, cx - r):min(w, cx + r)]
        if patch.size > 0:
            cheek_pixels.append(patch.reshape(-1, 3))
    if not cheek_pixels:
        return 0.0
    cheek_arr = np.concatenate(cheek_pixels, axis=0)
    cheek_lab = cv2.cvtColor(
        cheek_arr.reshape(1, -1, 3), cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3).astype(np.float32)
    cheek_l_mean = float(cheek_lab[:, 0].mean())
    cheek_chroma_mean = float(np.sqrt(
        (cheek_lab[:, 1] - 128) ** 2 + (cheek_lab[:, 2] - 128) ** 2
    ).mean())

    lab = cv2.cvtColor(band, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    chroma = np.sqrt((lab[:, :, 1] - 128) ** 2 + (lab[:, :, 2] - 128) ** 2)
    # Hair = (significantly darker than cheek skin) AND (less saturated
    # than cheek skin).  The 25-unit L gap and 8-unit chroma gap are
    # scale-invariant relative thresholds, not absolute - they hold up
    # whether the customer's skin is L=70 (deep shadow on dark skin) or
    # L=180 (bright lighting on lighter skin).
    hair = (l < cheek_l_mean - 25) & (chroma < cheek_chroma_mean - 8)
    return float(hair.mean())


def _blur_score(rgb: np.ndarray) -> float:
    """Laplacian variance.  Higher = sharper.  Below ~60 is unusable."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _save_jpeg(pil: Image.Image, target_dir: Path, hint: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    fp = tempfile.NamedTemporaryFile(
        prefix=f"{hint}_", suffix=".jpg", delete=False, dir=str(target_dir),
    )
    pil.save(fp, format="JPEG", quality=92, optimize=True)
    fp.close()
    return Path(fp.name)
