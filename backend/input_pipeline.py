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

# Pre-flight thresholds.  Tuned to be lenient enough that a competent phone
# shot under salon lighting passes, but strict enough to block the obvious
# failure modes (no face, tiny face, camera shake).
MIN_FACE_FRACTION = 0.10     # face bbox height >= 10% of frame height
MAX_FACE_FRACTION = 0.75     # face filling >75% means the camera is too close
MIN_BLUR_SCORE = 60.0        # Laplacian variance; <60 is camera-shake or oof

_mp_face_detection = mp.solutions.face_detection


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

    if blur_score < MIN_BLUR_SCORE:
        warnings.append(
            f"Photo looks soft (blur score {blur_score:.0f} < {MIN_BLUR_SCORE:.0f}). "
            "Hold the camera steady and try again if the preview is unclear.")

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
