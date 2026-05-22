"""Tests for backend.face_composite.paste_source_face."""
from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# This file must be a frontal face photo that MediaPipe FaceMesh can detect at
# min_detection_confidence=0.5.  If you replace the fixture, verify that
# _find_nose_tip_pixel() succeeds on the new file before committing.
SOURCE_MAN = PROJECT_ROOT / "tests" / "selfies" / "test_random_indian_man.jpg"


def _make_red_kontext_like(source_path: Path, out_path: Path) -> None:
    """Save an image the same size as the source, filled with pure red.
    Stands in for a Kontext output so we can verify the composite preserves
    source pixels in the face region."""
    src = Image.open(source_path).convert("RGB")
    red = Image.new("RGB", src.size, (220, 30, 30))
    red.save(out_path, format="PNG")


def _find_nose_tip_pixel(source_path: Path) -> tuple[int, int]:
    """Return (y, x) pixel coordinates of the source photo's nose tip.

    Uses MediaPipe FaceMesh landmark index 1 (nose tip) which is by definition
    inside the face polygon.  We compute this in the test instead of assuming
    the geometric centre because portrait fixtures often have the face in the
    upper portion of a landscape frame, not at h//2, w//2.
    """
    rgb = np.array(Image.open(source_path).convert("RGB"))
    h, w = rgb.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(rgb)
    assert result.multi_face_landmarks, "MediaPipe found no face in fixture"
    nose = result.multi_face_landmarks[0].landmark[1]
    return int(nose.y * h), int(nose.x * w)


def test_paste_source_face_preserves_face_replaces_background(tmp_path):
    from backend.face_composite import paste_source_face

    kontext_path = tmp_path / "fake_kontext.png"
    _make_red_kontext_like(SOURCE_MAN, kontext_path)

    out = paste_source_face(
        source_path=SOURCE_MAN,
        kontext_output_url_or_path=kontext_path,
        output_dir=tmp_path,
    )
    assert out.exists(), "composite output file was not written"

    src = np.array(Image.open(SOURCE_MAN).convert("RGB"))
    composed = np.array(Image.open(out).convert("RGB"))
    assert composed.shape == src.shape, "composite must match source dimensions"

    # Sample at the nose tip - guaranteed inside the face polygon regardless
    # of frame orientation.
    ny, nx = _find_nose_tip_pixel(SOURCE_MAN)
    src_face = src[ny, nx].astype(int)
    out_face = composed[ny, nx].astype(int)
    diff_face = int(np.abs(out_face - src_face).max())
    assert diff_face < 8, (
        f"face pixel at nose tip drifted from source by {diff_face}; "
        f"polygon may not cover the nose tip"
    )

    # Top-left corner - should be the red Kontext background (alpha = 0)
    out_corner = composed[8, 8].astype(int)
    assert out_corner[0] > 180 and out_corner[1] < 80 and out_corner[2] < 80, (
        f"top-left corner = {tuple(out_corner)}, expected red Kontext pixel"
    )


def test_no_face_in_source_returns_kontext_unchanged(tmp_path):
    """If MediaPipe can't find a face in the source, the function must NOT
    crash - it should ship the Kontext output as-is.  Tests defence-in-depth
    against unusual inputs that slipped past pre-flight."""
    from backend.face_composite import paste_source_face

    # Source with no face: solid grey.  Same dims as the man photo so we
    # don't accidentally exercise the resize path here.
    src_arr = np.full((800, 1216, 3), 128, dtype=np.uint8)
    blank_src = tmp_path / "blank.jpg"
    Image.fromarray(src_arr).save(blank_src, format="JPEG", quality=92)

    kontext = tmp_path / "kontext.png"
    Image.new("RGB", (1216, 800), (220, 30, 30)).save(kontext, format="PNG")

    out = paste_source_face(
        source_path=blank_src,
        kontext_output_url_or_path=kontext,
        output_dir=tmp_path,
    )
    out_arr = np.array(Image.open(out).convert("RGB"))
    # Should be the Kontext red, not the grey source - because we fell back
    # to shipping Kontext when face detection failed.
    assert out_arr[400, 600, 0] > 180, "expected Kontext red, got something else"
    assert out_arr[400, 600, 1] < 80
    assert out_arr[400, 600, 2] < 80
