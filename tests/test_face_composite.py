"""Tests for backend.face_composite.paste_source_face."""
from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
import pytest
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


def test_paste_source_face_beard_mode_preserves_upper_face_only(tmp_path):
    """In beard mode, the polygon must cover eyes + nose + brow (UPPER face),
    so a synthetic red Kontext output is preserved on the JAW/CHIN region
    while source pixels are preserved on the EYES.

    This is the contract that lets Kontext change the beard while we lock
    eye identity."""
    from backend.face_composite import paste_source_face
    from pathlib import Path
    import numpy as np
    from PIL import Image

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SOURCE_MAN = PROJECT_ROOT / "tests" / "selfies" / "test_random_indian_man.jpg"

    kontext_path = tmp_path / "fake_kontext.png"
    src = Image.open(SOURCE_MAN).convert("RGB")
    Image.new("RGB", src.size, (220, 30, 30)).save(kontext_path, format="PNG")

    out = paste_source_face(
        source_path=SOURCE_MAN,
        kontext_output_url_or_path=kontext_path,
        output_dir=tmp_path,
        mode="beard",
    )
    assert out.exists()

    src_arr = np.array(Image.open(SOURCE_MAN).convert("RGB"))
    composed = np.array(Image.open(out).convert("RGB"))
    assert composed.shape == src_arr.shape

    # Sample at the nose tip (MediaPipe landmark 1) - in beard mode the
    # upper-face polygon DOES cover the nose, so this pixel should match source.
    import mediapipe as mp
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(src_arr)
    landmarks = result.multi_face_landmarks[0].landmark
    nose = landmarks[1]
    h, w = src_arr.shape[:2]
    ny, nx = int(nose.y * h), int(nose.x * w)
    src_face = src_arr[ny, nx].astype(int)
    out_face = composed[ny, nx].astype(int)
    assert int(np.abs(out_face - src_face).max()) < 12, (
        "nose pixel must come from source (it's inside the upper-face polygon)"
    )

    # Sample at the chin (landmark 152) - in beard mode chin is OUTSIDE
    # the upper-face polygon, so this pixel should be the Kontext red.
    chin = landmarks[152]
    cy_, cx_ = int(chin.y * h), int(chin.x * w)
    out_chin = composed[cy_, cx_].astype(int)
    assert out_chin[0] > 150, (
        f"chin pixel should be red Kontext fill in beard mode; got {tuple(out_chin)}"
    )


def test_paste_source_face_skin_only_excludes_turban_pixels(tmp_path):
    """Regression for SP 9: turban fabric INSIDE the geometric face polygon
    must be excluded by the skin-only filter, not pasted back from source.

    Picks a pixel that is provably inside the polygon AND inside the
    turban fabric region.  Without the skin filter the composite would
    paste the dark fabric back; with the filter it stays Kontext-cyan.
    """
    import mediapipe as mp
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from backend.face_composite import paste_source_face, FACE_POLYGON_INDICES

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sikh_src = PROJECT_ROOT / "tests" / "selfies" / "young_indian_man.jpg"
    assert sikh_src.exists(), "missing test fixture"

    # Build the geometric polygon ourselves (without the skin filter) to
    # find a probe pixel guaranteed to be inside it.
    import cv2
    src_arr = np.array(Image.open(sikh_src).convert("RGB"))
    h, w = src_arr.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(src_arr)
    landmarks = result.multi_face_landmarks[0].landmark
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
    poly = pts[FACE_POLYGON_INDICES].astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    geom_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(geom_mask, [poly], 255)

    # Find a dark pixel that is INSIDE the polygon and at the top edge
    # (where the turban fabric sits over the temples).  Scan the top 30%
    # of the polygon's bounding box.
    inside = (geom_mask > 0)
    luminance = src_arr.astype(np.int32).sum(axis=2)
    rows_with_poly = np.where(inside.any(axis=1))[0]
    assert len(rows_with_poly) > 0, "no polygon rows found"
    top_y = rows_with_poly[0]
    band_end = top_y + int(0.30 * (rows_with_poly[-1] - top_y))
    # Mask: inside polygon AND in top band AND dark
    candidate = inside & (np.arange(h)[:, None] >= top_y) & (np.arange(h)[:, None] <= band_end) & (luminance < 130)
    ys, xs = np.where(candidate)
    assert len(ys) >= 50, (
        f"expected at least 50 dark-pixel candidates inside the upper "
        f"polygon, found {len(ys)}.  Fixture may have changed."
    )
    # Pick the centroid of the candidate cluster (most likely to be deep
    # inside the turban-fabric region, not a 1-pixel artifact)
    probe_y = int(np.median(ys))
    probe_x = int(np.median(xs))
    probe_pixel_src = src_arr[probe_y, probe_x].astype(int)
    assert probe_pixel_src.sum() < 150, (
        f"probe pixel at ({probe_y},{probe_x}) was supposed to be dark "
        f"turban fabric; got {tuple(probe_pixel_src)}"
    )
    assert geom_mask[probe_y, probe_x] > 0, "probe pixel not inside polygon"

    # Now run the composite with a cyan Kontext fill and verify the probe
    # pixel comes out CYAN (skin filter excluded the turban fabric) rather
    # than DARK FABRIC (it would have been pasted back without the filter).
    kontext_path = tmp_path / "fake_kontext.png"
    Image.new("RGB", (w, h), (40, 200, 220)).save(kontext_path, format="PNG")
    out = paste_source_face(
        source_path=sikh_src,
        kontext_output_url_or_path=kontext_path,
        output_dir=tmp_path,
        head_covering_type="turban",
    )
    composed = np.array(Image.open(out).convert("RGB"))
    probe_pixel_out = composed[probe_y, probe_x].astype(int)
    # Composite should be much closer to cyan (40,200,220) than to source dark fabric
    assert probe_pixel_out[1] > 100 and probe_pixel_out[2] > 100, (
        f"turban-region pixel at ({probe_y},{probe_x}) appears to have been "
        f"pasted from source.  Got {tuple(probe_pixel_out)}, expected closer "
        f"to cyan (40, 200, 220).  Source pixel: {tuple(probe_pixel_src)}. "
        f"The skin-only filter and/or head_covering shrink did not exclude "
        f"this region."
    )


@pytest.mark.parametrize("fixture_name", [
    "young_indian_woman.jpg",
    "young_indian_man.jpg",
    "curly_hair_indian_woman.jpg",
    "dark_skin_indian_man.jpg",
])
@pytest.mark.parametrize("landmark_idx,label", [
    (1,   "nose tip"),
    (50,  "left high cheek"),
    (280, "right high cheek"),
    (152, "chin"),
    (9,   "between eyebrows"),
])
def test_paste_source_face_preserves_skin_at_landmark(
    tmp_path, fixture_name, landmark_idx, label,
):
    """The skin filter must preserve source skin pixels at every face
    landmark across every fixture, even under specular highlight or
    shadow lighting on different parts of the face.

    A regression here means Kontext output is bleeding into source skin
    regions - the customer would see identity drift on cheek/chin/forehead."""
    import mediapipe as mp
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from backend.face_composite import paste_source_face

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    src_path_raw = PROJECT_ROOT / "tests" / "selfies" / fixture_name
    if not src_path_raw.exists():
        pytest.skip(f"fixture {fixture_name} missing")

    # Cap long-edge at 1536 px to mirror production's _resize_for_flux step.
    # Some fixtures are 4480x6720 raw - face_composite was never meant to be
    # called at that resolution and morphology kernels scale to image height.
    src_img = Image.open(src_path_raw).convert("RGB")
    sw, sh = src_img.size
    long_edge = max(sw, sh)
    if long_edge > 1536:
        scale = 1536.0 / long_edge
        new_w = int(round(sw * scale))
        new_h = int(round(sh * scale))
        src_img = src_img.resize((new_w, new_h), Image.LANCZOS)
    src_path = tmp_path / fixture_name
    src_img.save(src_path, format="JPEG", quality=92)

    kontext_path = tmp_path / "fake_kontext.png"
    Image.new("RGB", src_img.size, (255, 0, 255)).save(kontext_path, format="PNG")  # magenta

    out = paste_source_face(
        source_path=src_path,
        kontext_output_url_or_path=kontext_path,
        output_dir=tmp_path,
    )
    composed = np.array(Image.open(out).convert("RGB"))
    src_arr = np.array(Image.open(src_path).convert("RGB"))
    h, w = src_arr.shape[:2]

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(src_arr)
    if not result.multi_face_landmarks:
        pytest.skip(f"MediaPipe found no face in {fixture_name}")
    lm = result.multi_face_landmarks[0].landmark[landmark_idx]
    ly, lx = int(lm.y * h), int(lm.x * w)

    src_skin = src_arr[ly, lx].astype(int)
    out_skin = composed[ly, lx].astype(int)
    diff = int(np.abs(out_skin - src_skin).max())
    assert diff < 35, (
        f"{fixture_name} {label} pixel drifted from source by {diff} "
        f"(source RGB {tuple(src_skin)} -> composite RGB {tuple(out_skin)}). "
        f"Skin filter is over-rejecting legitimate skin at this landmark."
    )


def test_paste_source_face_accepts_head_covering_type_parameter():
    """The function signature must accept head_covering_type as an optional
    parameter.  Pure interface check - no Kontext call required."""
    from backend.face_composite import paste_source_face
    import inspect
    sig = inspect.signature(paste_source_face)
    assert "head_covering_type" in sig.parameters
    param = sig.parameters["head_covering_type"]
    assert param.default is None
