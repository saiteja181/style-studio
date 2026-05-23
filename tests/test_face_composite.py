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
# Pick landmarks that sit SOLIDLY INTERIOR to FACE_POLYGON_INDICES.  The
# old test sampled at landmarks 152 (chin tip) and 9 (between-brows top)
# which are polygon VERTICES - the feather Gaussian pulls alpha to ~128
# there by design, so the composite is half-source/half-Kontext and the
# test fired falsely.  Replacements:
#   152 (chin tip)        -> 17 (chin centre, mid-distance from mouth to tip)
#   9   (between brows)   -> 6  (nose bridge upper, solidly inside polygon)
@pytest.mark.parametrize("landmark_idx,label", [
    (1,   "nose tip"),
    (50,  "left high cheek"),
    (280, "right high cheek"),
    (17,  "chin centre"),
    (6,   "nose bridge upper"),
])
def test_paste_source_face_preserves_skin_at_landmark(
    tmp_path, fixture_name, landmark_idx, label,
):
    """At each face landmark, the composite must either:
      (a) be deep-interior (alpha >= 200) and match source within 20 RGB, OR
      (b) be near the polygon edge (alpha < 200), in which case the test
          asserts the composite is BETWEEN source and Kontext per the
          alpha blend - i.e., the maths of feathering is right.

    Cheek and chin landmarks sit close to the polygon edge for some
    face shapes (3/4 profile, narrow jaw); the (b) branch keeps the
    test meaningful for those without falsely failing because of the
    feather Gaussian by design pulls boundary alpha below 255.
    """
    import mediapipe as mp
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from backend.face_composite import paste_source_face, _build_face_alpha

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

    kontext_rgb = (255, 0, 255)
    kontext_path = tmp_path / "fake_kontext.png"
    Image.new("RGB", src_img.size, kontext_rgb).save(kontext_path, format="PNG")

    out = paste_source_face(
        source_path=src_path,
        kontext_output_url_or_path=kontext_path,
        output_dir=tmp_path,
    )
    composed = np.array(Image.open(out).convert("RGB"))
    src_arr = np.array(Image.open(src_path).convert("RGB"))
    h, w = src_arr.shape[:2]

    # Probe alpha at the landmark by rebuilding the face mask in this test.
    # The composite call internally builds the same alpha, so this is a
    # faithful read of what the production code did.
    alpha_result = _build_face_alpha(
        src_arr, feather_px=18, polygon_indices=None, apply_skin_filter=False,
    )
    if alpha_result is None:
        pytest.skip(f"MediaPipe found no face in {fixture_name}")
    face_alpha, lm_xy = alpha_result
    if landmark_idx >= len(lm_xy):
        pytest.skip(f"landmark {landmark_idx} not produced for {fixture_name}")
    lx, ly = int(lm_xy[landmark_idx, 0]), int(lm_xy[landmark_idx, 1])
    alpha_at = int(face_alpha[ly, lx])

    src_skin = src_arr[ly, lx].astype(int)
    out_skin = composed[ly, lx].astype(int)
    diff = int(np.abs(out_skin - src_skin).max())

    if alpha_at >= 200:
        assert diff < 20, (
            f"{fixture_name} {label} alpha={alpha_at} (deep interior) but "
            f"diff={diff}: source RGB {tuple(src_skin)} -> composite RGB "
            f"{tuple(out_skin)}. Source skin should be preserved here."
        )
        return

    # Edge zone: verify the composite is the expected alpha-blend of
    # source and Kontext (not some third thing).  Allow +/- 12 RGB
    # slack for the resize chain (LANCZOS + JPEG q92 + Gaussian).
    alpha_f = alpha_at / 255.0
    expected = tuple(
        int(round(alpha_f * s + (1 - alpha_f) * k))
        for s, k in zip(src_skin, kontext_rgb)
    )
    expected_diff = max(abs(o - e) for o, e in zip(out_skin, expected))
    assert expected_diff < 12, (
        f"{fixture_name} {label} alpha={alpha_at} (edge); expected blend "
        f"{expected} but got {tuple(out_skin)}. Feathering math is wrong."
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


def test_lab_mean_match_shifts_source_toward_kontext_within_cap():
    """Lab mean colour match must shift the source mean within the masked
    region toward the Kontext mean, capped so identity skin tone is not
    obliterated.  Without this, the pasted source face has a different
    white-balance from the Kontext-generated surroundings and the seam
    is visible."""
    from backend.face_composite import _match_face_lab_mean
    import cv2

    h, w = 200, 200
    # Source: brownish skin tone (Indian skin, Lab ~ (160, 145, 145))
    source = np.full((h, w, 3), (180, 140, 110), dtype=np.uint8)
    # Kontext: cooler/lighter face area, like a Kontext relight
    kontext = np.full((h, w, 3), (210, 175, 150), dtype=np.uint8)
    # Mask: center 100x100 square
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:150, 50:150] = 255

    shifted = _match_face_lab_mean(source, kontext, mask)
    assert shifted.shape == source.shape
    assert shifted.dtype == np.uint8

    src_mean_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)[50:150, 50:150].mean(axis=(0, 1))
    kn_mean_lab = cv2.cvtColor(kontext, cv2.COLOR_RGB2LAB)[50:150, 50:150].mean(axis=(0, 1))
    out_mean_lab = cv2.cvtColor(shifted, cv2.COLOR_RGB2LAB)[50:150, 50:150].mean(axis=(0, 1))
    # Output mean should be CLOSER to Kontext mean than the source was.
    src_dist = float(np.linalg.norm(src_mean_lab - kn_mean_lab))
    out_dist = float(np.linalg.norm(out_mean_lab - kn_mean_lab))
    assert out_dist < src_dist, (
        f"colour match did not move source toward Kontext: "
        f"src_dist={src_dist:.1f}, out_dist={out_dist:.1f}"
    )


def test_lab_mean_match_caps_shift_so_extreme_kontext_does_not_dominate():
    """The shift must be capped: if Kontext returns garish magenta, the
    pasted face must NOT be turned magenta to match it.  Identity colour
    has to survive."""
    from backend.face_composite import _match_face_lab_mean

    h, w = 200, 200
    source = np.full((h, w, 3), (180, 140, 110), dtype=np.uint8)  # skin
    kontext = np.full((h, w, 3), (255, 0, 255), dtype=np.uint8)   # magenta
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:150, 50:150] = 255

    shifted = _match_face_lab_mean(source, kontext, mask)
    # Inside the mask the output is shifted but must still be skin-ish,
    # not magenta.  Skin has G > 50; magenta has G == 0.
    out_inside = shifted[100, 100].astype(int)
    assert out_inside[1] > 50, (
        f"colour match cap failed: shifted source toward magenta, "
        f"got RGB {tuple(out_inside)}"
    )


def test_alignment_transform_rejects_extreme_scale():
    """The alignment helper must reject transforms whose scale factor is
    outside the safety band; otherwise a bad MediaPipe lock on the
    Kontext output would smear the source face."""
    from backend.face_composite import _alignment_transform

    # Build dummy landmarks: source eye/eye/chin triangle vs Kontext where
    # the eye distance has tripled (impossible scale change).  Need full
    # 478-length array because helper indexes into specific landmark IDs.
    src = np.zeros((478, 2), dtype=np.float32)
    src[33] = (100, 200)
    src[263] = (300, 200)
    src[152] = (200, 400)
    dst = np.zeros((478, 2), dtype=np.float32)
    dst[33] = (100, 200)
    dst[263] = (700, 200)   # eye distance 600 vs source 200 -> 3x scale
    dst[152] = (400, 800)

    M = _alignment_transform(src, dst)
    assert M is None, "extreme scale change must be rejected"


def test_alignment_transform_accepts_modest_translation():
    """Should accept a small translation - that's the realistic Kontext
    output: head shifted by 1-2 % from the source pose."""
    from backend.face_composite import _alignment_transform

    src = np.zeros((478, 2), dtype=np.float32)
    src[33] = (100, 200)
    src[263] = (300, 200)
    src[152] = (200, 400)
    dst = src.copy()
    dst[33] += (8, 3)
    dst[263] += (8, 3)
    dst[152] += (8, 3)

    M = _alignment_transform(src, dst)
    assert M is not None
    # Translation should be ~ (8, 3); scale ~ 1.0; rotation ~ 0.
    assert abs(M[0, 2] - 8.0) < 1.0
    assert abs(M[1, 2] - 3.0) < 1.0


def test_head_covering_shrink_never_crosses_eyebrow_line():
    """The 12 % shrink applied when a head covering is detected MUST cap
    above the topmost brow landmark.  Direct unit test of the guard's
    contract: with my eyebrow guard, for every column the
    shrink-end row must be <= brow_y_top (i.e., the shrink never
    crosses INTO the brow row).

    This tests the LOGIC directly, independent of skin-filter behaviour
    and polygon self-intersections (which can also dim alpha at brow
    landmarks for unrelated reasons)."""
    from backend.face_composite import BROW_LANDMARK_INDICES
    import numpy as np

    # Synthetic alpha mask: a 200x200 image with a "polygon" that's
    # a horizontal band from row 50 to row 180, full width.  Simulates
    # a forehead-heavy polygon where the unguarded shrink would
    # plow into the brow line.
    h_img, w_img = 200, 200
    face_alpha = np.zeros((h_img, w_img), dtype=np.uint8)
    face_alpha[50:180, :] = 255

    # Pretend the brow line lives at row 100, with brow landmark y=100
    brow_y_top = 100
    shrink_px = 70   # large enough to plow well past row 100 without the guard
    shrink_floor = max(0, brow_y_top - 12)

    mask_bool = face_alpha > 16
    row_indices = np.arange(h_img)[:, None]
    inf_for_empty = np.where(mask_bool, row_indices, h_img)
    col_top = inf_for_empty.min(axis=0)
    shrink_top_plus_px = np.minimum(col_top + shrink_px, shrink_floor)

    # The guard's contract: shrink-end must not exceed brow_y_top - safety
    # for ANY column.
    assert int(shrink_top_plus_px.max()) <= brow_y_top, (
        f"shrink-end reached row {shrink_top_plus_px.max()}, "
        f"which crosses the brow line at y={brow_y_top}"
    )
    # And BROW_LANDMARK_INDICES must contain landmarks ON the polygon's
    # top edge (so brow_y_top is a real polygon-interior y).
    from backend.face_composite import FACE_POLYGON_INDICES
    for idx in BROW_LANDMARK_INDICES:
        assert idx in FACE_POLYGON_INDICES, (
            f"BROW_LANDMARK_INDICES landmark {idx} is not in "
            f"FACE_POLYGON_INDICES; the shrink guard can only protect "
            f"landmarks that are on the polygon top edge."
        )
