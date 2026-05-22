"""Diagnostic: dump every intermediate of the pipeline so we can SEE
where the dramatic-transformation budget is being lost.

For the young Indian man source, produce:
  - source.jpg                  the pre-flight-normalised source
  - mask_alone.png              the binary hair mask, black on white
  - mask_overlay.png            mask shown semi-transparently OVER the source
  - bald_canvas.png             the result of pass-1 (erase to bald scalp)
  - after_pompadour.png         the final pass-2 + harmonise output

Looking at these answers three questions:
  1. Does the mask actually cover the customer's real hair?
  2. Is the bald pass actually erasing hair to scalp, or just lightly editing?
  3. Is the final result conservative because of the mask, the bald canvas,
     or FLUX itself?
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

import urllib.request  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from backend.input_pipeline import prepare_upload  # noqa: E402
from backend.inpaint import (  # noqa: E402
    build_shared_bald_canvas,
    generate_preview_erase_then_inpaint,
)
from backend.customer_analysis import analyze_customer  # noqa: E402

OUT_DIR = PROJECT_ROOT / "tests" / "debug_pipeline"
SRC_PATH = PROJECT_ROOT / "tests" / "selfies" / "test_random_indian_man.jpg"


def download(url: str, out_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp:
        out_path.write_bytes(resp.read())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = SRC_PATH.read_bytes()
    source_path, report = prepare_upload(
        raw_bytes=raw, target_dir=OUT_DIR, filename_hint="source",
    )
    print(f"preflight ok: {report.normalised_size} face={report.face_fraction:.2f}")

    profile = analyze_customer(selfie_path=source_path, use_vision_lm=False).to_dict()
    print(f"hair_color={profile['hair_color_rgb']}")

    # Build the shared bald canvas - this also saves the mask file.
    mask_save = OUT_DIR / "mask_alone.png"
    print("building bald canvas (1 FLUX call)...")
    shared = build_shared_bald_canvas(
        selfie_path=source_path, seed=42, save_mask_to=mask_save,
    )
    print(f"  bald_url={shared['bald_url']}")
    print(f"  mask_path={shared['mask_path']}")

    # Download the bald canvas so we can look at it.
    bald_local = OUT_DIR / "bald_canvas.png"
    download(shared["bald_url"], bald_local)
    print(f"saved bald canvas -> {bald_local}")

    # Build a mask overlay: red where mask=255, source where mask=0.
    src_rgb = np.array(Image.open(source_path).convert("RGB"))
    mask_g = np.array(Image.open(mask_save).convert("L"))
    if mask_g.shape != src_rgb.shape[:2]:
        mask_g = np.array(Image.fromarray(mask_g).resize(
            (src_rgb.shape[1], src_rgb.shape[0]), Image.LANCZOS))
    alpha = (mask_g.astype(np.float32) / 255.0)[..., None]
    red_layer = np.zeros_like(src_rgb)
    red_layer[..., 0] = 220
    overlay = (src_rgb.astype(np.float32) * (1 - 0.55 * alpha) +
               red_layer.astype(np.float32) * (0.55 * alpha))
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
        OUT_DIR / "mask_overlay.png")
    print(f"saved mask overlay -> {OUT_DIR / 'mask_overlay.png'}")

    # Run pompadour through the full pipeline using the shared bald.
    print("running pompadour pass 2...")
    result = generate_preview_erase_then_inpaint(
        selfie_path=source_path, style_id="mens_pompadour", seed=42,
        customer_profile=profile, shared_bald=shared,
        validate=False, max_retries=0,
    )
    print(f"  final url: {result.image_url}")

    # Locate the final harmonised file.
    final_name = Path(result.image_url).name
    src_dir = source_path.parent
    for c in (src_dir, PROJECT_ROOT / "tests" / "uploads"):
        cand = c / final_name
        if cand.exists():
            (OUT_DIR / "after_pompadour.png").write_bytes(cand.read_bytes())
            print(f"saved final -> {OUT_DIR / 'after_pompadour.png'}")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
