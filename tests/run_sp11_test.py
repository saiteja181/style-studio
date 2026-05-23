"""SP 11 smoke test: hair-masked inpainting on 1 face x 2 styles.

Saves outputs + a mask-overlay debug image to
tests/acceptance/sp11/.  Cost: ~$0.10 per run (1 segmentation + 2
FLUX Fill calls).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "tests" / "acceptance" / "sp11"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["STYLE_STUDIO_UPLOADS_DIR"] = str(OUT_DIR)

# Clear preview cache so we exercise the real pipeline both times.
for f in (PROJECT_ROOT / "catalogue" / "preview_cache").glob("preview_inp_*.png"):
    f.unlink(missing_ok=True)

from backend.input_pipeline import prepare_upload, PreflightError  # noqa: E402
from backend.customer_analysis import analyze_customer  # noqa: E402
from backend.inpaint_engine import generate_hair_preview  # noqa: E402
from backend.kontext_engine import GenerationError  # noqa: E402
from backend.hair_segmenter import segment_hair  # noqa: E402
from backend.mask_builder import build_inpaint_mask, save_mask_preview  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

SRC = PROJECT_ROOT / "tests" / "selfies" / "young_indian_woman.jpg"
STYLES = ["bridal_juda", "curtain_bangs_medium"]


def main() -> int:
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN missing")
        return 1

    try:
        src_norm, report = prepare_upload(
            raw_bytes=SRC.read_bytes(),
            target_dir=OUT_DIR,
            filename_hint="src",
        )
    except PreflightError as e:
        print(f"preflight blocked: {e.report.code} - {e.report.message}")
        return 1
    print(f"preflight: face={report.face_fraction:.2f} "
          f"blur={report.blur_score:.0f} size={report.normalised_size}")

    # First: visualise the segmenter + mask builder output so we can
    # eyeball what FLUX Fill will see.  This catches mask problems
    # (over-segmentation, missing hairline) before spending on Fill.
    print("\n--- mask preview ---")
    src_rgb = np.array(Image.open(src_norm).convert("RGB"))
    raw_mask = segment_hair(src_norm)
    for silhouette in ("short", "medium", "long"):
        m = build_inpaint_mask(raw_mask, expected_silhouette=silhouette,
                               source_height=src_rgb.shape[0])
        save_mask_preview(src_rgb, m,
                          OUT_DIR / f"mask_{silhouette}.png")
        white = int((m > 127).sum())
        print(f"  silhouette={silhouette}: {white} mask pixels "
              f"({100*white/(m.shape[0]*m.shape[1]):.1f}% of frame)")

    profile = analyze_customer(
        selfie_path=src_norm, use_vision_lm=False,
    ).to_dict()
    for style in STYLES:
        print(f"\n=== {style} ===")
        try:
            r = generate_hair_preview(
                source_path=src_norm,
                style_id=style,
                customer_profile=profile,
                seed=42,
            )
            print(f"  verdict={r.validator_verdict}  "
                  f"elapsed={r.elapsed_ms / 1000:.1f}s")
            print(f"  prompt: {r.prompt[:120]}...")
            src_img = OUT_DIR / Path(r.image_url).name
            if src_img.exists():
                dst = OUT_DIR / f"output_{style}.png"
                shutil.copy2(src_img, dst)
                print(f"  saved -> {dst.relative_to(PROJECT_ROOT)}")
        except GenerationError as e:
            print(f"  GenerationError: {e}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
