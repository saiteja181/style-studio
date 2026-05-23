"""Compare the default cdingram/face-swap (inswapper_128) against a
better Replicate face-swap model on one face + two styles.

Usage:
    python tests/run_model_compare.py
"""
from __future__ import annotations

import os
# Set env BEFORE importing backend so the face-swap model override
# takes effect on module load.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# codeplugtech/face-swap: ~$0.0012/call, fast (typically <15s), uses
# the modern inswapper variant.  Bare model name so Replicate auto-
# resolves the latest version.  Swap to "easel/advanced-face-swap"
# for the higher-fidelity paid model if you want to compare both.
os.environ["STYLE_STUDIO_FACESWAP_MODEL"] = "codeplugtech/face-swap"

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "tests" / "acceptance" / "model_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["STYLE_STUDIO_UPLOADS_DIR"] = str(OUT_DIR)

# Clear preview cache + validator cache so we get fresh outputs to
# evaluate the new model fairly.
for f in (PROJECT_ROOT / "catalogue" / "preview_cache").glob("preview_*.png"):
    f.unlink(missing_ok=True)
for f in (PROJECT_ROOT / "catalogue" / "validation_cache").glob("validate_*.json"):
    f.unlink(missing_ok=True)

from backend.input_pipeline import prepare_upload, PreflightError  # noqa: E402
from backend.customer_analysis import analyze_customer  # noqa: E402
from backend.kontext_engine import generate_preview, GenerationError  # noqa: E402
from backend.face_swap import PRIMARY_MODEL  # noqa: E402

print(f"face-swap model in use: {PRIMARY_MODEL}\n")

SRC = PROJECT_ROOT / "tests" / "selfies" / "young_indian_woman.jpg"
STYLES = ["bridal_juda", "curtain_bangs_medium"]


def main() -> int:
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN missing in .env")
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

    profile = analyze_customer(
        selfie_path=src_norm, use_vision_lm=False,
    ).to_dict()
    glasses = getattr(report, "glasses_detected", False)

    for style in STYLES:
        print(f"\n=== {style} ===")
        try:
            r = generate_preview(
                source_path=src_norm,
                style_id=style,
                customer_profile=profile,
                seed=42,
                max_retries=0,   # one shot per style for a clean comparison
                glasses_detected=glasses,
                head_covering_type=(
                    report.head_covering.get("covering_type")
                    if report.head_covering.get("detected") else None
                ),
            )
            print(f"  verdict={r.validator_verdict}  "
                  f"elapsed={r.elapsed_ms / 1000:.1f}s")
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
