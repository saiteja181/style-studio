"""Phase-by-phase smoke test: run the production pipeline on 2 faces and
save the outputs + validator verdict JSON into tests/acceptance/phase_N/.

Usage: python tests/run_phase_test.py <phase_number>
Cost:  ~$0.13 per run (2 faces x ~$0.065 each).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

# Salon-flow test fixtures: clear front-facing photos of one woman + one
# man, no posed bridal jewellery, no 3/4 profile, no fashion-shoot edits.
# These represent what an actual salon staff would capture on the
# customer's first visit.
CASES = [
    ("woman", PROJECT_ROOT / "tests" / "selfies" / "young_indian_woman.jpg",
     "bridal_juda"),
    ("man",   PROJECT_ROOT / "tests" / "selfies" / "round_face_indian_man.jpg",
     "mens_pompadour"),
]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_phase_test.py <phase_number>")
        return 2
    phase = int(sys.argv[1])
    out_dir = PROJECT_ROOT / "tests" / "acceptance" / f"phase_{phase}"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["STYLE_STUDIO_UPLOADS_DIR"] = str(out_dir)

    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN missing")
        return 1

    from backend.input_pipeline import prepare_upload, PreflightError
    from backend.customer_analysis import analyze_customer
    from backend.kontext_engine import generate_preview, GenerationError

    results = []
    for tag, src_path, style_id in CASES:
        print(f"\n=== {tag}: {style_id} ===")
        if not src_path.exists():
            print(f"  source missing: {src_path}")
            results.append({"tag": tag, "error": "source missing"})
            continue
        try:
            src_norm, report = prepare_upload(
                raw_bytes=src_path.read_bytes(),
                target_dir=out_dir,
                filename_hint=f"src_{tag}",
            )
        except PreflightError as e:
            print(f"  preflight BLOCKED: {e.report.code} - {e.report.message}")
            results.append({
                "tag": tag, "preflight_block": e.report.code,
                "preflight_message": e.report.message,
            })
            continue
        print(f"  preflight: face={report.face_fraction:.2f} blur={report.blur_score:.0f} "
              f"size={report.normalised_size}")

        profile = analyze_customer(selfie_path=src_norm, use_vision_lm=False).to_dict()
        t0 = time.time()
        try:
            r = generate_preview(
                source_path=src_norm, style_id=style_id,
                customer_profile=profile, seed=42, max_retries=1,
                head_covering_type=report.head_covering.get("covering_type")
                if report.head_covering.get("detected") else None,
            )
        except GenerationError as e:
            print(f"  generation FAILED: {e}")
            results.append({"tag": tag, "error": str(e)})
            continue
        elapsed = time.time() - t0
        # Copy the final image into the phase folder with a predictable name.
        final_name = Path(r.image_url).name
        for candidate in (out_dir, out_dir.parent / "uploads", PROJECT_ROOT / "tests" / "uploads"):
            cand = candidate / final_name
            if cand.exists():
                dst = out_dir / f"{tag}__{style_id}.png"
                shutil.copy2(cand, dst)
                print(f"  saved {dst.name} (verdict={r.validator_verdict}, "
                      f"{elapsed:.1f}s, retries={r.retries})")
                break
        results.append({
            "tag": tag, "style": style_id,
            "verdict": r.validator_verdict, "retries": r.retries,
            "elapsed_ms": r.elapsed_ms,
        })

    (out_dir / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    print(f"\nresults summary -> {out_dir / 'summary.json'}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
