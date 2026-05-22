"""Run face analysis on every selfie in tests/selfies/ and save annotated overlays.

Usage:
    python tests/run_batch_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.face_analysis import analyze_face, draw_landmarks_overlay  # noqa: E402
from backend.hair_estimation import estimate_hairline  # noqa: E402

SELFIE_DIR = ROOT / "tests" / "selfies"
DEBUG_DIR = ROOT / "tests" / "debug_out"
SUPPORTED = {".jpg", ".jpeg", ".png"}


def main() -> int:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    selfies = sorted(
        p for p in SELFIE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    if not selfies:
        print(f"No selfies found in {SELFIE_DIR}", file=sys.stderr)
        return 1

    print(f"Analyzing {len(selfies)} selfie(s)...\n")
    summary = []

    for path in selfies:
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  {path.name:50s}  ERROR: cannot open ({e})")
            continue

        arr = np.array(pil)
        result = analyze_face(arr)

        if result is None:
            row = {
                "file": path.name,
                "dims": f"{arr.shape[1]}x{arr.shape[0]}",
                "face_detected": False,
                "face_shape": None,
                "L/W": None,
            }
            summary.append(row)
            print(f"  {path.name:50s}  NO FACE DETECTED")
            continue

        # Save annotated overlay
        overlay_rgb = draw_landmarks_overlay(arr, result)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        debug_path = DEBUG_DIR / f"{path.stem}_annotated.jpg"
        cv2.imwrite(str(debug_path), overlay_bgr)

        # Also test hair estimation
        import mediapipe as mp
        mp_fm = mp.solutions.face_mesh
        with mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1,
                            refine_landmarks=True, min_detection_confidence=0.5) as fm:
            mp_result = fm.process(arr)
        landmarks = mp_result.multi_face_landmarks[0].landmark
        h, w = arr.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
        hairline = estimate_hairline(pts)

        row = {
            "file": path.name,
            "dims": f"{arr.shape[1]}x{arr.shape[0]}",
            "face_detected": True,
            "face_shape": result.face_shape,
            "L/W": result.metrics["length_to_width"],
            "fore/jaw": result.metrics["forehead_to_jaw"],
            "hairline_pts": len(hairline.points),
            "annotated": str(debug_path.relative_to(ROOT)),
        }
        summary.append(row)
        print(f"  {path.name:50s}  shape={result.face_shape:7s}  L/W={result.metrics['length_to_width']:.3f}  fore/jaw={result.metrics['forehead_to_jaw']:.3f}")

    print()
    print("Summary (JSON):")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Annotated overlays saved to: {DEBUG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
