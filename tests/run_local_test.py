"""Local sanity check for face analysis.

Usage:
    python tests/run_local_test.py tests/selfies/me.jpg
    python tests/run_local_test.py tests/selfies/me.jpg --debug-out tests/debug_out/me_annotated.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Make `backend` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.face_analysis import analyze_face, draw_landmarks_overlay  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Face analysis smoke test.")
    parser.add_argument("image", type=Path, help="Path to a selfie (jpg/png).")
    parser.add_argument(
        "--debug-out", type=Path, default=None,
        help="If set, saves the image with landmarks + label drawn.",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: file not found: {args.image}", file=sys.stderr)
        return 1

    try:
        pil = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"ERROR: could not open image ({e})", file=sys.stderr)
        return 1

    arr = np.array(pil)
    print(f"Loaded: {args.image.name}  ({arr.shape[1]}x{arr.shape[0]} px)")

    result = analyze_face(arr)
    if result is None:
        print("No face detected. Try a clearer front-facing photo.", file=sys.stderr)
        return 2

    print("Face detected.")
    print(json.dumps(result.to_dict(), indent=2))

    if args.debug_out:
        args.debug_out.parent.mkdir(parents=True, exist_ok=True)
        overlay_rgb = draw_landmarks_overlay(arr, result)
        # cv2 writes BGR, so convert.
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(args.debug_out), overlay_bgr)
        print(f"Annotated image saved to: {args.debug_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
