"""End-to-end generation smoke test.

Usage:
    python tests/run_generate_test.py tests/selfies/me.jpg indian_braid_long
    python tests/run_generate_test.py tests/selfies/me.jpg modern_chin_bob --seed 42

Requires REPLICATE_API_TOKEN in .env. Costs ~$0.04 per call (PhotoMaker).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from backend.generate import generate_preview, GenerationError  # noqa: E402


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(description="Hairstyle preview generation test.")
    parser.add_argument("image", type=Path, help="Path to selfie (jpg/png).")
    parser.add_argument("style_id", help="Catalogue style id (e.g. indian_braid_long).")
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed seed.")
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Diffusion steps (default 30). Higher is slower but cleaner.",
    )
    parser.add_argument(
        "--backend", choices=["instantid", "photomaker"], default=None,
        help="Generation backend. Default: instantid (env REPLICATE_BACKEND if set).",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: selfie not found: {args.image}", file=sys.stderr)
        return 1

    try:
        result = generate_preview(
            selfie_path=args.image,
            style_id=args.style_id,
            seed=args.seed,
            num_steps=args.steps,
            backend=args.backend,
        )
    except GenerationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2))
    print()
    print("Preview URL (open in browser):")
    print(f"  {result.image_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
