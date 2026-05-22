"""End-to-end reference-image inpainting test.

Usage:
    python tests/run_ref_test.py tests/selfies/me.jpg mens_pompadour catalogue/references/ref3_polished.jpg
    python tests/run_ref_test.py tests/selfies/me.jpg mens_curly_volume catalogue/references/ref1_full_beard.jpg --seed 42

Requires REPLICATE_API_TOKEN in .env. Costs ~$0.07-0.10 per call.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from backend.inpaint_with_reference import (  # noqa: E402
    generate_preview_with_reference, RefInpaintError,
)


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Reference-image hair inpainting test (IP-Adapter + inpaint)."
    )
    parser.add_argument("image", type=Path, help="Path to selfie (jpg/png).")
    parser.add_argument("style_id", help="Catalogue style id.")
    parser.add_argument("reference", type=Path, help="Path to hairstyle reference photo.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--ip-weight", type=float, default=1.0,
                        help="IP-Adapter weight (0-2, default 1.0).")
    parser.add_argument("--strength", type=float, default=0.95,
                        help="Inpaint strength (0-1, default 0.95).")
    parser.add_argument("--save-mask", type=Path, default=None,
                        help="Persist the hair mask to this PNG path.")
    args = parser.parse_args()

    for p in (args.image, args.reference):
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 1

    try:
        result = generate_preview_with_reference(
            selfie_path=args.image,
            style_id=args.style_id,
            reference_image_path=args.reference,
            seed=args.seed,
            steps=args.steps,
            guidance=args.guidance,
            ip_adapter_weight=args.ip_weight,
            inpainting_strength=args.strength,
            save_mask_to=args.save_mask,
        )
    except RefInpaintError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2))
    print()
    print(f"Mask: {result.mask_local_path}")
    print(f"Output URL: {result.image_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
