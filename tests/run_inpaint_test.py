"""End-to-end inpaint pipeline test (hair-only swap, face preserved).

Usage:
    python tests/run_inpaint_test.py tests/selfies/me.jpg mens_textured_crop
    python tests/run_inpaint_test.py tests/selfies/me.jpg modern_chin_bob --seed 42 --strength 0.9

Requires REPLICATE_API_TOKEN in .env. Costs ~$0.05-0.08 per call
(segmentation + inpainting combined).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from backend.inpaint import (  # noqa: E402
    generate_preview_inpaint, generate_preview_auto,
    generate_preview_expert, generate_preview_erase_then_inpaint,
    InpaintError,
)
from backend.inpaint_with_reference import (  # noqa: E402
    generate_preview_with_reference, RefInpaintError,
)


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Hair-only inpainting test (face/clothes/background preserved)."
    )
    parser.add_argument("image", type=Path, help="Path to selfie (jpg/png).")
    parser.add_argument("style_id", help="Catalogue style id.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50,
                        help="Denoising steps. FLUX: 15-50 (default 50). IP-Adapter: 20-50.")
    parser.add_argument("--guidance", type=float, default=35.0,
                        help="Guidance scale. FLUX: 1.5-100 (default 35). IP-Adapter: 4-12.")
    parser.add_argument("--save-mask", type=Path, default=None,
                        help="Persist the hair mask to this PNG path for QA.")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Path to a hairstyle reference photo. If set, routes "
                             "to the IP-Adapter pipeline (reference-driven inpaint) "
                             "instead of FLUX text-prompt inpaint.")
    parser.add_argument("--ip-weight", type=float, default=1.0,
                        help="IP-Adapter weight (only with --reference, default 1.0).")
    parser.add_argument("--strength", type=float, default=0.95,
                        help="Inpaint strength (only with --reference, default 0.95).")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-derive the prompt from the style's reference photo "
                             "via Qwen2-VL. Ignores --reference; uses catalogue.")
    parser.add_argument("--expert", action="store_true",
                        help="Top-quality: Anthropic vision sees both customer photo "
                             "and reference, writes adapted prompt. Requires "
                             "ANTHROPIC_API_KEY in .env.")
    parser.add_argument("--erase", action="store_true",
                        help="Two-pass: erase existing hair to bald scalp, then "
                             "inpaint the new style on the bald canvas. Forces "
                             "real hair-style transformations.")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: selfie not found: {args.image}", file=sys.stderr)
        return 1

    if args.erase:
        try:
            erase_result = generate_preview_erase_then_inpaint(
                selfie_path=args.image,
                style_id=args.style_id,
                seed=args.seed,
                save_mask_to=args.save_mask,
                steps=args.steps,
                guidance=args.guidance,
            )
        except InpaintError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        print(json.dumps(erase_result.to_dict(), indent=2))
        print()
        print(f"Local hair mask: {erase_result.mask_local_path}")
        print(f"Bald intermediate URL: {erase_result.raw_image_url}")
        print(f"Final image URL: {erase_result.image_url}")
        return 0

    if args.expert:
        try:
            expert_result = generate_preview_expert(
                selfie_path=args.image,
                style_id=args.style_id,
                seed=args.seed,
                save_mask_to=args.save_mask,
                steps=args.steps,
                guidance=args.guidance,
            )
        except InpaintError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        print(json.dumps(expert_result.to_dict(), indent=2))
        print()
        print(f"Local hair mask: {expert_result.mask_local_path}")
        print(f"Inpainted image URL: {expert_result.image_url}")
        return 0

    if args.auto:
        try:
            auto_result = generate_preview_auto(
                selfie_path=args.image,
                style_id=args.style_id,
                seed=args.seed,
                save_mask_to=args.save_mask,
                steps=args.steps,
                guidance=args.guidance,
            )
        except InpaintError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        print(json.dumps(auto_result.to_dict(), indent=2))
        print()
        print(f"Local hair mask: {auto_result.mask_local_path}")
        print(f"Inpainted image URL: {auto_result.image_url}")
        return 0

    if args.reference is not None:
        if not args.reference.exists():
            print(f"ERROR: reference not found: {args.reference}", file=sys.stderr)
            return 1
        try:
            ref_result = generate_preview_with_reference(
                selfie_path=args.image,
                style_id=args.style_id,
                reference_image_path=args.reference,
                seed=args.seed,
                steps=args.steps if args.steps <= 50 else 50,
                guidance=args.guidance if args.guidance <= 12 else 7.0,
                ip_adapter_weight=args.ip_weight,
                inpainting_strength=args.strength,
                save_mask_to=args.save_mask,
            )
        except RefInpaintError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        print(json.dumps(ref_result.to_dict(), indent=2))
        print()
        print(f"Local hair mask: {ref_result.mask_local_path}")
        print(f"Reference used: {ref_result.reference_path}")
        print(f"Inpainted image URL: {ref_result.image_url}")
        return 0

    try:
        result = generate_preview_inpaint(
            selfie_path=args.image,
            style_id=args.style_id,
            seed=args.seed,
            steps=args.steps,
            guidance=args.guidance,
            save_mask_to=args.save_mask,
        )
    except InpaintError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2))
    print()
    print(f"Local hair mask: {result.mask_local_path}")
    print(f"Inpainted image URL: {result.image_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
