"""Full salon consultation pipeline test.

Runs:
  1. Customer analysis (MediaPipe + optional vision LM)
  2. Style recommendation (rule-based matcher against tagged catalogue)
  3. Reports a consultation summary

Usage:
    python tests/run_consultation.py tests/selfies/me.jpg
    python tests/run_consultation.py tests/selfies/me.jpg --vision --gender male
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from backend.customer_analysis import analyze_customer, AnalysisError  # noqa: E402
from backend.style_matcher import recommend_styles  # noqa: E402


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Run customer analysis + style recommendation."
    )
    parser.add_argument("image", type=Path, help="Customer selfie path.")
    parser.add_argument("--vision", action="store_true",
                        help="Use vision LM to enrich profile (texture, hairline, gender).")
    parser.add_argument("--gender", default=None,
                        choices=["male", "female", "unknown"],
                        help="Override gender estimation.")
    parser.add_argument("--occasion", default=None,
                        help="Filter recommendations by occasion (daily/party/etc).")
    parser.add_argument("--top-n", type=int, default=5,
                        help="How many recommendations to return.")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: file not found: {args.image}", file=sys.stderr)
        return 1

    print("=" * 70)
    print(f"CUSTOMER ANALYSIS  -  {args.image.name}")
    print("=" * 70)

    try:
        profile = analyze_customer(
            selfie_path=args.image,
            use_vision_lm=args.vision,
            gender_hint=args.gender,
        )
    except AnalysisError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"  Face shape       : {profile.face_shape}")
    print(f"  Jawline          : {profile.jawline}")
    print(f"  Skin tone        : {profile.skin_tone_bucket}  (RGB {profile.skin_rgb})")
    print(f"  Hair color       : {profile.hair_color_descriptor}  (RGB {profile.hair_color_rgb})")
    print(f"  Hair texture     : {profile.hair_texture}")
    print(f"  Hairline shape   : {profile.hairline_shape}")
    print(f"  Estimated gender : {profile.estimated_gender}")
    print(f"  Landmark metrics : {json.dumps(profile.landmark_metrics)}")
    if profile.notes:
        print("  Notes:")
        for n in profile.notes:
            print(f"    - {n}")

    print()
    print("=" * 70)
    print("STYLE RECOMMENDATIONS")
    print("=" * 70)

    recs = recommend_styles(
        profile=profile,
        top_n=args.top_n,
        occasion=args.occasion,
    )

    if not recs:
        print("  No matching styles found in the catalogue.")
        return 0

    for i, r in enumerate(recs, 1):
        print(f"\n  {i}. {r.style_name}  (score: {r.suit_score}/100)")
        print(f"     id: {r.style_id}")
        print(f"     why: {r.reasoning}")
        traits = ", ".join(r.style_metadata.get("style_traits", []))
        print(f"     traits: {traits}")
        ref = r.style_metadata.get("reference_image_path") or "(no reference yet)"
        print(f"     reference: {ref}")

    print()
    print("=" * 70)
    print("CONSULTATION SUMMARY")
    print("=" * 70)
    print(json.dumps({
        "profile": profile.to_dict(),
        "recommendations": [r.to_dict() for r in recs],
    }, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
