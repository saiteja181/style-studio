"""Test: skip the bald-pass entirely, inpaint directly with very high
guidance on the new wider hair-aware mask.  If FLUX can deliver dramatic
style change in one pass, we don't need the broken bald canvas.

Generates 3 pompadour attempts at different guidance levels so we can see
exactly what the model is willing to do.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from backend.input_pipeline import prepare_upload  # noqa: E402
from backend.inpaint import generate_preview_inpaint  # noqa: E402
from backend.customer_analysis import analyze_customer  # noqa: E402

OUT = PROJECT_ROOT / "tests" / "debug_pipeline" / "direct"
SRC = PROJECT_ROOT / "tests" / "selfies" / "test_random_indian_man.jpg"


def download(url, p):
    with urllib.request.urlopen(url, timeout=60) as r:
        p.write_bytes(r.read())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p, _ = prepare_upload(
        raw_bytes=SRC.read_bytes(), target_dir=OUT, filename_hint="src",
    )
    profile = analyze_customer(selfie_path=p, use_vision_lm=False).to_dict()

    # Test 3 guidance levels with new wider mask + pompadour prompt.
    for g in [35, 55, 80]:
        print(f"\n--- guidance={g} ---")
        r = generate_preview_inpaint(
            selfie_path=p, style_id="mens_pompadour", seed=42,
            customer_profile=profile, guidance=float(g),
            harmonise=True, validate=False, max_retries=0,
        )
        # Resolve URL/path -> file
        url = r.image_url
        out_path = OUT / f"direct_g{g}.png"
        if url.startswith("http"):
            download(url, out_path)
        else:
            name = Path(url).name
            for d in (p.parent, PROJECT_ROOT / "tests" / "uploads"):
                cand = d / name
                if cand.exists():
                    out_path.write_bytes(cand.read_bytes())
                    break
        print(f"  saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
