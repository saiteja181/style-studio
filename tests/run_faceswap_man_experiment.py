"""Second single-cell experiment: Sikh man + mens_pompadour, with face-swap.

Goal: verify the Kontext + face-swap architecture generalises beyond the
curly-hair woman case.  This tests:
  - male source (different demographic from the woman experiment)
  - head-covering case (turban removed by Kontext)
  - short hairstyle (pompadour) vs the long-hair bridal juda case

Cost: ~$0.045 ($0.04 Kontext + $0.005 face-swap).
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")
import replicate  # noqa: E402

os.environ["STYLE_STUDIO_UPLOADS_DIR"] = str(PROJECT_ROOT / "tests" / "acceptance")

from backend.input_pipeline import prepare_upload  # noqa: E402
from backend.customer_analysis import analyze_customer  # noqa: E402
from backend.kontext_engine import _call_kontext, _load_style, _resolve_reference_path  # noqa: E402
from backend.prompt_builder import build_edit_prompt  # noqa: E402


SRC = PROJECT_ROOT / "tests" / "selfies" / "young_indian_man.jpg"  # Sikh, turban
STYLE_ID = "mens_pompadour"
OUT_DIR = PROJECT_ROOT / "tests" / "acceptance"
KONTEXT_RAW = OUT_DIR / "faceswap_man_kontext_raw.png"
FINAL = OUT_DIR / "faceswap_man_pompadour.png"

FACE_SWAP_MODEL = (
    "cdingram/face-swap:d1d6ea8c8be89d664a07a457526f7128109dee7030fdac424788d762c71ed111"
)


def main() -> int:
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN missing")
        return 2

    raw = SRC.read_bytes()
    src_norm, report = prepare_upload(
        raw_bytes=raw, target_dir=OUT_DIR, filename_hint="faceswap_src_man",
    )
    print(f"preflight: face={report.face_fraction:.2f} blur={report.blur_score:.0f}")
    if report.head_covering.get("detected"):
        print(f"  head-covering: {report.head_covering.get('covering_type')}")

    profile = analyze_customer(selfie_path=src_norm, use_vision_lm=False).to_dict()
    style = _load_style(STYLE_ID)
    ref_path = _resolve_reference_path(style)
    prompt = build_edit_prompt(
        style=style, customer_profile=profile,
        source_path=src_norm, reference_path=ref_path,
    )
    print(f"prompt: {prompt[:120]}...")

    # 1. Kontext draws the new style
    t0 = time.time()
    raw_url = _call_kontext(source_path=src_norm, prompt=prompt, seed=42, style=style)
    print(f"kontext: {time.time()-t0:.1f}s")
    with urllib.request.urlopen(raw_url, timeout=60) as resp:
        KONTEXT_RAW.write_bytes(resp.read())
    print(f"  raw Kontext saved: {KONTEXT_RAW.name}")

    # 2. Face-swap the customer's face onto the Kontext output
    t0 = time.time()
    with src_norm.open("rb") as id_f, KONTEXT_RAW.open("rb") as target_f:
        swap_out = replicate.run(
            FACE_SWAP_MODEL,
            input={
                "swap_image": id_f,
                "input_image": target_f,
            },
        )
    print(f"face-swap: {time.time()-t0:.1f}s")

    url = (swap_out if isinstance(swap_out, str)
           else (swap_out[0] if isinstance(swap_out, list) and swap_out
                 else getattr(swap_out, "url", None)))
    if not url:
        print(f"  no URL from face-swap: {swap_out!r}")
        return 3
    with urllib.request.urlopen(url, timeout=60) as resp:
        FINAL.write_bytes(resp.read())
    print(f"  final saved: {FINAL.name}")

    print()
    print("compare three images:")
    print(f"  SOURCE      : {SRC}")
    print(f"  KONTEXT RAW : {KONTEXT_RAW}")
    print(f"  FACE-SWAPPED: {FINAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
