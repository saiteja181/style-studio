"""Single-cell experiment: face-swap the customer's face onto the Kontext
output that has the bridal juda hair.

Goal: keep what Kontext does well (body pose, clothing, background,
hairstyle) but lock identity by swapping in the customer's face after
generation.  This is the standard "AI hairstyle preview" pattern.

We reuse the existing raw Kontext output saved by run_feather_experiment.py,
so the only spend is the face-swap call itself (~$0.005).
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

# Identity source - the customer's actual face
IDENTITY = PROJECT_ROOT / "tests" / "selfies" / "curly_hair_indian_woman.jpg"
# Target - the Kontext output that has the bridal juda hairstyle but on a
# different face.  We reuse the existing raw output from the previous
# feather experiment so we don't pay for Kontext again.
TARGET = PROJECT_ROOT / "tests" / "acceptance" / "feather4_curly_bridal_juda__kontext_raw.png"
OUT = PROJECT_ROOT / "tests" / "acceptance" / "faceswap_curly_bridal_juda.png"

# cdingram/face-swap version - inswapper_128 based, the standard
FACE_SWAP_MODEL = (
    "cdingram/face-swap:d1d6ea8c8be89d664a07a457526f7128109dee7030fdac424788d762c71ed111"
)


def main() -> int:
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN missing")
        return 2
    if not IDENTITY.exists():
        print(f"identity source missing: {IDENTITY}")
        return 1
    if not TARGET.exists():
        print(f"target image missing: {TARGET}")
        print(f"  run tests/run_feather_experiment.py first to produce it")
        return 1

    print(f"Face-swap experiment:")
    print(f"  identity (face)  : {IDENTITY.name}")
    print(f"  target (hair etc): {TARGET.name}")
    print()

    t0 = time.time()
    try:
        with IDENTITY.open("rb") as swap_f, TARGET.open("rb") as target_f:
            # cdingram/face-swap input shape:
            #   swap_image  = the face to PUT IN (identity / source face)
            #   input_image = the image to PUT THE FACE INTO (target with new hair)
            output = replicate.run(
                FACE_SWAP_MODEL,
                input={
                    "swap_image": swap_f,
                    "input_image": target_f,
                },
            )
    except Exception as e:
        print(f"ERROR: {e}")
        return 3
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    url = None
    if isinstance(output, str):
        url = output
    elif isinstance(output, list) and output:
        first = output[0]
        url = first if isinstance(first, str) else getattr(first, "url", None)
    else:
        url = getattr(output, "url", None)

    if not url:
        print(f"  no URL in output: {output!r}")
        return 4

    print(f"  url: {url[:100]}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        OUT.write_bytes(resp.read())
    print(f"  saved: {OUT.name}")
    print()
    print("compare four images:")
    print(f"  SOURCE          : {IDENTITY}")
    print(f"  KONTEXT RAW     : {TARGET}")
    print(f"  KONTEXT+composite (feather=4) : tests/acceptance/feather4_curly_bridal_juda.png")
    print(f"  KONTEXT+FACESWAP: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
