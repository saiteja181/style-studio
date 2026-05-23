"""Single-cell experiment: bytedance/flux-pulid for identity-locked generation.

Goal: test whether passing the customer face as an identity reference (PuLID)
gives us preserved face + new hairstyle in one shot, instead of the
Kontext + paste-polygon approach (which can't preserve face SHAPE).

The model takes the customer's face as `main_face_image`, plus a hair-styling
prompt, and produces a new image with that identity + the styled hair.
Body pose / clothing may change since PuLID regenerates the whole image -
this is the trade-off vs Kontext.

Cost: ~$0.05 per run.
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

SRC = PROJECT_ROOT / "tests" / "selfies" / "curly_hair_indian_woman.jpg"
OUT = PROJECT_ROOT / "tests" / "acceptance" / "pulid_curly_bridal_juda.png"

# A few prompt variants to try if needed - we want to anchor on the
# customer's identity and only change the hair to a traditional Indian bridal
# bun (juda).
PROMPT = (
    "A young Indian woman with the SAME face, eyes, jawline, nose, and lips as the reference photo, "
    "wearing a traditional INDIAN BRIDAL JUDA hairstyle: hair pulled back tightly from a centre parting, "
    "gathered into a high decorated bun at the crown of the head, decorated with fresh red and white flowers, "
    "small string of mogra blossoms wrapping the bun, ornate maang tikka hanging from a centre parting onto "
    "the forehead, large traditional jhumka earrings. Outdoor portrait, soft daylight, photoreal, sharp focus. "
    "Indian salon bridal styling, elegant, polished. Preserve the exact face identity from the reference image."
)


def main() -> int:
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN missing")
        return 2

    if not SRC.exists():
        print(f"source missing: {SRC}")
        return 1

    print(f"Running bytedance/flux-pulid on {SRC.name}")
    print(f"prompt: {PROMPT[:140]}...")

    t0 = time.time()
    try:
        with SRC.open("rb") as f:
            # Common PuLID-FLUX input shape on Replicate.  The exact key names
            # vary - we pass the most-likely set; Replicate will reject extras.
            output = replicate.run(
                "bytedance/flux-pulid:8baa7ef2255075b46f4d91cd238c21d31181b3e6a864463f967960bb0112525b",
                input={
                    "prompt": PROMPT,
                    "main_face_image": f,
                    "num_steps": 20,
                    "guidance_scale": 4.0,
                    "true_cfg": 1.0,
                    "id_weight": 1.0,
                    "seed": 42,
                    "max_sequence_length": 128,
                    "output_format": "png",
                },
            )
    except Exception as e:
        print(f"ERROR: {e}")
        return 3

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    # Extract URL from Replicate output
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
    print(f"  saved: {OUT}")
    print()
    print("compare three images:")
    print(f"  SOURCE         : {SRC}")
    print(f"  KONTEXT (best) : tests/acceptance/feather4_curly_bridal_juda.png")
    print(f"  PULID          : {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
