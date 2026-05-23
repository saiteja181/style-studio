"""Two-source face-swap test on FRESH Pexels portraits we haven't seen
before.  Verifies the Kontext + face-swap architecture generalises beyond
the optimised acceptance fixtures.

Sources:
  - tests/selfies/sp_test/new_woman.jpg (Pexels 17261596 - saree portrait)
  - tests/selfies/sp_test/new_man.jpg   (Pexels 5354069 - young clean-shaven)

Styles:
  - woman: bridal_juda (highest commercial value for Indian salons)
  - man  : mens_pompadour

Cost: 2 cells x $0.045 = ~$0.09.
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

OUT_DIR = PROJECT_ROOT / "tests" / "acceptance"
FACE_SWAP_MODEL = (
    "cdingram/face-swap:d1d6ea8c8be89d664a07a457526f7128109dee7030fdac424788d762c71ed111"
)

CASES = [
    ("new_woman", PROJECT_ROOT / "tests" / "selfies" / "sp_test" / "new_woman.jpg",
     "bridal_juda"),
    ("new_man",   PROJECT_ROOT / "tests" / "selfies" / "sp_test" / "new_man.jpg",
     "mens_pompadour"),
]


def run_one(tag: str, src: Path, style_id: str) -> bool:
    print(f"\n=== {tag} + {style_id} ===")
    if not src.exists():
        print(f"  source missing: {src}")
        return False
    raw = src.read_bytes()
    src_norm, report = prepare_upload(
        raw_bytes=raw, target_dir=OUT_DIR, filename_hint=f"newtest_src_{tag}",
    )
    print(f"  preflight: face={report.face_fraction:.2f} blur={report.blur_score:.0f} "
          f"size={report.normalised_size}")
    if report.head_covering.get("detected"):
        print(f"  head-covering: {report.head_covering.get('covering_type')}")

    profile = analyze_customer(selfie_path=src_norm, use_vision_lm=False).to_dict()
    style = _load_style(style_id)
    ref_path = _resolve_reference_path(style)
    prompt = build_edit_prompt(
        style=style, customer_profile=profile,
        source_path=src_norm, reference_path=ref_path,
    )

    t0 = time.time()
    raw_url = _call_kontext(source_path=src_norm, prompt=prompt, seed=42, style=style)
    print(f"  kontext: {time.time()-t0:.1f}s")

    kontext_path = OUT_DIR / f"newtest_{tag}__{style_id}__kontext.png"
    with urllib.request.urlopen(raw_url, timeout=60) as resp:
        kontext_path.write_bytes(resp.read())

    t0 = time.time()
    with src_norm.open("rb") as id_f, kontext_path.open("rb") as target_f:
        swap_out = replicate.run(
            FACE_SWAP_MODEL,
            input={"swap_image": id_f, "input_image": target_f},
        )
    print(f"  face-swap: {time.time()-t0:.1f}s")
    url = (swap_out if isinstance(swap_out, str)
           else (swap_out[0] if isinstance(swap_out, list) and swap_out
                 else getattr(swap_out, "url", None)))
    if not url:
        print(f"  face-swap returned no URL")
        return False
    final_path = OUT_DIR / f"newtest_{tag}__{style_id}__final.png"
    with urllib.request.urlopen(url, timeout=60) as resp:
        final_path.write_bytes(resp.read())
    print(f"  saved: {final_path.name}")
    return True


def main() -> int:
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN missing")
        return 2
    ok_count = 0
    for tag, src, sid in CASES:
        if run_one(tag, src, sid):
            ok_count += 1
    print(f"\n{ok_count} / {len(CASES)} cells completed")
    return 0 if ok_count == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
