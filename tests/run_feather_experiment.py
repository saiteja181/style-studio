"""Single-cell experiment: bridal_juda on curly-hair woman with feather_px=4.

Goal: test the hypothesis that the polygon edge feather (default 18 px) is
softening alpha at face landmarks (chin 152 etc.) so Kontext bleeds through
and produces face drift.  Reducing feather to 4 should keep source face
pixels at alpha ~1.0 deeper into the polygon.

Saves output as tests/acceptance/feather4_curly_bridal_juda.png alongside
the previous-run grid in tests/acceptance/runs/.

Cost: 1 Kontext call + 1 validator call ~= INR 4-5.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

import os
os.environ["STYLE_STUDIO_UPLOADS_DIR"] = str(PROJECT_ROOT / "tests" / "acceptance")

from backend.input_pipeline import prepare_upload  # noqa: E402
from backend.customer_analysis import analyze_customer  # noqa: E402
from backend.kontext_engine import _call_kontext, _load_style, _resolve_reference_path  # noqa: E402
from backend.prompt_builder import build_edit_prompt  # noqa: E402
from backend.face_composite import paste_source_face  # noqa: E402


SRC = PROJECT_ROOT / "tests" / "selfies" / "curly_hair_indian_woman.jpg"
OUT_DIR = PROJECT_ROOT / "tests" / "acceptance"


def main() -> int:
    print(f"experiment: bridal_juda on {SRC.name} with feather_px=4")

    raw = SRC.read_bytes()
    src_norm, report = prepare_upload(
        raw_bytes=raw, target_dir=OUT_DIR, filename_hint="feather4_src",
    )
    print(f"  preflight: face={report.face_fraction:.2f} blur={report.blur_score:.0f}")

    profile = analyze_customer(selfie_path=src_norm, use_vision_lm=False).to_dict()

    style = _load_style("bridal_juda")
    ref_path = _resolve_reference_path(style)
    prompt = build_edit_prompt(
        style=style, customer_profile=profile,
        source_path=src_norm, reference_path=ref_path,
    )
    print(f"  prompt: {prompt[:140]}...")

    t0 = time.time()
    raw_url = _call_kontext(source_path=src_norm, prompt=prompt, seed=42, style=style)
    print(f"  kontext: {time.time()-t0:.1f}s -> {raw_url[:80]}")

    # Save the raw Kontext output (BEFORE face composite) so we can see
    # exactly how much face drift came from Kontext itself.
    raw_out = OUT_DIR / "feather4_curly_bridal_juda__kontext_raw.png"
    with urllib.request.urlopen(raw_url, timeout=60) as resp:
        raw_out.write_bytes(resp.read())
    print(f"  raw Kontext saved: {raw_out.name}")

    # Now do the face composite with feather_px=4 (was 18)
    composited = paste_source_face(
        source_path=src_norm,
        kontext_output_url_or_path=raw_url,
        output_dir=OUT_DIR,
        feather_px=4,
        head_covering_type=None,
    )
    # Rename to a predictable filename
    final_out = OUT_DIR / "feather4_curly_bridal_juda.png"
    if final_out.exists():
        final_out.unlink()
    composited.rename(final_out)
    print(f"  composited (feather=4) saved: {final_out.name}")

    print()
    print(f"compare:")
    print(f"  SOURCE       : {src_norm}")
    print(f"  KONTEXT RAW  : {raw_out}")
    print(f"  COMPOSITE f=4: {final_out}")
    print()
    print(f"previous run (feather=18) preserved at:")
    print(f"  tests/acceptance/runs/2026-05-23-pre-feather-fix/curly__bridal_juda.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
