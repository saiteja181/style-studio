"""Acceptance test for the Kontext core engine.

Generates 6 previews on the unmodified Indian-face corpus:
  - tests/selfies/test_random_indian_man.jpg with 3 men's styles
  - tests/selfies/test_indian_woman_b.jpg   with 3 women's styles

Outputs:
  tests/acceptance/grid.png    -- 2 rows x 4 cols (source + 3 styles)
  tests/acceptance/summary.json -- per-cell verdict + elapsed_ms + retries

Cost: ~\$0.30 per run.

After running, eyeball grid.png.  Ship criterion: 'would I show this to a
paying customer?' for each cell.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from backend.input_pipeline import prepare_upload, PreflightError  # noqa: E402
from backend.customer_analysis import analyze_customer  # noqa: E402
from backend.kontext_engine import generate_preview, GenerationError  # noqa: E402

OUT = PROJECT_ROOT / "tests" / "acceptance"
SELFIES = PROJECT_ROOT / "tests" / "selfies"

CASES = [
    ("man",   SELFIES / "young_indian_man.jpg",
     "Young Indian man",
     ["mens_pompadour", "mens_korean_fringe", "mens_textured_crop",
      "mens_classic_side_part", "mens_buzz_cut"]),
    ("woman", SELFIES / "young_indian_woman.jpg",
     "Young Indian woman",
     ["indian_braid_long", "bridal_juda", "curtain_bangs_medium",
      "modern_chin_bob", "side_swept_layers"]),
    ("round", SELFIES / "round_face_indian_man.jpg",
     "Round-face Indian man",
     ["mens_pompadour", "mens_korean_fringe", "mens_textured_crop",
      "mens_classic_side_part", "mens_buzz_cut"]),
    ("curly", SELFIES / "curly_hair_indian_woman.jpg",
     "Curly-hair Indian woman",
     ["indian_braid_long", "bridal_juda", "curtain_bangs_medium",
      "modern_chin_bob", "side_swept_layers"]),
]


def run_one(tag, src_path, label, style_ids):
    print(f"\n=== {tag}: {label} ===")
    raw = src_path.read_bytes()
    src_norm, report = prepare_upload(
        raw_bytes=raw, target_dir=OUT, filename_hint=f"src_{tag}",
    )
    print(f"  preflight: face={report.face_fraction:.2f} blur={report.blur_score:.0f}")
    profile = analyze_customer(selfie_path=src_norm, use_vision_lm=False).to_dict()

    cells = []
    for sid in style_ids:
        t0 = time.time()
        try:
            r = generate_preview(
                source_path=src_norm, style_id=sid,
                customer_profile=profile, seed=42, max_retries=1,
            )
        except GenerationError as e:
            print(f"  [fail] {sid}: {e}")
            cells.append({"style_id": sid, "error": str(e)})
            continue
        out_name = Path(r.image_url).name
        for d in (PROJECT_ROOT / "tests" / "uploads", src_norm.parent, OUT):
            cand = d / out_name
            if cand.exists():
                dst = OUT / f"{tag}__{sid}.png"
                dst.write_bytes(cand.read_bytes())
                break
        cells.append({
            "style_id": sid, "style_name": r.style_name,
            "validator_verdict": r.validator_verdict,
            "retries": r.retries, "elapsed_ms": r.elapsed_ms,
        })
        print(f"  {sid}: verdict={r.validator_verdict} "
              f"retries={r.retries} {r.elapsed_ms} ms")
    return {"tag": tag, "label": label, "source": src_norm.name, "cells": cells}


def compose_grid(summaries):
    TILE = 480
    PAD = 14
    LABEL_H = 38
    rows = len(summaries)
    cols = 1 + max(len(s["cells"]) for s in summaries)
    W = cols * TILE + (cols + 1) * PAD
    H = rows * (TILE + PAD) + LABEL_H * rows + PAD

    canvas = Image.new("RGB", (W, H), (245, 245, 248))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for ri, s in enumerate(summaries):
        y_label = ri * (TILE + LABEL_H + PAD) + PAD
        draw.text((PAD, y_label), s["label"], fill=(15, 15, 25), font=font)
        src_path = OUT / s["source"]
        src_img = Image.open(src_path).convert("RGB").resize(
            (TILE, TILE), Image.LANCZOS,
        )
        canvas.paste(src_img, (PAD, y_label + LABEL_H))
        for ci, cell in enumerate(s["cells"]):
            x = PAD + (ci + 1) * (TILE + PAD)
            label = cell.get("style_name", cell["style_id"]).upper()
            draw.text((x, y_label), label, fill=(15, 15, 25), font=font)
            out = OUT / f"{s['tag']}__{cell['style_id']}.png"
            if out.exists():
                img = Image.open(out).convert("RGB").resize(
                    (TILE, TILE), Image.LANCZOS,
                )
                canvas.paste(img, (x, y_label + LABEL_H))

    grid_path = OUT / "grid.png"
    canvas.save(grid_path)
    return grid_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    try:
        for tag, p, label, sids in CASES:
            summaries.append(run_one(tag, p, label, sids))
    except PreflightError as e:
        print(f"preflight blocked a source: {e.report.message}")
    grid = compose_grid(summaries)
    (OUT / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\ngrid: {grid}\nsummary: {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
