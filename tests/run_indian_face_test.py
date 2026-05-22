"""Comparison test on Indian face sources.

Runs 3 women's catalogue styles through both pipelines (BEFORE = current
manual mode without customer_profile, AFTER = recommended default with shared
erase + 2-stage + customer-aware prompts) on two Indian sources:

  source_a = tests/selfies/test_indian_woman_a.jpg (young woman, dark curly hair)
  source_b = tests/selfies/test_indian_woman_b.jpg (older woman, grey hair - hard case)

Output:
  tests/indian_comparison/grid_a.png   - young woman, 3 styles, before+after
  tests/indian_comparison/grid_b.png   - older woman, 3 styles, before+after
  tests/indian_comparison/after_*.png  - downloaded "AFTER" results, full res

Cost: 2 sources x (3 BEFORE + 1 erase + 3 AFTER) = 14 FLUX calls ~= $0.70.
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

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from backend.input_pipeline import prepare_upload, PreflightError  # noqa: E402
from backend.inpaint import (  # noqa: E402
    generate_preview_inpaint,
    generate_preview_erase_then_inpaint,
    build_shared_bald_canvas,
    InpaintError,
)
from backend.customer_analysis import analyze_customer  # noqa: E402

OUT_DIR = PROJECT_ROOT / "tests" / "indian_comparison"
SELFIE_DIR = PROJECT_ROOT / "tests" / "selfies"

SOURCES = [
    ("a_young",  SELFIE_DIR / "test_indian_woman_a_padded.jpg",
     "young woman (dark curly hair)"),
]

STYLES = [
    "indian_braid_long",
    "bridal_juda",
    "curtain_bangs_medium",
]


def download(url: str, out_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp:
        out_path.write_bytes(resp.read())


def _local_or_download(url_or_path: str, fallback_dir: Path, out_path: Path) -> Path:
    if url_or_path.startswith(("http://", "https://")):
        download(url_or_path, out_path)
        return out_path
    if url_or_path.startswith("/uploads/"):
        name = Path(url_or_path).name
        for d in (PROJECT_ROOT / "tests" / "uploads", fallback_dir):
            cand = d / name
            if cand.exists():
                out_path.write_bytes(cand.read_bytes())
                return out_path
        matches = sorted(fallback_dir.glob("harmonised_*.png"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            out_path.write_bytes(matches[0].read_bytes())
            return out_path
    raise RuntimeError(f"could not resolve image at {url_or_path}")


def compose_grid(
    source: Image.Image,
    rows: list[tuple[str, Image.Image, Image.Image]],
    out_path: Path,
    title: str,
) -> None:
    TILE = 384
    PAD = 12
    LABEL_W = 240
    TITLE_H = 50

    n_rows = len(rows)
    grid_w = LABEL_W + 3 * TILE + 4 * PAD
    grid_h = TITLE_H + TILE + PAD * 2 + n_rows * (TILE + PAD) + PAD

    grid = Image.new("RGB", (grid_w, grid_h), (245, 245, 248))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
        bold = ImageFont.truetype("arialbd.ttf", 26)
        title_font = ImageFont.truetype("arialbd.ttf", 28)
    except Exception:
        font = bold = title_font = ImageFont.load_default()

    draw.text((PAD, PAD), title, fill=(15, 15, 25), font=title_font)

    src_tile = source.resize((TILE, TILE), Image.LANCZOS)
    headers = ["SOURCE", "BEFORE (manual)", "AFTER (recommended)"]
    for i, h in enumerate(headers):
        x = LABEL_W + PAD + i * (TILE + PAD)
        grid.paste(src_tile, (x, TITLE_H + PAD))
        draw.text((x + 6, TITLE_H + PAD + 6),
                  h, fill=(255, 255, 255), font=bold,
                  stroke_width=2, stroke_fill=(0, 0, 0))

    y = TITLE_H + PAD + TILE + PAD
    for style_label, before, after in rows:
        draw.text((PAD, y + TILE // 2 - 12),
                  style_label, fill=(20, 20, 30), font=font)
        grid.paste(src_tile, (LABEL_W + PAD, y))
        grid.paste(before.resize((TILE, TILE), Image.LANCZOS),
                   (LABEL_W + PAD + TILE + PAD, y))
        grid.paste(after.resize((TILE, TILE), Image.LANCZOS),
                   (LABEL_W + PAD + 2 * (TILE + PAD), y))
        y += TILE + PAD

    grid.save(out_path, format="PNG", optimize=True)


def run_one_source(tag: str, src_path: Path, description: str) -> bool:
    if not src_path.exists():
        print(f"[skip] {src_path} not found")
        return False

    print(f"\n=== {tag}: {description} ===")
    raw_bytes = src_path.read_bytes()
    try:
        source_path, report = prepare_upload(
            raw_bytes=raw_bytes, target_dir=OUT_DIR,
            filename_hint=f"src_{tag}",
        )
    except PreflightError as e:
        print(f"  preflight blocked: {e.report.code} {e.report.message}")
        return False
    print(f"  preflight ok: face={report.face_fraction:.2f} "
          f"blur={report.blur_score:.0f} size={report.normalised_size}")

    profile = analyze_customer(selfie_path=source_path, use_vision_lm=False)
    profile_dict = profile.to_dict()
    print(f"  hair_color={profile_dict['hair_color_rgb']} "
          f"texture={profile_dict['hair_texture']}")

    # BEFORE: manual mode, no customer_profile.
    before_imgs: dict[str, Path] = {}
    for sid in STYLES:
        t0 = time.time()
        print(f"  BEFORE {sid}...")
        try:
            r = generate_preview_inpaint(
                selfie_path=source_path, style_id=sid, seed=42,
                customer_profile=None,
                validate=False, max_retries=0,
            )
        except InpaintError as e:
            print(f"    [fail] {e}")
            continue
        out = OUT_DIR / f"before_{tag}__{sid}.png"
        before_imgs[sid] = _local_or_download(
            r.image_url, fallback_dir=source_path.parent, out_path=out,
        )
        print(f"    {time.time()-t0:.1f}s")

    # AFTER: shared erase + 3 inpaints with customer_profile.
    print(f"  AFTER: shared bald canvas (1 FLUX call)...")
    t0 = time.time()
    shared = build_shared_bald_canvas(selfie_path=source_path, seed=42)
    print(f"    erase done in {time.time()-t0:.1f}s")

    after_imgs: dict[str, Path] = {}
    for sid in STYLES:
        t0 = time.time()
        print(f"  AFTER {sid}...")
        try:
            r = generate_preview_erase_then_inpaint(
                selfie_path=source_path, style_id=sid, seed=42,
                customer_profile=profile_dict,
                shared_bald=shared,
                validate=False, max_retries=0,
            )
        except InpaintError as e:
            print(f"    [fail] {e}")
            continue
        out = OUT_DIR / f"after_{tag}__{sid}.png"
        after_imgs[sid] = _local_or_download(
            r.image_url, fallback_dir=source_path.parent, out_path=out,
        )
        print(f"    {time.time()-t0:.1f}s")

    # Build per-source grid.
    source_img = Image.open(source_path).convert("RGB")
    rows = []
    for sid in STYLES:
        if sid not in before_imgs or sid not in after_imgs:
            continue
        rows.append((
            sid.replace("indian_", "").replace("_", " "),
            Image.open(before_imgs[sid]).convert("RGB"),
            Image.open(after_imgs[sid]).convert("RGB"),
        ))
    if rows:
        grid_path = OUT_DIR / f"grid_{tag}.png"
        compose_grid(source_img, rows, grid_path, title=description)
        print(f"  grid saved at {grid_path}")
    return bool(rows)


def main() -> int:
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN not set in .env - aborting.")
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    any_ok = False
    for tag, path, desc in SOURCES:
        if run_one_source(tag, path, desc):
            any_ok = True
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
