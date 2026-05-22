"""Side-by-side: shipped manual mode vs new recommended-default pipeline.

Goal: make the quality delta visible to the user.

For ONE fresh TPDNE portrait, generate 3 different styles two ways:
  A) "BEFORE" = the manual-mode call as shipped today (single FLUX inpaint,
     no customer-colour anchor in the prompt, no two-stage erase).  Mask is
     the current geometric U-band with selfie-segmentation refinement (we
     can't disable that without ripping out the module - and it costs $0).
  B) "AFTER"  = the recommended default: erase-then-inpaint with the bald
     canvas SHARED across all 3 styles, customer hair-colour hex woven into
     the prompt, negative-prompt clause, refined mask.

Output: tests/comparison/grid.png - a 3-row by 3-column image
  rows = [textured_crop, buzz_cut, korean_fringe]
  cols = [source | BEFORE | AFTER]

Cost: ~$0.35 in Replicate (3 FLUX for BEFORE + 1 erase + 3 inpaints for AFTER).
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

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
from backend.customer_analysis import analyze_customer, AnalysisError  # noqa: E402

OUT_DIR = PROJECT_ROOT / "tests" / "comparison"
TPDNE = "https://thispersondoesnotexist.com"

STYLES = [
    "mens_textured_crop",
    "mens_buzz_cut",
    "mens_korean_fringe",
]


def fetch_portrait(out_path: Path) -> None:
    req = urllib.request.Request(
        TPDNE, headers={"User-Agent": "style-studio-comparison/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        out_path.write_bytes(resp.read())


def download(url: str, out_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp:
        out_path.write_bytes(resp.read())


def _local_or_download(url_or_path: str, fallback_dir: Path, out_path: Path) -> Path:
    """Resolve a generator-returned URL/path into an actual file on disk."""
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


def label(img: Image.Image, text: str) -> Image.Image:
    """Draw a small label band at the top-left of the image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    pad = 8
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle((0, 0, tw + 2 * pad, th + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def compose_grid(
    source: Image.Image,
    rows: list[tuple[str, Image.Image, Image.Image]],
    out_path: Path,
) -> None:
    """Compose source + (before, after) rows into a single grid PNG.

    Layout (each cell = TILE x TILE):
        [source ][source ][source ]
        [label  ][BEFORE ][AFTER  ]   <- per row, with style label on the left
    """
    TILE = 384
    PAD = 12
    LABEL_W = 220

    n_rows = len(rows)
    grid_w = LABEL_W + 3 * TILE + 4 * PAD
    grid_h = TILE + PAD * 2 + n_rows * (TILE + PAD) + PAD

    grid = Image.new("RGB", (grid_w, grid_h), (245, 245, 248))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
        bold = ImageFont.truetype("arialbd.ttf", 26)
    except Exception:
        font = bold = ImageFont.load_default()

    # Top row: source repeated under each column with header labels.
    src_tile = source.resize((TILE, TILE), Image.LANCZOS)
    headers = ["SOURCE", "BEFORE (manual)", "AFTER (recommended)"]
    for i, h in enumerate(headers):
        x = LABEL_W + PAD + i * (TILE + PAD)
        grid.paste(src_tile, (x, PAD))
        draw.text((x + 6, PAD + 6),
                  h, fill=(255, 255, 255), font=bold,
                  stroke_width=2, stroke_fill=(0, 0, 0))

    # Subsequent rows: per-style label + before + after.
    y = PAD + TILE + PAD
    for style_label, before, after in rows:
        # Style name column
        draw.text((PAD, y + TILE // 2 - 12),
                  style_label, fill=(20, 20, 30), font=font)
        # Source thumbnail to anchor identity
        grid.paste(src_tile, (LABEL_W + PAD, y))
        # Before
        grid.paste(before.resize((TILE, TILE), Image.LANCZOS),
                   (LABEL_W + PAD + TILE + PAD, y))
        # After
        grid.paste(after.resize((TILE, TILE), Image.LANCZOS),
                   (LABEL_W + PAD + 2 * (TILE + PAD), y))
        y += TILE + PAD

    grid.save(out_path, format="PNG", optimize=True)


def main() -> int:
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN not set in .env - aborting.")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pull source portrait and pre-flight it.
    raw_path = OUT_DIR / "source_raw.jpg"
    print("fetching TPDNE portrait...")
    fetch_portrait(raw_path)
    raw_bytes = raw_path.read_bytes()
    try:
        source_path, report = prepare_upload(
            raw_bytes=raw_bytes, target_dir=OUT_DIR, filename_hint="source",
        )
    except PreflightError as e:
        print(f"preflight blocked: {e.report.code} {e.report.message}")
        return 1
    print(f"preflight ok: face={report.face_fraction:.2f} "
          f"blur={report.blur_score:.0f} size={report.normalised_size}")

    # 2. Customer analysis (local sampling - no Anthropic call required).
    try:
        profile = analyze_customer(selfie_path=source_path, use_vision_lm=False)
    except AnalysisError as e:
        print(f"customer analysis failed: {e}")
        return 1
    profile_dict = profile.to_dict()
    print(f"customer profile: hair_color={profile_dict['hair_color_rgb']} "
          f"texture={profile_dict['hair_texture']}")

    # 3. BEFORE column - one FLUX manual call per style, no customer_profile.
    before_imgs: dict[str, Path] = {}
    for sid in STYLES:
        t0 = time.time()
        print(f"  BEFORE {sid}...")
        try:
            r = generate_preview_inpaint(
                selfie_path=source_path, style_id=sid, seed=42,
                customer_profile=None,        # match shipped behaviour
                validate=False, max_retries=0,
            )
        except InpaintError as e:
            print(f"    [fail] {e}")
            continue
        out = OUT_DIR / f"before__{sid}.png"
        before_imgs[sid] = _local_or_download(
            r.image_url, fallback_dir=source_path.parent, out_path=out,
        )
        print(f"    {time.time()-t0:.1f}s -> {before_imgs[sid].name}")

    # 4. AFTER column - shared erase + 3 inpaints with customer_profile.
    print("AFTER: building shared bald canvas (1 FLUX call)...")
    t0 = time.time()
    shared = build_shared_bald_canvas(selfie_path=source_path, seed=42)
    print(f"  shared bald ready in {time.time()-t0:.1f}s")

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
        out = OUT_DIR / f"after__{sid}.png"
        after_imgs[sid] = _local_or_download(
            r.image_url, fallback_dir=source_path.parent, out_path=out,
        )
        print(f"    {time.time()-t0:.1f}s -> {after_imgs[sid].name}")

    # 5. Compose grid.
    source_img = Image.open(source_path).convert("RGB")
    rows = []
    for sid in STYLES:
        if sid not in before_imgs or sid not in after_imgs:
            continue
        rows.append((
            sid.replace("mens_", "").replace("_", " "),
            Image.open(before_imgs[sid]).convert("RGB"),
            Image.open(after_imgs[sid]).convert("RGB"),
        ))
    grid_path = OUT_DIR / "grid.png"
    compose_grid(source_img, rows, grid_path)
    print(f"\ngrid saved at {grid_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
