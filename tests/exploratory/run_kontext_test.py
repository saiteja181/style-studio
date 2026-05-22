"""Test FLUX Kontext Pro on the Indian male source.

Kontext is natural-language image editing - no mask required.  If it can
deliver dramatic hair shape changes while preserving identity, this becomes
the new core gen path and we can simplify the pipeline a lot.

Tests 3 styles to see if Kontext actually differentiates them.
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from backend.input_pipeline import prepare_upload  # noqa: E402
from backend.inpaint import _load_style, _extract_first_url  # noqa: E402
from backend.customer_analysis import analyze_customer  # noqa: E402

OUT = PROJECT_ROOT / "tests" / "debug_pipeline" / "kontext"
SRC = PROJECT_ROOT / "tests" / "selfies" / "test_random_indian_man.jpg"
MODEL = "black-forest-labs/flux-kontext-pro"

STYLES_TO_TRY = ["mens_pompadour", "mens_korean_fringe", "mens_textured_crop"]


def _style_edit_prompt(style: dict, profile: dict) -> str:
    """Build a Kontext-friendly EDIT prompt: 'change the hair to X, keep
    the face / clothing / background exactly the same'."""
    name = style.get("name", style["id"])
    tmpl = style.get("prompt_template") or ""
    hair_color_rgb = profile.get("hair_color_rgb") or (40, 30, 25)
    r, g, b = (int(c) for c in hair_color_rgb)
    hex_code = f"#{r:02x}{g:02x}{b:02x}"
    return (
        f"Change ONLY the hairstyle to: {tmpl}. "
        f"Keep the hair colour the customer's natural dark colour ({hex_code}). "
        f"Keep the face, eyes, expression, beard, clothing, hands, paper, and "
        f"background exactly identical to the original photo - do not change "
        f"anything below the eyebrows. Photoreal, natural ambient indoor "
        f"lighting matching the source photo. No studio lighting, no halo."
    )


def download(url: str, p: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as r:
        p.write_bytes(r.read())


def run_style(src_path: Path, profile: dict, style_id: str) -> Path | None:
    style = _load_style(style_id)
    if not style:
        return None
    prompt = _style_edit_prompt(style, profile)
    print(f"\n=== {style_id} ===")
    print(f"prompt: {prompt[:160]}...")
    t0 = time.time()
    try:
        with src_path.open("rb") as img_f:
            payload = {
                "prompt": prompt,
                "input_image": img_f,
                "aspect_ratio": "match_input_image",
                "output_format": "png",
                "safety_tolerance": 2,
                "prompt_upsampling": False,
                "seed": 42,
            }
            output = replicate.run(MODEL, input=payload)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    elapsed = time.time() - t0
    url = _extract_first_url(output) or (output if isinstance(output, str) else None)
    if not url:
        print(f"  no url: {output!r}")
        return None
    print(f"  {elapsed:.1f}s -> {url[:80]}...")
    out = OUT / f"kontext_{style_id}.png"
    download(url, out)
    print(f"  saved {out}")
    return out


def compose(src_path: Path, outputs: list[tuple[str, Path | None]]) -> Path:
    TILE = 460
    PAD = 12
    LABEL_H = 36
    cols = 1 + len(outputs)
    W = cols * TILE + (cols + 1) * PAD
    H = TILE + LABEL_H + 2 * PAD
    canvas = Image.new("RGB", (W, H), (245, 245, 248))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    src_img = Image.open(src_path).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
    cells = [("SOURCE", src_img)]
    for label, p in outputs:
        if p and p.exists():
            cells.append((label.replace("mens_", "").replace("_", " ").upper(),
                          Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS)))

    for i, (label, img) in enumerate(cells):
        x = PAD + i * (TILE + PAD)
        canvas.paste(img, (x, LABEL_H + PAD))
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (TILE - (bbox[2] - bbox[0])) // 2, PAD),
                  label, fill=(15, 15, 25), font=font)
    out = OUT / "kontext_comparison.png"
    canvas.save(out)
    return out


def main() -> int:
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN not set")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    p, report = prepare_upload(
        raw_bytes=SRC.read_bytes(), target_dir=OUT, filename_hint="src",
    )
    profile = analyze_customer(selfie_path=p, use_vision_lm=False).to_dict()
    print(f"source: {report.normalised_size} hair_color={profile['hair_color_rgb']}")

    outputs = [(s, run_style(p, profile, s)) for s in STYLES_TO_TRY]
    grid = compose(p, outputs)
    print(f"\ngrid -> {grid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
