"""Test alternative inpaint models on the same source + mask + prompt.

We've confirmed FLUX Fill Pro is too conservative even with the new wide
hair-aware mask + guidance up to 80.  This script calls a few alternative
inpaint models directly and saves the raw outputs side-by-side so we can see
which one actually produces a different hair shape.
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
from backend.inpaint import (  # noqa: E402
    _build_local_hair_mask, _build_flux_prompt, _build_default_prompt_from_style,
    _load_style, _extract_first_url,
)
from backend.customer_analysis import analyze_customer  # noqa: E402
from backend.hair_mask import refine_with_selfie_segmentation  # noqa: E402

import numpy as np  # noqa: E402

OUT = PROJECT_ROOT / "tests" / "debug_pipeline" / "model_swap"
SRC = PROJECT_ROOT / "tests" / "selfies" / "test_random_indian_man.jpg"

# Models to compare.  Each entry: (label, model_ref, input_dict_builder)
def _flux_fill_pro_inputs(img_f, mask_f, prompt):
    return {
        "image": img_f, "mask": mask_f, "prompt": prompt,
        "steps": 50, "guidance": 55.0, "output_format": "png",
        "safety_tolerance": 2, "prompt_upsampling": False, "seed": 42,
    }

def _zsxkib_flux_dev_inputs(img_f, mask_f, prompt):
    # zsxkib/flux-dev-inpainting expects: prompt, image, mask_image,
    # num_inference_steps, guidance_scale, strength, seed.
    return {
        "prompt": prompt, "image": img_f, "mask_image": mask_f,
        "num_inference_steps": 35, "guidance_scale": 6.0,
        "strength": 0.95,  # high strength = more deviation from source
        "seed": 42, "output_format": "png",
    }

MODELS = [
    ("flux_fill_pro",
     "black-forest-labs/flux-fill-pro",
     _flux_fill_pro_inputs),
    ("zsxkib_flux_dev_inpaint",
     "zsxkib/flux-dev-inpainting",
     _zsxkib_flux_dev_inputs),
]

POMPADOUR_STYLE_ID = "mens_pompadour"


def download(url, p):
    with urllib.request.urlopen(url, timeout=60) as r:
        p.write_bytes(r.read())


def build_inputs():
    """Pre-flight source, build refined hair mask, return paths + prompt."""
    OUT.mkdir(parents=True, exist_ok=True)
    p, _ = prepare_upload(
        raw_bytes=SRC.read_bytes(), target_dir=OUT, filename_hint="src",
    )
    profile = analyze_customer(selfie_path=p, use_vision_lm=False).to_dict()

    pil = Image.open(p).convert("RGB")
    rgb = np.array(pil)
    u_band, pts = _build_local_hair_mask(
        rgb, offset_ratio=-0.05, extend_ratio=0.90, lateral_extend=0.25,
        feather_px=24, feather_frac=0.03, ear_level_ratio=0.80,
        return_landmarks=True,
    )
    mask = refine_with_selfie_segmentation(rgb, u_band, pts)
    mask_path = OUT / "shared_mask.png"
    Image.fromarray(mask).save(mask_path)

    style = _load_style(POMPADOUR_STYLE_ID)
    raw_prompt = style.get("prompt_template") or _build_default_prompt_from_style(style)
    prompt = _build_flux_prompt(raw_prompt, customer_profile=profile, style=style)
    print(f"prompt: {prompt[:140]}...")
    return p, mask_path, prompt


def run_model(label, model_ref, build_inputs_fn, src_path, mask_path, prompt):
    print(f"\n=== {label} ({model_ref}) ===")
    t0 = time.time()
    try:
        with src_path.open("rb") as img_f, mask_path.open("rb") as mask_f:
            payload = build_inputs_fn(img_f, mask_f, prompt)
            output = replicate.run(model_ref, input=payload)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    elapsed = time.time() - t0
    url = _extract_first_url(output) or (output if isinstance(output, str) else None)
    if not url:
        print(f"  no url returned: {output!r}")
        return None
    print(f"  {elapsed:.1f}s -> {url[:80]}...")
    out_file = OUT / f"out_{label}.png"
    download(url, out_file)
    print(f"  saved {out_file}")
    return out_file


def compose(src_path, mask_path, model_outputs):
    TILE = 480
    PAD = 12
    LABEL_H = 36
    cols = 2 + len(model_outputs)
    W = cols * TILE + (cols + 1) * PAD
    H = TILE + LABEL_H + 2 * PAD

    canvas = Image.new("RGB", (W, H), (245, 245, 248))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    src_img = Image.open(src_path).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
    mask_img = Image.open(mask_path).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
    tiles = [("SOURCE", src_img), ("MASK", mask_img)]
    for label, p in model_outputs:
        if p and p.exists():
            tiles.append((label.upper(), Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS)))

    for i, (label, img) in enumerate(tiles):
        x = PAD + i * (TILE + PAD)
        canvas.paste(img, (x, LABEL_H + PAD))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (TILE - tw) // 2, PAD), label, fill=(15, 15, 25), font=font)

    out_path = OUT / "model_comparison.png"
    canvas.save(out_path)
    print(f"\ncomparison saved at {out_path}")
    return out_path


def main():
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN not set in .env")
        return 2
    src, mask, prompt = build_inputs()
    outputs = []
    for label, ref, build_fn in MODELS:
        outputs.append((label, run_model(label, ref, build_fn, src, mask, prompt)))
    compose(src, mask, outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
