"""Build a side-by-side composite of source vs generated image.

Usage:
    python tests/build_comparison.py <source.jpg> <generated.png> <out.png> [label]
"""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    if len(sys.argv) < 4:
        print("Usage: build_comparison.py <source> <generated> <out> [label]", file=sys.stderr)
        return 1

    source_path = Path(sys.argv[1])
    generated_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else ""

    if not source_path.exists() or not generated_path.exists():
        print("ERROR: input image missing", file=sys.stderr)
        return 1

    src = Image.open(source_path).convert("RGB")
    gen = Image.open(generated_path).convert("RGB")

    target_h = 700
    src = _resize_to_h(src, target_h)
    gen = _resize_to_h(gen, target_h)

    gap = 24
    header_h = 56 if label else 0
    composite = Image.new(
        "RGB",
        (src.width + gen.width + gap, target_h + header_h),
        (245, 245, 245),
    )
    composite.paste(src, (0, header_h))
    composite.paste(gen, (src.width + gap, header_h))

    draw = ImageDraw.Draw(composite)
    try:
        font_label = ImageFont.truetype("arial.ttf", 28)
        font_caption = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font_label = ImageFont.load_default()
        font_caption = font_label

    if label:
        draw.text((16, 14), label, fill=(20, 20, 20), font=font_label)

    cap_y = header_h + target_h - 38
    draw.rectangle([(0, cap_y), (src.width, cap_y + 36)], fill=(0, 0, 0, 180))
    draw.text((12, cap_y + 6), "SOURCE", fill=(255, 255, 255), font=font_caption)
    draw.rectangle(
        [(src.width + gap, cap_y), (src.width + gap + gen.width, cap_y + 36)],
        fill=(0, 0, 0, 180),
    )
    draw.text((src.width + gap + 12, cap_y + 6), "GENERATED",
              fill=(255, 255, 255), font=font_caption)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path, quality=90)
    print(f"Saved comparison to {out_path}")
    return 0


def _resize_to_h(img: Image.Image, h: int) -> Image.Image:
    w = int(img.width * (h / img.height))
    return img.resize((w, h), Image.LANCZOS)


if __name__ == "__main__":
    sys.exit(main())
