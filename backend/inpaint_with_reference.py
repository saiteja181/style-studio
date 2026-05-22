"""Reference-image-driven hair inpainting (the barber-grade pipeline).

Same architecture as inpaint.py (local hair mask + remote inpaint), but instead
of describing the target hairstyle with text alone, we feed FLUX-class IP-Adapter
a REFERENCE PHOTO of the target hairstyle. The model transfers the visual style
of the reference's hair (shape, texture, length, fade pattern) into the masked
region of the customer's photo.

This addresses the limitation we found with text-only inpainting: models like
FLUX Fill Pro produce "generally styled" hair but don't render specific cut
features (clean parts, skin fades, tapered sides, layer structure). Reference
images give the model concrete visual targets.

Pipeline:
  1. LOCAL  : MediaPipe-derived U-band hair mask (same as inpaint.py)
  2. REMOTE : usamaehsan/controlnet-x-ip-adapter-realistic-vision-v5
              - source image + mask + reference image + IP-Adapter conditioning

Cost: ~$0.07-0.10 per preview (Realistic Vision v5 + IP-Adapter + inpaint).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Reuse mask construction from inpaint.py to keep both pipelines aligned.
from backend.inpaint import (
    _build_local_hair_mask,
    DEFAULT_OFFSET_RATIO,
    DEFAULT_EXTEND_RATIO,
    DEFAULT_LATERAL_EXTEND,
    DEFAULT_FEATHER_PX,
    DEFAULT_EAR_LEVEL_RATIO,
)

logger = logging.getLogger(__name__)

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "catalogue" / "styles.json"
REFERENCES_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "references"

# Pinned model version (Realistic Vision v5 + ControlNet + IP-Adapter).
REFERENCE_INPAINT_MODEL = (
    "usamaehsan/controlnet-x-ip-adapter-realistic-vision-v5:"
    "50ac06bb9bcf30e7b5dc66d3fe6e67262059a11ade572a35afa0ef686f55db82"
)

# Defaults tuned for "transfer the reference hair onto this person, blend cleanly."
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.0
DEFAULT_IP_ADAPTER_WEIGHT = 1.0          # 0-2, 1.0 = strong reference influence
DEFAULT_IP_ADAPTER_CKPT = "ip-adapter-plus_sd15.bin"  # plus = best style transfer
DEFAULT_INPAINTING_STRENGTH = 0.95       # 0-1, high = fully repaint hair region
DEFAULT_INPAINTING_CONDITIONING = 1.0    # 0-1, controlnet conditioning strength


@dataclass
class RefInpaintResult:
    image_url: str
    style_id: str
    style_name: str
    reference_path: str
    seed: Optional[int]
    prompt: str
    ip_adapter_weight: float
    inpainting_strength: float
    steps: int
    mask_local_path: str
    model_ref: str

    def to_dict(self) -> dict:
        return asdict(self)


class RefInpaintError(RuntimeError):
    """Raised for any failure during reference inpainting."""


def generate_preview_with_reference(
    selfie_path: Path,
    style_id: str,
    reference_image_path: Optional[Path] = None,
    seed: Optional[int] = None,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    ip_adapter_weight: float = DEFAULT_IP_ADAPTER_WEIGHT,
    inpainting_strength: float = DEFAULT_INPAINTING_STRENGTH,
    inpainting_conditioning: float = DEFAULT_INPAINTING_CONDITIONING,
    ip_adapter_ckpt: str = DEFAULT_IP_ADAPTER_CKPT,
    save_mask_to: Optional[Path] = None,
) -> RefInpaintResult:
    """Generate a preview using a reference hairstyle photo for IP-Adapter conditioning.

    Args:
        selfie_path: source customer photo.
        style_id: catalogue entry id. If reference_image_path is omitted, looks for
            `reference_image_path` field on the style entry.
        reference_image_path: explicit path to the reference hairstyle photo.
            Overrides any path stored in the catalogue.
        seed: optional fixed seed.
        steps: denoising steps.
        guidance: classifier-free guidance scale.
        ip_adapter_weight: how strongly the reference influences the output (0-2).
        inpainting_strength: how aggressively to repaint the masked region (0-1).
        inpainting_conditioning: ControlNet inpaint conditioning scale (0-1).
        ip_adapter_ckpt: which IP-Adapter checkpoint - "ip-adapter-plus_sd15.bin"
            gives the strongest style transfer; -face is for face transfer.
        save_mask_to: optional path to persist the generated hair mask.

    Raises:
        RefInpaintError: any failure during mask construction or inference.
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise RefInpaintError("REPLICATE_API_TOKEN is not set in .env")

    if not selfie_path.exists():
        raise RefInpaintError(f"Selfie not found: {selfie_path}")

    style = _load_style(style_id)
    if style is None:
        raise RefInpaintError(f"Unknown style_id: {style_id}")

    # Resolve reference photo: explicit arg wins, else catalogue field.
    if reference_image_path is None:
        ref_field = style.get("reference_image_path")
        if not ref_field:
            raise RefInpaintError(
                f"Style {style_id!r} has no reference_image_path and none was "
                f"passed to the function."
            )
        reference_image_path = Path(ref_field)
        if not reference_image_path.is_absolute():
            reference_image_path = REFERENCES_DIR / reference_image_path

    if not reference_image_path.exists():
        raise RefInpaintError(f"Reference image not found: {reference_image_path}")

    try:
        import replicate
    except ImportError as e:
        raise RefInpaintError("`replicate` package not installed") from e

    # Lighter text prompt - reference photo does most of the visual work.
    text_prompt = _build_reference_prompt(style)

    # --- Step 1: build the hair-zone mask locally -----------------------------
    logger.info("ref-inpaint step 1/2: local hair mask")
    try:
        pil = Image.open(selfie_path).convert("RGB")
    except Exception as e:
        raise RefInpaintError(f"could not open selfie: {e}") from e

    image_rgb = np.array(pil)
    mask = _build_local_hair_mask(
        image_rgb,
        offset_ratio=DEFAULT_OFFSET_RATIO,
        extend_ratio=DEFAULT_EXTEND_RATIO,
        lateral_extend=DEFAULT_LATERAL_EXTEND,
        feather_px=DEFAULT_FEATHER_PX,
        ear_level_ratio=DEFAULT_EAR_LEVEL_RATIO,
    )

    if save_mask_to is not None:
        mask_path = Path(save_mask_to)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix="_mask.png", delete=False)
        tmp.close()
        mask_path = Path(tmp.name)
    cv2.imwrite(str(mask_path), mask)

    # --- Step 2: IP-Adapter inpaint with reference photo ----------------------
    logger.info(
        "ref-inpaint step 2/2: IP-Adapter inpaint style=%s ref=%s",
        style_id, reference_image_path.name,
    )
    try:
        with (
            selfie_path.open("rb") as src_f,
            mask_path.open("rb") as mask_f,
            reference_image_path.open("rb") as ref_f,
        ):
            payload = {
                "inpainting_image": src_f,
                "mask_image": mask_f,
                "ip_adapter_image": ref_f,
                "ip_adapter_weight": ip_adapter_weight,
                "ip_adapter_ckpt": ip_adapter_ckpt,
                "prompt": text_prompt,
                "negative_prompt": (
                    "lowres, blurry, deformed face, distorted features, "
                    "extra fingers, extra ears, plastic skin, oversmooth"
                ),
                "inpainting_strength": inpainting_strength,
                "inpainting_conditioning_scale": inpainting_conditioning,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "disable_safety_checker": True,
            }
            if seed is not None:
                payload["seed"] = seed
            output = replicate.run(REFERENCE_INPAINT_MODEL, input=payload)
    except Exception as e:
        raise RefInpaintError(f"IP-Adapter inpaint call failed: {e}") from e

    image_url = _extract_first_url(output)
    if not image_url:
        raise RefInpaintError(f"no image URL returned: {output!r}")

    return RefInpaintResult(
        image_url=image_url,
        style_id=style_id,
        style_name=style["name"],
        reference_path=str(reference_image_path),
        seed=seed,
        prompt=text_prompt,
        ip_adapter_weight=ip_adapter_weight,
        inpainting_strength=inpainting_strength,
        steps=steps,
        mask_local_path=str(mask_path),
        model_ref=REFERENCE_INPAINT_MODEL,
    )


# ---- helpers ----

def _build_reference_prompt(style: dict) -> str:
    """Compose a short text prompt to pair with the reference image.

    The reference photo carries the visual hair information; the text just
    nudges the model toward photorealism + style category.
    """
    name = style.get("name", "modern hairstyle").lower()
    return (
        f"photorealistic {name} on an Indian person, salon barbershop "
        "photography, sharp focus, natural hair texture, clean hairline, "
        "professional grooming, RAW photo, high detail"
    )


def _load_style(style_id: str) -> Optional[dict]:
    if not CATALOGUE_PATH.exists():
        return None
    with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
        styles = json.load(f)
    for s in styles:
        if s["id"] == style_id:
            return s
    return None


def _extract_first_url(output) -> Optional[str]:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str):
            return first
        url = getattr(first, "url", None)
        if isinstance(url, str):
            return url
    url = getattr(output, "url", None)
    if isinstance(url, str):
        return url
    return None
