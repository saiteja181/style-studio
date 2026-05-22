"""Hairstyle preview generation via Replicate (face-preserving image gen).

Supports two backends:
  - "instantid"  (default) : zsxkib/instant-id - strong face identity preservation,
                              ControlNet-based pose preservation. Best for our salon
                              use case where the customer must still look like themselves.
  - "photomaker"           : tencentarc/photomaker - faster but weaker face fidelity.
                              Useful for stylized outputs (paintings, avatars).

Backend is chosen via the `backend` argument, the REPLICATE_BACKEND env var, or
defaults to "instantid".

`generate_preview()` returns the URL of the generated preview plus diagnostics.
No image is generated until the function is called, so the module imports
safely even without REPLICATE_API_TOKEN configured.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "catalogue" / "styles.json"

# Pinned model versions. Update these when a better version ships, and bump
# the matching schema dispatch if input shapes change.
BACKENDS: dict[str, str] = {
    "instantid": (
        "zsxkib/instant-id:"
        "2e4785a4d80dadf580077b2244c8d7c05d8e3faac04a04c02d8e099dd2876789"
    ),
    "photomaker": (
        "tencentarc/photomaker:"
        "ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4"
    ),
}

DEFAULT_BACKEND = "instantid"

# PhotoMaker requires the literal token "img" anchored in the prompt.
PHOTOMAKER_TRIGGER = "img"


@dataclass
class GenerationResult:
    image_url: str
    style_id: str
    style_name: str
    backend: str
    model_ref: str
    seed: Optional[int]
    prompt: str
    negative_prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


class GenerationError(RuntimeError):
    """Raised for any failure during preview generation."""


def generate_preview(
    selfie_path: Path,
    style_id: str,
    seed: Optional[int] = None,
    backend: Optional[str] = None,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    ip_adapter_scale: float = 0.8,
    controlnet_conditioning_scale: float = 0.8,
    style_strength: int = 25,
) -> GenerationResult:
    """Generate one preview image and return its URL + metadata.

    Args:
        selfie_path: path to a JPEG or PNG of the customer's face.
        style_id: catalogue entry id (see catalogue/styles.json).
        seed: optional fixed seed for reproducible outputs.
        backend: "instantid" (default) or "photomaker". Overrides
            REPLICATE_BACKEND env var.
        num_steps: diffusion steps. Higher = better quality, slower.
        guidance_scale: prompt adherence strength.
        ip_adapter_scale: InstantID only - face-detail strength (0-1.5, default 0.8).
        controlnet_conditioning_scale: InstantID only - face fidelity (0-1.5, default 0.8).
        style_strength: PhotoMaker only - 0-100. Lower keeps face identity stronger.

    Raises:
        GenerationError: any failure during the API call or response handling.
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise GenerationError(
            "REPLICATE_API_TOKEN is not set. Add it to .env before generating."
        )

    if not selfie_path.exists():
        raise GenerationError(f"Selfie not found: {selfie_path}")

    style = _load_style(style_id)
    if style is None:
        raise GenerationError(f"Unknown style_id: {style_id}")

    chosen_backend = (
        backend
        or os.getenv("REPLICATE_BACKEND")
        or DEFAULT_BACKEND
    ).lower()
    if chosen_backend not in BACKENDS:
        raise GenerationError(
            f"Unknown backend: {chosen_backend!r}. Choose from {list(BACKENDS)}."
        )
    model_ref = BACKENDS[chosen_backend]

    raw_prompt = style.get("prompt_template", "")
    negative = style.get("negative_prompt", "")

    try:
        import replicate
    except ImportError as e:
        raise GenerationError(
            "The `replicate` package is not installed. Run setup.ps1 first."
        ) from e

    logger.info(
        "generate_preview start | style=%s backend=%s model=%s seed=%s",
        style_id, chosen_backend, model_ref, seed,
    )

    if chosen_backend == "instantid":
        sent_prompt = raw_prompt
        input_payload = {
            "image": None,  # filled below from open file handle
            "prompt": raw_prompt,
            "negative_prompt": negative,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "ip_adapter_scale": ip_adapter_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "enable_pose_controlnet": True,
            "enhance_nonface_region": True,
            "output_format": "png",
            "output_quality": 95,
            "seed": seed if seed is not None else -1,
        }
        image_key = "image"
    elif chosen_backend == "photomaker":
        sent_prompt = _compose_photomaker_prompt(raw_prompt)
        input_payload = {
            "prompt": sent_prompt,
            "negative_prompt": negative,
            "input_image": None,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "style_strength_ratio": style_strength,
            "seed": seed if seed is not None else -1,
        }
        image_key = "input_image"
    else:  # pragma: no cover - guarded above
        raise GenerationError(f"Backend dispatch missing: {chosen_backend}")

    try:
        with selfie_path.open("rb") as f:
            input_payload[image_key] = f
            output = replicate.run(model_ref, input=input_payload)
    except Exception as e:
        raise GenerationError(f"Replicate call failed: {e}") from e

    image_url = _extract_first_url(output)
    if not image_url:
        raise GenerationError(f"Could not extract image URL from output: {output!r}")

    return GenerationResult(
        image_url=image_url,
        style_id=style_id,
        style_name=style["name"],
        backend=chosen_backend,
        model_ref=model_ref,
        seed=seed,
        prompt=sent_prompt,
        negative_prompt=negative,
    )


# ---- helpers ----

def _compose_photomaker_prompt(raw: str) -> str:
    """Inject the PhotoMaker face-anchor token if it isn't already present."""
    if PHOTOMAKER_TRIGGER in raw.split():
        return raw
    return f"a person {PHOTOMAKER_TRIGGER}, {raw}"


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
    """Replicate returns either a single URL string, a list of URLs, or FileOutputs."""
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
