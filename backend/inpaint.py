"""Inpainting-based hairstyle preview - the correct pipeline for salon use.

Two-step process (ONE remote API call):
  1. LOCAL: derive a hair-region mask from MediaPipe face landmarks.
       - Mask covers the area above the actual hairline, with enough headroom
         for tall hairstyles. Generated on the user's machine - no API cost.
  2. REMOTE: FLUX Fill Pro inpaints ONLY the masked region with the new style.
       - Face, skin, clothes, lighting, background stay byte-preserved.

Outcome: the customer's photo, with only the hair area regenerated. This is
exactly what a salon wants - "same photo of my customer, different hairstyle."

Why local mask:
  - The remote segmenter (grounded_sam) is unreliable with cold-start disconnects.
  - MediaPipe landmarks give us a deterministic hair-zone derived from the actual
    hairline of THIS customer's face. The dilated polygon above the hairline is
    a reliable "this is where new hair grows" region.
  - One API call = lower cost (~$0.05) and fewer failure modes.

Why FLUX Fill Pro:
  - SOTA inpainting model on Replicate in 2026 (Black Forest Labs).
  - Trained specifically for "edit only this region" - won't drift the rest.
  - 50 denoising steps available for max detail.

Cost: ~$0.05 per preview.
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

logger = logging.getLogger(__name__)

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "catalogue" / "styles.json"
REFERENCES_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "references"
RESULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "result_cache"

INPAINT_MODEL = (
    "black-forest-labs/flux-fill-pro:"
    "41c767bcbfffe54ef8f05eb4d0100f9314790f7fc43a7b88d73ec06839deddb9"
)

# FLUX Fill Pro tuning - hardened for "preserve face, change hair, photoreal."
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 35.0          # moderate - lets FLUX produce natural styling

# Local mask tuning - U-shaped band wrapping head: top + temples + sides to ear level.
DEFAULT_OFFSET_RATIO = -0.05     # push mask DOWN ~5% of face height into upper forehead so stray hair pixels get repainted (negative = downward)
DEFAULT_EXTEND_RATIO = 0.90      # mask reaches well above head (90% face height up)
DEFAULT_LATERAL_EXTEND = 0.25    # widen left+right by 25% of face width for fade/volume room
DEFAULT_FEATHER_PX = 24          # legacy fixed-pixel feather; superseded by feather_frac when face_width is known
DEFAULT_FEATHER_FRAC = 0.03      # preferred: feather size as a fraction of face width (resolution-invariant)
DEFAULT_EAR_LEVEL_RATIO = 0.80   # how far down to drop the side walls (frac of face height from forehead) - 0.80 ~= ear bottom

# Post-FLUX boundary colour match + source-pixel composite outside mask.
HARMONISE_DEFAULT = True

_mp_face_mesh = mp.solutions.face_mesh

# Mesh indices for the upper face boundary curve (left ear -> top -> right ear)
UPPER_FACE_ARC_INDICES = [
    234, 127, 162, 21, 54, 103, 67, 109,
    10,
    338, 297, 332, 284, 251, 389, 356, 454,
]
LM_FOREHEAD_TOP = 10
LM_CHIN_BOTTOM = 152
LM_LEFT_CHEEK = 234
LM_RIGHT_CHEEK = 454


@dataclass
class InpaintResult:
    image_url: str
    style_id: str
    style_name: str
    seed: Optional[int]
    prompt: str
    steps: int
    guidance: float
    mask_local_path: str          # for QA: open this to see where we inpainted
    model_ref: str
    enhanced: bool = False        # whether clarity-upscaler ran
    raw_image_url: Optional[str] = None  # FLUX output before enhancement

    def to_dict(self) -> dict:
        return asdict(self)


class InpaintError(RuntimeError):
    """Raised for any failure during the inpaint pipeline."""


def generate_preview_inpaint(
    selfie_path: Path,
    style_id: str,
    seed: Optional[int] = None,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    offset_ratio: float = DEFAULT_OFFSET_RATIO,
    extend_ratio: float = DEFAULT_EXTEND_RATIO,
    lateral_extend: float = DEFAULT_LATERAL_EXTEND,
    feather_px: int = DEFAULT_FEATHER_PX,
    feather_frac: Optional[float] = DEFAULT_FEATHER_FRAC,
    ear_level_ratio: float = DEFAULT_EAR_LEVEL_RATIO,
    save_mask_to: Optional[Path] = None,
    prompt_override: Optional[str] = None,
    enhance: bool = False,
    harmonise: bool = HARMONISE_DEFAULT,
    validate: bool = False,
    max_retries: int = 1,
) -> InpaintResult:
    """Local hair-mask + FLUX Fill Pro inpaint.

    Args:
        selfie_path: source photo (JPEG/PNG). Front-facing face required.
        style_id: catalogue entry id (see catalogue/styles.json).
        seed: optional fixed seed for reproducible outputs.
        steps: FLUX denoising steps (15-50). 50 = best quality.
        guidance: FLUX guidance scale (1.5-100). 60 default.
        offset_ratio: how far above face arc to start the hair zone (frac of face height).
        extend_ratio: how far above the head the hair zone reaches (frac of face height).
        lateral_extend: how much to widen the hair zone on each side (frac of face width).
        feather_px: gaussian blur radius for soft mask edges.
        save_mask_to: if set, also persist the binary mask PNG here for QA.

    Raises:
        InpaintError: any failure during face detection or inpainting.
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise InpaintError("REPLICATE_API_TOKEN is not set in .env")

    if not selfie_path.exists():
        raise InpaintError(f"Selfie not found: {selfie_path}")

    style = _load_style(style_id)
    if style is None:
        raise InpaintError(f"Unknown style_id: {style_id}")

    try:
        import replicate
    except ImportError as e:
        raise InpaintError("`replicate` package not installed") from e

    raw_prompt = prompt_override if prompt_override else style.get("prompt_template", "")
    if not raw_prompt:
        raw_prompt = _build_default_prompt_from_style(style)
    style_prompt = _build_flux_prompt(raw_prompt)

    # Apply per-style mask overrides (fringe needs offset above forehead, long
    # styles need more headroom + lateral room, etc.).  Caller-supplied kwargs
    # still win because we only fill the param if it wasn't overridden upstream.
    style_overrides = _style_mask_params(style)
    offset_ratio = style_overrides.get("offset_ratio", offset_ratio)
    extend_ratio = style_overrides.get("extend_ratio", extend_ratio)
    lateral_extend = style_overrides.get("lateral_extend", lateral_extend)
    ear_level_ratio = style_overrides.get("ear_level_ratio", ear_level_ratio)
    feather_frac = style_overrides.get("feather_frac", feather_frac)
    logger.info("mask params for %s: offset=%.2f extend=%.2f lateral=%.2f ear=%.2f feather_frac=%.3f",
                style_id, offset_ratio, extend_ratio, lateral_extend, ear_level_ratio,
                feather_frac if feather_frac is not None else -1.0)

    # --- Step 1: build the hair-zone mask LOCALLY -----------------------------
    logger.info("inpaint step 1/2: local hair mask from MediaPipe landmarks")
    try:
        pil = Image.open(selfie_path).convert("RGB")
    except Exception as e:
        raise InpaintError(f"could not open selfie: {e}") from e

    image_rgb = np.array(pil)
    mask = _build_local_hair_mask(
        image_rgb,
        offset_ratio=offset_ratio,
        extend_ratio=extend_ratio,
        lateral_extend=lateral_extend,
        feather_px=feather_px,
        feather_frac=feather_frac,
        ear_level_ratio=ear_level_ratio,
    )

    # Persist mask to a temp PNG (or user-specified path) for upload.
    if save_mask_to is not None:
        mask_path = Path(save_mask_to)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix="_mask.png", delete=False)
        tmp.close()
        mask_path = Path(tmp.name)

    cv2.imwrite(str(mask_path), mask)
    logger.info("hair mask saved to %s", mask_path)

    # --- Step 2: inpaint only the masked hair region with FLUX Fill Pro -------
    # Seed schedule: first the requested seed, then up to max_retries alternates
    # so the validator (if enabled) can ask FLUX for a fresh draw on failure.
    seed_schedule = [seed]
    if validate and max_retries > 0:
        base = seed if seed is not None else 42
        for i in range(max_retries):
            seed_schedule.append(base + 1000 + i)

    ref_path = resolve_reference_path(style)
    raw_url = None
    final_url = None
    last_validation: Optional[dict] = None

    for attempt_idx, attempt_seed in enumerate(seed_schedule):
        logger.info("inpaint attempt %d/%d (seed=%s): FLUX Fill Pro -> %s",
                    attempt_idx + 1, len(seed_schedule), attempt_seed, style_id)
        try:
            with selfie_path.open("rb") as img_f, mask_path.open("rb") as mask_f:
                inpaint_payload = {
                    "image": img_f,
                    "mask": mask_f,
                    "prompt": style_prompt,
                    "steps": max(15, min(50, steps)),
                    "guidance": max(1.5, min(100.0, guidance)),
                    "output_format": "png",
                    "safety_tolerance": 2,
                    "prompt_upsampling": False,
                }
                if attempt_seed is not None:
                    inpaint_payload["seed"] = attempt_seed
                inpaint_output = replicate.run(INPAINT_MODEL, input=inpaint_payload)
        except Exception as e:
            raise InpaintError(f"FLUX Fill Pro call failed: {e}") from e

        raw_url = _extract_first_url(inpaint_output)
        if not raw_url:
            raise InpaintError(f"no image URL returned: {inpaint_output!r}")

        # Boundary colour match + clean source composite (cheap, always worth it).
        attempt_final_url = raw_url
        if harmonise:
            try:
                from backend.colour_match import harmonise_with_source
                harmonised_path = harmonise_with_source(
                    source_path=selfie_path,
                    generated_url_or_path=raw_url,
                    mask_path=mask_path,
                    output_dir=selfie_path.parent,
                )
                # Serve the local harmonised file via the /uploads mount.
                attempt_final_url = f"/uploads/{harmonised_path.name}"
                logger.info("harmonised output saved at %s", harmonised_path)
            except Exception as e:
                logger.warning("harmonise step failed, using raw FLUX URL: %s", e)

        final_url = attempt_final_url

        # Stop here unless we're validating + a reference is available.
        if not validate or ref_path is None or attempt_idx == len(seed_schedule) - 1:
            break
        try:
            from backend.output_validator import validate_generation
            last_validation = validate_generation(
                source_path=selfie_path,
                reference_path=ref_path,
                generated_url=raw_url,
            )
            verdict = last_validation.get("verdict", "uncertain")
            logger.info("validator attempt %d: %s (%s)",
                        attempt_idx + 1, verdict,
                        last_validation.get("one_line_reason", ""))
            if verdict == "pass":
                break
        except Exception as e:
            logger.warning("validator unavailable, accepting current result: %s", e)
            break

    was_enhanced = False
    if enhance:
        try:
            from backend.enhance import enhance_image
            logger.info("inpaint step 2.5/3: clarity upscaler (hair-region only)")
            final_url = enhance_image(raw_url, preserve_mask_path=mask_path)
            was_enhanced = True
        except Exception as e:
            logger.warning("upscaler failed, returning raw FLUX output: %s", e)

    return InpaintResult(
        image_url=final_url,
        raw_image_url=raw_url if (was_enhanced or final_url != raw_url) else None,
        enhanced=was_enhanced,
        style_id=style_id,
        style_name=style["name"],
        seed=seed,
        prompt=style_prompt,
        steps=steps,
        guidance=guidance,
        mask_local_path=str(mask_path),
        model_ref=INPAINT_MODEL,
    )


# ---- mask construction ---------------------------------------------------

def _build_local_hair_mask(
    image_rgb: np.ndarray,
    offset_ratio: float,
    extend_ratio: float,
    lateral_extend: float,
    feather_px: int,
    ear_level_ratio: float,
    feather_frac: Optional[float] = None,
) -> np.ndarray:
    """Build a U-shaped band mask covering the hair zone around the head.

    Geometry: a band that wraps the head from one ear, up over the forehead
    (just above the hairline), around the top of the head with headroom for
    tall styles, and down to the other ear. The face itself (eyes, nose,
    cheeks, mouth, chin) is in the preserved black region.

    This shape gives FLUX canvas above the head AND on the temples/sides so
    that side fades, sideburns, ear-covering volume, and full hair silhouette
    can all be rendered.

    Mask convention: white (255) = inpaint, black (0) = preserve.
    """
    h, w = image_rgb.shape[:2]

    with _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(image_rgb)

    if not result.multi_face_landmarks:
        raise InpaintError("No face detected in the selfie - cannot build hair mask.")

    landmarks = result.multi_face_landmarks[0].landmark
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

    forehead_y = float(pts[LM_FOREHEAD_TOP][1])
    chin_y = float(pts[LM_CHIN_BOTTOM][1])
    face_height = float(abs(chin_y - forehead_y))
    face_width = float(abs(pts[LM_RIGHT_CHEEK][0] - pts[LM_LEFT_CHEEK][0]))

    # Arc along the upper face boundary, lifted upward toward the hairline.
    arc = pts[UPPER_FACE_ARC_INDICES].copy()
    arc[:, 1] -= face_height * offset_ratio

    # Side wall x coordinates - extend slightly past the face perimeter.
    lateral_px = face_width * lateral_extend
    left_x = arc[0, 0] - lateral_px
    right_x = arc[-1, 0] + lateral_px

    # Top of mask - well above the head for tall styles.
    top_y = forehead_y - face_height * extend_ratio

    # Bottom of side walls - drop to approximately ear level (mid-cheek).
    # Computed as forehead_y + ear_level_ratio * face_height.
    ear_y = forehead_y + face_height * ear_level_ratio

    # Build the U-band polygon counter-clockwise:
    # upper-left -> upper-right -> right-ear -> rightmost arc point ->
    # arc traversed right-to-left across forehead -> leftmost arc point ->
    # left-ear -> back to upper-left.
    polygon_pts = []
    polygon_pts.append([left_x, top_y])           # upper-left
    polygon_pts.append([right_x, top_y])          # upper-right
    polygon_pts.append([right_x, ear_y])          # right-ear (down right side)
    # Traverse arc from right end to left end (right-to-left along forehead)
    for p in arc[::-1]:
        polygon_pts.append(p.tolist())
    polygon_pts.append([left_x, ear_y])           # left-ear (after arc curls back)
    # Implicit close back to [left_x, top_y]

    polygon = np.array(polygon_pts, dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
    polygon = polygon.astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # Prefer feather_frac (resolution-invariant) when supplied; fall back to
    # legacy fixed pixel feather otherwise.
    effective_feather = feather_px
    if feather_frac is not None and feather_frac > 0:
        effective_feather = max(2, int(round(face_width * feather_frac)))

    if effective_feather and effective_feather > 0:
        k = max(3, effective_feather * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask


def _style_mask_params(style: dict) -> dict:
    """Per-style overrides for hair-mask geometry.

    Lookup order: style.mask_params (highest), then a small fallback table for
    common shapes (fringe, long, very short).  Returns only the keys that
    differ from defaults; callers merge with their own kwargs.
    """
    overrides = dict(style.get("mask_params") or {})
    length = (style.get("length") or "").lower()
    traits = {t.lower() for t in (style.get("style_traits") or [])}

    # Fringe / bangs / curtain bangs: start mask ABOVE the forehead so we don't
    # repaint forehead skin as hair.
    if traits & {"fringe", "curtain bangs", "bangs"}:
        overrides.setdefault("offset_ratio", 0.08)
        overrides.setdefault("feather_frac", 0.025)

    # Buzz / very-short: keep the mask snug to the head; less lateral room.
    if length in ("very short",) or "buzz cut" in traits or "shaved" in traits:
        overrides.setdefault("extend_ratio", 0.35)
        overrides.setdefault("lateral_extend", 0.12)
        overrides.setdefault("feather_frac", 0.02)

    # Long / very-long / braid / bun: needs lots of headroom + width.
    if length in ("long", "very long") or traits & {"braid", "long", "updo", "bun"}:
        overrides.setdefault("ear_level_ratio", 1.55)
        overrides.setdefault("lateral_extend", 0.40)
        overrides.setdefault("extend_ratio", 0.75)

    return overrides


# ---- helpers ----

def _build_flux_prompt(raw_style_prompt: str) -> str:
    """Compose a FLUX-optimized prompt for hair-only inpainting.

    FLUX prefers natural-language descriptions. The mask scopes regeneration to
    the hair area only, so the prompt should describe ONLY hair appearance and
    emphasize blending with the surrounding pixels (NOT studio lighting that
    would clash with the source photo's actual lighting).
    """
    base = raw_style_prompt.strip().rstrip(".")
    return (
        f"{base}. The hair sits naturally on the person's head with the SAME "
        "ambient indoor lighting as the rest of the photo, no studio lighting, "
        "no rim light, no glamour highlights. The hair colour stays in the "
        "natural dark-brown to black family typical of South Asian hair unless "
        "the style explicitly calls for a different colour. Hairline meets the "
        "forehead with a soft realistic blend, no hard edge, no painted-on "
        "look, no halo of stray pixels around the head. Photorealistic, sharp "
        "focus on individual hair strands, casual everyday photo."
    )


def _build_default_prompt_from_style(style: dict) -> str:
    """Compose a description from catalogue metadata when no explicit prompt
    is supplied.  Manual mode used to ship an empty prompt for most styles
    because few catalogue entries have a prompt_template; this fills the gap
    without requiring a vision-LM.
    """
    name = style.get("name") or "hairstyle"
    traits = style.get("style_traits") or []
    length = style.get("length") or ""
    cultural = style.get("cultural") or []
    gender = style.get("gender") or ""

    trait_clause = ", ".join(traits[:6]) if traits else ""
    cultural_clause = ", ".join(cultural[:3]) if cultural else ""
    bits = [f"A {name}"]
    if length:
        bits.append(f"{length} length")
    if trait_clause:
        bits.append(trait_clause)
    if cultural_clause:
        bits.append(f"{cultural_clause} style")
    if gender:
        bits.append(f"suited to {gender} customer")
    return ", ".join(bits)


def _result_cache_key(source_path: Path, style_id: str,
                      seed: Optional[int], mode: str) -> str:
    import hashlib as _hashlib
    h = _hashlib.sha1()
    h.update(mode.encode())
    h.update(b"|")
    h.update(style_id.encode())
    h.update(b"|")
    h.update(str(seed or "auto").encode())
    h.update(b"|")
    h.update(source_path.read_bytes())
    return f"r_{style_id}_{h.hexdigest()[:12]}"


def _get_cached_result(source_path: Path, style_id: str,
                       seed: Optional[int], mode: str) -> Optional[str]:
    try:
        key = _result_cache_key(source_path, style_id, seed, mode)
        cache_file = RESULT_CACHE_DIR / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


def _cache_result(source_path: Path, style_id: str, seed: Optional[int],
                  mode: str, url: str) -> None:
    key = _result_cache_key(source_path, style_id, seed, mode)
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_CACHE_DIR / f"{key}.txt").write_text(url, encoding="utf-8")


def _load_style(style_id: str) -> Optional[dict]:
    if not CATALOGUE_PATH.exists():
        return None
    with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
        styles = json.load(f)
    for s in styles:
        if s["id"] == style_id:
            return s
    return None


def resolve_reference_path(style: dict) -> Optional[Path]:
    """Resolve the reference image path from a catalogue style entry."""
    ref_field = style.get("reference_image_path")
    if not ref_field:
        return None
    p = Path(ref_field)
    if not p.is_absolute():
        p = REFERENCES_DIR / p
    return p if p.exists() else None


BALD_PROMPT = (
    "Smooth bare scalp completely without any hair, like a freshly shaved head with "
    "zero stubble. Skin tone exactly matches the person's forehead and natural face "
    "color, seamlessly blending into the existing skin with no visible boundary. "
    "Absolutely no hair strands, no fringe, no stubble shadow, no dark hair color "
    "anywhere - just smooth bare scalp skin. Natural soft ambient indoor lighting "
    "matching the source photo. Photorealistic skin texture, casual everyday photo, "
    "not glamour, not studio lit."
)


def generate_preview_erase_then_inpaint(
    selfie_path: Path,
    style_id: str,
    seed: Optional[int] = None,
    save_mask_to: Optional[Path] = None,
    expert_model: Optional[str] = None,
    validate: bool = True,
    max_retries: int = 1,
    **inpaint_kwargs,
) -> InpaintResult:
    """Two-pass FLUX: erase existing hair to bald scalp, then inpaint the new
    style on the bald canvas. Forces FLUX to DRAW new hair instead of merely
    editing the existing hair (which is the conservative default).

    Uses Claude expert consult for the second-pass style prompt.

    Cost: 2 FLUX calls (~$0.10) + 1 Claude consult ($0.01, cached).
    """
    style = _load_style(style_id)
    if style is None:
        raise InpaintError(f"Unknown style_id: {style_id}")

    ref_path = resolve_reference_path(style)
    if ref_path is None:
        raise InpaintError(
            f"Style {style_id!r} has no reference photo; erase-mode requires one."
        )

    # ---- RESULT CACHE: same (source, style, seed) returns instant URL ----
    cached = _get_cached_result(selfie_path, style_id, seed, "transform")
    if cached:
        logger.info("result cache hit: returning %s for %s", cached, style_id)
        return InpaintResult(
            image_url=cached, raw_image_url=None, enhanced=False,
            style_id=style_id, style_name=style["name"],
            seed=seed, prompt="(cached)", steps=0, guidance=0.0,
            mask_local_path="(cached)", model_ref="cached",
        )

    # ---- PASS 1: erase existing hair to bald scalp ----
    # Use high guidance to force the model to actually render bare scalp instead
    # of conservatively keeping existing hair pixels.
    logger.info("erase-mode pass 1/2: erasing existing hair (high guidance)")
    bald_kwargs = dict(inpaint_kwargs)
    bald_kwargs["guidance"] = 75.0  # aggressive prompt adherence for clean erase
    bald_kwargs["extend_ratio"] = 1.15  # extend mask higher so top-of-head hair is covered
    bald_kwargs["harmonise"] = False    # bald pass MUST keep raw FLUX scalp; never composite hair back
    bald_kwargs["validate"] = False     # bald output validated against a hairstyle ref makes no sense
    bald_result = generate_preview_inpaint(
        selfie_path=selfie_path,
        style_id=style_id,
        seed=seed,
        save_mask_to=save_mask_to,
        prompt_override=BALD_PROMPT,
        enhance=False,
        **bald_kwargs,
    )

    bald_url = bald_result.image_url
    if not bald_url:
        raise InpaintError("Pass 1 (erase) produced no URL")

    # ---- PASS 2: ask Claude for the adapted style prompt, then inpaint
    #             the bald canvas with the new hairstyle. ----
    from backend.expert_consult import consult_for_style, ConsultError
    try:
        kwargs = {"model": expert_model} if expert_model else {}
        adapted_prompt = consult_for_style(
            source_image_path=selfie_path,
            reference_image_path=ref_path,
            **kwargs,
        )
    except ConsultError as e:
        raise InpaintError(f"Claude consult failed: {e}") from e

    logger.info("erase-mode pass 2/2: drawing new hair on bald canvas")
    attempt_seeds = [seed]
    if validate and max_retries > 0:
        for i in range(max_retries):
            attempt_seeds.append((seed if seed is not None else 42) + 1000 + i)

    final_result = None
    validation = None
    for attempt_idx, attempt_seed in enumerate(attempt_seeds):
        final_result = _inpaint_on_remote_url(
            source_url=bald_url,
            mask_local_path=Path(bald_result.mask_local_path),
            prompt=adapted_prompt,
            style=style,
            seed=attempt_seed,
            **{k: v for k, v in inpaint_kwargs.items()
               if k in ("steps", "guidance")},
        )
        if not validate or attempt_idx == len(attempt_seeds) - 1:
            break
        try:
            from backend.output_validator import validate_generation, ValidationError as VE
            validation = validate_generation(
                source_path=selfie_path,
                reference_path=ref_path,
                generated_url=final_result["image_url"],
            )
            logger.info("validation attempt %d: %s (%s)",
                        attempt_idx + 1, validation.get("verdict"),
                        validation.get("one_line_reason"))
            if validation.get("verdict") == "pass":
                break
        except Exception as e:
            logger.warning("validator unavailable, accepting current attempt: %s", e)
            break

    # Final harmonisation: pull lighting/colour back toward the ORIGINAL
    # customer photo (not the bald canvas) and composite source pixels outside
    # the mask so face/skin/clothes stay byte-preserved.
    final_image_url = final_result["image_url"]
    try:
        from backend.colour_match import harmonise_with_source
        harmonised_path = harmonise_with_source(
            source_path=selfie_path,
            generated_url_or_path=final_image_url,
            mask_path=Path(bald_result.mask_local_path),
            output_dir=selfie_path.parent,
        )
        final_image_url = f"/uploads/{harmonised_path.name}"
        logger.info("transform harmonised output saved at %s", harmonised_path)
    except Exception as e:
        logger.warning("transform harmonise step failed: %s", e)

    result = InpaintResult(
        image_url=final_image_url,
        raw_image_url=bald_url,
        enhanced=False,
        style_id=style_id,
        style_name=style["name"],
        seed=attempt_seeds[-1] if final_result else seed,
        prompt=adapted_prompt,
        steps=final_result["steps"],
        guidance=final_result["guidance"],
        mask_local_path=str(bald_result.mask_local_path),
        model_ref=INPAINT_MODEL,
    )
    # Cache successful result so repeat customer/style requests are free
    try:
        _cache_result(selfie_path, style_id, seed, "transform", result.image_url)
    except Exception as e:
        logger.warning("result cache write failed: %s", e)
    return result


def _inpaint_on_remote_url(
    source_url: str,
    mask_local_path: Path,
    prompt: str,
    style: dict,
    seed: Optional[int],
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
) -> dict:
    """Run FLUX Fill Pro inpainting using a remote URL as the source.

    Reuses our local hair mask. Returns dict with image_url + params.
    """
    import replicate
    style_prompt = _build_flux_prompt(prompt)

    try:
        with mask_local_path.open("rb") as mask_f:
            payload = {
                "image": source_url,         # URL passthrough
                "mask": mask_f,
                "prompt": style_prompt,
                "steps": max(15, min(50, steps)),
                "guidance": max(1.5, min(100.0, guidance)),
                "output_format": "png",
                "safety_tolerance": 2,
                "prompt_upsampling": False,
            }
            if seed is not None:
                payload["seed"] = seed
            output = replicate.run(INPAINT_MODEL, input=payload)
    except Exception as e:
        raise InpaintError(f"Pass 2 (inpaint on bald) failed: {e}") from e

    url = _extract_first_url(output)
    if not url:
        raise InpaintError(f"Pass 2 returned no URL: {output!r}")
    return {"image_url": url, "steps": steps, "guidance": guidance}


def generate_preview_expert(
    selfie_path: Path,
    style_id: str,
    seed: Optional[int] = None,
    save_mask_to: Optional[Path] = None,
    expert_model: Optional[str] = None,
    **inpaint_kwargs,
) -> InpaintResult:
    """Top-quality pipeline: a vision-LM (via backend.expert_consult) acts as
    the expert stylist, generating a customer-adapted prompt by looking at
    BOTH the source photo and the style's reference photo, then FLUX inpaints.

    Requires the provider API key in the environment (.env).
    """
    style = _load_style(style_id)
    if style is None:
        raise InpaintError(f"Unknown style_id: {style_id}")

    ref_path = resolve_reference_path(style)
    if ref_path is None:
        raise InpaintError(
            f"Style {style_id!r} has no reference_image_path or the file is "
            f"missing. Expert consult requires a reference photo."
        )

    from backend.expert_consult import consult_for_style, ConsultError
    try:
        kwargs = {"model": expert_model} if expert_model else {}
        adapted_prompt = consult_for_style(
            source_image_path=selfie_path,
            reference_image_path=ref_path,
            **kwargs,
        )
    except ConsultError as e:
        raise InpaintError(f"expert consult failed: {e}") from e

    logger.info("expert prompt for %s: %s", style_id, adapted_prompt[:140])

    return generate_preview_inpaint(
        selfie_path=selfie_path,
        style_id=style_id,
        seed=seed,
        save_mask_to=save_mask_to,
        prompt_override=adapted_prompt,
        **inpaint_kwargs,
    )


def generate_preview_auto(
    selfie_path: Path,
    style_id: str,
    seed: Optional[int] = None,
    save_mask_to: Optional[Path] = None,
    **inpaint_kwargs,
) -> InpaintResult:
    """Software-grade path: derive the inpaint prompt automatically from the
    style's reference image via Florence-2, then inpaint.

    No hand-written prompts. Salons just drop a reference photo into the
    catalogue; Florence-2 describes it; FLUX inpaints. Captions are cached
    per reference image so we pay Florence-2 only once.

    Raises InpaintError on any failure.
    """
    style = _load_style(style_id)
    if style is None:
        raise InpaintError(f"Unknown style_id: {style_id}")

    ref_path = resolve_reference_path(style)
    if ref_path is None:
        raise InpaintError(
            f"Style {style_id!r} has no reference_image_path or the file is "
            f"missing. Auto mode requires a reference photo."
        )

    # Import lazily so users can still call the manual-prompt path without
    # paying for the Florence-2 dependency.
    from backend.auto_caption import describe_hair_only, CaptionError
    try:
        auto_prompt = describe_hair_only(ref_path)
    except CaptionError as e:
        raise InpaintError(f"auto-caption failed: {e}") from e

    logger.info("auto-prompt for %s: %s", style_id, auto_prompt[:120])

    return generate_preview_inpaint(
        selfie_path=selfie_path,
        style_id=style_id,
        seed=seed,
        save_mask_to=save_mask_to,
        prompt_override=auto_prompt,
        **inpaint_kwargs,
    )


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
