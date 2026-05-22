# Core engine swap to FLUX Kontext Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace FLUX Fill Pro with FLUX Kontext Pro as the only generation engine, keep customer face byte-perfect via a local OpenCV composite, drop the `mode` API parameter, delete ~1,400 lines of legacy inpaint code, and ship with a manual visual acceptance test on 6 outputs against unmodified Indian-face sources.

**Architecture:** Three small new modules — `kontext_engine.py` (sole Replicate caller), `face_composite.py` (MediaPipe polygon + alpha-blend, no network), `prompt_builder.py` (catalogue → optional Anthropic expert rewrite → Kontext wrapper). `main.py` drops the `mode` param and routes through `kontext_engine.generate_preview`. Validator + retry loop is integrated inside the engine.

**Tech Stack:** Python 3.11, FastAPI, Replicate SDK (`black-forest-labs/flux-kontext-pro`), MediaPipe FaceMesh, OpenCV, PIL, Anthropic SDK (Haiku 4.5 for validator, Sonnet 4.6 for expert prompt rewrite — both optional via `ANTHROPIC_API_KEY`), pytest.

---

## Prerequisites

Before starting Task 1, verify the working tree is clean:

```bash
cd C:/Users/Asus/Desktop/style-studio
git status            # should show clean working tree on `main`
git log --oneline -1  # should show "Spec: core engine swap..." commit
```

Also verify the env is loaded:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('replicate_set=', bool(os.getenv('REPLICATE_API_TOKEN'))); print('anthropic_set=', bool(os.getenv('ANTHROPIC_API_KEY')))"
```

Expected: `replicate_set= True` (Anthropic may be False; the plan handles that).

The source files used by tests must exist:

```bash
ls tests/selfies/test_random_indian_man.jpg tests/selfies/test_indian_woman_b.jpg
```

Both should print without error.

---

## Task 1: `face_composite.paste_source_face` — basic composite

**Files:**
- Create: `backend/face_composite.py`
- Create: `tests/test_face_composite.py`

This task delivers the smallest valuable composite: build a MediaPipe-derived face polygon and alpha-blend source pixels back over a Kontext-style "different" image. We verify by using a synthetic all-red image as the "Kontext output" — after compositing, the central face region should still be source pixels, and the corners should still be red.

- [ ] **Step 1: Write the failing test**

Create `tests/test_face_composite.py`:

```python
"""Tests for backend.face_composite.paste_source_face."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_MAN = PROJECT_ROOT / "tests" / "selfies" / "test_random_indian_man.jpg"


def _make_red_kontext_like(source_path: Path, out_path: Path) -> None:
    """Save an image the same size as the source, filled with pure red.
    Stands in for a Kontext output so we can verify the composite preserves
    source pixels in the face region."""
    src = Image.open(source_path).convert("RGB")
    red = Image.new("RGB", src.size, (220, 30, 30))
    red.save(out_path, format="PNG")


def test_paste_source_face_preserves_face_replaces_background(tmp_path):
    from backend.face_composite import paste_source_face

    kontext_path = tmp_path / "fake_kontext.png"
    _make_red_kontext_like(SOURCE_MAN, kontext_path)

    out = paste_source_face(
        source_path=SOURCE_MAN,
        kontext_output_url_or_path=kontext_path,
        output_dir=tmp_path,
    )
    assert out.exists(), "composite output file was not written"

    src = np.array(Image.open(SOURCE_MAN).convert("RGB"))
    composed = np.array(Image.open(out).convert("RGB"))
    assert composed.shape == src.shape, "composite must match source dimensions"

    h, w = src.shape[:2]
    # Central face pixel - should be close to source (alpha = 1 in face polygon)
    cy, cx = h // 2, w // 2
    src_center = src[cy, cx].astype(int)
    out_center = composed[cy, cx].astype(int)
    diff_center = int(np.abs(out_center - src_center).max())
    assert diff_center < 8, (
        f"central face pixel drifted from source by {diff_center}; "
        f"face polygon may not cover the centre"
    )

    # Top-left corner - should be the red Kontext background (alpha = 0)
    out_corner = composed[8, 8].astype(int)
    assert out_corner[0] > 180 and out_corner[1] < 80 and out_corner[2] < 80, (
        f"top-left corner = {tuple(out_corner)}, expected red Kontext pixel"
    )
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
cd C:/Users/Asus/Desktop/style-studio
python -m pytest tests/test_face_composite.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.face_composite'` (or `ImportError`). The test file imports a module that doesn't exist yet.

- [ ] **Step 3: Implement `backend/face_composite.py`**

Create `backend/face_composite.py`:

```python
"""Paste the source customer's face polygon onto a Kontext-generated image.

This is what makes the "identity" guarantee hold despite Kontext regenerating
the whole image: we composite source pixels back over a polygon covering
eyes, nose, mouth, cheeks, and jaw.  The forehead, hairline, and ears stay
as Kontext output so the new hairstyle can render freely.
"""
from __future__ import annotations

import io
import logging
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Union

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_mp_face_mesh = mp.solutions.face_mesh

# Face polygon indices, in counter-clockwise order, covering eyes/nose/mouth/
# cheeks/jaw but EXCLUDING forehead/hairline/ears.  Traverses jawline from one
# ear to the other (along the chin), then back across the eyebrow line.
# Source: MediaPipe Face Mesh canonical 478-point map.
FACE_POLYGON_INDICES = [
    # right side jaw (from ear down to chin)
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
    # chin
    152,
    # left side jaw (chin back up to ear)
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
    # along left brow to mid-forehead at brow line, then across right brow
    356, 389, 251, 284, 332, 297,
    9,    # between-brows, mid line
    67, 109, 103, 54, 21, 162, 127,
]


def paste_source_face(
    source_path: Path,
    kontext_output_url_or_path: Union[str, Path],
    output_dir: Path,
    feather_px: int = 18,
) -> Path:
    """Composite the customer's face polygon (from MediaPipe) onto a Kontext
    output image.

    Args:
        source_path: pre-flight-normalised customer photo.
        kontext_output_url_or_path: Replicate URL or local Path of the
            Kontext-generated image.
        output_dir: where to write the composited PNG.
        feather_px: Gaussian blur radius for the polygon edge, in pixels.
            ~18 gives a soft seam between the source face and the new hair.

    Returns:
        Path to the composited PNG.

    Raises:
        FileNotFoundError: if source_path is missing.
        ValueError: if either image fails to decode.
    """
    source_rgb = np.array(Image.open(source_path).convert("RGB"))
    kontext_rgb = _load_rgb(kontext_output_url_or_path)

    # Match dimensions to source (Kontext may return a slightly different size
    # depending on the input aspect ratio).
    h, w = source_rgb.shape[:2]
    if kontext_rgb.shape[:2] != (h, w):
        kontext_rgb = cv2.resize(
            kontext_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4,
        )

    face_alpha = _build_face_alpha(source_rgb, feather_px=feather_px)
    if face_alpha is None:
        logger.warning(
            "face_composite: no face detected in source; returning raw Kontext"
        )
        return _save_png(kontext_rgb, output_dir, prefix="kontext_only_")

    alpha = (face_alpha.astype(np.float32) / 255.0)[..., None]
    composed = (
        kontext_rgb.astype(np.float32) * (1.0 - alpha)
        + source_rgb.astype(np.float32) * alpha
    )
    composed = np.clip(composed, 0, 255).astype(np.uint8)
    return _save_png(composed, output_dir, prefix="composed_")


def _build_face_alpha(
    image_rgb: np.ndarray, feather_px: int,
) -> Optional[np.ndarray]:
    """Build a feathered alpha mask covering the face polygon.

    Returns None if MediaPipe finds no face in the image.
    """
    h, w = image_rgb.shape[:2]
    with _mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(image_rgb)
    if not result.multi_face_landmarks:
        return None
    landmarks = result.multi_face_landmarks[0].landmark
    if len(landmarks) <= max(FACE_POLYGON_INDICES):
        return None

    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
    poly = pts[FACE_POLYGON_INDICES].astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    if feather_px > 0:
        k = max(3, feather_px * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def _load_rgb(src: Union[str, Path]) -> np.ndarray:
    """Load an RGB image from a local path or http(s) URL."""
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=60) as resp:
            return np.array(Image.open(io.BytesIO(resp.read())).convert("RGB"))
    raise FileNotFoundError(f"image source not found: {src}")


def _save_png(rgb: np.ndarray, output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp = tempfile.NamedTemporaryFile(
        prefix=prefix, suffix=".png", delete=False, dir=str(output_dir),
    )
    Image.fromarray(rgb).save(fp, format="PNG", optimize=False)
    fp.close()
    return Path(fp.name)
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
python -m pytest tests/test_face_composite.py -v
```

Expected: `1 passed in <X>s`. If the central-pixel diff assertion trips, increase `feather_px` to 24 or widen the polygon — but the chosen indices should put the centre well inside the face.

- [ ] **Step 5: Commit**

```bash
git add backend/face_composite.py tests/test_face_composite.py
git commit -m "Add face_composite: paste source face polygon over Kontext output

MediaPipe FaceMesh -> 478 landmarks -> polygon covering eyes, nose, mouth,
cheeks, and jaw (NOT forehead, hairline, ears).  Feathered alpha-blend
composites source pixels back over a Kontext-generated image so customer
identity stays byte-perfect on the face while hair can change freely above
the brow line."
```

---

## Task 2: `face_composite` — handle the missing-face fallback explicitly

**Files:**
- Modify: `tests/test_face_composite.py` (add one test)

The Task 1 implementation already returns the raw Kontext output when MediaPipe finds no face. This task pins that behaviour with a regression test so a later refactor can't silently change it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_composite.py`:

```python
def test_no_face_in_source_returns_kontext_unchanged(tmp_path):
    """If MediaPipe can't find a face in the source, the function must NOT
    crash - it should ship the Kontext output as-is.  Tests defence-in-depth
    against unusual inputs that slipped past pre-flight."""
    from backend.face_composite import paste_source_face

    # Source with no face: solid grey.  Same dims as the man photo so we
    # don't accidentally exercise the resize path here.
    src_arr = np.full((800, 1216, 3), 128, dtype=np.uint8)
    blank_src = tmp_path / "blank.jpg"
    Image.fromarray(src_arr).save(blank_src, format="JPEG", quality=92)

    kontext = tmp_path / "kontext.png"
    Image.new("RGB", (1216, 800), (220, 30, 30)).save(kontext, format="PNG")

    out = paste_source_face(
        source_path=blank_src,
        kontext_output_url_or_path=kontext,
        output_dir=tmp_path,
    )
    out_arr = np.array(Image.open(out).convert("RGB"))
    # Should be the Kontext red, not the grey source - because we fell back
    # to shipping Kontext when face detection failed.
    assert out_arr[400, 600, 0] > 180, "expected Kontext red, got something else"
    assert out_arr[400, 600, 1] < 80
    assert out_arr[400, 600, 2] < 80
```

- [ ] **Step 2: Run the test and verify it passes**

```bash
python -m pytest tests/test_face_composite.py -v
```

Expected: `2 passed`. The Task 1 implementation already handles this path; this test pins the behaviour.

- [ ] **Step 3: Commit**

```bash
git add tests/test_face_composite.py
git commit -m "Pin face_composite no-face fallback behaviour with a test"
```

---

## Task 3: `prompt_builder` — default style prompt + colour + texture clauses

**Files:**
- Create: `backend/prompt_builder.py`
- Create: `tests/test_prompt_builder.py`

Most of this logic exists today in `backend/inpaint.py`. We relocate it into a focused module with a clear public surface and unit-test the parts that drive prompt quality.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prompt_builder.py`:

```python
"""Tests for backend.prompt_builder."""
from __future__ import annotations


def test_default_from_style_includes_name_length_traits_culture_gender():
    from backend.prompt_builder import _default_from_style
    style = {
        "id": "mens_pompadour",
        "name": "Modern Pompadour with Skin Fade",
        "gender": "male",
        "length": "medium",
        "style_traits": ["pompadour", "fade", "structured", "height on top"],
        "cultural": ["modern"],
    }
    p = _default_from_style(style)
    assert "Modern Pompadour" in p
    assert "medium length" in p
    assert "pompadour" in p
    assert "modern" in p
    assert "male" in p


def test_default_from_style_handles_missing_fields():
    from backend.prompt_builder import _default_from_style
    style = {"id": "mystery", "name": "Mystery Cut"}
    p = _default_from_style(style)
    assert "Mystery Cut" in p
    assert isinstance(p, str)


def test_colour_clause_uses_hex_when_rgb_supplied():
    from backend.prompt_builder import _colour_clause
    c = _colour_clause({"hair_color_rgb": (50, 40, 38)})
    assert "#322826" in c
    assert "natural" in c.lower()


def test_colour_clause_blank_when_missing():
    from backend.prompt_builder import _colour_clause
    assert _colour_clause({}) == ""
    assert _colour_clause({"hair_color_rgb": None}) == ""


def test_texture_contrast_fires_when_source_disagrees_with_target():
    from backend.prompt_builder import _texture_contrast_clause
    style = {
        "compat_texture": ["straight"],
        "style_traits": ["straight", "sleek", "modern"],
    }
    profile = {"hair_texture": "curly"}
    c = _texture_contrast_clause(style, profile)
    assert "straight" in c
    assert "curly" in c


def test_texture_contrast_quiet_when_unknown():
    from backend.prompt_builder import _texture_contrast_clause
    style = {"compat_texture": ["straight"], "style_traits": ["straight"]}
    profile = {"hair_texture": "unknown"}
    assert _texture_contrast_clause(style, profile) == ""
```

- [ ] **Step 2: Run the tests and verify they fail**

```bash
python -m pytest tests/test_prompt_builder.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.prompt_builder'`.

- [ ] **Step 3: Implement `backend/prompt_builder.py`**

Create `backend/prompt_builder.py`:

```python
"""Build the natural-language edit prompt sent to FLUX Kontext.

Layers (lightest to heaviest):
  1. Base description - catalogue `prompt_template` if present, else built
     from style name + traits + length + cultural + gender.
  2. Optional Anthropic expert rewrite when ANTHROPIC_API_KEY is set AND the
     style has a reference photo on disk.  Reuses backend.expert_consult.
  3. Customer hair-colour hex anchor (no bleach / no colour drift).
  4. Texture contrast clause when source texture disagrees with target.
  5. Kontext "Change ONLY the hairstyle to:" wrapper + identity-preservation
     clause.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def build_edit_prompt(
    style: dict,
    customer_profile: dict,
    source_path: Path,
    reference_path: Optional[Path],
) -> str:
    """Compose the full Kontext prompt.  See module docstring for layers."""
    base = style.get("prompt_template")
    if not base:
        base = _default_from_style(style)

    if (os.getenv("ANTHROPIC_API_KEY") and reference_path
            and Path(reference_path).exists()):
        try:
            from backend import expert_consult
            base = expert_consult.consult_for_style(
                source_image_path=source_path,
                reference_image_path=reference_path,
            )
        except Exception as e:
            logger.info("expert_consult unavailable, using catalogue base: %s", e)

    colour = _colour_clause(customer_profile)
    texture = _texture_contrast_clause(style, customer_profile)
    return (
        f"Change ONLY the hairstyle to: {base}.{colour}{texture} "
        "Keep the face, eyes, expression, beard, eyebrows, glasses, "
        "clothing, hands, and background exactly identical to the original "
        "photo - do not change anything below the eyebrows. Photoreal, same "
        "ambient indoor lighting as the source, no studio lighting, no halo."
    )


def _default_from_style(style: dict) -> str:
    name = style.get("name") or "hairstyle"
    traits = style.get("style_traits") or []
    length = style.get("length") or ""
    cultural = style.get("cultural") or []
    gender = style.get("gender") or ""

    parts = [f"A {name}"]
    if length:
        parts.append(f"{length} length")
    if traits:
        parts.append(", ".join(traits[:6]))
    if cultural:
        parts.append(f"{', '.join(cultural[:3])} style")
    if gender:
        parts.append(f"suited to {gender} customer")
    return ", ".join(parts)


def _colour_clause(customer_profile: dict) -> str:
    rgb = customer_profile.get("hair_color_rgb")
    if not rgb or len(rgb) != 3:
        return ""
    try:
        r, g, b = (int(c) for c in rgb)
    except (TypeError, ValueError):
        return ""
    hex_code = f"#{r:02x}{g:02x}{b:02x}"
    return (
        f" Keep the hair colour the customer's natural shade ({hex_code}); "
        "no bleach, no highlights, no colour drift."
    )


def _texture_contrast_clause(style: dict, customer_profile: dict) -> str:
    src = (customer_profile.get("hair_texture") or "").lower()
    if not src or src == "unknown":
        return ""
    compat = [t.lower() for t in style.get("compat_texture", [])]
    traits = [t.lower() for t in style.get("style_traits", [])]
    if src in compat:
        return ""
    target = next((t for t in traits
                   if t in ("straight", "wavy", "curly", "coiled")), None)
    if not target:
        return ""
    return (
        f" The new hair texture is visibly {target}, clearly different from "
        f"the customer's source {src} texture - do not retain the original "
        "hair shape."
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
python -m pytest tests/test_prompt_builder.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/prompt_builder.py tests/test_prompt_builder.py
git commit -m "Add prompt_builder: Kontext edit prompt assembly

Relocates the smart-default, colour-hex, and texture-contrast logic out of
inpaint.py into a focused module with unit tests.  Adds the Kontext-specific
'Change ONLY the hairstyle to:' wrapper + identity-preservation clause.
Optionally calls expert_consult when ANTHROPIC_API_KEY is set and a
reference photo exists."
```

---

## Task 4: `kontext_engine` — Replicate call + `PreviewResult` dataclass

**Files:**
- Create: `backend/kontext_engine.py`
- Create: `tests/test_kontext_engine.py`

This task wires the actual Replicate call.  We exercise it once against the real API (smoke test, ~$0.04 cost) so we know the call signature works and we can extract the URL.

- [ ] **Step 1: Write the smoke test**

Create `tests/test_kontext_engine.py`:

```python
"""Smoke tests for backend.kontext_engine.  These hit the real Replicate API
and cost ~$0.04 per run; skipped automatically when REPLICATE_API_TOKEN is
not configured so CI can still pass without billing keys."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

SOURCE_MAN = PROJECT_ROOT / "tests" / "selfies" / "test_random_indian_man.jpg"


@pytest.mark.skipif(
    not os.getenv("REPLICATE_API_TOKEN"),
    reason="REPLICATE_API_TOKEN not set; skipping live Replicate test",
)
def test_call_kontext_returns_url():
    from backend.kontext_engine import _call_kontext

    url = _call_kontext(
        source_path=SOURCE_MAN,
        prompt=("Change ONLY the hairstyle to: a short textured crop with a "
                "fade. Keep the face exactly identical."),
        seed=42,
    )
    assert isinstance(url, str)
    assert url.startswith(("http://", "https://"))
    assert ".png" in url.lower() or ".jpg" in url.lower() or "replicate" in url


def test_preview_result_dataclass_has_required_fields():
    from backend.kontext_engine import PreviewResult
    r = PreviewResult(
        image_url="/uploads/foo.png", style_id="x", style_name="X",
        prompt="p", seed=42, validator_verdict="skipped",
        retries=0, elapsed_ms=1234,
    )
    d = r.to_dict()
    for k in ("image_url", "style_id", "style_name", "prompt", "seed",
              "validator_verdict", "retries", "elapsed_ms"):
        assert k in d
```

- [ ] **Step 2: Run the tests and verify they fail**

```bash
python -m pytest tests/test_kontext_engine.py -v
```

Expected: both fail with `ModuleNotFoundError: No module named 'backend.kontext_engine'`.

- [ ] **Step 3: Implement `backend/kontext_engine.py` (Replicate call + dataclass only — retry loop in Task 5)**

Create `backend/kontext_engine.py`:

```python
"""Core generation engine: FLUX Kontext Pro via Replicate.

This module is the ONLY place that imports `replicate`.  Public surface is
`generate_preview()` (added in Task 5) and the `PreviewResult` dataclass.
Failures raise `GenerationError`; callers map that to HTTP 502.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KONTEXT_MODEL = "black-forest-labs/flux-kontext-pro"


@dataclass
class PreviewResult:
    image_url: str           # served path /uploads/<file>.png
    style_id: str
    style_name: str
    prompt: str
    seed: int
    validator_verdict: str   # "pass" | "fail" | "uncertain" | "skipped"
    retries: int
    elapsed_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


class GenerationError(RuntimeError):
    """Raised when the Kontext call cannot produce any image at all."""


def _call_kontext(
    source_path: Path,
    prompt: str,
    seed: int,
    safety_tolerance: int = 2,
) -> str:
    """Single Replicate call.  Returns the output URL string.

    Raises GenerationError on any failure (network, API rejection, missing
    URL in the response).
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise GenerationError("REPLICATE_API_TOKEN not set in environment")
    try:
        import replicate
    except ImportError as e:
        raise GenerationError("replicate SDK not installed") from e

    try:
        with Path(source_path).open("rb") as img_f:
            output = replicate.run(
                KONTEXT_MODEL,
                input={
                    "prompt": prompt,
                    "input_image": img_f,
                    "aspect_ratio": "match_input_image",
                    "output_format": "png",
                    "safety_tolerance": safety_tolerance,
                    "prompt_upsampling": False,
                    "seed": seed,
                },
            )
    except Exception as e:
        raise GenerationError(f"Kontext call failed: {e}") from e

    url = _extract_first_url(output)
    if not url:
        raise GenerationError(f"Kontext returned no URL: {output!r}")
    return url


def _extract_first_url(output) -> Optional[str]:
    """Replicate may return a string, list of strings, or an object with .url."""
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
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
python -m pytest tests/test_kontext_engine.py -v
```

Expected: `2 passed` if `REPLICATE_API_TOKEN` is configured (the smoke test costs ~$0.04). If the token is missing, expected: `1 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add backend/kontext_engine.py tests/test_kontext_engine.py
git commit -m "Add kontext_engine skeleton: PreviewResult + _call_kontext

Wires the single Replicate call to black-forest-labs/flux-kontext-pro.
GenerationError replaces InpaintError as the failure type.  Smoke test
hits Replicate live when token is configured (~\$0.04)."
```

---

## Task 5: `kontext_engine.generate_preview` — full pipeline with validator + retry

**Files:**
- Modify: `backend/kontext_engine.py` (add `generate_preview`)
- Modify: `tests/test_kontext_engine.py` (add an end-to-end smoke test)

This task makes the engine production-shaped: load the style, build the prompt, call Kontext, face-composite, validate, retry once on fail.

- [ ] **Step 1: Write the failing end-to-end smoke test**

Append to `tests/test_kontext_engine.py`:

```python
@pytest.mark.skipif(
    not os.getenv("REPLICATE_API_TOKEN"),
    reason="REPLICATE_API_TOKEN not set; skipping live test",
)
def test_generate_preview_end_to_end(tmp_path, monkeypatch):
    """Live end-to-end on the Indian-male source + textured_crop style.
    Cost: ~\$0.04 (Kontext) + ~\$0.006 (validator, if Anthropic configured)."""
    from backend.kontext_engine import generate_preview, PreviewResult
    from backend.customer_analysis import analyze_customer

    # Use a private uploads dir so the test doesn't pollute /uploads/.
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))

    profile = analyze_customer(
        selfie_path=SOURCE_MAN, use_vision_lm=False,
    ).to_dict()

    result = generate_preview(
        source_path=SOURCE_MAN,
        style_id="mens_textured_crop",
        customer_profile=profile,
        seed=42,
        max_retries=1,
    )
    assert isinstance(result, PreviewResult)
    assert result.image_url.startswith(("/uploads/", "http"))
    assert result.style_id == "mens_textured_crop"
    assert result.retries in (0, 1)
    assert result.validator_verdict in ("pass", "fail", "uncertain", "skipped")
    assert result.elapsed_ms > 0
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
python -m pytest tests/test_kontext_engine.py::test_generate_preview_end_to_end -v
```

Expected: `AttributeError: module 'backend.kontext_engine' has no attribute 'generate_preview'`.

- [ ] **Step 3: Implement `generate_preview`**

Append to `backend/kontext_engine.py`:

```python
import json
import time
from typing import Optional

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "catalogue" / "styles.json"
REFERENCES_DIR = Path(__file__).resolve().parent.parent / "catalogue" / "references"


def generate_preview(
    source_path: Path,
    style_id: str,
    customer_profile: dict,
    seed: int = 42,
    max_retries: int = 1,
) -> PreviewResult:
    """Run a full preview: build prompt -> Kontext -> face composite ->
    validate -> retry on fail.

    Raises GenerationError if every attempt fails to produce an image.
    """
    style = _load_style(style_id)
    if style is None:
        raise GenerationError(f"Unknown style_id: {style_id}")
    ref_path = _resolve_reference_path(style)

    from backend.prompt_builder import build_edit_prompt
    from backend.face_composite import paste_source_face

    uploads_dir = Path(
        os.getenv("STYLE_STUDIO_UPLOADS_DIR")
        or (Path(__file__).resolve().parent.parent / "tests" / "uploads")
    )

    started = time.time()
    verdict = "skipped"
    retries = 0
    final_image_url = None
    final_prompt = ""

    for attempt_idx in range(max_retries + 1):
        attempt_seed = seed if attempt_idx == 0 else seed + 1000 + attempt_idx
        final_prompt = build_edit_prompt(
            style=style, customer_profile=customer_profile,
            source_path=source_path, reference_path=ref_path,
        )

        raw_url = _call_kontext(source_path, final_prompt, attempt_seed)
        composited = paste_source_face(
            source_path=source_path,
            kontext_output_url_or_path=raw_url,
            output_dir=uploads_dir,
        )
        final_image_url = f"/uploads/{composited.name}"

        if os.getenv("ANTHROPIC_API_KEY") and ref_path is not None:
            verdict = _validate(source_path, ref_path, composited)
            logger.info("validator attempt %d: %s", attempt_idx + 1, verdict)
            if verdict in ("pass", "uncertain"):
                # 'uncertain' counts as ship - validator parse error shouldn't
                # burn a second Kontext call.
                break
        else:
            verdict = "skipped"
            break
        retries = attempt_idx + 1  # one retry consumed when we loop again

    elapsed_ms = int((time.time() - started) * 1000)
    return PreviewResult(
        image_url=final_image_url,
        style_id=style_id,
        style_name=style.get("name", style_id),
        prompt=final_prompt,
        seed=seed,
        validator_verdict=verdict,
        retries=retries,
        elapsed_ms=elapsed_ms,
    )


def _validate(
    source_path: Path, reference_path: Path, composited_path: Path,
) -> str:
    try:
        from backend.output_validator import validate_generation
        verdict_dict = validate_generation(
            source_path=source_path, reference_path=reference_path,
            generated_url=str(composited_path),
        )
        return verdict_dict.get("verdict", "uncertain")
    except Exception as e:
        logger.warning("validator unavailable: %s", e)
        return "uncertain"


def _load_style(style_id: str) -> Optional[dict]:
    if not CATALOGUE_PATH.exists():
        return None
    with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
        styles = json.load(f)
    for s in styles:
        if s.get("id") == style_id:
            return s
    return None


def _resolve_reference_path(style: dict) -> Optional[Path]:
    ref = style.get("reference_image_path")
    if not ref:
        return None
    p = Path(ref)
    if not p.is_absolute():
        p = REFERENCES_DIR / p
    return p if p.exists() else None
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
python -m pytest tests/test_kontext_engine.py -v
```

Expected: `3 passed` (or `1 passed, 2 skipped` if no Replicate token).

- [ ] **Step 5: Commit**

```bash
git add backend/kontext_engine.py tests/test_kontext_engine.py
git commit -m "kontext_engine: generate_preview with face composite + validator retry

End-to-end pipeline now wired: load catalogue style -> prompt_builder ->
Kontext -> face_composite -> output_validator -> retry once on 'fail' with
seed+1000.  Validator is skipped gracefully when ANTHROPIC_API_KEY is unset
or no reference photo exists (verdict='skipped').  validate() parse error
returns 'uncertain' instead of looping forever."
```

---

## Task 6: `main.py` — drop `mode`, route through `kontext_engine`

**Files:**
- Modify: `backend/main.py`
- Modify: `tests/test_kontext_engine.py` (smoke test the HTTP route)

- [ ] **Step 1: Write the failing HTTP smoke test**

Append to `tests/test_kontext_engine.py`:

```python
@pytest.mark.skipif(
    not os.getenv("REPLICATE_API_TOKEN"),
    reason="REPLICATE_API_TOKEN not set; skipping live test",
)
def test_generate_route_returns_preview(tmp_path, monkeypatch):
    """The /generate route accepts no `mode` parameter and returns a
    PreviewResult dict shape."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))
    from backend.main import app
    client = TestClient(app)

    with SOURCE_MAN.open("rb") as f:
        resp = client.post(
            "/generate",
            files={"image": ("man.jpg", f, "image/jpeg")},
            data={"style_id": "mens_textured_crop", "seed": "42"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for k in ("image_url", "style_id", "validator_verdict", "elapsed_ms"):
        assert k in body, f"missing key {k} in {body!r}"
    assert body["style_id"] == "mens_textured_crop"
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
python -m pytest tests/test_kontext_engine.py::test_generate_route_returns_preview -v
```

Expected: either 422 (current `/generate` still requires a `mode` parameter) or 500 (route still references deleted helpers).

- [ ] **Step 3: Replace the `/generate` and `/generate-batch` routes in `backend/main.py`**

In `backend/main.py`, replace the existing `/generate` and `/generate-batch` handler bodies (the ones that branch on `mode`) with this. Keep `/consult`, `/catalogue`, `/health`, `/analyze` unchanged.

Find and delete the existing `@app.post("/generate")` and `@app.post("/generate-batch")` handlers (the entire `async def generate(...)` and `async def generate_batch(...)` functions).

Replace the `inpaint` import block near the top of the file:

```python
# BEFORE:
from backend.inpaint import (
    generate_preview_inpaint,
    generate_preview_auto,
    generate_preview_expert,
    generate_preview_erase_then_inpaint,
    InpaintError,
)

# AFTER:
from backend.kontext_engine import generate_preview, GenerationError, PreviewResult
```

Then add the new handlers (replacing the old ones):

```python
@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    style_id: str = Form(...),
    seed: Optional[int] = Form(42),
) -> dict:
    """Generate a single hairstyle preview using FLUX Kontext.

    Returns PreviewResult.to_dict() shape (see backend.kontext_engine).
    """
    _validate_image_upload(image)
    saved_path = await _save_upload(image)

    profile = analyze_customer(
        selfie_path=saved_path,
        use_vision_lm=bool(os.getenv("ANTHROPIC_API_KEY")),
    ).to_dict()

    try:
        result = generate_preview(
            source_path=saved_path,
            style_id=style_id,
            customer_profile=profile,
            seed=seed if seed is not None else 42,
        )
    except GenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return result.to_dict()


@app.post("/generate-batch")
async def generate_batch(
    image: UploadFile = File(...),
    style_ids: str = Form(...),     # comma-separated list
    seed: Optional[int] = Form(42),
) -> dict:
    """Generate multiple previews in parallel.  Returns
    {style_id: PreviewResult.to_dict() | {"error": "..."}}.
    """
    import asyncio
    _validate_image_upload(image)
    saved_path = await _save_upload(image)

    profile = analyze_customer(
        selfie_path=saved_path,
        use_vision_lm=bool(os.getenv("ANTHROPIC_API_KEY")),
    ).to_dict()

    ids = [s.strip() for s in style_ids.split(",") if s.strip()]
    loop = asyncio.get_event_loop()

    def _gen_one(sid: str) -> dict:
        try:
            r = generate_preview(
                source_path=saved_path, style_id=sid,
                customer_profile=profile, seed=seed if seed is not None else 42,
            )
            return r.to_dict()
        except Exception as e:
            return {"error": str(e), "style_id": sid}

    results = await asyncio.gather(
        *[loop.run_in_executor(None, _gen_one, sid) for sid in ids],
        return_exceptions=False,
    )
    return {"results": {ids[i]: results[i] for i in range(len(ids))}}
```

Also bump the FastAPI version string near the top of the file:

```python
app = FastAPI(
    title="Style Studio API",
    version="0.3.0",  # was 0.2.0
    description="Indian hairstyle consultation + preview generation for salons.",
)
```

And update the `/health` response to include the engine name (find the existing `HealthResponse` class + `health` function and add the `engine` field):

```python
class HealthResponse(BaseModel):
    status: str
    version: str
    engine: str
    catalogue_styles: int
    replicate_configured: bool
    anthropic_configured: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=app.version,
        engine="flux-kontext-pro",
        catalogue_styles=_load_catalogue_count(),
        replicate_configured=bool(os.getenv("REPLICATE_API_TOKEN")),
        anthropic_configured=bool(os.getenv("ANTHROPIC_API_KEY")),
    )
```

- [ ] **Step 4: Run the test and verify it passes**

```bash
python -m pytest tests/test_kontext_engine.py -v
```

Expected: `4 passed` (or some skipped if tokens are missing). The new HTTP test costs ~$0.04 extra in Replicate.

Also sanity-check the app boots:

```bash
python -c "from backend.main import app; print(app.title, app.version, [r.path for r in app.routes if hasattr(r,'path')])"
```

Expected to print `Style Studio API 0.3.0` and a list including `/generate` and `/generate-batch`.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py tests/test_kontext_engine.py
git commit -m "main.py: drop mode param, route /generate + /generate-batch via kontext_engine

Removes the four-way branching on mode=transform|expert|auto|manual.
Single path: pre-flight -> customer_analysis -> kontext_engine.generate_preview.
Bumps API version to 0.3.0 and adds 'engine' field to /health."
```

---

## Task 7: Delete the legacy code

**Files:**
- Delete: `backend/inpaint.py`
- Delete: `backend/colour_match.py`
- Delete: `backend/enhance.py`
- Delete: `backend/auto_caption.py`
- Delete: `backend/inpaint_with_reference.py`
- Delete: `backend/generate.py`
- Delete: `backend/hair_estimation.py`
- Delete: `backend/hair_mask.py` (logic now lives in `face_composite.py`)
- Delete: `backend/expert_consult.py` ONLY IF no other code path references it (verify in Step 1)

- [ ] **Step 1: Verify the surviving modules don't reference the legacy ones**

Run these greps. Each should return ZERO matches (or only matches inside files we're about to delete):

```bash
python -m pip install --quiet ripgrep 2>/dev/null || true
```

```bash
# After deletion these must NOT match anywhere in backend/ or tests/test_*:
python -c "
import subprocess, pathlib, sys
targets = ['inpaint', 'colour_match', 'enhance', 'auto_caption',
           'inpaint_with_reference', 'hair_estimation', 'hair_mask',
           'InpaintError', 'generate_preview_inpaint',
           'generate_preview_auto', 'generate_preview_expert',
           'generate_preview_erase_then_inpaint', 'build_shared_bald_canvas']
keep = ['backend/main.py', 'backend/kontext_engine.py',
        'backend/face_composite.py', 'backend/prompt_builder.py',
        'backend/customer_analysis.py', 'backend/customer_vision.py',
        'backend/face_analysis.py', 'backend/style_matcher.py',
        'backend/input_pipeline.py', 'backend/output_validator.py',
        'backend/expert_consult.py', 'backend/__init__.py']
bad = []
for f in keep:
    try:
        text = pathlib.Path(f).read_text(encoding='utf-8')
    except FileNotFoundError:
        continue
    for t in targets:
        if t in text:
            bad.append((f, t))
print('OK' if not bad else 'STILL_REFERENCES:')
for b in bad: print(' ', b)
sys.exit(0 if not bad else 1)
"
```

Expected: prints `OK`.  If anything other than `OK` prints, fix the surviving module to not depend on the legacy code, then re-run this check before continuing.

Verify `expert_consult.py` is still referenced (`prompt_builder.py` imports it):

```bash
python -c "from backend import expert_consult; print('expert_consult kept')"
```

Expected: `expert_consult kept`. If this errors, do NOT delete it.

- [ ] **Step 2: Delete the legacy files**

```bash
git rm backend/inpaint.py backend/colour_match.py backend/enhance.py \
       backend/auto_caption.py backend/inpaint_with_reference.py \
       backend/generate.py backend/hair_estimation.py backend/hair_mask.py
```

- [ ] **Step 3: Smoke-test the app still imports cleanly**

```bash
python -c "from backend.main import app; print('app ok'); print('routes:', sorted(r.path for r in app.routes if hasattr(r,'path')))"
```

Expected: prints `app ok` and the route list, with no `ModuleNotFoundError`.

Run the full unit suite:

```bash
python -m pytest tests/test_face_composite.py tests/test_prompt_builder.py -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git commit -m "Delete legacy inpaint code paths (FLUX Fill Pro era)

~1,400 lines removed across inpaint.py, colour_match.py, enhance.py,
auto_caption.py, inpaint_with_reference.py, generate.py,
hair_estimation.py, and hair_mask.py.  All surviving modules import
cleanly; pytest passes."
```

---

## Task 8: Acceptance test runner — 6 outputs on the Indian-face corpus

**Files:**
- Create: `tests/run_acceptance_test.py`
- Create: `tests/acceptance/` (output dir, gitignored — already covered by `tests/uploads/` rule, add explicit rule if needed)

This is the manual acceptance gate from the spec.  The script generates 6 previews (1 Indian male × 3 men's styles + 1 Indian woman × 3 women's styles), composes a 2×4 grid PNG, and writes a JSON summary.  User reviews the grid and decides "ship / no-ship".

- [ ] **Step 1: Add an ignore for `tests/acceptance/`**

Append to `.gitignore`:

```gitignore
tests/acceptance/
```

- [ ] **Step 2: Create the acceptance script**

Create `tests/run_acceptance_test.py`:

```python
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
    ("man",   SELFIES / "test_random_indian_man.jpg",
     "Indian man",
     ["mens_pompadour", "mens_korean_fringe", "mens_textured_crop"]),
    ("woman", SELFIES / "test_indian_woman_b.jpg",
     "Indian woman (grey hair, hard case)",
     ["indian_braid_long", "bridal_juda", "curtain_bangs_medium"]),
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
```

- [ ] **Step 3: Run the acceptance test**

```bash
python tests/run_acceptance_test.py
```

Expected: prints 6 generation log lines, then `grid: ...acceptance/grid.png` and `summary: ...summary.json`.

Cost: ~$0.30 in Replicate + ~$0.04 in Anthropic (if configured).

If `mens_pompadour` (or another style with explicit `prompt_template` from the catalogue) returns a `GenerationError`, the engine is broken — investigate the Replicate error in the traceback before continuing.

- [ ] **Step 4: Manual review**

Open `tests/acceptance/grid.png` in an image viewer. For each of the 6 cells, ask:

1. Is the face clearly the same person as the source?
2. Is the hair visibly different from the source?
3. Does the hair match the style's name and look like a real haircut?
4. Are there obvious artefacts (halos, painted-on edges, double hair)?

Ship criterion: each cell should pass 1, 2, and 3 and fail 4.  Three of six failing is "do not ship; iterate on prompts or guidance".  This is the final acceptance gate.

- [ ] **Step 5: Commit**

```bash
git add tests/run_acceptance_test.py .gitignore
git commit -m "Add acceptance test: 6 outputs on unmodified Indian-face corpus

Runs 1 Indian man x 3 men's styles + 1 Indian woman x 3 women's styles
through the new Kontext engine.  Composes a 2x4 grid PNG + JSON summary.
Cost ~\$0.30 per run.  This is the manual visual gate for sub-project 1
shipping."
```

---

## Self-Review Notes

Verified before saving:

- **Spec coverage**: every spec section (decisions, architecture, components 6.1–6.4, deletions 6.5, error handling, testing, implementation order) is covered by Tasks 1–8.
- **Placeholder scan**: no `TBD` / `TODO` / "implement later"; every code block is complete.
- **Type consistency**: `PreviewResult` fields used in Tasks 5, 6, 8 match the dataclass definition in Task 4.  `generate_preview` signature is identical in Tasks 5, 6, 8.  `paste_source_face` signature is identical in Task 1 and Task 5.

---

## Done When

- All 8 tasks committed.
- `python -m pytest tests/test_face_composite.py tests/test_prompt_builder.py -v` passes locally.
- `python tests/run_acceptance_test.py` produces a `grid.png` that you judge "ship-able" for at least 5 of 6 cells.
