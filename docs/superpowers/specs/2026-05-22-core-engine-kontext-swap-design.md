# Design — Core engine swap to FLUX Kontext + identity composite

- **Sub-project**: 1 of 6 in the broader Indian-salon product plan
- **Date**: 2026-05-22
- **Status**: Design approved, awaiting written-spec review
- **Author**: brainstormed with the user (style-studio owner)

## 1. Context

style-studio is a FastAPI service that previews hairstyle changes on a
customer photo for Indian salons. The current core uses FLUX Fill Pro
(`black-forest-labs/flux-fill-pro`) for inpainting on a hair-region mask.

Diagnostic work on a young Indian male source verified three problems with
this engine:

- The U-band geometric mask covered only above-the-head, missing the temples
  and sides. The actual hair region was never paintable.
- After fixing the mask to be hair-aware (`selfie_segmentation - face + headroom`),
  the bald-canvas pass still failed to render bald scalp. FLUX Fill Pro
  preserves source context too aggressively even at `guidance=80` and a
  strong bald prompt.
- Direct inpaint (skipping bald) at guidance 35/55/80 produced near-identical
  conservative output. The bottleneck is the model, not the mask, prompt, or
  guidance.

A one-off test against `black-forest-labs/flux-kontext-pro` (FLUX Kontext) on
the same source produced three visibly different haircuts (pompadour, korean
fringe, textured crop) while preserving face identity. Kontext is
natural-language image editing rather than masked inpainting; it has no
context-preservation bias to fight.

This design captures the decision to make Kontext the core engine and the
shape of the resulting pipeline.

## 2. Goals

- Replace FLUX Fill Pro with FLUX Kontext Pro as the only generation engine.
- Preserve the customer's **face** byte-for-byte vs the source.
- Drop the `mode` API parameter and delete the legacy code paths it served.
- Keep total per-preview cost within today's budget (≈ ₹17/customer for 3
  styles).
- Acceptance: 6 outputs on the unmodified Indian-face corpus (1 male × 3
  men's styles + 1 female × 3 women's styles) pass manual visual review.

## 3. Non-goals

- Background pixels are **not** preserved byte-for-byte. Kontext regenerates
  the whole image; that is accepted.
- Beard transforms, length classifier, skin-tone-conditioned colour palette,
  4-dimension recommendation engine, and Indian-catalogue depth are
  **separate sub-projects** and out of scope here.
- A/B testing infrastructure, CI gates, or quantified acceptance metrics are
  not required for first ship.

## 4. Decisions (locked during brainstorming)

| # | Question | Choice | Rationale |
|---|---|---|---|
| 1 | Identity strictness | **Face byte-perfect via composite** | Customers scrutinise their own face. Composite costs ~50 ms local, no API. |
| 2 | Background strictness | **Accept Kontext-regenerated background** | Simplest. Salon use cares about the haircut, not the wall. |
| 3 | Quality control | **Claude vision validator + auto-retry on fail** | Existing `output_validator.py`. Catches obvious failures. Assumes Anthropic key configured. |
| 4 | API surface | **Drop `mode` param, delete legacy code** | One mental model. Old code lives in git history if ever needed. |
| 5 | Acceptance criterion | **Manual visual review of 6 fixed outputs** | First ship; user's eye is the right judge. Quantified gates are a later upgrade. |

## 5. Architecture

Single linear pipeline. One module owns the Replicate call, one owns identity
preservation, one owns prompt assembly; the rest is shared with the existing
system.

```
HTTP /generate
       │
       ▼
input_pipeline.prepare_upload          ← existing
       │ (normalised source path)
       ▼
customer_analysis.analyze_customer     ← existing
       │ (CustomerProfile)
       ▼
prompt_builder.build_edit_prompt       ← NEW (extracted)
       │ (final Kontext string)
       ▼
kontext_engine.generate_preview        ← NEW (the only place that imports replicate)
       │ ↘
       │   _call_kontext(source, prompt, seed)
       │ ↙
       │ (raw Kontext output URL)
       ▼
face_composite.paste_source_face       ← NEW
       │ (final URL served from /uploads)
       ▼
output_validator.validate_generation   ← existing, Anthropic
       │ (verdict: pass / fail / uncertain / skipped)
       │
       │ on fail + retries_left: bump seed, loop to _call_kontext
       ▼
PreviewResult { image_url, validator_verdict, prompt, seed, retries, elapsed_ms }
```

### Module boundaries

- `backend/kontext_engine.py` — Replicate is imported nowhere else.
- `backend/face_composite.py` — MediaPipe FaceMesh + OpenCV; no network.
- `backend/prompt_builder.py` — Pure-Python string assembly; optional
  Anthropic call via existing `expert_consult` when key is configured.
- `backend/main.py` — HTTP routing only; delegates to `kontext_engine`.

## 6. Component design

### 6.1 `backend/kontext_engine.py`

Public surface:

```python
@dataclass
class PreviewResult:
    image_url: str           # served path /uploads/<file>.png
    style_id: str
    style_name: str
    prompt: str              # final string sent to Kontext
    seed: int
    validator_verdict: str   # "pass" | "fail" | "uncertain" | "skipped"
    retries: int             # 0 or 1
    elapsed_ms: int


class GenerationError(RuntimeError):
    """Raised when the Kontext call cannot produce any image at all."""


def generate_preview(
    source_path: Path,
    style_id: str,
    customer_profile: dict,
    seed: int = 42,
    max_retries: int = 1,
) -> PreviewResult: ...
```

Internal flow:

```
style = catalogue.load(style_id)
ref_path = catalogue.resolve_reference(style)  # may be None
attempt = 0
for seed_attempt in (seed, seed + 1000):
    prompt = prompt_builder.build_edit_prompt(style, customer_profile,
                                              source_path, ref_path)
    raw_url = _call_kontext(source_path, prompt, seed_attempt)
    composited = face_composite.paste_source_face(
        source_path, raw_url, output_dir=UPLOADS_DIR,
    )
    if anthropic_configured() and ref_path is not None:
        verdict = output_validator.validate_generation(
            source_path, ref_path, composited_url,
        )["verdict"]
    else:
        verdict = "skipped"
    if verdict in ("pass", "skipped"):
        break
    attempt += 1
    if attempt > max_retries:
        break
return PreviewResult(...)
```

Kontext call (single function `_call_kontext`):

```python
output = replicate.run(
    "black-forest-labs/flux-kontext-pro",
    input={
        "prompt": prompt,
        "input_image": open(source_path, "rb"),
        "aspect_ratio": "match_input_image",
        "output_format": "png",
        "safety_tolerance": 2,
        "prompt_upsampling": False,
        "seed": seed,
    },
)
return _extract_first_url(output)
```

### 6.2 `backend/face_composite.py`

One public function, no external APIs.

```python
def paste_source_face(
    source_path: Path,
    kontext_output_url_or_path,
    output_dir: Path,
    feather_px: int = 12,
) -> Path:
    """
    Composite source face polygon onto a Kontext-generated image.

    1. Open source and Kontext output. Resize Kontext to source dims if needed.
    2. Run MediaPipe FaceMesh on source -> 478 landmarks.
    3. Build a face polygon from landmarks covering eyes + nose + mouth +
       cheeks + jaw. The polygon explicitly EXCLUDES the hairline, forehead
       above eyebrows, and ears so the new hair can still meet the face.
    4. Feather the polygon edge with feather_px-pixel Gaussian blur to soften
       the seam between source face and new hair.
    5. alpha = polygon_mask / 255
       out = kontext * (1 - alpha) + source * alpha
    6. Save as PNG in output_dir with a unique name; return the Path.
    """
```

Failure mode: if MediaPipe finds no face in the source (shouldn't happen
post-preflight, but defence in depth), return the raw Kontext output path
unchanged + log a warning.

### 6.3 `backend/prompt_builder.py`

Single public function. Mostly relocation of logic currently scattered in
`inpaint.py` (`_build_default_prompt_from_style`, the colour-hex clause, the
texture-contrast clause).

```python
def build_edit_prompt(
    style: dict,
    customer_profile: dict,
    source_path: Path,
    reference_path: Optional[Path],
) -> str:
    # 1. Base description
    base = style.get("prompt_template")
    if not base:
        base = _default_from_style(style)
    if (os.getenv("ANTHROPIC_API_KEY") and reference_path
            and reference_path.exists()):
        try:
            base = expert_consult.consult_for_style(
                source_image_path=source_path,
                reference_image_path=reference_path,
            )
        except expert_consult.ConsultError:
            pass  # fall back to base

    # 2. Customer colour hex anchor
    rgb = customer_profile.get("hair_color_rgb")
    colour_clause = ""
    if rgb and len(rgb) == 3:
        hex_code = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in rgb])
        colour_clause = (
            f" Keep the hair colour the customer's natural shade ({hex_code}); "
            f"no bleach, no highlights, no colour drift."
        )

    # 3. Texture-contrast clause (when source ≠ target texture)
    texture_clause = _texture_contrast_clause(style, customer_profile)

    # 4. Kontext wrapper - explicit "change ONLY hair" framing
    return (
        f"Change ONLY the hairstyle to: {base}.{colour_clause}{texture_clause} "
        f"Keep the face, eyes, expression, beard, eyebrows, glasses, "
        f"clothing, hands, and background exactly identical to the original "
        f"photo. Photoreal, same ambient indoor lighting as the source, "
        f"no studio lighting, no halo."
    )
```

### 6.4 `backend/main.py` changes

- `/generate` loses the `mode` parameter entirely. Signature becomes
  `image: UploadFile, style_id: str, seed: int = 42`. Body delegates to
  `kontext_engine.generate_preview()` and returns its `PreviewResult.to_dict()`.
- `/generate-batch` fires N parallel `generate_preview` calls (no shared erase
  optimisation any more — Kontext does everything in one call).
- `/consult` and `/catalogue` unchanged.
- `/health` reports `engine: "flux-kontext-pro"` and `version: "0.3.0"`.

### 6.5 Files deleted in one commit

- `backend/inpaint.py` (the entire FLUX Fill Pro path, mask-builder helpers)
- `backend/colour_match.py` (no longer needed — face composite replaces it)
- `backend/enhance.py` (clarity-upscaler was an inpaint-era post-step)
- `backend/auto_caption.py` (Florence-2 auto-prompt path; obsoleted)
- `backend/inpaint_with_reference.py` (legacy variant)
- `backend/generate.py` (legacy variant)
- `backend/hair_estimation.py` (legacy variant)
- `backend/hair_mask.py` is kept; mask logic stays here in case `face_composite`
  reuses parts. (If unused after refactor, deleted in the same commit.)

Total: ~1,400 lines removed, ~350 lines added (net ~ -1,050 lines).

## 7. Error handling

| Failure | Behaviour |
|---|---|
| Pre-flight rejects upload | 422 with actionable `code` + `message` (existing behaviour, unchanged). |
| Kontext API error / timeout | `GenerationError` → `/generate` returns 502 with the underlying reason. UI shows "Retry". |
| Output URL missing in Kontext response | `GenerationError` (treated same as API error). |
| Face composite finds no face in source | Log warning; ship the raw Kontext URL as-is. Should be impossible after pre-flight. |
| Validator unreachable / non-JSON / timeout | Log; ship the current composite with `validator_verdict="uncertain"`. No retry on validator errors (they're not Kontext quality issues). |
| Validator says `fail` and retries remain | Loop with `seed + 1000`. After max_retries, ship the last attempt with `validator_verdict="fail"`. Frontend can show an amber pill. |
| Style id not in catalogue | 404 with `Unknown style: {id}`. |

## 8. Testing / acceptance

Acceptance script: `tests/run_acceptance_test.py`.

- Pulls both unmodified sources from `tests/selfies/`:
  - `test_random_indian_man.jpg` × 3 men's styles: `mens_pompadour`,
    `mens_korean_fringe`, `mens_textured_crop`
  - `test_indian_woman_b.jpg` × 3 women's styles: `indian_braid_long`,
    `bridal_juda`, `curtain_bangs_medium`
- For each (source, style) pair: run `kontext_engine.generate_preview`,
  save the result + `validator_verdict` per cell.
- Compose `tests/acceptance/grid.png` — 2 rows × 4 cols (source + 3 styles).
- Write `tests/acceptance/summary.json` with per-cell verdict, elapsed time,
  retries used.

User reviews the grid manually. Ship criterion: "Would I be willing to show
this to a paying customer?" for each cell.

Cost: 6 Kontext calls + ~6 validator calls + ~10% retries ≈ **$0.30** per
acceptance run.

## 9. Cost model

Per single style preview (post-shipping):

| Step | Provider | Cost | Notes |
|---|---|---|---|
| Pre-flight, customer analysis, prompt assembly, face composite | local | $0 | |
| Expert prompt rewrite | Anthropic (Sonnet) | ~$0.018 | Only when key configured + reference exists; cached per (source, ref). |
| Kontext call | Replicate | ~$0.04 | |
| Validator | Anthropic (Haiku) | ~$0.006 | Only when key configured; cached per (source, ref, gen-url). |
| Retry (10% of calls) | Replicate + Anthropic | ~$0.046 | One full repeat. |
| **Blended per preview** | | **~$0.07** | with Anthropic on |
| **Blended per preview** | | **~$0.044** | Anthropic off (validator skipped) |

For one customer session × 3 styles: ~$0.21 (≈ ₹18) with Anthropic, ~$0.13
(≈ ₹11) without. Acceptable.

## 10. Implementation order (for the writing-plans skill)

1. **`backend/face_composite.py`** + a minimal smoke test that pastes a
   source face onto an arbitrary "stand-in" image and confirms the central
   face region is pixel-identical to source.
2. **`backend/prompt_builder.py`** by extracting/relocating from
   `inpaint.py`. Unit-test prompts for a male and a female style.
3. **`backend/kontext_engine.py`**. Smoke test against Replicate with the
   man source + `mens_pompadour`. Verify a `PreviewResult` is returned.
4. **`backend/main.py` route update** — drop `mode`, route through
   `kontext_engine`. Smoke test `/generate` end-to-end.
5. **Delete obsolete files in one commit** — `inpaint.py`, `colour_match.py`,
   `enhance.py`, `auto_caption.py`, `inpaint_with_reference.py`, `generate.py`,
   `hair_estimation.py`, and `hair_mask.py` if unused.
6. **`tests/run_acceptance_test.py`** — runs both Indian sources × 3 styles,
   composes `acceptance/grid.png`. User reviews.
7. Commit + ship.

## 11. Out of scope (subsequent sub-projects)

These are explicitly **not** in this spec and will get their own
spec → plan → implementation cycles:

- 4-dimension recommendation engine (face shape + jawline + hairline + skin
  tone weighting in `style_matcher.py`).
- Indian catalogue depth (30+ bridal, regional traditional, festive styles).
- Skin-tone-conditioned colour palette suggestion.
- Beard / facial-hair preview.
- Service-card → booking integration.

## 12. Open questions / risks

- **Kontext output dimensions** vs source dimensions: Kontext returns at the
  input image's aspect ratio when `aspect_ratio="match_input_image"`. The
  face composite assumes matching dimensions; if Kontext returns at a
  different resolution we resize. Verified during smoke test.
- **Kontext rate limits / availability**: Kontext Pro is in active Replicate
  rotation; no fallback model is provided in this spec (decision #4 was to
  delete legacy). If Kontext has an outage post-ship, the workaround is a
  follow-up hotfix that reintroduces FLUX Fill Pro behind an emergency flag.
  Out of scope for this design.
- **Reference photo availability for expert prompt rewrite**: not every
  catalogue style has a reference image. Those styles fall back to the
  catalogue `prompt_template` (or the smart-default). The validator is also
  skipped for these styles (validator needs a reference to compare). This is
  a known gap; expanding reference photos is part of sub-project #3.
