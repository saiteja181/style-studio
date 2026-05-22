# Acceptance Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle three thematic commits that fix the 4 Important code-review issues from sub-project 1, deepen Kontext prompt quality so the three men's styles render as visibly different cuts, and re-run acceptance against a corpus of clean young Indian portraits (the real salon target audience).

**Architecture:** No new modules. Three commits, each touching files already shipped by sub-project 1. Commit 1 fixes bugs in `kontext_engine.py` + `main.py`. Commit 2 deepens the prompt pipeline (`expert_consult.py` system prompt rewrite, `prompt_upsampling=True`, imperative clause in `prompt_builder.py`). Commit 3 sources two new CC-licensed photos, repoints `tests/run_acceptance_test.py`, and re-runs acceptance.

**Tech Stack:** Same as sub-project 1 — Python 3.11, FastAPI, FLUX Kontext Pro via Replicate, MediaPipe FaceMesh, OpenCV, PIL, optional Anthropic (Haiku 4.5 validator + Sonnet 4.6 expert prompt), pytest. Photo sourcing via WebFetch + curl, fallback to user-provided.

---

## Prerequisites

Verify working tree is clean and on `main` with sub-project 1 + spec already committed:

```bash
cd C:/Users/Asus/Desktop/style-studio
git status            # should show clean
git log -1 --oneline  # should show "Spec: sub-project 1.5 acceptance polish" or similar
```

Confirm the source files this plan modifies exist:

```bash
ls backend/kontext_engine.py backend/main.py backend/expert_consult.py backend/prompt_builder.py catalogue/styles.json tests/test_kontext_engine.py tests/test_prompt_builder.py tests/run_acceptance_test.py
```

All eight should print without error.

Confirm sub-project 1's unit suite is green at the starting point:

```bash
python -m pytest tests/test_face_composite.py tests/test_prompt_builder.py tests/test_kontext_engine.py -v --no-header --no-summary -q | tail -5
```

Expected: at least the non-live tests pass. Live tests may skip if `REPLICATE_API_TOKEN` is absent.

---

## Task 1: Commit 1 — code-review hot fixes

**Files:**
- Modify: `backend/kontext_engine.py` — file-URI for validator, retries counter, lazy catalogue cache, `StyleNotFoundError` class
- Modify: `backend/main.py` — `asyncio.get_running_loop()`, route `StyleNotFoundError` to HTTP 404
- Modify: `tests/test_kontext_engine.py` — add two regression tests (404 route, retries counter)

This single commit addresses all four Important issues from the final code review of sub-project 1, plus the lazy-catalogue Minor issue from the same review.

- [ ] **Step 1: Write the failing 404-route regression test**

Append to `tests/test_kontext_engine.py` (at the end of the file):

```python
def test_generate_route_returns_404_for_unknown_style(tmp_path, monkeypatch):
    """Unknown style_id must return 404 (client error), not 502 (server error).
    This is a $0 test - the catalogue lookup fails before any Replicate call."""
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)

    with SOURCE_MAN.open("rb") as f:
        resp = client.post(
            "/generate",
            files={"image": ("man.jpg", f, "image/jpeg")},
            data={"style_id": "does_not_exist_xyz", "seed": "42"},
        )
    assert resp.status_code == 404, (
        f"expected 404 for unknown style, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert "Unknown style" in body.get("detail", ""), body
```

- [ ] **Step 2: Write the failing retries-counter regression test**

Append to `tests/test_kontext_engine.py` (after the test added in Step 1):

```python
def test_retries_counter_capped_at_max_retries(monkeypatch, tmp_path):
    """When the validator says 'fail' on every attempt, the result's retries
    field must equal max_retries (not max_retries + 1).  Regression for the
    off-by-one observed in sub-project 1's final code review."""
    import backend.kontext_engine as ke
    import backend.face_composite as fc

    # Force the Anthropic + reference-path branch so the validator runs.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-for-test")
    monkeypatch.setenv("STYLE_STUDIO_UPLOADS_DIR", str(tmp_path))

    # Stub out the actual Replicate + composite + validator calls.
    monkeypatch.setattr(
        ke, "_call_kontext",
        lambda source_path, prompt, seed: "https://example.test/fake.png",
    )
    fake_png = tmp_path / "fake_output.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header bytes
    monkeypatch.setattr(
        fc, "paste_source_face",
        lambda source_path, kontext_output_url_or_path, output_dir, **kw: fake_png,
    )
    monkeypatch.setattr(
        ke, "_validate",
        lambda source_path, reference_path, composited_path: "fail",
    )

    # Pick a real style id that has a reference photo so the validator branch fires.
    profile = {"hair_color_rgb": (40, 30, 25), "hair_texture": "unknown"}
    result = ke.generate_preview(
        source_path=SOURCE_MAN,
        style_id="mens_pompadour",   # has reference_image_path in catalogue
        customer_profile=profile,
        seed=42,
        max_retries=1,
    )
    assert result.retries == 1, (
        f"expected retries=1 after two failing attempts with max_retries=1, "
        f"got {result.retries}"
    )
    assert result.validator_verdict == "fail"
```

- [ ] **Step 3: Run the two new tests and verify they fail**

```bash
cd C:/Users/Asus/Desktop/style-studio
python -m pytest tests/test_kontext_engine.py::test_generate_route_returns_404_for_unknown_style tests/test_kontext_engine.py::test_retries_counter_capped_at_max_retries -v
```

Expected:
- The 404 test fails: actual response is 502 (because current code maps `GenerationError` to 502 for all causes).
- The retries test fails: `result.retries` is `2` instead of `1`.

- [ ] **Step 4: Apply all six edits across `backend/kontext_engine.py` and `backend/main.py`**

In `backend/kontext_engine.py`:

**Sub-step 4a:** Replace the existing `_load_style` function with a cached version. Find:

```python
def _load_style(style_id: str) -> Optional[dict]:
    if not CATALOGUE_PATH.exists():
        return None
    with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
        styles = json.load(f)
    for s in styles:
        if s.get("id") == style_id:
            return s
    return None
```

Replace with:

```python
_CATALOGUE_CACHE: Optional[list[dict]] = None


def _load_style(style_id: str) -> Optional[dict]:
    """Return the catalogue entry for style_id, or None if not found.
    The full catalogue is parsed once and cached at module scope."""
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is None:
        if not CATALOGUE_PATH.exists():
            return None
        with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
            _CATALOGUE_CACHE = json.load(f)
    for s in _CATALOGUE_CACHE:
        if s.get("id") == style_id:
            return s
    return None
```

**Sub-step 4b:** Add the `StyleNotFoundError` exception class. Find the existing `class GenerationError(RuntimeError):` block and add directly below it:

```python
class StyleNotFoundError(GenerationError):
    """Raised when style_id is not present in the catalogue."""
```

**Sub-step 4c:** Update `generate_preview` to raise `StyleNotFoundError`. Find:

```python
    style = _load_style(style_id)
    if style is None:
        raise GenerationError(f"Unknown style_id: {style_id}")
```

Replace with:

```python
    style = _load_style(style_id)
    if style is None:
        raise StyleNotFoundError(f"Unknown style: {style_id}")
```

**Sub-step 4d:** Fix `_validate` to convert the local path to a `file:///` URI. Find:

```python
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
```

Replace `generated_url=str(composited_path)` with `generated_url=composited_path.as_uri()`:

```python
def _validate(
    source_path: Path, reference_path: Path, composited_path: Path,
) -> str:
    try:
        from backend.output_validator import validate_generation
        verdict_dict = validate_generation(
            source_path=source_path, reference_path=reference_path,
            generated_url=composited_path.as_uri(),
        )
        return verdict_dict.get("verdict", "uncertain")
    except Exception as e:
        logger.warning("validator unavailable: %s", e)
        return "uncertain"
```

**Sub-step 4e:** Fix the `retries` counter in `generate_preview`. Find the retry loop body containing:

```python
        retries = attempt_idx + 1  # one retry consumed when we loop again
```

Delete that line. Then after the `for attempt_idx in range(max_retries + 1):` loop closes, add a single line:

```python
    retries = min(attempt_idx, max_retries)
```

Concretely, the relevant block goes from this (current):

```python
    for attempt_idx in range(max_retries + 1):
        attempt_seed = seed if attempt_idx == 0 else seed + 1000 + attempt_idx
        final_prompt = build_edit_prompt(
            ...
        )
        raw_url = _call_kontext(source_path, final_prompt, attempt_seed)
        composited = paste_source_face(...)
        final_image_url = f"/uploads/{composited.name}"
        if os.getenv("ANTHROPIC_API_KEY") and ref_path is not None:
            verdict = _validate(source_path, ref_path, composited)
            logger.info("validator attempt %d: %s", attempt_idx + 1, verdict)
            if verdict in ("pass", "uncertain"):
                break
        else:
            verdict = "skipped"
            break
        retries = attempt_idx + 1  # one retry consumed when we loop again

    elapsed_ms = int((time.time() - started) * 1000)
```

To this (after edits):

```python
    for attempt_idx in range(max_retries + 1):
        attempt_seed = seed if attempt_idx == 0 else seed + 1000 + attempt_idx
        final_prompt = build_edit_prompt(
            ...
        )
        raw_url = _call_kontext(source_path, final_prompt, attempt_seed)
        composited = paste_source_face(...)
        final_image_url = f"/uploads/{composited.name}"
        if os.getenv("ANTHROPIC_API_KEY") and ref_path is not None:
            verdict = _validate(source_path, ref_path, composited)
            logger.info("validator attempt %d: %s", attempt_idx + 1, verdict)
            if verdict in ("pass", "uncertain"):
                break
        else:
            verdict = "skipped"
            break

    retries = min(attempt_idx, max_retries)
    elapsed_ms = int((time.time() - started) * 1000)
```

(Use the existing `build_edit_prompt(...)` and `paste_source_face(...)` arg lists in the file — keep them unchanged.  Only the two specific lines change.)

In `backend/main.py`:

**Sub-step 4f:** Update the import line. Find:

```python
from backend.kontext_engine import generate_preview, GenerationError
```

Replace with:

```python
from backend.kontext_engine import generate_preview, GenerationError, StyleNotFoundError
```

**Sub-step 4g:** Update the `/generate` handler to return 404 on `StyleNotFoundError`. Find the `try`/`except` block inside the existing `async def generate(...)`:

```python
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
```

Replace with:

```python
    try:
        result = generate_preview(
            source_path=saved_path,
            style_id=style_id,
            customer_profile=profile,
            seed=seed if seed is not None else 42,
        )
    except StyleNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return result.to_dict()
```

**Sub-step 4h:** Replace deprecated `get_event_loop()` in `/generate-batch`. Find:

```python
    loop = asyncio.get_event_loop()
```

Replace with:

```python
    loop = asyncio.get_running_loop()
```

- [ ] **Step 5: Run the two new tests and verify they pass**

```bash
python -m pytest tests/test_kontext_engine.py::test_generate_route_returns_404_for_unknown_style tests/test_kontext_engine.py::test_retries_counter_capped_at_max_retries -v
```

Expected: `2 passed`.

- [ ] **Step 6: Run the full unit suite to confirm no regressions**

```bash
python -m pytest tests/test_face_composite.py tests/test_prompt_builder.py tests/test_kontext_engine.py -v
```

Expected: all non-live tests pass. The live Replicate tests (`test_call_kontext_returns_url`, `test_generate_preview_end_to_end`, `test_generate_route_returns_preview`) may pass (consuming ~$0.12 in Replicate) or be skipped depending on whether `REPLICATE_API_TOKEN` is set. Total: 11 passed, or fewer with some skipped.

- [ ] **Step 7: Commit**

```bash
git add backend/kontext_engine.py backend/main.py tests/test_kontext_engine.py
git commit -m "kontext_engine + main: code-review hot fixes (Important issues)

- _validate: convert local Path to file:/// URI so urlopen accepts it on Windows
- generate_preview: fix off-by-one in retries counter
- _load_style: cache parsed catalogue at module scope (avoid disk read per req)
- StyleNotFoundError raised on unknown style_id; main.py maps it to HTTP 404
- /generate-batch: replace deprecated get_event_loop with get_running_loop
- New tests pin the 404 behaviour and retries counter bound"
```

---

## Task 2: Commit 2 — prompt depth for Kontext

**Files:**
- Modify: `backend/expert_consult.py` — system prompt rewritten for Kontext edits (was Fill Pro inpaint); docstring sweep
- Modify: `backend/kontext_engine.py` — `prompt_upsampling=True` in `_call_kontext` payload
- Modify: `backend/prompt_builder.py` — prepend imperative "complete hairstyle change" clause
- Modify: `tests/test_prompt_builder.py` — add one regression test for the imperative clause
- Audit: `catalogue/styles.json` — verify six target styles have anatomically-specific `prompt_template`; restore if missing

- [ ] **Step 1: Write the failing imperative-clause regression test**

Append to `tests/test_prompt_builder.py`:

```python
def test_build_edit_prompt_includes_imperative_clause():
    """The output of build_edit_prompt must contain the imperative clause that
    pushes Kontext to commit to a real hair change instead of editing
    conservatively.  This was the lever that unblocked men's style
    differentiation in sub-project 1.5."""
    from backend.prompt_builder import build_edit_prompt
    from pathlib import Path

    style = {"name": "Test Cut", "prompt_template": "a short crop"}
    profile = {"hair_color_rgb": (40, 30, 25), "hair_texture": "unknown"}
    out = build_edit_prompt(
        style=style, customer_profile=profile,
        source_path=Path("/tmp/nope.jpg"),  # not opened; expert_consult skipped
        reference_path=None,
    )
    assert "visibly different from the source" in out, (
        f"missing imperative clause; got: {out!r}"
    )
    assert "Keep the face" in out, "identity-preservation clause must remain"
    assert "Change ONLY the hairstyle to:" in out, "Kontext wrapper must remain"
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
cd C:/Users/Asus/Desktop/style-studio
python -m pytest tests/test_prompt_builder.py::test_build_edit_prompt_includes_imperative_clause -v
```

Expected: AssertionError missing imperative clause.

- [ ] **Step 3: Apply the imperative-clause edit to `backend/prompt_builder.py`**

Find the `return` block at the end of `build_edit_prompt`:

```python
    return (
        f"Change ONLY the hairstyle to: {base}.{colour}{texture} "
        "Keep the face, eyes, expression, beard, eyebrows, glasses, "
        "clothing, hands, and background exactly identical to the original "
        "photo - do not change anything below the eyebrows. Photoreal, same "
        "ambient indoor lighting as the source, no studio lighting, no halo."
    )
```

Replace with:

```python
    return (
        f"Change ONLY the hairstyle to: {base}.{colour}{texture} "
        "This is a complete hairstyle change. The new hair must look "
        "visibly different from the source hair in shape, length, or styling "
        "- do not preserve the original silhouette. "
        "Keep the face, eyes, expression, beard, eyebrows, glasses, "
        "clothing, hands, and background exactly identical to the original "
        "photo - do not change anything below the eyebrows. Photoreal, same "
        "ambient indoor lighting as the source, no studio lighting, no halo."
    )
```

- [ ] **Step 4: Run the imperative-clause test and verify it passes**

```bash
python -m pytest tests/test_prompt_builder.py -v
```

Expected: `7 passed` (the 6 existing tests + the new one).

- [ ] **Step 5: Apply the `prompt_upsampling=True` edit to `backend/kontext_engine.py`**

Find inside `_call_kontext`:

```python
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
```

Change `"prompt_upsampling": False,` to `"prompt_upsampling": True,`:

```python
            output = replicate.run(
                KONTEXT_MODEL,
                input={
                    "prompt": prompt,
                    "input_image": img_f,
                    "aspect_ratio": "match_input_image",
                    "output_format": "png",
                    "safety_tolerance": safety_tolerance,
                    "prompt_upsampling": True,
                    "seed": seed,
                },
            )
```

- [ ] **Step 6: Rewrite `backend/expert_consult.py`'s system prompt for Kontext**

First, read the current file to locate the system prompt constant. The variable is typically named `SYSTEM_PROMPT` or similar near the top of the file.

```bash
python -c "import backend.expert_consult as e; print([n for n in dir(e) if 'PROMPT' in n.upper() or 'SYSTEM' in n.upper()])"
```

Find the multi-line string assignment for the system prompt — it currently contains language like "inpainting prompt", "FLUX Fill Pro", "hair region only". Replace the entire system-prompt string with:

```python
SYSTEM_PROMPT = """You are an expert hair stylist writing an EDIT INSTRUCTION for an AI image editor (FLUX Kontext).  The editor will modify the customer's photo to apply the target hairstyle while keeping the rest of the photo intact.

You will see TWO images:
1. SOURCE: the customer's current photo.
2. REFERENCE: the target hairstyle the customer wants.

Write a single-paragraph edit instruction that:
- Describes the target hairstyle in anatomically specific terms: approximate length in cm, hair direction (forward/back/up/parted), fade location and height for cuts that have one, fringe extent for styles with bangs, exposed-temple vs covered-temple decisions.
- Names the styling explicitly (e.g. "pompadour", "korean fringe", "textured crop"), but does not rely on the name alone - describe what it looks like.
- References the customer's face shape, jawline, or hairline visible in SOURCE when it helps adapt the cut (e.g. "the customer has a square jaw; soften the sides slightly").
- Ends with this exact sentence: "Keep face, eyes, expression, beard, eyebrows, glasses, clothing, hands, and background identical to source."

Output ONLY the edit instruction.  No preamble, no markdown, no quotes, no list bullets.  One paragraph, 80-160 words."""
```

If the variable in the file is named something else, rename your replacement to match the existing name. Do not add a new constant — replace the existing one.

Also update the module docstring at the top of the file. Find any sentence mentioning "FLUX Fill Pro", "inpaint", or "inpainting" in the docstring and rewrite to reference "FLUX Kontext" and "edit". Concretely, if the module docstring opens with something like:

```python
"""Expert hair stylist consult: given a source photo and a reference style
photo, produce an inpaint prompt for FLUX Fill Pro that adapts the style
to this specific customer."""
```

Replace with:

```python
"""Expert hair stylist consult: given a source photo and a reference style
photo, produce an EDIT INSTRUCTION for FLUX Kontext that adapts the style
to this specific customer's face shape and hairline."""
```

Leave the rest of the file (function signatures, caching logic, Anthropic SDK calls) unchanged.

- [ ] **Step 7: Audit `catalogue/styles.json` for the six target styles**

Run this audit script:

```bash
python -c "
import json
with open('catalogue/styles.json', 'r', encoding='utf-8') as f:
    styles = json.load(f)
targets = ['mens_pompadour', 'mens_korean_fringe', 'mens_textured_crop',
           'indian_braid_long', 'bridal_juda', 'curtain_bangs_medium']
for sid in targets:
    s = next((s for s in styles if s.get('id') == sid), None)
    if s is None:
        print(f'{sid}: MISSING from catalogue')
        continue
    pt = s.get('prompt_template', '')
    is_anatomic = len(pt) >= 80
    print(f'{sid}: {len(pt)} chars, anatomic={is_anatomic}')
"
```

Expected: each style prints a length >= 80 and `anatomic=True`.

If any of the six prints `anatomic=False` or `MISSING`, restore an anatomic `prompt_template` for that style by editing `catalogue/styles.json` directly. Use these as references (the exact text we used in the earlier round before sub-project 1's cleanup):

- `mens_pompadour`: `"A bold modern pompadour: hair on top stands 6-8 centimetres tall, dramatic vertical volume sweeping up and back from the forehead in a single high wave. Razor-sharp skin fade on the sides and back - hair clipped to near-zero on the temples and around the ears, with a crisp horizontal fade line above the ear. Pronounced undercut. Hairline sharp and defined. Heavy structured shape, polished and styled with product"`
- `mens_korean_fringe`: `"A modern Korean two-block fringe haircut: straight soft fringe falling forward across the forehead and reaching down to just above the eyebrows, completely covering the upper forehead. Medium-length layered hair on top, soft texture, slightly side-swept. Closely tapered sides and back, but NOT shaved - longer than a fade, sitting flat against the head. No volume on top, no pompadour shape, hair lies down naturally. K-pop styling, young modern look"`
- `mens_textured_crop`: `"A textured crop haircut: hair on top cropped to 2-3 centimetres maximum, choppy uneven ends giving a deliberately messy texture, hair pushed slightly forward over the forehead in a short blunt fringe. Tight low skin fade on the sides and back, sharp horizontal fade line, very short above the ears. Modern barbershop cut. Hair is short and structured, NOT raised tall, NOT swept back"`

For the three women's styles, if any `prompt_template` is missing, write one in the same anatomically-specific tone (length, parting, where the silhouette sits, what's exposed vs covered). Do not invent details that contradict the existing style metadata (gender, length, occasion, cultural).

- [ ] **Step 8: Run the full unit suite to confirm no regressions**

```bash
python -m pytest tests/test_face_composite.py tests/test_prompt_builder.py tests/test_kontext_engine.py -v
```

Expected: 12 passed (or fewer with live-test skips if no Replicate token). The new imperative-clause test is the seventh in `test_prompt_builder.py`; everything else carries over from Task 1.

- [ ] **Step 9: Commit**

```bash
git add backend/expert_consult.py backend/kontext_engine.py backend/prompt_builder.py tests/test_prompt_builder.py catalogue/styles.json
git commit -m "prompt_builder + expert_consult + kontext_engine: prompt depth for Kontext

- expert_consult.py: rewrite system prompt for Kontext edit instructions
  (was framed for FLUX Fill Pro inpaint).  Module docstring updated.
- kontext_engine._call_kontext: enable prompt_upsampling=True so Kontext
  expands short prompts internally.
- prompt_builder.build_edit_prompt: prepend a strong imperative clause
  ('this is a complete hairstyle change ... do not preserve the original
  silhouette') ahead of the final identity-preservation sentence.
- catalogue/styles.json: verified six target styles still carry
  anatomically-specific prompt_template fields (restored any that drifted).
- Test pins the new imperative clause."
```

Stage `catalogue/styles.json` only if Step 7 found any styles needing restoration.  If all six audits returned `anatomic=True`, drop `catalogue/styles.json` from the `git add` list.

---

## Task 3: Commit 3 — new test corpus + acceptance re-run

**Files:**
- Create: `tests/selfies/young_indian_man.jpg` — CC-licensed young Indian man portrait, no watermarks
- Create: `tests/selfies/young_indian_woman.jpg` — CC-licensed young Indian woman portrait
- Create: `tests/selfies/CREDITS.md` — attribution log for the two new photos
- Modify: `tests/run_acceptance_test.py` — repoint `CASES` to the new fixtures

- [ ] **Step 1: Source the two new photos**

Sourcing strategy — try in this order. Stop at the first that produces two clean, attribution-clear, frontal Indian portraits without watermarks:

**Strategy A: Pexels CDN direct URLs.** Pexels images are free for any use (their license is permissive; attribution appreciated but not required).  Search for "indian man portrait" and "indian woman portrait" and download two specific image IDs.

Find candidate Pexels image IDs by visiting the search results page in a browser:
- https://www.pexels.com/search/indian%20man%20portrait/
- https://www.pexels.com/search/indian%20woman%20portrait/

For each result that looks like a young (18-35) frontal Indian portrait with clear lighting and visible shoulders, note the image ID (it appears in the URL on the photo's own page, like `https://www.pexels.com/photo/<title>-<id>/`).

Download via direct CDN URL (Pexels CDN serves `https://images.pexels.com/photos/<id>/pexels-photo-<id>.jpeg?cs=srgb&dl=portrait.jpg`):

```bash
# Replace <id> + photographer attributes once chosen
curl -sL -o tests/selfies/young_indian_man.jpg \
    "https://images.pexels.com/photos/<id>/pexels-photo-<id>.jpeg?cs=srgb&dl=man.jpg"
curl -sL -o tests/selfies/young_indian_woman.jpg \
    "https://images.pexels.com/photos/<id>/pexels-photo-<id>.jpeg?cs=srgb&dl=woman.jpg"
```

Verify both files:

```bash
python -c "
from PIL import Image
for p in ('tests/selfies/young_indian_man.jpg', 'tests/selfies/young_indian_woman.jpg'):
    img = Image.open(p)
    print(p, img.size, img.format)
"
```

Each should print dimensions and `JPEG`.

**Strategy B: Wikimedia Commons.** If Pexels search fails or the candidates are watermarked, fall back to Wikimedia Commons (CC-BY-SA or public domain). Search at:
- https://commons.wikimedia.org/wiki/Special:Search?search=indian+man+portrait
- https://commons.wikimedia.org/wiki/Special:Search?search=indian+woman+portrait

Use the file's "Original file" link (right-click → save) to grab the largest version, then resize to ~1024-1536 long edge before saving into `tests/selfies/`. Note the photographer / license / source URL in `CREDITS.md`.

**Strategy C: Manual provision.** If neither Strategy A nor B yields suitable photos in a reasonable time, escalate to the controller:

> "Photo sourcing automation didn't find suitable images.  Please drop two photos into `tests/selfies/young_indian_man.jpg` and `tests/selfies/young_indian_woman.jpg` and confirm.  Constraints: frontal pose, young (18-35), shoulders visible, no watermarks."

Wait for confirmation before proceeding.

- [ ] **Step 2: Verify both photos pass pre-flight on the new pipeline**

Run a quick local check:

```bash
python -c "
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from pathlib import Path
from backend.input_pipeline import prepare_upload, PreflightError

for name in ('young_indian_man', 'young_indian_woman'):
    src = Path('tests/selfies') / f'{name}.jpg'
    try:
        p, report = prepare_upload(
            raw_bytes=src.read_bytes(), target_dir=Path('/tmp'), filename_hint=name,
        )
        print(f'{name}: OK face={report.face_fraction:.2f} blur={report.blur_score:.0f}')
    except PreflightError as e:
        print(f'{name}: BLOCKED code={e.report.code} msg={e.report.message}')
"
```

Expected: both print `OK` with `face` between 0.10 and 0.75 and `blur` >= 60.

If either is BLOCKED, replace the offending photo (back to Step 1) before continuing. Do not proceed with a photo that fails pre-flight — the acceptance test won't be able to use it.

- [ ] **Step 3: Write `tests/selfies/CREDITS.md`**

Create `tests/selfies/CREDITS.md`:

```markdown
# Test selfie credits

The two acceptance fixtures used by `tests/run_acceptance_test.py` were
sourced under permissive licenses.  This file records attribution so the
project complies with each source's terms.

## young_indian_man.jpg
- Source: <Pexels URL or Wikimedia URL>
- Photographer: <name>
- License: <Pexels free-to-use OR CC-BY-SA 4.0 OR public domain>
- Downloaded: 2026-05-23

## young_indian_woman.jpg
- Source: <Pexels URL or Wikimedia URL>
- Photographer: <name>
- License: <Pexels free-to-use OR CC-BY-SA 4.0 OR public domain>
- Downloaded: 2026-05-23

## Legacy fixtures
- `test_random_indian_man.jpg` (watermarked, no longer acceptance gate)
- `test_indian_woman_a.jpg` (tight portrait, blocked by pre-flight)
- `test_indian_woman_b.jpg` (older grey-haired woman, no longer acceptance gate)

These remain on disk for historical reference but are not used by current
acceptance tests.  Delete in a future cleanup if storage matters.
```

Fill in the four `<...>` placeholders with the actual source URL, photographer, and license noted during Step 1.

- [ ] **Step 4: Update `tests/run_acceptance_test.py` to point at the new fixtures**

Open `tests/run_acceptance_test.py`. Find the existing `CASES` list:

```python
CASES = [
    ("man",   SELFIES / "test_random_indian_man.jpg",
     "Indian man",
     ["mens_pompadour", "mens_korean_fringe", "mens_textured_crop"]),
    ("woman", SELFIES / "test_indian_woman_b.jpg",
     "Indian woman (grey hair, hard case)",
     ["indian_braid_long", "bridal_juda", "curtain_bangs_medium"]),
]
```

Replace with:

```python
CASES = [
    ("man",   SELFIES / "young_indian_man.jpg",
     "Young Indian man",
     ["mens_pompadour", "mens_korean_fringe", "mens_textured_crop"]),
    ("woman", SELFIES / "young_indian_woman.jpg",
     "Young Indian woman",
     ["indian_braid_long", "bridal_juda", "curtain_bangs_medium"]),
]
```

- [ ] **Step 5: Run the acceptance test**

```bash
cd C:/Users/Asus/Desktop/style-studio
python tests/run_acceptance_test.py
```

Expected console output: 6 generation log lines (one per cell), then `grid: ...acceptance/grid.png` and `summary: ...summary.json`.

Cost: ~$0.30 Replicate.  If `ANTHROPIC_API_KEY` is set, also ~$0.04 Anthropic (expert_consult fires for each style with a reference photo).

If a cell errors (verdict captured in summary.json as an `error` key), continue — the grid composer skips missing cells.  Most likely failure modes: a transient Replicate rate limit (retry by re-running) or pre-flight rejecting the new photo (back to Step 2).

- [ ] **Step 6: Manual review of `tests/acceptance/grid.png`**

Open `tests/acceptance/grid.png` in an image viewer. For each of the 6 cells, ask:

1. Is the face clearly the same person as the source?
2. Is the hair visibly different from the source?
3. Within each row (man / woman), are the three styles clearly different from EACH OTHER?
4. Are there obvious artefacts (halos, painted-on edges, double hair, flat-colour bands)?

Ship criterion: at least 5 of 6 cells pass criteria 1, 2, and 3, and zero cells fail criterion 4. If the gate fails, do NOT proceed to commit — escalate to the controller with the grid file path and a one-sentence description of what went wrong. The controller decides whether to iterate or accept.

- [ ] **Step 7: Commit (only if Step 6 passed)**

```bash
git add tests/selfies/young_indian_man.jpg tests/selfies/young_indian_woman.jpg tests/selfies/CREDITS.md tests/run_acceptance_test.py
git commit -m "Acceptance corpus: replace fixtures with young Indian man + young woman

Salon target audience is men + young women, not older customers.  Old
sources (test_random_indian_man.jpg watermarked; test_indian_woman_b.jpg
older grey-haired) stay on disk as legacy but are no longer the acceptance
gate.  run_acceptance_test.py CASES now point at the new photos.
CREDITS.md records source + license + photographer for the two new files."
```

`tests/acceptance/*` is gitignored from sub-project 1's Task 8 — do NOT add the new grid/summary files.

---

## Self-Review Notes

Verified before saving:

- **Spec coverage**: every decision in `docs/superpowers/specs/2026-05-23-acceptance-polish-design.md` is implemented by one of the three Tasks. Code-review fixes (§6.1) → Task 1; prompt depth (§6.2) → Task 2; new corpus + acceptance (§6.3) → Task 3.
- **Placeholder scan**: no `TBD` / `TODO`; every code block contains the actual content. The `<...>` placeholders in `CREDITS.md` are user-fillable runtime values, not plan placeholders.
- **Type consistency**: `StyleNotFoundError` defined in Task 1 sub-step 4b is imported in Task 1 sub-step 4f and caught in 4g. `_call_kontext` signature unchanged across tasks. `_validate` signature unchanged across tasks. `build_edit_prompt` signature unchanged across tasks.

---

## Done When

- All three commits land on `main`.
- Unit suite passes: `python -m pytest tests/test_face_composite.py tests/test_prompt_builder.py tests/test_kontext_engine.py -v` shows 12 passed (or fewer with live-test skips).
- `python tests/run_acceptance_test.py` produces a `grid.png` that you judge ship-able for ≥ 5 of 6 cells.
- `tests/selfies/CREDITS.md` records the two new photo sources, photographers, and licenses.
