# Design — Sub-project 1.5: acceptance polish

- **Sub-project**: 1.5 of the broader Indian-salon product plan (follow-up to
  sub-project 1 "core engine swap to FLUX Kontext")
- **Date**: 2026-05-23
- **Status**: Design approved, awaiting written-spec review
- **Author**: brainstormed with the user

## 1. Context

Sub-project 1 shipped the FLUX Kontext engine, deleted ~2,289 lines of legacy
inpaint code, and produced an acceptance test grid on two Indian sources.
Two real issues surfaced when the user reviewed the grid:

- **Older woman's face drifted.** The face composite preserved face shape
  byte-perfect, but the polygon excludes the upper cheek + forehead, so
  Kontext's regenerated cheek shape produced visible identity drift on a
  high-detail older face (wrinkles, bindi, distinct smile).
- **Men's three styles looked near-identical.** Pompadour, korean fringe,
  textured crop all rendered as "slightly raised brushed-back medium hair"
  on the same male source. Real product use needs each style to read as a
  distinct cut.

A separate code review surfaced four Important bugs in the shipped sub-project 1
code that should be fixed before turning the Anthropic key on in production:

- `kontext_engine._validate` passes a Windows file path to `urllib.urlopen`,
  which raises `URLError: unknown url type: c`.  The validator is silently a
  no-op on Windows when `ANTHROPIC_API_KEY` is set.
- `retries` counter in `generate_preview` is off-by-one in the double-fail
  case.
- `main.py` uses deprecated `asyncio.get_event_loop()` (Python 3.12 will
  raise).
- Unknown `style_id` returns HTTP 502 (should be 404).

Plus one stale-prompt issue in `backend/expert_consult.py`: its system prompt
still instructs Claude Sonnet to write "an inpaint prompt for FLUX Fill Pro",
which is the wrong framing for FLUX Kontext.

The salon target audience is **men + young women**.  The current acceptance
corpus (older grey-haired woman + heavily-watermarked man) doesn't represent
that demographic, so the acceptance gate isn't actually measuring what
matters.

This sub-project bundles those quality issues + the code review fixes into
three thematic commits, then re-runs acceptance against a corpus that
represents the real target users.

## 2. Goals

- Three men's styles (`mens_pompadour`, `mens_korean_fringe`,
  `mens_textured_crop`) render as **visibly different** cuts on a clean
  young Indian male source.
- Three women's styles (`indian_braid_long`, `bridal_juda`,
  `curtain_bangs_medium`) render as **visibly different** looks on a young
  Indian female source.
- Face identity is **preserved** on both new sources (verified by manual
  review of the new acceptance grid).
- All four code-review Important issues are fixed.
- `expert_consult.py`'s system prompt is rewritten for Kontext.
- Acceptance gate passes for 5 of 6 cells minimum.

## 3. Non-goals

- **Face composite changes** are explicitly deferred.  We test on the young-
  face corpus first.  If identity still drifts on the new sources, that
  becomes a separate iteration.
- **Switching to FLUX Kontext Max** is parked.  Higher cost (~$0.08 vs
  ~$0.04 per preview) without proof we need it.
- **Catalogue expansion** (bridal sub-catalogue, regional styles, beard
  transforms, 4-D recommendation) — all remain sub-projects 2-6.
- **Frontend changes** are out of scope.
- **Adding a face-similarity check** on top of the composite is out of
  scope.

## 4. Decisions (locked during brainstorming)

| # | Question | Choice |
|---|---|---|
| 1 | Priority order for the three observed issues | Lead with men's differentiation; identity preservation is secondary; test corpus is foundational and addressed below. |
| 2 | Men's differentiation lever | Prompt tightening (free) **+** fix `expert_consult.py` for Kontext (requires Anthropic on, ~$0.018/preview).  Skip FLUX Kontext Max for now. |
| 3 | Test corpus shape | Source **new** young Indian man + young Indian woman photos.  Replace both current acceptance fixtures. |
| 4 | Identity composite changes | **Defer.**  Verify on the new young-face corpus first; widen polygon only if drift persists. |
| 5 | Implementation cadence | **Three thematic commits**: code-review hot fixes; prompt depth for Kontext; new test corpus + acceptance re-run. |

## 5. Architecture (unchanged)

No new modules.  Same pipeline as sub-project 1:

```
source → prompt_builder.build_edit_prompt
       → kontext_engine.generate_preview
           → _call_kontext (Replicate / Kontext Pro)
           → face_composite.paste_source_face
           → output_validator.validate_generation (optional)
       → PreviewResult
```

Changes touch existing files only.

## 6. Three commits

### 6.1 Commit 1 — code-review hot fixes

Five surgical edits.  No behaviour change visible to a normal user except
the 502 → 404 routing for unknown `style_id`.

**`backend/kontext_engine.py`:**

1. `_validate(...)`: replace
   ```python
   generated_url=str(composited_path)
   ```
   with
   ```python
   generated_url=composited_path.as_uri()
   ```
   so `urllib.request.urlopen` accepts a Windows path as a `file:///` URI.

2. `generate_preview(...)` retry loop: the current `retries = attempt_idx + 1`
   inside the loop body produces `retries=2` after a double-fail with
   `max_retries=1`.  Replace with a single line **after** the loop:
   ```python
   retries = min(attempt_idx, max_retries)
   ```
   where `attempt_idx` is the loop variable.  Result: `retries ∈ {0, 1}` for
   `max_retries=1`.

3. `_load_style(...)`: cache the parsed catalogue at module scope:
   ```python
   _CATALOGUE_CACHE: Optional[list[dict]] = None

   def _load_style(style_id: str) -> Optional[dict]:
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
   Saves one disk read + JSON parse per request.

4. New exception class:
   ```python
   class StyleNotFoundError(GenerationError):
       """Raised when style_id is not in the catalogue."""
   ```
   `generate_preview` raises this (instead of plain `GenerationError`) when
   `_load_style` returns `None`.

**`backend/main.py`:**

5. `/generate-batch`: replace
   ```python
   loop = asyncio.get_event_loop()
   ```
   with
   ```python
   loop = asyncio.get_running_loop()
   ```

6. `/generate` handler: catch `StyleNotFoundError` before the generic
   `GenerationError` handler and map it to:
   ```python
   raise HTTPException(status_code=404, detail=f"Unknown style: {style_id}")
   ```

**Tests:**

7. Add one pytest case to `tests/test_kontext_engine.py`:
   ```python
   def test_generate_route_returns_404_for_unknown_style(tmp_path, monkeypatch):
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
       assert resp.status_code == 404
       assert "Unknown style" in resp.json()["detail"]
   ```
   Cost: $0 (no Replicate call — `_load_style` returns None before that).

**Commit message:**

```
kontext_engine + main: code-review hot fixes (Important issues)

- _validate: convert local Path to file:/// URI so urlopen accepts it on Windows
- generate_preview: fix off-by-one in retries counter
- _load_style: cache parsed catalogue at module scope (avoid disk read per req)
- StyleNotFoundError raised on unknown style_id; main.py maps it to HTTP 404
- /generate-batch: replace deprecated get_event_loop with get_running_loop
- New test pins the 404 behaviour
```

### 6.2 Commit 2 — prompt depth for Kontext

**`backend/expert_consult.py` — rewrite the system prompt.**

Current system prompt (paraphrased): "You are an expert hair stylist.
Looking at the customer's photo and the reference style photo, write a
short inpainting prompt for FLUX Fill Pro that will inpaint the hair region
only..."

New system prompt:

```
You are an expert hair stylist writing an EDIT INSTRUCTION for an AI image
editor (FLUX Kontext).  The editor will modify the customer's photo to apply
the target hairstyle while keeping the rest of the photo intact.

You will see TWO images:
1. SOURCE: the customer's current photo.
2. REFERENCE: the target hairstyle the customer wants.

Write a single-paragraph edit instruction that:
- Describes the target hairstyle in anatomically specific terms:
  approximate length in cm, hair direction (forward/back/up/parted),
  fade location and height for cuts that have one, fringe extent for
  styles with bangs, exposed-temple vs covered-temple decisions.
- Names the styling explicitly (e.g. "pompadour", "korean fringe",
  "textured crop"), but does not rely on the name alone - describe what
  it looks like.
- References the customer's face shape, jawline, or hairline visible in
  SOURCE when it helps adapt the cut (e.g. "the customer has a square
  jaw; soften the sides slightly").
- Ends with this exact sentence: "Keep face, eyes, expression, beard,
  eyebrows, glasses, clothing, hands, and background identical to source."

Output ONLY the edit instruction.  No preamble, no markdown, no quotes,
no list bullets.  One paragraph, 80-160 words.
```

The user-message content stays the same (two images + a short "Adapt this
reference style to this customer" prompt).

Also update the module docstring at the top of `expert_consult.py` to
reflect Kontext, not Fill Pro.

**`backend/kontext_engine.py` `_call_kontext` payload:**

Change one parameter:
```python
"prompt_upsampling": False,   # before
"prompt_upsampling": True,    # after
```
Lets Kontext expand short prompts internally.  Improves style fidelity
when the catalogue `prompt_template` is concise.

**`backend/prompt_builder.py` `build_edit_prompt`:**

Append one new clause to the return string, right before the final
boilerplate sentence:

```
This is a complete hairstyle change.  The new hair must look visibly
different from the source hair in shape, length, or styling - do not
preserve the original silhouette.
```

So the full prompt becomes:
```
Change ONLY the hairstyle to: {base}.{colour}{texture} This is a complete
hairstyle change.  The new hair must look visibly different from the source
hair in shape, length, or styling - do not preserve the original silhouette.
Keep the face, eyes, expression, beard, eyebrows, glasses, clothing, hands,
and background exactly identical to the original photo - do not change
anything below the eyebrows. Photoreal, same ambient indoor lighting as
the source, no studio lighting, no halo.
```

**`catalogue/styles.json` audit (no edits unless drift found):**

Verify these six styles still have anatomically-specific `prompt_template`
fields (added earlier in the session, before sub-project 1's Task 7
cleanup):

- `mens_pompadour`
- `mens_korean_fringe`
- `mens_textured_crop`
- `indian_braid_long`
- `bridal_juda`
- `curtain_bangs_medium`

If any of the six is missing `prompt_template`, restore an anatomic
description.  Reference: the language we used in earlier rounds
("hair raised 6-8 cm tall in a pompadour with a razor-sharp skin fade
on the sides...").

**Tests:**

Add one pytest case to `tests/test_prompt_builder.py`:
```python
def test_build_edit_prompt_includes_imperative_clause():
    from backend.prompt_builder import build_edit_prompt
    style = {"name": "Test Cut", "prompt_template": "a short crop"}
    out = build_edit_prompt(style, {"hair_color_rgb": (40, 30, 25)},
                            source_path=None, reference_path=None)
    assert "visibly different from the source" in out
    assert "Keep the face" in out
```

**Commit message:**

```
prompt_builder + expert_consult + kontext_engine: prompt depth for Kontext

- expert_consult.py: rewrite system prompt for Kontext edit instructions
  (was framed for FLUX Fill Pro inpaint).  Module docstring updated.
- kontext_engine._call_kontext: enable prompt_upsampling=True so Kontext
  expands short prompts internally.
- prompt_builder.build_edit_prompt: prepend a strong imperative clause
  ('this is a complete hairstyle change ... do not preserve the original
  silhouette') ahead of the final identity-preservation sentence.
- catalogue/styles.json: verified six target styles still carry
  anatomically-specific prompt_template fields.
- Test pins the new imperative clause.
```

### 6.3 Commit 3 — new test corpus + acceptance re-run

**Source new photos.**

Two CC-licensed Indian portraits, no watermarks, frontal pose, shoulders
visible:

- `tests/selfies/young_indian_man.jpg` — young Indian man (18-35),
  short to medium hair, dark hair, clear lighting.
- `tests/selfies/young_indian_woman.jpg` — young Indian woman (18-35),
  any hair length, clear lighting.

Sourcing strategy:
1. Try Unsplash / Pexels / Wikimedia Commons direct CDN URLs via
   `WebFetch` + `curl`.
2. If automated sourcing fails (URLs require API keys, copyright
   ambiguous, etc.), prompt the user to drop two files into
   `tests/selfies/` manually.

License requirement: CC0, CC-BY (with attribution), or equivalent
public-domain.  Attribution notes go in a new comment header in the
filenames or a `tests/selfies/CREDITS.md`.

**`tests/run_acceptance_test.py` changes:**

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

Old `test_random_indian_man.jpg` and `test_indian_woman_b.jpg` are not
deleted — they stay in `tests/selfies/` as legacy fixtures, but the
acceptance gate no longer uses them.

**Run the acceptance:**

```bash
cd C:/Users/Asus/Desktop/style-studio
python tests/run_acceptance_test.py
```

Expected cost: ~$0.30 Replicate + ~$0.04 Anthropic if the key is set
(expert_consult fires for each style with a reference photo).

**Manual review of `tests/acceptance/grid.png`:**

Ship criterion for each of 6 cells:

1. Face clearly the same person as source?
2. Hair visibly different from source?
3. Three styles per source clearly different from each other?
4. No obvious artefacts (halos, painted-on edges, double hair)?

Ship if 5 of 6 cells pass criteria 1-3 and zero cells fail criterion 4.

**Commit message:**

```
Acceptance corpus: replace fixtures with young Indian man + young woman

Salon target audience is men + young women, not older customers.  Old
sources (test_random_indian_man.jpg watermarked; test_indian_woman_b.jpg
older grey-haired) stay on disk as legacy but are no longer the acceptance
gate.  run_acceptance_test.py CASES now point at the new photos.
```

## 7. Error handling — changes only

| Failure | Before sub-project 1.5 | After sub-project 1.5 |
|---|---|---|
| Unknown `style_id` | HTTP 502 with `Unknown style_id` message | HTTP 404 with `Unknown style: {id}` |
| Validator file path URL | Silently `verdict="uncertain"` on Windows | Validator actually runs (Path.as_uri()) |
| Double-fail retries counter | `retries` may be 2 (off-by-one) | `retries ∈ {0, max_retries}` |
| `asyncio.get_event_loop()` | DeprecationWarning on 3.11, RuntimeError on 3.12 | `asyncio.get_running_loop()` works on both |

All other error handling is unchanged from sub-project 1.

## 8. Testing

- **`tests/test_face_composite.py`** — unchanged.  Still 2 tests passing.
- **`tests/test_prompt_builder.py`** — add 1 test (imperative clause).
  Total 7 tests.
- **`tests/test_kontext_engine.py`** — add 1 test (404 route).  Total 5
  tests including the 3 live Replicate-call tests.
- **`tests/run_acceptance_test.py`** — re-pointed at new fixtures.  Ship
  gate is manual visual review of the resulting grid.

Per-commit verification:

- After Commit 1: `python -m pytest tests/test_face_composite.py
  tests/test_prompt_builder.py tests/test_kontext_engine.py -v` shows 10
  passed (or 8 + 2 skipped if no token).
- After Commit 2: same as Commit 1, +1 test for imperative clause = 11
  passed.
- After Commit 3: acceptance grid produced and manually reviewed.

## 9. Cost

Per real customer preview (3 styles), after sub-project 1.5 ships:

| Mode | Per-style cost | 3-style session | Notes |
|---|---|---|---|
| Anthropic key OFF | $0.04 Kontext | $0.12 (~₹10) | expert_consult skipped, smart-default prompts |
| Anthropic key ON | $0.04 Kontext + $0.018 Sonnet + $0.006 Haiku | $0.192 (~₹16) | expert_consult fires per style with a reference photo |
| Plus ~10% retry budget | as above + $0.04 | + $0.012 avg | only when validator says fail |

Acceptance test cost per run: ~$0.30 + Anthropic if configured.  Within
existing budget — no change versus sub-project 1's quoted ₹17/customer.

## 10. Out of scope (subsequent iterations)

- **FLUX Kontext Max swap** — parked.  Would jump cost from ~$0.04 to
  ~$0.08 per preview.  Try only if A+B+C from this sub-project fail to
  produce buyable differentiation.
- **Face polygon widening + feather tightening** — deferred.  Only if the
  new young-face corpus still shows identity drift.
- **Face-similarity check** as a third gate after the validator — out of
  scope.
- **4-D recommendation matching, Indian catalogue depth, beard transforms,
  skin-tone palette, booking** — all separate sub-projects per the
  original plan.

## 11. Open questions / risks

- **Photo sourcing automation may fail.**  Pexels and Unsplash CDN URLs
  may require API keys or HTTP referer headers.  Mitigation: fall back to
  the user dropping files manually.
- **Even with `prompt_upsampling=True` and the expert_consult rewrite, men's
  differentiation may still be weak.**  If acceptance still shows three
  similar-looking men's outputs on the new clean source, the Kontext Pro
  model itself is the ceiling — and the right next step is the parked
  Kontext Max experiment (cost decision required from user).
- **Older-style code that referenced the legacy `mode` parameter** may
  exist in `frontend/index.html`.  Out of scope here, but worth a one-line
  grep before declaring the sub-project done.
