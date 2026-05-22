# Design — Sub-project 8: Core quality polish

- **Date**: 2026-05-23
- **Status**: Approved verbally, no separate user review gate
- **Goal**: Take output quality from current 7.5-8/10 to 9+/10 sellable.

## 1. Context

After 7 prior sub-projects (1, 1.5, 1.6, 2, 3, 4, 5, 1.7, 6-data), the
pipeline ships 49 passing unit tests and a working acceptance grid on two
Indian sources.  Honest review found three quality issues:

- **Forelock-strand artifact** — recurring asymmetric strand crossing the
  face in all three male hair outputs (pompadour / korean fringe / textured
  crop).  Reads as "AI generated" rather than "my haircut."
- **Beard pipeline has no retry loop** — single Kontext flop ships.
- **Acceptance corpus too narrow** — only 2 source faces.  No data on round
  face / curly hair / dark skin / older male.  Quality could collapse on
  any of those demographics.

The user has explicitly said deployment is downstream; this sub-project
focuses purely on the core AI output quality being sellable.

## 2. Goals

- Remove the forelock-strand artifact from male short cuts.
- Beard pipeline retries on validator fail when configured.
- Wider acceptance corpus (4 sources × 5 styles = 20 outputs) surfaces
  quality outliers across more demographics.
- Final visual review of the grid + per-image scores by the user.

## 3. Non-goals

- Kontext Max A/B comparison (deferred per user choice).
- Frontend changes (deferred — explicitly downstream).
- Catalogue reference photos (out of scope, separate sourcing work).
- CORS / signed URLs / booking (deferred).

## 4. Decisions (locked from brainstorming)

| Q | Choice | Why |
|---|---|---|
| Wider corpus scope | B: 4 sources × 5 styles = 20 outputs (~$0.80) | Catches most quality issues, affordable to re-run after fixes. |
| Forelock fix lever | C: both negative-prompt clause + `prompt_upsampling=False` for short male cuts | Tiny edits, belt-and-braces. |
| Kontext Max comparison | A: defer | Premature optimisation before seeing post-fix Pro outputs. |

## 5. Architecture

No new modules.  Three engineering fixes + one corpus extension + one
acceptance run + visual review.

### 5.1 Forelock-strand fix

`backend/prompt_builder.py`: append to the return string of
`build_edit_prompt`, before the existing identity-preservation sentence:

> ` Avoid asymmetric forelock locks, do not draw a single dramatic strand falling across the face, no stylegan2 watermark, no extra hair locks beyond what the style describes. `

`backend/kontext_engine.py:_call_kontext`: read `prompt_upsampling` from the
style metadata if provided, otherwise default `True`:

```python
upsampling = style.get("upsampling", True) if style is not None else True
```

The function currently doesn't receive the style.  Change the signature to
accept an optional `style` parameter; pass it from `generate_preview`.

`catalogue/styles.json`: add `"upsampling": false` to these styles:
`mens_pompadour`, `mens_korean_fringe`, `mens_textured_crop`,
`mens_buzz_cut`, `mens_classic_side_part`.  Leave other styles at the
default `True` (the upsampler helps for longer / more elaborate cuts).

### 5.2 Beard validator-retry loop

`backend/beard_engine.py:generate_beard_preview`: add `max_retries=1`
parameter and a retry loop mirroring `kontext_engine.generate_preview`:

```python
for attempt_idx in range(max_retries + 1):
    seed_attempt = seed if attempt_idx == 0 else seed + 1000 + attempt_idx
    raw_url = _call_kontext(source_path, prompt, seed_attempt)
    composited = paste_source_face(... mode="beard" ...)
    if not os.getenv("ANTHROPIC_API_KEY"):
        verdict = "skipped_no_anthropic_key"
        break
    # Beard catalogue has no reference photos today, so validator skips.
    # Infrastructure is in place for when references are added.
    verdict = "skipped_no_reference"
    break
```

Today this is dormant infrastructure (beard catalogue has no reference
photos).  Wired so that when references are added in a future sub-project,
retry just works.

### 5.3 New acceptance corpus

Source 3 new CC-licensed Indian portraits from Pexels:

- **Round face**: a young Indian man with a visibly round face shape.
- **Curly hair**: a young Indian woman with curly / coiled hair.
- **Dark skin or older male**: dusky/dark complexion source for skin-tone
  diversity; if dark-skin sourcing fails, substitute an older male
  (40-55) to add age diversity instead.

Save as:

- `tests/selfies/round_face_indian_man.jpg`
- `tests/selfies/curly_hair_indian_woman.jpg`
- `tests/selfies/dark_skin_indian_man.jpg` (or `older_indian_man.jpg`)

Update `tests/selfies/CREDITS.md` with attribution.

Update `tests/run_acceptance_test.py` `CASES`:

```python
CASES = [
    ("man",   SELFIES / "young_indian_man.jpg",      "Young Indian man",
     ["mens_pompadour", "mens_korean_fringe", "mens_textured_crop",
      "mens_classic_side_part", "mens_buzz_cut"]),
    ("woman", SELFIES / "young_indian_woman.jpg",   "Young Indian woman",
     ["indian_braid_long", "bridal_juda", "curtain_bangs_medium",
      "modern_chin_bob", "side_swept_layers"]),
    ("round", SELFIES / "round_face_indian_man.jpg", "Round-face Indian man",
     ["mens_pompadour", "mens_korean_fringe", "mens_textured_crop",
      "mens_classic_side_part", "mens_buzz_cut"]),
    ("curly", SELFIES / "curly_hair_indian_woman.jpg",
     "Curly-hair Indian woman",
     ["indian_braid_long", "bridal_juda", "curtain_bangs_medium",
      "modern_chin_bob", "side_swept_layers"]),
]
```

(Drop one of these case-rows if the dark-skin / older male source replaces
one of the existing ones; final corpus is 4 sources × 5 styles = 20 cells.)

### 5.4 Run + visual review

After all three engineering commits land:

1. Run `python tests/run_acceptance_test.py`.
2. Controller (me) reads the grid + each of the 20 individual outputs.
3. Controller displays the grid + 6-10 individual outputs in chat with
   per-cell scores against the four ship criteria from sub-project 1.5:
   identity preserved, hair visibly different, three styles per source
   clearly different, no obvious artefacts.

## 6. Error handling

| Failure | Behaviour |
|---|---|
| Pexels sourcing fails | Fall back to user manually providing photos.  Same pattern as SP 1.5 Task 3. |
| New photo fails pre-flight | Replace it; do not commit a fixture that pre-flight rejects. |
| Acceptance run hits Replicate rate limit | Wait 60s, retry the run from where it failed (the `summary.json` shows which cells succeeded). |
| Acceptance run produces >3 failed cells | Don't ship.  Iterate on the prompts / catalogue overrides; re-run. |
| Forelock fix doesn't help | Surface in the visual review; user decides whether to escalate (Kontext Max A/B, deeper prompt-engineering, etc.). |

## 7. Testing / acceptance

- All existing 49 unit tests must continue to pass.
- 1 new unit test for the negative-prompt clause in `prompt_builder`.
- 1 new unit test for the `prompt_upsampling=False` override in
  `kontext_engine`.
- 1 new unit test for the beard retry counter (mirror of the kontext
  retries test).
- Acceptance gate: visual review of the 20-output grid.  Ship criterion
  unchanged from SP 1.5: 5 of 6 cells per source pass the four criteria,
  no cells fail criterion 4 (artefacts).

## 8. Cost

- Replicate: ~$0.80 for the 20-output corpus run + ~$0.05 for unit-test
  live calls.
- Anthropic: ~$0.10 if key is on (validator + expert_consult + head-cover
  detection on the corpus).
- **Total: ~$1.00 for the full sub-project.**

## 9. Implementation order

Three thematic commits, then the acceptance run:

1. **Commit 1**: forelock fix (negative prompt + per-style upsampling
   override) + beard retry loop + the three new unit tests.
2. **Commit 2**: 3 new acceptance sources + CREDITS.md + `CASES` update.
3. **Acceptance run + visual review** (not a commit).

## 10. Out of scope (subsequent sub-projects)

- Kontext Max A/B comparison (sub-project 8.5 if post-fix Pro is still
  inadequate).
- Reference photos for the 20 new SP 3 catalogue entries (sourcing work).
- Frontend live camera (sub-project 7 — deferred until core is sellable).
- CORS / signed URLs / booking / multi-language UI / per-style LoRA.
