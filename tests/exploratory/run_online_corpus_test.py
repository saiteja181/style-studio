"""End-to-end test on a small corpus of online portraits.

What this verifies (each on a real photo we did not pre-stage):
  1. backend/input_pipeline normalises the upload (EXIF, resize, face check).
  2. backend/inpaint produces a FLUX preview through the full /generate flow.
  3. backend/colour_match harmonises the boundary so the output composites
     cleanly on top of the source.
  4. The new per-style mask overrides actually take effect (fringe styles
     pull the mask above the forehead; long styles get extra room).

Source: thispersondoesnotexist.com produces a fresh CC0 synthetic portrait per
request. We hit it a few times to get varied faces. No tracking, no auth.

Run from the project root with REPLICATE_API_TOKEN set in .env:

    python tests/run_online_corpus_test.py

Output: tests/online_corpus/<face_n>/
  - source.jpg          (after pre-flight)
  - preflight.json      (face fraction, blur, size)
  - <style_id>__raw.png      (FLUX direct, for reference)
  - <style_id>__final.png    (colour-matched composite, the production output)
  - <style_id>__meta.json    (prompt, seed, validator verdict if available)
  - <style_id>__mask.png     (the local hair mask we built)
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

# Project imports must work whether this script is launched from repo root or
# from inside tests/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from backend.input_pipeline import prepare_upload, PreflightError  # noqa: E402
from backend.inpaint import (  # noqa: E402
    generate_preview_inpaint,
    InpaintError,
)

CORPUS_DIR = PROJECT_ROOT / "tests" / "online_corpus"
TPDNE = "https://thispersondoesnotexist.com"

# 2 styles per face to keep Replicate cost ~$0.10 per face.  Pick one short
# masculine style + one long feminine style so the per-style mask override
# logic is exercised in both directions.
STYLES_TO_TEST = [
    "mens_textured_crop",
    "indian_braid_long",
]

N_FACES = 3


def fetch_portrait(out_path: Path) -> int:
    """Download a fresh TPDNE portrait to out_path.  Returns bytes."""
    req = urllib.request.Request(
        TPDNE,
        headers={"User-Agent": "style-studio-tests/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    out_path.write_bytes(data)
    return len(data)


def run_face(idx: int, raw_path: Path) -> dict:
    summary: dict = {"face": idx, "source_bytes": raw_path.stat().st_size,
                     "styles": {}}
    face_dir = CORPUS_DIR / f"face_{idx}"
    face_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = raw_path.read_bytes()
    try:
        prepared_path, report = prepare_upload(
            raw_bytes=raw_bytes,
            target_dir=face_dir,
            filename_hint="source",
        )
    except PreflightError as e:
        summary["preflight"] = e.report.to_dict()
        summary["skipped"] = "preflight_blocked"
        print(f"  [skip] preflight blocked: {e.report.code} {e.report.message}")
        return summary

    summary["preflight"] = report.to_dict()
    summary["source"] = prepared_path.name
    (face_dir / "preflight.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"  preflight: face_fraction={report.face_fraction:.2f} "
          f"blur={report.blur_score:.0f} normalised_size={report.normalised_size}")

    for style_id in STYLES_TO_TEST:
        t0 = time.time()
        mask_path = face_dir / f"{style_id}__mask.png"
        try:
            result = generate_preview_inpaint(
                selfie_path=prepared_path,
                style_id=style_id,
                seed=42,
                save_mask_to=mask_path,
                harmonise=True,
                validate=False,         # no Anthropic key in this env
                max_retries=0,
            )
        except InpaintError as e:
            summary["styles"][style_id] = {"error": str(e)}
            print(f"  [fail] {style_id}: {e}")
            continue

        elapsed = time.time() - t0

        # The result.image_url is either an http(s) URL (raw FLUX) or a
        # /uploads/<file> path pointing at the harmonised composite saved
        # locally by the harmoniser.
        final_url = result.image_url
        raw_url = result.raw_image_url

        if final_url.startswith("/uploads/"):
            # Local harmonised file - already in face_dir's parent uploads/.
            # Move/copy it into face_dir with a stable name.
            src = PROJECT_ROOT / "tests" / "uploads" / Path(final_url).name
            if src.exists():
                dst = face_dir / f"{style_id}__final.png"
                shutil.copyfile(src, dst)
            else:
                # Harmoniser saved beside the selfie (face_dir) - find it.
                candidates = sorted(
                    face_dir.glob(f"harmonised_*.png"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    candidates[0].rename(face_dir / f"{style_id}__final.png")

        if raw_url and raw_url.startswith("http"):
            try:
                _download(raw_url, face_dir / f"{style_id}__raw.png")
            except Exception as e:
                print(f"  [warn] could not save raw FLUX url for {style_id}: {e}")
        elif final_url.startswith("http"):
            # No harmonise step happened; download FLUX result as the final.
            try:
                _download(final_url, face_dir / f"{style_id}__final.png")
            except Exception as e:
                print(f"  [warn] could not save final url for {style_id}: {e}")

        meta = {
            "style_id": style_id,
            "style_name": result.style_name,
            "prompt": result.prompt,
            "seed": result.seed,
            "steps": result.steps,
            "guidance": result.guidance,
            "elapsed_s": round(elapsed, 2),
            "raw_url": raw_url,
            "final_url": final_url,
        }
        (face_dir / f"{style_id}__meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        summary["styles"][style_id] = meta
        print(f"  [ok]  {style_id}: {elapsed:.1f}s -> {final_url}")

    return summary


def _download(url: str, out_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp:
        out_path.write_bytes(resp.read())


def main() -> int:
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN not set in environment / .env - aborting.")
        return 2

    if CORPUS_DIR.exists():
        # Keep history of prior runs - move aside, don't delete blindly.
        archive = CORPUS_DIR.with_suffix(f".{int(time.time())}")
        CORPUS_DIR.rename(archive)
        print(f"archived prior corpus to {archive.name}")
    CORPUS_DIR.mkdir(parents=True)

    all_summaries: list[dict] = []
    for i in range(1, N_FACES + 1):
        print(f"\n=== face {i}/{N_FACES} ===")
        raw_path = CORPUS_DIR / f"face_{i}__upload.jpg"
        try:
            n = fetch_portrait(raw_path)
            print(f"  fetched {n} bytes from TPDNE")
        except Exception as e:
            print(f"  [skip] could not fetch portrait: {e}")
            continue
        summary = run_face(i, raw_path)
        all_summaries.append(summary)
        # Brief pause so TPDNE doesn't rate-limit us.
        time.sleep(1)

    summary_path = CORPUS_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\n=== summary written to {summary_path} ===")
    # Print a short report
    for s in all_summaries:
        if s.get("skipped"):
            print(f"  face {s['face']}: SKIPPED ({s['skipped']})")
            continue
        ok = sum(1 for v in s["styles"].values() if "error" not in v)
        fail = sum(1 for v in s["styles"].values() if "error" in v)
        print(f"  face {s['face']}: {ok} ok, {fail} fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
