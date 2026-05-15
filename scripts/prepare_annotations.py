"""
Convert intervisions_annotations_v01.csv → data/annotations.json

The output JSON is keyed by the exact path string that RetrievalEngine.load_combined_pool()
stores in engine.image_paths (i.e. str(Path(curated_folder) / filename)).

Run from the project root:
    python scripts/prepare_annotations.py

Output: data/annotations.json
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_annotations")

# ── Paths (relative to project root) ─────────────────────────────────────────
CSV_PATH      = Path("data/situated-usecase-image-pool-v01/intervisions_annotations_v01.csv")
IMAGES_FOLDER = Path("data/situated-usecase-image-pool-v01/images_v01")
OUT_PATH      = Path("data/annotations.json")

# ── Gender mapping (CSV numeric → FAIR calculator string) ────────────────────
# 0 = M.Male  |  1 = NB  |  2 = M.Female
GENDER_MAP = {"0": "M.Male", "1": "NB", "2": "M.Female"}


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    if not IMAGES_FOLDER.exists():
        raise SystemExit(f"Images folder not found: {IMAGES_FOLDER}")

    annotations: dict[str, dict] = {}
    skipped = 0

    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["image_path"].strip()
            img_path = IMAGES_FOLDER / filename

            if not img_path.exists():
                log.warning("Image file not found, skipping: %s", img_path)
                skipped += 1
                continue

            # Gender: map numeric string → label; leave None if unrecognised
            gender_raw = row.get("perceived_gender", "").strip()
            gender = GENDER_MAP.get(gender_raw)

            # Age: keep the full string as-is (FAIR calculator maps it internally)
            age_raw = row.get("perceived_age", "").strip()
            age = age_raw if age_raw else None

            # Skin tone: integer 1-6; None if blank
            st_raw = row.get("perceived_skin_tone", "").strip()
            try:
                skin_tone = int(st_raw) if st_raw else None
            except ValueError:
                skin_tone = None

            # Use the same path string the engine will produce
            key = str(img_path)

            entry: dict = {"is_curated": 1}
            if gender is not None:
                entry["gender"] = gender
            if age is not None:
                entry["age"] = age
            if skin_tone is not None:
                entry["skin_tone"] = skin_tone

            annotations[key] = entry

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    log.info("Wrote %d annotations → %s  (%d skipped)", len(annotations), OUT_PATH, skipped)

    # ── Quick validation ──────────────────────────────────────────────────────
    no_gender    = sum(1 for v in annotations.values() if "gender"    not in v)
    no_age       = sum(1 for v in annotations.values() if "age"       not in v)
    no_skin_tone = sum(1 for v in annotations.values() if "skin_tone" not in v)
    log.info(
        "Missing annotations: gender=%d  age=%d  skin_tone=%d",
        no_gender, no_age, no_skin_tone,
    )


if __name__ == "__main__":
    main()
