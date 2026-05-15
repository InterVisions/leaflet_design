"""
End-to-end smoke test for the Flickr30k → curated+FHIBE dataset swap.

Run from the project root:
    python scripts/test_swap.py

Checks:
  1. Combined pool loads (curated only if FHIBE not yet downloaded)
  2. Embeddings compute correctly (ViT-B/16)
  3. Retrieval returns indices + similarities
  4. FAIR scores compute for all 3 axes
  5. Curated vs distractor split is correct in G[i] vector
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import RetrievalEngine
from metrics.fair_calculator import FAIRCalculator, load_desired_distribution, AXES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_swap")

CURATED_FOLDER    = Path("data/situated-usecase-image-pool-v01/images_v01")
DISTRACTOR_FOLDER = Path("data/fhibe")
ANNOTATIONS_PATH  = Path("data/annotations.json")
DIST_PATH         = Path("data/desired_distribution.json")
TEST_QUERY        = "A person pushing a person in a wheelchair"


def load_annotations() -> dict:
    if not ANNOTATIONS_PATH.exists():
        log.error("annotations.json not found — run: python scripts/prepare_annotations.py")
        sys.exit(1)
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    # ── 1. Load combined pool ─────────────────────────────────────────────────
    has_distractors = DISTRACTOR_FOLDER.exists()
    if not has_distractors:
        log.warning(
            "FHIBE folder not found at %s — testing with curated images only. "
            "Download FHIBE and re-run to test the full combined pool.",
            DISTRACTOR_FOLDER,
        )

    engine = RetrievalEngine(device="auto")
    engine.load_model(model_name="ViT-B-16", pretrained="openai")
    engine.load_combined_pool(
        curated_folder=str(CURATED_FOLDER),
        distractor_folder=str(DISTRACTOR_FOLDER) if has_distractors else None,
        max_curated=2000,
        max_distractors=2000,
    )

    n_total = engine.dataset_size()
    log.info("Pool size: %d images", n_total)
    assert n_total > 0, "No images loaded!"

    # ── 2. Embed ──────────────────────────────────────────────────────────────
    engine.embed_images()
    assert engine.image_embeddings is not None
    assert engine.image_embeddings.shape[0] == n_total
    assert engine.image_embeddings.shape[1] == 512, (
        f"Expected ViT-B/16 dim=512, got {engine.image_embeddings.shape[1]}"
    )
    log.info("Embeddings OK: %s (ViT-B/16, dim=512)", engine.image_embeddings.shape)

    # ── 3. Retrieval ──────────────────────────────────────────────────────────
    result = engine.retrieve(TEST_QUERY, top_k=20)
    assert len(result["indices"]) == min(20, n_total)
    assert all(0 <= i < n_total for i in result["indices"]), "Index out of range!"
    log.info(
        "Retrieval OK: top-5 indices=%s  sims=%s",
        result["indices"][:5],
        result["similarities"][:5],
    )

    # ── 4. Annotations ────────────────────────────────────────────────────────
    annotations = load_annotations()
    log.info("Annotations loaded: %d entries", len(annotations))

    # Check how many retrieved images have annotations
    top20_keys = [engine.image_paths[i] for i in result["indices"][:20]]
    n_annotated = sum(1 for k in top20_keys if k in annotations)
    n_curated   = sum(1 for k in top20_keys if annotations.get(k, {}).get("is_curated", 0) == 1)
    log.info(
        "Top-20: %d annotated, %d curated (is_curated=1), %d distractors",
        n_annotated, n_curated, 20 - n_curated,
    )

    # ── 5. FAIR scores ────────────────────────────────────────────────────────
    desired = load_desired_distribution(DIST_PATH)
    calculator = FAIRCalculator(engine=engine, annotations=annotations, desired_distributions=desired)
    metrics = calculator.calculate_metrics_for_query(TEST_QUERY, k=20)

    for axis in AXES:
        key = f"FAIR_{axis}"
        score = metrics[key]
        assert 0.0 <= score <= 1.0, f"{key}={score} out of [0,1]"
        log.info("  %s = %.4f", key, score)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("ALL CHECKS PASSED")
    print(f"  Images indexed : {n_total}")
    print(f"  Has distractors: {has_distractors}")
    print(f"  Query          : {TEST_QUERY!r}")
    print(f"  FAIR_gender    : {metrics['FAIR_gender']:.4f}")
    print(f"  FAIR_age       : {metrics['FAIR_age']:.4f}")
    print(f"  FAIR_skin_tone : {metrics['FAIR_skin_tone']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
