"""
Pre-workshop FAIR / NDKL metric calculation.

Run once before each workshop to calculate CLIP retrieval fairness metrics
for every query participants will use.  Results are saved to a JSON file
that analyze_outcomes.py reads after the workshop.

Usage examples
--------------
# Image folder dataset
python scripts/precalculate_metrics.py \\
    --queries     data/queries.json \\
    --annotations data/annotations.json \\
    --folder      data/image_pool \\
    --output      data/metrics/query_metrics.json

# HuggingFace dataset
python scripts/precalculate_metrics.py \\
    --queries     data/queries.json \\
    --annotations data/annotations.json \\
    --hf-repo     nlphuji/flickr30k \\
    --output      data/metrics/query_metrics.json

# Override model / evaluation depth
python scripts/precalculate_metrics.py \\
    --queries     data/queries.json \\
    --annotations data/annotations.json \\
    --folder      data/image_pool \\
    --output      data/metrics/query_metrics.json \\
    --model       ViT-B-16 \\
    --pretrained  openai \\
    --k           20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow importing from the project root regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import RetrievalEngine
from metrics.fair_calculator import FAIRCalculator, load_desired_distribution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("precalculate_metrics")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-calculate FAIR/NDKL metrics for a list of queries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input data
    p.add_argument(
        "--queries", required=True,
        help="Path to queries JSON file (list of strings, or dict with a 'queries' key)",
    )
    p.add_argument(
        "--annotations", required=True,
        help="Path to annotations JSON file ({image_key: {gender, age, skin_tone}})",
    )

    # Image dataset — pick one mode
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--curated-folder", help="Curated images folder (is_curated=1); combine with --distractor-folder for FHIBE")
    src.add_argument("--folder",         help="Local folder of images (legacy single-pool mode)")
    src.add_argument("--hf-repo",        help="HuggingFace dataset repo id")

    # Combined-pool options (used with --curated-folder)
    p.add_argument("--distractor-folder", default=None,  help="FHIBE distractor folder (is_curated=0)")
    p.add_argument("--max-curated",       type=int, default=2000)
    p.add_argument("--max-distractors",   type=int, default=40)

    # HuggingFace options
    p.add_argument("--hf-split",     default="train",  help="Dataset split")
    p.add_argument("--hf-config",    default=None,     help="Dataset config name")
    p.add_argument("--image-column", default="image",  help="Image column name")

    # CLIP model
    p.add_argument("--model",      default="ViT-B-16", help="CLIP model architecture")
    p.add_argument("--pretrained", default="openai",   help="CLIP pretrained weights")
    p.add_argument("--device",     default="auto",     help="'auto', 'cpu', or 'cuda'")

    # Dataset size
    p.add_argument("--max-images", type=int, default=2000, help="Image cap")

    # Metric parameters
    p.add_argument(
        "--k", type=int, default=20,
        help="Ranking depth at which to evaluate FAIR/NDKL",
    )
    p.add_argument(
        "--desired-distribution",
        default="data/desired_distribution.json",
        help="Path to desired_distribution.json (nulls fall back to uniform)",
    )

    # Output
    p.add_argument(
        "--output", default="data/metrics/query_metrics.json",
        help="Path to write metric results JSON",
    )

    return p.parse_args()


def load_queries(path: str) -> list[str]:
    """Load queries from a JSON file.

    Accepts either a bare list of strings or a dict with a 'queries' key.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        queries = data
    elif isinstance(data, dict) and "queries" in data:
        queries = data["queries"]
    else:
        raise ValueError(
            f"queries file must be a JSON list or a dict with a 'queries' key — got {type(data)}"
        )

    if not queries:
        raise ValueError("queries list is empty")
    if not all(isinstance(q, str) for q in queries):
        raise ValueError("all queries must be strings")

    return queries


def load_annotations(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("annotations file must be a JSON object mapping image keys to attribute dicts")
    return data


def main() -> None:
    args = parse_args()

    # ── Load queries ──────────────────────────────────────────────────────────
    log.info("Loading queries from %s", args.queries)
    queries = load_queries(args.queries)
    log.info("%d queries loaded", len(queries))
    for i, q in enumerate(queries, 1):
        log.info("  %d. %s", i, q)

    # ── Load annotations ──────────────────────────────────────────────────────
    log.info("Loading annotations from %s", args.annotations)
    annotations = load_annotations(args.annotations)
    log.info("%d images annotated", len(annotations))

    # ── Build retrieval engine ────────────────────────────────────────────────
    engine = RetrievalEngine(device=args.device)
    engine.load_model(model_name=args.model, pretrained=args.pretrained)

    if args.curated_folder:
        engine.load_combined_pool(
            curated_folder=args.curated_folder,
            distractor_folder=args.distractor_folder,
            max_curated=args.max_curated,
            max_distractors=args.max_distractors,
        )
    elif args.folder:
        engine.load_dataset_from_folder(args.folder, max_images=args.max_images)
    else:
        engine.load_dataset_from_huggingface(
            repo=args.hf_repo,
            split=args.hf_split,
            image_column=args.image_column,
            max_images=args.max_images,
            hf_config=args.hf_config,
        )

    engine.embed_images()
    log.info("Engine ready — %d images indexed", engine.dataset_size())

    # Warn if annotation coverage is low
    n_ann = len(annotations)
    n_img = engine.dataset_size()
    coverage = n_ann / n_img if n_img else 0
    if coverage < 0.5:
        log.warning(
            "Only %d/%d (%.0f%%) images have annotations — "
            "FAIR/NDKL scores will be based on this subset",
            n_ann, n_img, 100 * coverage,
        )

    # ── Load desired distribution ─────────────────────────────────────────────
    log.info("Loading desired distribution from %s", args.desired_distribution)
    desired_dist = load_desired_distribution(args.desired_distribution)

    # ── Calculate metrics ─────────────────────────────────────────────────────
    calculator = FAIRCalculator(
        engine=engine,
        annotations=annotations,
        desired_distributions=desired_dist,
    )

    output_path = Path(args.output)
    results = calculator.calculate_metrics_batch(
        queries=queries,
        k=args.k,
        output_path=output_path,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print(f"{'Query':<45} {'FAIR_g':>7} {'FAIR_a':>7} {'FAIR_st':>7}")
    print("-" * 68)
    for query, m in results.items():
        label = (query[:42] + "…") if len(query) > 43 else query
        print(
            f"{label:<45} "
            f"{m['FAIR_gender']:>7.4f} {m['FAIR_age']:>7.4f} {m['FAIR_skin_tone']:>7.4f}"
        )
    print()
    log.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
