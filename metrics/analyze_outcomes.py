"""
Post-workshop correlation analysis: model FAIR vs user-model ranking agreement.

For each query, this script computes mean Spearman ρ across all participant
sessions that used that query, then correlates that per-query mean with the
model's pre-calculated FAIR score.

Research question:
  Do users show higher ranking agreement with CLIP when CLIP's retrieval is
  more demographically fair?

Reads:
  - data/metrics/query_metrics.json  (pre-calculated FAIR over top-20 retrieval)
  - data/rankings.db                  (participant sessions with metrics_json)

Outputs:
  - results/metric_alignment_analysis.csv
  - results/correlation_plots/  (optional — requires matplotlib)

Usage
-----
python metrics/analyze_outcomes.py

python metrics/analyze_outcomes.py \\
    --metrics data/metrics/query_metrics.json \\
    --db      data/rankings.db \\
    --output  results/metric_alignment_analysis.csv \\
    --plots   results/correlation_plots
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("analyze_outcomes")

DB_PATH      = Path(__file__).parent.parent / "data" / "rankings.db"
METRICS_PATH = Path(__file__).parent.parent / "data" / "metrics" / "query_metrics.json"
OUT_CSV      = Path(__file__).parent.parent / "results" / "metric_alignment_analysis.csv"
PLOTS_DIR    = Path(__file__).parent.parent / "results" / "correlation_plots"


# ── Spearman correlation (scipy optional) ─────────────────────────────────────

try:
    from scipy.stats import spearmanr as _scipy_spearman

    def spearman_r(x: list[float], y: list[float]) -> tuple[float, float]:
        result = _scipy_spearman(x, y)
        return float(result.statistic), float(result.pvalue)

except ImportError:
    log.warning("scipy not installed — p-values will be NaN")

    def spearman_r(x: list[float], y: list[float]) -> tuple[float, float]:  # type: ignore[misc]
        ax = np.array(x, dtype=float)
        ay = np.array(y, dtype=float)
        rx = np.argsort(np.argsort(ax)).astype(float)
        ry = np.argsort(np.argsort(ay)).astype(float)
        rx -= rx.mean()
        ry -= ry.mean()
        denom = math.sqrt(float(np.sum(rx ** 2)) * float(np.sum(ry ** 2)))
        r = float(np.dot(rx, ry)) / denom if denom > 0 else 0.0
        return r, float("nan")


# ── Database loading ──────────────────────────────────────────────────────────

def load_session_outcomes(db_path: Path) -> list[dict]:
    """
    Return one row per session with prompt, workshop info, and spearman_r
    extracted from metrics_json. Sessions with no spearman_r are included
    but their spearman_r field is None.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT
                s.id           AS session_id,
                s.prompt,
                s.workshop_id,
                s.metrics_json,
                w.name         AS workshop_name,
                w.community_context
            FROM sessions s
            LEFT JOIN workshops w ON w.id = s.workshop_id
            ORDER BY s.id
        """).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        metrics = json.loads(d.pop("metrics_json") or "{}")
        d["spearman_r"] = metrics.get("spearman_r")   # None if not stored
        result.append(d)
    return result


# ── Aggregation ───────────────────────────────────────────────────────────────

METRIC_COLS  = ["FAIR_gender", "FAIR_age", "FAIR_skin_tone"]
OUTCOME_COLS = ["mean_spearman_r"]


def aggregate_outcomes(
    sessions: list[dict],
    workshop_filter: int | None = None,
) -> dict[str, dict]:
    """
    Average Spearman ρ per query, optionally filtered to one workshop.
    Returns {query: {mean_spearman_r, n_sessions, n_with_spearman}}
    """
    if workshop_filter is not None:
        sessions = [s for s in sessions if s["workshop_id"] == workshop_filter]

    groups: dict[str, list] = defaultdict(list)
    for s in sessions:
        groups[s["prompt"]].append(s)

    result = {}
    for query, grp in groups.items():
        rhos = [s["spearman_r"] for s in grp if s["spearman_r"] is not None]
        result[query] = {
            "mean_spearman_r":   float(np.mean(rhos)) if rhos else float("nan"),
            "n_sessions":        len(grp),
            "n_with_spearman":   len(rhos),
        }
    return result


# ── Correlation analysis ──────────────────────────────────────────────────────

def compute_correlations(
    query_metrics:  dict[str, dict],
    query_outcomes: dict[str, dict],
    label: str,
) -> list[dict]:
    """
    Spearman correlations between each FAIR metric and mean_spearman_r
    for all queries present in both dicts.
    """
    common = sorted(set(query_metrics) & set(query_outcomes))
    n = len(common)
    if n < 3:
        log.warning("[%s] Only %d shared queries — correlations unreliable (need ≥3)", label, n)

    rows = []
    for metric in METRIC_COLS:
        metric_vals = [query_metrics[q].get(metric, float("nan")) for q in common]
        for outcome in OUTCOME_COLS:
            outcome_vals = [query_outcomes[q].get(outcome, float("nan")) for q in common]

            valid = [
                (m, o) for m, o in zip(metric_vals, outcome_vals)
                if not (math.isnan(m) or math.isnan(o))
            ]
            n_valid = len(valid)

            if n_valid < 3:
                r, p = float("nan"), float("nan")
            else:
                xs, ys = zip(*valid)
                r, p = spearman_r(list(xs), list(ys))

            rows.append({
                "group":     label,
                "metric":    metric,
                "outcome":   outcome,
                "r":         round(r, 4) if not math.isnan(r) else "nan",
                "p_value":   round(p, 4) if not math.isnan(p) else "nan",
                "n_queries": n_valid,
            })

    return rows


def analyze_metric_alignment(
    metrics_path: Path,
    db_path:      Path,
    output_csv:   Path,
    plots_dir:    Path | None = None,
) -> list[dict]:
    # ── Load inputs ───────────────────────────────────────────────────────────
    log.info("Loading pre-calculated metrics from %s", metrics_path)
    with open(metrics_path, encoding="utf-8") as f:
        query_metrics: dict[str, dict] = json.load(f)

    log.info("Loading session outcomes from %s", db_path)
    sessions = load_session_outcomes(db_path)
    log.info("%d sessions loaded", len(sessions))

    if not sessions:
        log.error("No session data — run workshops first")
        return []

    n_with_spearman = sum(1 for s in sessions if s["spearman_r"] is not None)
    log.info("%d / %d sessions have Spearman ρ stored", n_with_spearman, len(sessions))

    # ── Warn about query mismatches ───────────────────────────────────────────
    metric_queries  = set(query_metrics)
    outcome_queries = {s["prompt"] for s in sessions}
    for q in sorted(metric_queries - outcome_queries):
        log.warning("No workshop sessions for query: %s", q)
    for q in sorted(outcome_queries - metric_queries):
        log.warning("No pre-calculated metrics for query: %s  (run precalculate_metrics.py)", q)

    # ── Pooled + per-community correlations ───────────────────────────────────
    all_results: list[dict] = []

    pooled_outcomes = aggregate_outcomes(sessions)
    all_results.extend(compute_correlations(query_metrics, pooled_outcomes, label="POOLED"))

    workshops = {
        (s["workshop_id"], s["workshop_name"], s["community_context"])
        for s in sessions
        if s["workshop_id"] is not None
    }
    for wid, wname, _ in sorted(workshops):
        label = wname or f"workshop_{wid}"
        all_results.extend(
            compute_correlations(query_metrics, aggregate_outcomes(sessions, workshop_filter=wid), label=label)
        )

    # ── Write CSV ─────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "metric", "outcome", "r", "p_value", "n_queries"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Correlations written to %s  (%d rows)", output_csv, len(all_results))

    # ── Print summary ─────────────────────────────────────────────────────────
    pooled = [r for r in all_results if r["group"] == "POOLED"]
    print()
    print("Pooled Spearman correlations (model FAIR vs mean user-model agreement)")
    print(f"{'Metric':<22} {'r':>7} {'p':>8} {'n':>4}")
    print("─" * 46)
    for row in pooled:
        print(f"{row['metric']:<22} {str(row['r']):>7} {str(row['p_value']):>8} {row['n_queries']:>4}")

    if plots_dir is not None:
        _try_plot(query_metrics, pooled_outcomes, plots_dir)

    return all_results


def _try_plot(
    query_metrics:   dict[str, dict],
    pooled_outcomes: dict[str, dict],
    plots_dir:       Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed — skipping plots")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    common = sorted(set(query_metrics) & set(pooled_outcomes))

    for metric in METRIC_COLS:
        xs = [query_metrics[q].get(metric, float("nan")) for q in common]
        ys = [pooled_outcomes[q].get("mean_spearman_r", float("nan")) for q in common]
        labels = [q[:30] for q in common]

        valid = [(x, y, l) for x, y, l in zip(xs, ys, labels)
                 if not (math.isnan(x) or math.isnan(y))]
        if len(valid) < 2:
            continue

        xv, yv, lv = zip(*valid)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(xv, yv, s=60, zorder=3)
        for x, y, label in zip(xv, yv, lv):
            ax.annotate(label, (x, y), fontsize=7, textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel(metric)
        ax.set_ylabel("mean Spearman ρ (user-model agreement)")
        ax.set_title(f"{metric} vs user-model agreement  (pooled, n={len(valid)})")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric}_vs_spearman.png", dpi=150)
        plt.close(fig)

    log.info("Plots saved to %s", plots_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correlate model FAIR with user-model ranking agreement (Spearman ρ)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metrics", default=str(METRICS_PATH))
    p.add_argument("--db",      default=str(DB_PATH))
    p.add_argument("--output",  default=str(OUT_CSV))
    p.add_argument("--plots",   default=None,
                   help="Directory for scatter plots (requires matplotlib)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_metric_alignment(
        metrics_path = Path(args.metrics),
        db_path      = Path(args.db),
        output_csv   = Path(args.output),
        plots_dir    = Path(args.plots) if args.plots else None,
    )
