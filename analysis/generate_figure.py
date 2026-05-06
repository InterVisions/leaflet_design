"""
Generate a heatmap of mean Kendall's tau by prompt × workshop.

Reads: analysis/tau_by_prompt_and_workshop.csv
       (run compute_tau_by_workshop.py first)

Output: analysis/tau_heatmap.png  (also shown interactively)

Usage:
    python analysis/generate_figure.py [--no-show]

Requires: numpy, matplotlib  (pip install --force-reinstall matplotlib)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:  pip install --force-reinstall matplotlib numpy")
    sys.exit(1)

CSV_PATH = Path(__file__).parent / "tau_by_prompt_and_workshop.csv"
OUT_PATH = Path(__file__).parent / "tau_heatmap.png"


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"CSV not found: {path}")
        print("Run:  python analysis/compute_tau_by_workshop.py")
        sys.exit(1)
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["mean_tau"]   = float(row["mean_tau"])
            row["n_sessions"] = int(row["n_sessions"])
            rows.append(row)
    return rows


def build_matrix(rows: list[dict]):
    prompts   = sorted({r["prompt"] for r in rows})
    workshops = sorted(
        {(r["workshop_id"], r["workshop_name"], r["community_context"]) for r in rows},
        key=lambda t: t[0] or ""
    )

    n_p = len(prompts)
    n_w = len(workshops)
    matrix   = np.full((n_w, n_p), np.nan)
    n_matrix = np.zeros((n_w, n_p), dtype=int)

    for row in rows:
        p_i = prompts.index(row["prompt"])
        w_i = next(i for i, (wid, wn, _) in enumerate(workshops)
                   if str(wid) == str(row["workshop_id"]))
        matrix[w_i, p_i]   = row["mean_tau"]
        n_matrix[w_i, p_i] = row["n_sessions"]

    return matrix, n_matrix, prompts, workshops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-show", action="store_true", help="Save only, don't open window")
    args = p.parse_args()

    rows = load_csv(CSV_PATH)
    matrix, n_matrix, prompts, workshops = build_matrix(rows)

    n_p = len(prompts)
    n_w = len(workshops)

    fig_w = max(9, n_p * 1.8)
    fig_h = max(4, n_w * 1.1 + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1f2937")

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Mean Kendall's τ", color="#d1d5db", fontsize=10)
    cb.ax.tick_params(colors="#9ca3af")
    cb.ax.yaxis.label.set_color("#d1d5db")

    # Cell annotations
    for i in range(n_w):
        for j in range(n_p):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            n      = n_matrix[i, j]
            bright = abs(v) > 0.6
            ax.text(j, i, f"{v:+.2f}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if bright else "#111")
            ax.text(j, i + 0.32, f"n={n}",
                    ha="center", va="center", fontsize=7,
                    color="#eee" if bright else "#333")

    # Prompt labels (truncated to fit)
    prompt_labels = [pr[:28] + "…" if len(pr) > 28 else pr for pr in prompts]
    ax.set_xticks(range(n_p))
    ax.set_xticklabels(prompt_labels, rotation=38, ha="right",
                        color="#d1d5db", fontsize=9)

    # Workshop labels: "Name\ncommunity"
    w_labels = [f"{wn}\n({ctx})" if ctx else wn for _, wn, ctx in workshops]
    ax.set_yticks(range(n_w))
    ax.set_yticklabels(w_labels, color="#d1d5db", fontsize=9)

    ax.set_title(
        "Kendall's τ: AI Ranking Agreement per Prompt × Workshop\n"
        "Green = community order matches AI  ·  Red = community disagrees  ·  "
        "τ ∈ [−1, 1]",
        color="#f3f4f6", fontsize=11, pad=14,
    )
    ax.tick_params(colors="#6b7280")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")

    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
