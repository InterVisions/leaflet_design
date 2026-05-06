"""
Compute Kendall's tau between AI ranking and user ranking for each session,
then aggregate by (prompt, workshop) to measure how much each community
agrees with the model's ordering.

Usage:
    python analysis/compute_tau_by_workshop.py

Output:
    analysis/tau_by_prompt_and_workshop.csv

Columns:
    prompt, workshop_id, workshop_name, community_context,
    mean_tau, std_tau, n_sessions
"""
from __future__ import annotations

import csv
import sqlite3
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import defaultdict
from pathlib import Path

import numpy as np

DB_PATH  = Path(__file__).parent.parent / "data" / "rankings.db"
OUT_PATH = Path(__file__).parent / "tau_by_prompt_and_workshop.csv"


# ── Kendall's tau (pure Python fallback if scipy is unavailable) ──────────────

def _kendall_tau_pure(x: list, y: list) -> float:
    """Kendall's tau-b, O(n²). Fine for n ≤ 9 (our session size)."""
    n = len(x)
    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if   dx * dy > 0: concordant += 1
            elif dx * dy < 0: discordant += 1
            elif dx == 0 and dy != 0: tied_x += 1
            elif dy == 0 and dx != 0: tied_y += 1
    denom = (concordant + discordant + tied_x) * (concordant + discordant + tied_y)
    return 0.0 if denom == 0 else (concordant - discordant) / denom ** 0.5


try:
    from scipy.stats import kendalltau as _scipy_kt
    def compute_tau(x: list, y: list) -> float:
        return float(_scipy_kt(x, y).statistic)
except Exception:
    compute_tau = _kendall_tau_pure


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> list[dict]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT
                s.id          AS session_id,
                s.prompt,
                s.workshop_id,
                w.name        AS workshop_name,
                w.community_context,
                r.model_rank,
                r.user_rank
            FROM sessions s
            JOIN rankings r ON r.session_id = s.id
            LEFT JOIN workshops w ON w.id = s.workshop_id
            ORDER BY s.id
        """).fetchall()

    if not rows:
        print("No data found in database.")
        return []

    # Group by session
    sessions: dict = defaultdict(lambda: {"meta": {}, "pairs": []})
    for row in rows:
        sid = row["session_id"]
        sessions[sid]["meta"] = {
            "prompt":            row["prompt"],
            "workshop_id":       row["workshop_id"],
            "workshop_name":     row["workshop_name"],
            "community_context": row["community_context"],
        }
        sessions[sid]["pairs"].append((row["model_rank"], row["user_rank"]))

    # Compute tau per session
    session_taus = []
    for sid, data in sessions.items():
        pairs = data["pairs"]
        if len(pairs) < 2:
            continue
        mr  = [p[0] for p in pairs]
        ur  = [p[1] for p in pairs]
        tau = compute_tau(mr, ur)
        session_taus.append({**data["meta"], "tau": tau})

    # Aggregate by (prompt, workshop_id)
    groups: dict = defaultdict(list)
    group_meta:  dict = {}
    for s in session_taus:
        key = (s["prompt"], s["workshop_id"])
        groups[key].append(s["tau"])
        if key not in group_meta:
            group_meta[key] = {
                "prompt":            s["prompt"],
                "workshop_id":       s["workshop_id"],
                "workshop_name":     s["workshop_name"],
                "community_context": s["community_context"],
            }

    output_rows = []
    for key, taus in groups.items():
        output_rows.append({
            **group_meta[key],
            "mean_tau":   round(float(np.mean(taus)), 4),
            "std_tau":    round(float(np.std(taus)),  4),
            "n_sessions": len(taus),
        })
    output_rows.sort(key=lambda r: (r["prompt"], r["workshop_id"] or 0))

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        fields = ["prompt", "workshop_id", "workshop_name", "community_context",
                  "mean_tau", "std_tau", "n_sessions"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {OUT_PATH}  ({len(output_rows)} rows, {len(session_taus)} sessions)")

    # Print a quick summary table to stdout
    print()
    prompts   = sorted({r["prompt"]       for r in output_rows})
    workshops = sorted({r["workshop_name"] for r in output_rows if r["workshop_name"]})
    col_w = 14
    print(f"{'Prompt':<35}", end="")
    for w in workshops:
        print(f"{w[:col_w]:^{col_w}}", end="")
    print()
    print("─" * (35 + col_w * len(workshops)))
    for prompt in prompts:
        label = prompt[:33] + "…" if len(prompt) > 33 else prompt
        print(f"{label:<35}", end="")
        for w in workshops:
            match = next((r for r in output_rows
                          if r["prompt"] == prompt and r["workshop_name"] == w), None)
            if match:
                cell = f"{match['mean_tau']:+.3f} (n={match['n_sessions']})"
                print(f"{cell:^{col_w}}", end="")
            else:
                print(f"{'—':^{col_w}}", end="")
        print()

    return output_rows


if __name__ == "__main__":
    main()
