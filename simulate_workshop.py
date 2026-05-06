"""
simulate_workshop.py — Visual preview of post-workshop analysis.

Generates synthetic data for 4 workshop communities with different biases,
runs the tau analysis, prints a table, and produces a self-contained HTML
report you can open in any browser.

Usage:
    python simulate_workshop.py                   # generate + open browser
    python simulate_workshop.py --no-browser      # generate, don't open browser
    python simulate_workshop.py --sessions 30     # more sessions per workshop
    python simulate_workshop.py --seed 7          # different random seed

No extra dependencies beyond numpy (already in requirements.txt).
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Kendall's tau (pure Python — no scipy needed) ────────────────────────────

def kendall_tau(x: list, y: list) -> float:
    """Kendall's tau-b. O(n²), fine for n ≤ 9."""
    n = len(x)
    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if   dx * dy > 0: concordant += 1
            elif dx * dy < 0: discordant += 1
            elif dx == 0 and dy != 0: tied_x += 1
            elif dy == 0 and dx != 0: tied_y += 1
    denom = (concordant + discordant + tied_x) * (concordant + discordant + tied_y)
    return 0.0 if denom == 0 else (concordant - discordant) / denom ** 0.5


# ── Community personas ────────────────────────────────────────────────────────
# noise_std controls how much each community deviates from the AI's ranking.
# Higher noise → lower Kendall's tau.

WORKSHOPS = [
    dict(name="LGBT+ Barcelona",    context="LGBTQ+",   noise=2.5),   # strongly reorders
    dict(name="Roma Communities",   context="Roma",     noise=1.8),   # moderately reorders
    dict(name="Migrant Collective", context="Migrants", noise=1.2),   # mild reordering
    dict(name="Youth Group",        context="Youth",    noise=0.4),   # close to AI
]

COMMUNITY_COLORS = ["#e85d4a", "#f4a93a", "#4a9fe8", "#48b068"]

# Shared prompts across all workshops — this is what makes the comparison meaningful
PROMPTS = [
    "create leaflet to promote a community park",
    "design flyer for a neighbourhood health fair",
    "make poster for cultural diversity festival",
    "build campaign for accessible public transport",
]


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(con: sqlite3.Connection):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS workshops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, community_context TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT, workshop_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER, image_index INTEGER,
            model_rank INTEGER, user_rank INTEGER
        );
    """)


def noisy_permutation(n: int, noise_std: float, rng: random.Random) -> list[int]:
    """Return a permutation of [1..n] with Gaussian noise applied to the sorted order."""
    noisy = [i + rng.gauss(0, noise_std) for i in range(n)]
    order = sorted(range(n), key=lambda i: noisy[i])
    result = [0] * n
    for rank, idx in enumerate(order, 1):
        result[idx] = rank
    return result


def seed_database(con: sqlite3.Connection, n_sessions: int, seed: int):
    init_db(con)
    rng = random.Random(seed)

    workshop_ids = []
    for w in WORKSHOPS:
        cur = con.execute(
            "INSERT INTO workshops (name, community_context) VALUES (?,?)",
            (w["name"], w["context"]),
        )
        workshop_ids.append(cur.lastrowid)

    for wid, workshop in zip(workshop_ids, WORKSHOPS):
        for _ in range(n_sessions):
            prompt = rng.choice(PROMPTS)
            cur    = con.execute(
                "INSERT INTO sessions (prompt, workshop_id) VALUES (?,?)",
                (prompt, wid),
            )
            session_id = cur.lastrowid

            n_images    = rng.randint(5, 9)
            model_ranks = list(range(1, n_images + 1))
            user_ranks  = noisy_permutation(n_images, workshop["noise"], rng)
            img_idxs    = rng.sample(range(200), n_images)

            con.executemany(
                "INSERT INTO rankings (session_id, image_index, model_rank, user_rank) VALUES (?,?,?,?)",
                [(session_id, img_idxs[i], model_ranks[i], user_ranks[i]) for i in range(n_images)],
            )
    con.commit()


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_tau_by_workshop(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute("""
        SELECT s.id, s.prompt, s.workshop_id, w.name AS wname, w.community_context,
               r.model_rank, r.user_rank
        FROM sessions s
        JOIN rankings r ON r.session_id = s.id
        LEFT JOIN workshops w ON w.id = s.workshop_id
        ORDER BY s.id
    """).fetchall()

    sessions: dict = defaultdict(lambda: {"meta": {}, "pairs": []})
    for row in rows:
        sid = row[0]
        sessions[sid]["meta"] = {
            "prompt": row[1], "workshop_id": row[2],
            "workshop_name": row[3], "community_context": row[4],
        }
        sessions[sid]["pairs"].append((row[5], row[6]))

    session_taus = []
    for sid, data in sessions.items():
        pairs = data["pairs"]
        if len(pairs) < 2:
            continue
        tau = kendall_tau([p[0] for p in pairs], [p[1] for p in pairs])
        session_taus.append({**data["meta"], "tau": tau})

    groups: dict = defaultdict(list)
    meta:   dict = {}
    for s in session_taus:
        key = (s["prompt"], s["workshop_id"])
        groups[key].append(s["tau"])
        if key not in meta:
            meta[key] = {k: s[k] for k in ("prompt", "workshop_id", "workshop_name", "community_context")}

    output = []
    for key, taus in groups.items():
        output.append({
            **meta[key],
            "mean_tau":   float(np.mean(taus)),
            "std_tau":    float(np.std(taus)),
            "n_sessions": len(taus),
        })
    output.sort(key=lambda r: (r["prompt"], r["workshop_id"] or 0))
    return output


# ── Terminal table ────────────────────────────────────────────────────────────

def print_table(rows: list[dict]):
    workshops = [w["name"] for w in WORKSHOPS]
    prompts   = sorted({r["prompt"] for r in rows})
    col_w     = 20

    print("\n\033[1mKendall's τ by Prompt × Workshop\033[0m")
    print("  +1 = community order matches AI  ·  0 = no correlation  ·  −1 = reversed")
    print("─" * (38 + col_w * len(workshops)))
    print(f"{'Prompt':<38}", end="")
    for w in workshops:
        print(f"{w[:col_w - 2]:^{col_w}}", end="")
    print()
    print("─" * (38 + col_w * len(workshops)))

    for prompt in prompts:
        label = prompt[:36] + "…" if len(prompt) > 36 else prompt
        print(f"{label:<38}", end="")
        for wname in workshops:
            match = next((r for r in rows if r["prompt"] == prompt and r["workshop_name"] == wname), None)
            if match:
                cell = f"{match['mean_tau']:+.3f} (n={match['n_sessions']})"
                print(f"{cell:^{col_w}}", end="")
            else:
                print(f"{'—':^{col_w}}", end="")
        print()

    print("─" * (38 + col_w * len(workshops)))

    # Overall per workshop
    overall: dict = defaultdict(list)
    for r in rows:
        overall[r["workshop_name"]].append(r["mean_tau"])
    print(f"{'Overall mean':38}", end="")
    for wname in workshops:
        vals = overall.get(wname, [])
        cell = f"{np.mean(vals):+.3f}" if vals else "—"
        print(f"{cell:^{col_w}}", end="")
    print("\n")


# ── HTML report ───────────────────────────────────────────────────────────────

def _tau_color(tau: float) -> str:
    t = max(-1.0, min(1.0, tau))
    if t >= 0:
        r, g, b = int(255 * (1 - t)), 185, int(60 * (1 - t))
    else:
        r, g, b = 210, int(185 * (1 + t)), 0
    return f"rgb({r},{g},{b})"


def generate_html(rows: list[dict], n_sessions: int, seed: int) -> str:
    workshops = [w["name"] for w in WORKSHOPS]
    prompts   = sorted({r["prompt"] for r in rows})

    # Build overall tau per workshop (for the bar chart section)
    overall: dict = defaultdict(list)
    for r in rows:
        overall[r["workshop_name"]].append(r["mean_tau"])

    # ── Heatmap table ─────────────────────────────────────────────────────────
    header_cells = "".join(
        f'<th style="padding:8px 14px;color:{c};font-size:13px">{w}</th>'
        for w, c in zip(workshops, COMMUNITY_COLORS)
    )
    body_rows = ""
    for prompt in prompts:
        cells = f'<td style="padding:8px 12px;font-size:12px;color:#9ca3af;white-space:nowrap">{prompt}</td>'
        for wname in workshops:
            match = next((r for r in rows if r["prompt"] == prompt and r["workshop_name"] == wname), None)
            if match:
                v   = match["mean_tau"]
                n   = match["n_sessions"]
                bg  = _tau_color(v)
                fg  = "black" if 0.1 < v < 0.75 else "white"
                cells += (f'<td style="background:{bg};color:{fg};text-align:center;'
                          f'padding:10px 16px;font-weight:700;font-size:13px">'
                          f'{v:+.3f}<br><span style="font-size:10px;font-weight:400">n={n}</span></td>')
            else:
                cells += '<td style="color:#4b5563;text-align:center">—</td>'
        body_rows += f"<tr>{cells}</tr>\n"

    # Overall row
    overall_cells = '<td style="padding:8px 12px;font-size:12px;color:#6b7280;font-style:italic">Overall mean</td>'
    for wname in workshops:
        vals = overall.get(wname, [])
        if vals:
            v   = float(np.mean(vals))
            bg  = _tau_color(v)
            fg  = "black" if 0.1 < v < 0.75 else "white"
            overall_cells += (f'<td style="background:{bg};color:{fg};text-align:center;'
                              f'padding:10px 16px;font-weight:700;font-size:14px;'
                              f'border-top:2px solid #374151">{v:+.3f}</td>')
        else:
            overall_cells += '<td>—</td>'
    body_rows += f"<tr>{overall_cells}</tr>"

    heatmap = f"""
    <table style="border-collapse:collapse;width:100%">
      <thead><tr style="border-bottom:2px solid #374151">
        <th style="text-align:left;padding:8px 12px;color:#6b7280;font-size:12px">Prompt</th>
        {header_cells}
      </tr></thead>
      <tbody>{body_rows}</tbody>
    </table>"""

    # ── Bar chart section ──────────────────────────────────────────────────────
    bar_rows = ""
    for wname, color in zip(workshops, COMMUNITY_COLORS):
        vals = overall.get(wname, [])
        if not vals:
            continue
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        bar_w = max(2, int((mean + 1) / 2 * 280))
        bar_rows += f"""
        <tr>
          <td style="width:180px;color:{color};font-weight:600;font-size:13px;padding:6px 10px">{wname}</td>
          <td style="padding:6px 10px">
            <div style="display:flex;align-items:center;gap:10px">
              <div style="background:{color};width:{bar_w}px;height:24px;border-radius:4px;opacity:0.85"></div>
              <span style="color:#9ca3af;font-size:12px">{mean:+.3f} ± {std:.3f}</span>
            </div>
          </td>
        </tr>"""

    bars = f"""
    <table style="border-collapse:collapse;width:100%">
      <thead><tr style="border-bottom:1px solid #374151">
        <th style="text-align:left;padding:6px 10px;color:#6b7280;font-size:12px">Workshop</th>
        <th style="text-align:left;padding:6px 10px;color:#6b7280;font-size:12px">Mean τ across all prompts (mean ± std)</th>
      </tr></thead>
      <tbody>{bar_rows}</tbody>
    </table>"""

    # ── Legend / interpretation ───────────────────────────────────────────────
    scale_steps = [(-1.0, "#c0392b", "−1.0  Complete reversal — community does the opposite of AI"),
                   (-0.5, "#e67e22", "−0.5  Moderate disagreement"),
                   ( 0.0, "#f1c40f", " 0.0  No correlation — community orders randomly relative to AI"),
                   (+0.5, "#a9d18e", "+0.5  Moderate agreement"),
                   (+1.0, "#27ae60", "+1.0  Perfect agreement — community mirrors AI exactly")]
    scale_html = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
        f'<div style="width:32px;height:16px;background:{c};border-radius:3px"></div>'
        f'<span style="font-size:12px;color:#9ca3af">{label}</span></div>'
        for _, c, label in scale_steps
    )

    workshop_dots = "".join(
        f'<span style="margin-right:16px"><span style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background:{c};margin-right:5px"></span>'
        f'<span style="font-size:12px;color:#d1d5db">{w["name"]} ({w["context"]})</span></span>'
        for w, c in zip(WORKSHOPS, COMMUNITY_COLORS)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Las Agencias — Workshop Simulation</title>
<style>
  body {{ background:#111827; color:#e5e7eb; font-family:'Segoe UI',Arial,sans-serif;
          margin:0; padding:28px 36px; max-width:1200px; }}
  h1   {{ color:#f9fafb; font-size:22px; margin-bottom:4px; }}
  h2   {{ color:#d1d5db; font-size:14px; font-weight:600; margin:28px 0 10px;
          border-bottom:1px solid #374151; padding-bottom:6px; }}
  .sub {{ color:#6b7280; font-size:13px; margin-bottom:24px; }}
  .card {{ background:#1f2937; border:1px solid #374151; border-radius:8px;
           padding:20px 24px; margin-bottom:22px; }}
  .note {{ font-size:12px; color:#6b7280; margin-top:10px; }}
</style>
</head>
<body>

<h1>Las Agencias — Workshop Simulation</h1>
<p class="sub">
  {len(WORKSHOPS)} communities · {n_sessions} sessions per workshop · seed {seed} ·
  This shows what your post-workshop analysis will look like with real data.
</p>

<div style="margin-bottom:20px">{workshop_dots}</div>

<div class="card">
  <h2>Kendall's τ by Prompt × Workshop</h2>
  <p class="note">
    How closely does each community's image ordering match the AI model's ranking?
    τ = +1 means identical order. τ = −1 means complete reversal. τ ≈ 0 means no relationship.
    Colour scale: <span style="color:#27ae60">green = agreement</span> ·
    <span style="color:#f1c40f">yellow = neutral</span> ·
    <span style="color:#c0392b">red = disagreement</span>
  </p>
  {heatmap}
</div>

<div class="card">
  <h2>Overall Agreement per Community (all prompts averaged)</h2>
  <p class="note">
    Bar length represents mean τ mapped to [0, 280px].
    Communities with lower noise in the simulation appear more aligned with the AI.
  </p>
  {bars}
</div>

<div class="card">
  <h2>Tau Scale Reference</h2>
  {scale_html}
  <p class="note">
    Expected real-world range: communities actively correcting AI bias typically produce τ ∈ [0.1, 0.5].
    Near-zero or negative values indicate systematic disagreement with the model's ordering.
  </p>
</div>

</body>
</html>"""


def generate_report(rows: list[dict], n_sessions: int, seed: int,
                    out_path: Path, open_browser: bool):
    html = generate_html(rows, n_sessions, seed)
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML report → {out_path}")
    if open_browser:
        import webbrowser
        webbrowser.open(out_path.as_uri())


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Las Agencias — workshop simulation")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--sessions",   type=int, default=20)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n\033[1mLas Agencias — Workshop Simulation\033[0m")
    print(f"  {len(WORKSHOPS)} communities · {args.sessions} sessions each · seed={args.seed}\n")

    tmp    = tempfile.mkdtemp()
    db_path = Path(tmp) / "sim.db"

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        print("  Generating data … ", end="", flush=True)
        seed_database(con, n_sessions=args.sessions, seed=args.seed)
        print("done.")
        print("  Computing tau … ",  end="", flush=True)
        rows = compute_tau_by_workshop(con)
        print("done.")

    print_table(rows)

    out = Path(__file__).parent / "simulation_results.html"
    generate_report(rows, args.sessions, args.seed,
                    out_path=out, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
