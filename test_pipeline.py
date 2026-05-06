"""
Integration test for the workshop data collection pipeline.

Creates 4 synthetic workshops with 20 sessions each (all using the same
shared prompt pool), runs compute_tau_by_workshop.py, and validates the output.

Run: python test_pipeline.py
"""
from __future__ import annotations

import random
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── Synthetic workshops and prompts ───────────────────────────────────────────

WORKSHOPS = [
    {"name": "LGBT+ Barcelona",    "community_context": "LGBTQ+",   "location": "Barcelona"},
    {"name": "Roma Communities",   "community_context": "Roma",     "location": "Girona"},
    {"name": "Migrant Collective", "community_context": "Migrants", "location": "Tarragona"},
    {"name": "Youth Group",        "community_context": "Youth",    "location": "Lleida"},
]

# Same prompts used across all workshops so tau values are comparable
PROMPTS = [
    "create leaflet to promote a community park",
    "design flyer for a neighbourhood health fair",
    "make poster for cultural diversity festival",
]

SESSIONS_PER_WORKSHOP = 20
N_IMAGES = 9   # max images per session


def build_user_ranks(model_ranks: list[int], noise_std: float, rng: random.Random) -> list[int]:
    n     = len(model_ranks)
    noisy = [r + rng.gauss(0, noise_std) for r in model_ranks]
    order = sorted(range(n), key=lambda i: noisy[i])
    user_ranks = [0] * n
    for rank, idx in enumerate(order, 1):
        user_ranks[idx] = rank
    return user_ranks


def init_db(con: sqlite3.Connection):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS workshops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, community_context TEXT, location TEXT,
            date TEXT, facilitator TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            workshop_id INTEGER REFERENCES workshops(id),
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            image_index INTEGER NOT NULL,
            model_rank INTEGER NOT NULL,
            user_rank INTEGER NOT NULL
        );
    """)


def seed_database(db_path: Path, seed: int = 0):
    rng = random.Random(seed)
    with sqlite3.connect(db_path) as con:
        init_db(con)

        workshop_ids = []
        for w in WORKSHOPS:
            cur = con.execute(
                "INSERT INTO workshops (name, community_context, location, created_at) VALUES (?,?,?,datetime('now'))",
                (w["name"], w["community_context"], w["location"]),
            )
            workshop_ids.append(cur.lastrowid)

        # Each community has a distinct agreement bias with the model
        biases = [2.5, 1.5, 0.4, 1.8]   # higher noise → lower tau

        for wid, noise_std in zip(workshop_ids, biases):
            for _ in range(SESSIONS_PER_WORKSHOP):
                prompt = rng.choice(PROMPTS)
                cur = con.execute(
                    "INSERT INTO sessions (prompt, workshop_id) VALUES (?,?)",
                    (prompt, wid),
                )
                session_id = cur.lastrowid

                n          = rng.randint(5, N_IMAGES)
                img_idxs   = rng.sample(range(200), n)
                model_ranks = list(range(1, n + 1))
                user_ranks  = build_user_ranks(model_ranks, noise_std, rng)

                con.executemany(
                    "INSERT INTO rankings (session_id, image_index, model_rank, user_rank) VALUES (?,?,?,?)",
                    [(session_id, img_idxs[i], model_ranks[i], user_ranks[i]) for i in range(n)],
                )
        con.commit()


# ── Assertions ────────────────────────────────────────────────────────────────

def ok(label: str):
    print(f"  PASS  {label}")


def fail(label: str, msg: str):
    print(f"  FAIL  {label}: {msg}")
    return False


def check(label: str, cond: bool, msg: str = "") -> bool:
    if cond:
        ok(label)
        return True
    return fail(label, msg)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_db_structure(db_path: Path) -> bool:
    passed = True
    with sqlite3.connect(db_path) as con:
        n_workshops = con.execute("SELECT COUNT(*) FROM workshops").fetchone()[0]
        n_sessions  = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        n_per_ws    = con.execute(
            "SELECT MIN(c), MAX(c) FROM (SELECT COUNT(*) AS c FROM sessions GROUP BY workshop_id)"
        ).fetchone()
        has_workshop_id = con.execute(
            "SELECT COUNT(*) FROM sessions WHERE workshop_id IS NOT NULL"
        ).fetchone()[0]

    passed &= check("4 workshops created",        n_workshops == 4)
    passed &= check("80 sessions total",           n_sessions  == SESSIONS_PER_WORKSHOP * 4)
    passed &= check("20 sessions per workshop",    n_per_ws == (20, 20))
    passed &= check("all sessions linked to workshop", has_workshop_id == n_sessions)
    return passed


def test_analysis_output(db_path: Path) -> bool:
    import analysis.compute_tau_by_workshop as cta
    cta.DB_PATH  = db_path
    cta.OUT_PATH = db_path.parent / "tau_by_prompt_and_workshop.csv"

    rows = cta.main()
    if not rows:
        return fail("analysis produced rows", "got empty list")

    passed = True
    # Should have up to len(PROMPTS) × len(WORKSHOPS) rows
    passed &= check("output rows produced",        len(rows) > 0)
    passed &= check("all 4 workshops represented",
                    len({r["workshop_id"] for r in rows}) == 4)
    passed &= check("all prompts represented",
                    len({r["prompt"] for r in rows}) == len(PROMPTS))
    passed &= check("mean_tau in [-1, 1]",
                    all(-1 <= r["mean_tau"] <= 1 for r in rows))
    passed &= check("n_sessions > 0 for all rows",
                    all(r["n_sessions"] > 0 for r in rows))

    # Communities with different noise levels should have different tau
    by_ws = {}
    for r in rows:
        by_ws.setdefault(r["workshop_id"], []).append(r["mean_tau"])
    overall = {wid: float(np.mean(vals)) for wid, vals in by_ws.items()}
    spread  = max(overall.values()) - min(overall.values())
    passed &= check("tau varies across workshops (spread > 0.05)",
                    spread > 0.05,
                    f"spread={spread:.3f}")

    csv_written = cta.OUT_PATH.exists()
    passed &= check("CSV file written", csv_written)
    return passed


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    tmp     = tempfile.mkdtemp()
    db_path = Path(tmp) / "test.db"

    print("\n=== Seeding synthetic data ===")
    seed_database(db_path)
    print(f"  DB at {db_path}")

    print("\n=== Database structure ===")
    p1 = test_db_structure(db_path)

    print("\n=== Analysis output ===")
    p2 = test_analysis_output(db_path)

    passed = sum([p1, p2])
    failed = 2 - passed
    print(f"\n=== {passed}/2 test groups passed", "===" if not failed else f" — {failed} FAILED ===")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
