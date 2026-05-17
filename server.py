from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval import RetrievalEngine
from metrics.fair_calculator import (
    AXES,
    spearman_r,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("server")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://representacion.intervisions.eu"],  # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],   # this is what fixes the OPTIONS 405
    allow_headers=["*"],
)
ENGINE:             RetrievalEngine | None = None
ACTIVE_WORKSHOP_ID: int | None = None
QUERY_METRICS:      dict = {}   # loaded from data/metrics/query_metrics.json (pre-calculated FAIR)
STATIC_DIR = Path(__file__).parent / "static"
DB_PATH    = Path(__file__).parent / "data" / "rankings.db"
DATA_DIR   = Path(__file__).parent / "data"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Database ──────────────────────────────────────────────────────────────────

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS workshops (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                name              TEXT NOT NULL,
                community_context TEXT,
                location          TEXT,
                date              TEXT,
                facilitator       TEXT,
                created_at        TEXT
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt                 TEXT NOT NULL,
                created_at             TEXT DEFAULT (datetime('now')),
                selection_time_seconds REAL,
                metrics_json           TEXT
            );
            CREATE TABLE IF NOT EXISTS rankings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                image_index INTEGER NOT NULL,
                model_rank  INTEGER NOT NULL,
                user_rank   INTEGER NOT NULL
            );
        """)
        # Migrations for columns added after initial schema
        for migration in [
            "ALTER TABLE sessions ADD COLUMN workshop_id INTEGER REFERENCES workshops(id)",
            "ALTER TABLE sessions ADD COLUMN selection_time_seconds REAL",
            "ALTER TABLE sessions ADD COLUMN metrics_json TEXT",
            "ALTER TABLE sessions ADD COLUMN nickname TEXT NOT NULL DEFAULT ''",
        ]:
            try:
                con.execute(migration)
            except sqlite3.OperationalError:
                pass
        con.commit()


@contextmanager
def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ── Request / response models ─────────────────────────────────────────────────

class RankingItem(BaseModel):
    imageIndex: int
    modelRank:  int
    userRank:   int


class SubmitRequest(BaseModel):
    prompt:     str
    nickname:   str = ""
    selections: List[RankingItem]


class WorkshopCreate(BaseModel):
    name:              str
    community_context: str = ""
    location:          str = ""
    date:              str = ""
    facilitator:       str = ""


# ── Session metric computation ────────────────────────────────────────────────

def compute_session_metrics(body: "SubmitRequest") -> dict | None:
    """
    Build the metrics blob stored with each session:
      - FAIR (gender/age/skin_tone): looked up from pre-calculated query_metrics.json
      - Spearman r/p: computed from model rank vs user rank over the 9 selected images

    Never raises — metric failures must not block saving.
    Returns None only when ENGINE is unavailable.
    """
    if ENGINE is None:
        return None

    try:
        prompt = body.prompt.strip()
        selected = sorted(body.selections, key=lambda s: s.modelRank)
        model_ranks = [s.modelRank for s in selected]
        user_ranks  = [s.userRank  for s in selected]

        metrics: dict = {}

        # ── Pre-calculated FAIR (full top-20 retrieval, not just 9 images) ──
        qm = QUERY_METRICS.get(prompt)
        if qm:
            for axis in AXES:
                key = f"FAIR_{axis}"
                if key in qm:
                    metrics[key] = qm[key]
        else:
            log.debug("No pre-calculated metrics for query: %r — run precalculate_metrics.py", prompt)

        # ── Spearman (model rank vs participant rank over 9 selected images) ──
        if len(model_ranks) >= 3:
            r, p = spearman_r(model_ranks, user_ranks)
            metrics["spearman_r"] = round(r, 4)
            metrics["spearman_p"] = None if math.isnan(p) else round(p, 4)

        return metrics or None

    except Exception as exc:
        log.warning("Metric computation failed (session still saved): %s", exc)
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/flipbook")
async def flipbook():
    return FileResponse(str(STATIC_DIR / "flipbook.html"))


@app.get("/admin")
async def admin():
    return FileResponse(str(STATIC_DIR / "admin.html"))


@app.get("/api/queries")
async def get_queries():
    queries_path = DATA_DIR / "active_queries.json"
    if not queries_path.exists():
        return []
    with open(queries_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/search")
async def search(query: str, top_k: int | None = None):
    if not query.strip():
        raise HTTPException(400, "query cannot be empty")
    return ENGINE.retrieve(query, top_k=top_k)


@app.get("/api/image/{index}")
async def image(index: int):
    if index < 0 or index >= ENGINE.dataset_size():
        raise HTTPException(404, "image not found")
    return Response(content=ENGINE.get_image_bytes(index), media_type="image/jpeg")


@app.post("/api/submit")
async def submit(body: SubmitRequest):
    if not body.prompt.strip():
        raise HTTPException(400, "prompt cannot be empty")
    if not body.selections:
        raise HTTPException(400, "no selections provided")
    if len(body.selections) > 9:
        raise HTTPException(400, "maximum 9 selections")

    metrics = compute_session_metrics(body)

    with get_db() as con:
        cur = con.execute(
            "INSERT INTO sessions (prompt, nickname, workshop_id, metrics_json) VALUES (?, ?, ?, ?)",
            (body.prompt.strip(), body.nickname.strip(), ACTIVE_WORKSHOP_ID,
             json.dumps(metrics) if metrics else None),
        )
        session_id = cur.lastrowid
        con.executemany(
            "INSERT INTO rankings (session_id, image_index, model_rank, user_rank) VALUES (?,?,?,?)",
            [(session_id, s.imageIndex, s.modelRank, s.userRank) for s in body.selections],
        )

    log.info(f"Saved session {session_id}: workshop={ACTIVE_WORKSHOP_ID} "
             f"prompt='{body.prompt}' selections={len(body.selections)}")
    return {"session_id": session_id, "saved": len(body.selections), "metrics": metrics}


@app.get("/api/session/{session_id}/metrics")
async def get_session_metrics(session_id: int):
    with get_db() as con:
        row = con.execute(
            "SELECT metrics_json FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "session not found")
    if not row["metrics_json"]:
        return {}
    return json.loads(row["metrics_json"])


@app.post("/api/workshop/create")
async def create_workshop(body: WorkshopCreate):
    with get_db() as con:
        cur = con.execute(
            """INSERT INTO workshops (name, community_context, location, date, facilitator, created_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))""",
            (body.name, body.community_context, body.location, body.date, body.facilitator),
        )
        workshop_id = cur.lastrowid
    log.info(f"Created workshop {workshop_id}: {body.name} ({body.community_context})")
    return {"workshop_id": workshop_id}


@app.post("/api/workshop/set_active")
async def set_active_workshop(workshop_id: int):
    global ACTIVE_WORKSHOP_ID
    with get_db() as con:
        row = con.execute("SELECT id, name FROM workshops WHERE id=?", (workshop_id,)).fetchone()
        if not row:
            raise HTTPException(404, "workshop not found")
    ACTIVE_WORKSHOP_ID = workshop_id
    log.info(f"Active workshop → {workshop_id}: {row['name']}")
    return {"active_workshop_id": ACTIVE_WORKSHOP_ID}


@app.get("/api/workshop/active")
async def get_active_workshop():
    if ACTIVE_WORKSHOP_ID is None:
        return {"active": False, "workshop_id": None, "name": None, "community_context": None}
    with get_db() as con:
        row = con.execute(
            "SELECT id, name, community_context, location, date FROM workshops WHERE id=?",
            (ACTIVE_WORKSHOP_ID,)
        ).fetchone()
    if not row:
        return {"active": False, "workshop_id": None, "name": None, "community_context": None}
    return {"active": True, **dict(row)}


@app.post("/api/workshop/deactivate")
async def deactivate_workshop():
    global ACTIVE_WORKSHOP_ID
    previous = ACTIVE_WORKSHOP_ID
    ACTIVE_WORKSHOP_ID = None
    log.info(f"Workshop {previous} deactivated — recording stopped")
    return {"active": False, "previous_workshop_id": previous}


@app.get("/api/workshops")
async def list_workshops():
    with get_db() as con:
        rows = con.execute("""
            SELECT w.id, w.name, w.community_context, w.location, w.date,
                   COUNT(s.id) AS n_sessions
            FROM workshops w
            LEFT JOIN sessions s ON s.workshop_id = w.id
            GROUP BY w.id
            ORDER BY w.id DESC
        """).fetchall()
    return [dict(r) for r in rows]


def _build_csv(rows) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "session_id", "nickname", "workshop_id", "workshop_name", "community_context",
        "prompt", "selection_time_seconds", "image_index", "model_rank", "user_rank",
    ])
    for row in rows:
        writer.writerow(list(row))
    return buf.getvalue()


_RANKINGS_QUERY = """
    SELECT
        r.session_id,
        s.nickname,
        s.workshop_id,
        w.name                   AS workshop_name,
        w.community_context,
        s.prompt,
        s.selection_time_seconds,
        r.image_index,
        r.model_rank,
        r.user_rank
    FROM rankings r
    JOIN sessions  s ON r.session_id  = s.id
    LEFT JOIN workshops w ON s.workshop_id = w.id
"""


@app.get("/api/export_analysis")
async def export_for_analysis():
    with get_db() as con:
        rows = con.execute(
            _RANKINGS_QUERY + " ORDER BY r.session_id, r.user_rank"
        ).fetchall()
    return Response(
        content=_build_csv(rows),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rankings_all.csv"},
    )


@app.get("/api/export_workshop/{workshop_id}")
async def export_workshop(workshop_id: int):
    with get_db() as con:
        row = con.execute("SELECT name FROM workshops WHERE id=?", (workshop_id,)).fetchone()
        if not row:
            raise HTTPException(404, "workshop not found")
        safe_name = re.sub(r"[^\w\-]+", "_", row["name"]).strip("_")[:40]
        rows = con.execute(
            _RANKINGS_QUERY + " WHERE s.workshop_id=? ORDER BY r.session_id, r.user_rank",
            (workshop_id,),
        ).fetchall()
    filename = f"rankings_{safe_name}.csv"
    return Response(
        content=_build_csv(rows),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Startup / main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Image retrieval server")
    # Combined-pool mode (curated + FHIBE distractors)
    p.add_argument("--curated-folder",    default=None, help="Curated image folder (is_curated=1)")
    p.add_argument("--distractor-folder", default=None, help="FHIBE distractor folder (is_curated=0)")
    p.add_argument("--max-curated",       type=int, default=2000)
    p.add_argument("--max-distractors",   type=int, default=20000)
    # Legacy single-folder / HuggingFace modes
    p.add_argument("--folder",       default=None)
    p.add_argument("--hf-repo",      default=None)
    p.add_argument("--hf-split",     default="train")
    p.add_argument("--hf-config",    default=None)
    p.add_argument("--image-column", default="image")
    p.add_argument("--max-images",   type=int, default=2000)
    p.add_argument("--model",        default="ViT-B-16")
    p.add_argument("--pretrained",   default="openai")
    p.add_argument("--device",       default="auto")
    p.add_argument("--port",         type=int, default=8080)
    p.add_argument("--host",         default="127.0.0.1")
    return p.parse_args()


def main():
    global ENGINE, QUERY_METRICS
    args = parse_args()

    if not args.curated_folder and not args.folder and not args.hf_repo:
        raise SystemExit(
            "Provide one of: --curated-folder, --folder, or --hf-repo"
        )

    init_db()

    # Load pre-calculated FAIR scores (produced by scripts/precalculate_metrics.py)
    metrics_path = DATA_DIR / "metrics" / "query_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            QUERY_METRICS = json.load(f)
        log.info("Loaded pre-calculated FAIR for %d queries from %s", len(QUERY_METRICS), metrics_path)
    else:
        log.info(
            "No pre-calculated metrics at %s — run scripts/precalculate_metrics.py before the workshop",
            metrics_path,
        )

    ENGINE = RetrievalEngine(device=args.device)
    ENGINE.load_model(model_name=args.model, pretrained=args.pretrained)

    if args.curated_folder:
        ENGINE.load_combined_pool(
            curated_folder=args.curated_folder,
            distractor_folder=args.distractor_folder,
            max_curated=args.max_curated,
            max_distractors=args.max_distractors,
        )
    elif args.folder:
        ENGINE.load_dataset_from_folder(args.folder, max_images=args.max_images)
    else:
        ENGINE.load_dataset_from_huggingface(
            repo=args.hf_repo,
            split=args.hf_split,
            image_column=args.image_column,
            max_images=args.max_images,
            hf_config=args.hf_config,
        )

    ENGINE.embed_images()

    log.info(f"Ready — {ENGINE.dataset_size()} images indexed")
    log.info(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
