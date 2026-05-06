from __future__ import annotations

import argparse
import csv
import io
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retrieval import RetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("server")

app = FastAPI()
ENGINE: RetrievalEngine | None = None
ACTIVE_WORKSHOP_ID: int | None = None
STATIC_DIR = Path(__file__).parent / "static"
DB_PATH    = Path(__file__).parent / "data" / "rankings.db"

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
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt     TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS rankings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                image_index INTEGER NOT NULL,
                model_rank  INTEGER NOT NULL,
                user_rank   INTEGER NOT NULL
            );
        """)
        # Add workshop_id to sessions if this DB predates the workshops feature
        try:
            con.execute("ALTER TABLE sessions ADD COLUMN workshop_id INTEGER REFERENCES workshops(id)")
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
    selections: List[RankingItem]


class WorkshopCreate(BaseModel):
    name:              str
    community_context: str = ""
    location:          str = ""
    date:              str = ""
    facilitator:       str = ""


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

    with get_db() as con:
        cur = con.execute(
            "INSERT INTO sessions (prompt, workshop_id) VALUES (?, ?)",
            (body.prompt.strip(), ACTIVE_WORKSHOP_ID),
        )
        session_id = cur.lastrowid
        con.executemany(
            "INSERT INTO rankings (session_id, image_index, model_rank, user_rank) VALUES (?,?,?,?)",
            [(session_id, s.imageIndex, s.modelRank, s.userRank) for s in body.selections],
        )

    log.info(f"Saved session {session_id}: workshop={ACTIVE_WORKSHOP_ID} "
             f"prompt='{body.prompt}' selections={len(body.selections)}")
    return {"session_id": session_id, "saved": len(body.selections)}


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


@app.get("/api/export_analysis")
async def export_for_analysis():
    with get_db() as con:
        rows = con.execute("""
            SELECT
                r.session_id,
                s.workshop_id,
                w.name            AS workshop_name,
                w.community_context,
                s.prompt,
                r.image_index,
                r.model_rank,
                r.user_rank
            FROM rankings r
            JOIN sessions  s ON r.session_id  = s.id
            LEFT JOIN workshops w ON s.workshop_id = w.id
            ORDER BY r.session_id, r.user_rank
        """).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "session_id", "workshop_id", "workshop_name", "community_context",
        "prompt", "image_index", "model_rank", "user_rank",
    ])
    for row in rows:
        writer.writerow(list(row))

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rankings_export.csv"},
    )


# ── Startup / main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Image retrieval server")
    p.add_argument("--folder",       default=None)
    p.add_argument("--hf-repo",      default=None)
    p.add_argument("--hf-split",     default="train")
    p.add_argument("--hf-config",    default=None)
    p.add_argument("--image-column", default="image")
    p.add_argument("--max-images",   type=int, default=2000)
    p.add_argument("--model",        default="ViT-B-32")
    p.add_argument("--pretrained",   default="openai")
    p.add_argument("--device",       default="auto")
    p.add_argument("--port",         type=int, default=8080)
    p.add_argument("--host",         default="127.0.0.1")
    return p.parse_args()


def main():
    global ENGINE
    args = parse_args()

    if not args.folder and not args.hf_repo:
        raise SystemExit("Provide --folder <path> or --hf-repo <repo>")

    init_db()

    ENGINE = RetrievalEngine(device=args.device)
    ENGINE.load_model(model_name=args.model, pretrained=args.pretrained)

    if args.folder:
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
