from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from retrieval import RetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("server")

app = FastAPI()
ENGINE: RetrievalEngine | None = None
STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/search")
async def search(query: str, top_k: int = 20):
    if not query.strip():
        raise HTTPException(400, "query cannot be empty")
    result = ENGINE.retrieve(query, top_k=top_k)
    return result


@app.get("/api/image/{index}")
async def image(index: int):
    if index < 0 or index >= ENGINE.dataset_size():
        raise HTTPException(404, "image not found")
    return Response(content=ENGINE.get_image_bytes(index), media_type="image/jpeg")


def parse_args():
    p = argparse.ArgumentParser(description="Image retrieval server")
    p.add_argument("--folder", default=None, help="Path to local image folder")
    p.add_argument("--hf-repo", default=None, help="HuggingFace dataset repo (e.g. nlphuji/flickr30k)")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-config", default=None)
    p.add_argument("--image-column", default="image")
    p.add_argument("--max-images", type=int, default=2000)
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--device", default="auto")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    return p.parse_args()


def main():
    global ENGINE
    args = parse_args()

    if not args.folder and not args.hf_repo:
        raise SystemExit("Provide --folder <path> or --hf-repo <repo>")

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
