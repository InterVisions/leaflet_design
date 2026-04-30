
# Las Agencias

Semantic image retrieval and flipbook presentation tool.
Uses CLIP embeddings to rank images by similarity to a text prompt.
Users browse results, pick up to 9, and export them as a 5-page flipbook.

---

## Run

From the repo root:

```bash
python server.py --hf-repo nlphuji/flickr30k
```

Then open http://127.0.0.1:8080

---

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-repo` | — | HuggingFace dataset repo (e.g. `nlphuji/flickr30k`) |
| `--hf-split` | `train` | Dataset split (Flickr30k auto-switches to `test`) |
| `--hf-config` | — | Dataset config name if required |
| `--image-column` | `image` | Column name that holds the images |
| `--folder` | — | Local image folder instead of HuggingFace |
| `--max-images` | `0` | Max images to load (0 = entire dataset) |
| `--model` | `ViT-B-32` | OpenCLIP model name |
| `--pretrained` | `openai` | Pretrained weights key |
| `--device` | `auto` | `cuda`, `cpu`, or `auto` |
| `--host` | `127.0.0.1` | Server host |
| `--port` | `8080` | Server port |

Use `--folder` OR `--hf-repo`, not both.

### Examples

```bash
# Flickr30k from HuggingFace (full dataset)
python server.py --hf-repo nlphuji/flickr30k

# Local folder
python server.py --folder /path/to/images

# Cap at 500 images, different model
python server.py --hf-repo nlphuji/flickr30k --max-images 500 --model ViT-L-14 --pretrained openai

# Expose on local network
python server.py --hf-repo nlphuji/flickr30k --host 0.0.0.0 --port 8080
```

---

## How it works

1. **Startup** — loads the dataset, computes CLIP image embeddings, caches them to `data/<dataset>.pt`
   Subsequent runs load from cache instantly (cache is invalidated if the model or image count changes).

2. **Search** — user types a prompt; server encodes it with the same CLIP text encoder, computes cosine similarity
   against all image embeddings, and returns every image sorted by score. Ranking happens once per query.

3. **Browse** — results load progressively via infinite scroll (30 at a time from the pre-ranked local list).
   No re-ranking, no extra API calls as you scroll.

4. **Select** — click up to 9 images. Each gets a numbered badge (your personal rank order).

5. **Flipbook** — hit Done; selections are saved to SQLite (`data/rankings.db`) and the browser
   navigates to a 5-page animated flipbook at `/flipbook`.

---

## Project structure

```
leaflet_design/
├── server.py          # FastAPI server + routes
├── retrieval.py       # CLIP embedding + retrieval engine
├── requirements.txt
├── data/
│   ├── *.pt           # Cached embeddings (auto-generated)
│   └── rankings.db    # SQLite: user sessions + selections
└── static/
    ├── index.html     # Search & selection UI
    ├── flipbook.html  # Flipbook shell
    ├── flipbook.js    # Flipbook page-flip logic
    ├── flipbook.css   # Flipbook styles + 3D animations
    ├── i18n.js        # Localisation system
    └── locales/
        ├── en.json
        └── es.json
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Search UI |
| `GET` | `/flipbook?images=1,2,3&prompt=...` | Flipbook view |
| `GET` | `/api/search?query=...` | Returns all ranked `{indices, similarities}` |
| `GET` | `/api/image/{index}` | Returns image as JPEG (max 400×400) |
| `POST` | `/api/submit` | Saves session + selections to DB |

---

## Requirements

```bash
pip install -r requirements.txt
```

Main deps: `fastapi`, `uvicorn`, `open_clip_torch`, `torch`, `Pillow`, `datasets`, `numpy`

---

## Usage

### Step 1
Write a prompt for retrieval, then high similarity images get retrieved and you can choose up to 9 for later display in the flipbook.
![alt text](resources/part_1_screenshot.png)

### Step 2
After images are chosen, user can order them from 1 to 9, which corresponds to the order in which they are displayed in the flipbook.
![alt text](resources/part_2_screenshot.png)

### Step 3
Flipbook is generated with the retrieved images in the order chosen.
![alt text](resources/part_3_screenshot.png)
