
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

# Expose on local network (so participants can connect from their own devices)
python server.py --hf-repo nlphuji/flickr30k --host 0.0.0.0 --port 8080
```

---

## How it works

1. **Startup** — loads the dataset, computes CLIP image embeddings, caches them to `data/<dataset>.pt`.
   Subsequent runs load from cache instantly (cache is invalidated if the model or image count changes).

2. **Search** — user types a prompt; server encodes it with the same CLIP text encoder, computes cosine similarity
   against all image embeddings, and returns every image sorted by score. Ranking happens once per query.

3. **Browse** — results load progressively via infinite scroll (30 at a time from the pre-ranked local list).
   No re-ranking, no extra API calls as you scroll.

4. **Select** — click up to 9 images. Each gets a numbered badge (your personal rank order).

5. **Flipbook** — hit Done; selections are saved to SQLite (`data/rankings.db`) and the browser
   navigates to a 5-page animated flipbook at `/flipbook`.

---

## Workshop guide

This section documents how to run Las Agencias as a data collection instrument across multiple communities.
The goal is to compare how different groups order AI-retrieved images for the same prompts, measuring agreement with the model using Kendall's τ.

### Before the workshops

**1. Preview the analysis output**

Run the simulation to see what your post-workshop tables and report will look like before any real data is collected:

```bash
python simulate_workshop.py
```

This generates 4 synthetic communities × 20 sessions and opens `simulation_results.html` in your browser,
showing the tau heatmap and summary table. Use it to sanity-check the setup and calibrate expectations.

**2. Prepare the image dataset**

Choose a dataset that is culturally neutral enough to allow comparison across communities.
Flickr30k works well for general prompts. If you have a domain-specific image set, use `--folder`.

**3. Prepare your prompts**

Use the same prompts across all workshops — this is what makes the comparison valid.
Write them down before the first session so they are consistent.

---

### On the day: facilitator setup

The facilitator controls data collection through the **admin panel** at `/admin` — no terminal commands needed on the day.

**Step 1 — Open the admin panel**

Navigate to `https://your-domain.com/admin` (or `http://localhost:8080/admin` when running locally).

**Step 2 — Start recording**

Fill in the workshop details and click **Start recording**:

- **Workshop name** — e.g. "LGBT+ Barcelona"
- **Community context** — e.g. "LGBTQ+", "Roma", "Migrants", "Youth"
- **Location** and **Date** — for your records
- **Facilitator** — your name

The banner turns green and shows a pulsing dot. All participant sessions from this point are automatically linked to this workshop.

The admin panel also displays the **Participant URL** to share with the room.

**Step 3 — Run the session**

Participants go to the main URL and complete the task. The admin panel refreshes every 8 seconds — you can watch the session count rise in the Past workshops table.

**Step 4 — Stop recording**

When the workshop is done, click **Stop recording**. The database retains all data; no new sessions will be linked until you start the next workshop.

**Between workshops** — repeat steps 2–4 for each community. All data stays in the same database and is separated by workshop ID automatically.

---

### During the workshop: participant flow

Participants follow the same steps each time:

1. **Enter a prompt** — the facilitator reads the prompt aloud; participants type it in exactly.
2. **Browse and select** — pick up to 9 images that best fit the prompt.
3. **Order them** — drag to rank from most to least relevant, then submit.

The flipbook is a receipt for the participant. The ranking data is what matters for analysis.

**Facilitator notes:**
- Use identical prompts across all workshops. Wording differences will confuse the comparison.
- Participants should not see each other's screens while selecting.
- Each prompt is a separate session — participants submit and start fresh for the next prompt.
- There is no login — each submission is a new anonymous session, linked to the active workshop.

---

### After all workshops: analysis

**Step 1 — Export the raw data (optional)**

```bash
curl http://localhost:8080/api/export_analysis -o rankings_export.csv
```

This gives you a flat CSV with every image selection across all workshops and prompts.

**Step 2 — Compute Kendall's τ**

```bash
python analysis/compute_tau_by_workshop.py
```

Reads `data/rankings.db`, computes tau per session, aggregates by (prompt, workshop),
and writes `analysis/tau_by_prompt_and_workshop.csv`. Also prints a summary table to the terminal.

**Step 3 — Generate the heatmap figure**

```bash
python analysis/generate_figure.py
```

Reads the CSV from step 2 and saves `analysis/tau_heatmap.png`.
Requires matplotlib (`pip install --force-reinstall matplotlib`).

**Reading the results:**

| τ value | Meaning |
|---------|---------|
| `+1.0` | Community's ordering is identical to the AI's |
| `+0.5` | Moderate agreement — community and AI mostly agree |
| `0.0` | No correlation — community order is unrelated to AI ranking |
| `−0.5` | Moderate disagreement |
| `−1.0` | Complete reversal — community inverts the AI's ranking |

A consistently low τ across a community (relative to others) indicates that the group
systematically reorders images in ways the model did not anticipate.

---

## Project structure

```
leaflet_design/
├── server.py                        # FastAPI server + routes
├── retrieval.py                     # CLIP embedding + retrieval engine
├── simulate_workshop.py             # Preview analysis with synthetic data
├── test_pipeline.py                 # Integration tests
├── requirements.txt
├── analysis/
│   ├── compute_tau_by_workshop.py   # Main analysis: tau per prompt × workshop
│   └── generate_figure.py          # Heatmap figure from analysis CSV
├── data/
│   ├── *.pt                         # Cached embeddings (auto-generated)
│   └── rankings.db                  # SQLite: workshops, sessions, rankings
└── static/
    ├── index.html                   # Participant UI (search & select)
    ├── admin.html                   # Facilitator control panel (/admin)
    ├── flipbook.html                # Flipbook shell
    ├── flipbook.js                  # Page-flip logic
    ├── flipbook.css                 # Styles + 3D animations
    ├── i18n.js                      # Localisation system
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
| `POST` | `/api/submit` | Saves session + selections, links to active workshop |
| `GET` | `/admin` | Facilitator control panel |
| `POST` | `/api/workshop/create` | Creates a workshop record, returns `workshop_id` |
| `POST` | `/api/workshop/set_active?workshop_id=N` | Sets active workshop for new sessions |
| `POST` | `/api/workshop/deactivate` | Stops recording (clears active workshop) |
| `GET` | `/api/workshop/active` | Returns currently active workshop |
| `GET` | `/api/workshops` | Lists all workshops with session counts |
| `GET` | `/api/export_analysis` | Downloads full rankings CSV |

---

## Requirements

```bash
pip install -r requirements.txt
```

Main deps: `fastapi`, `uvicorn`, `open_clip_torch`, `torch`, `Pillow`, `datasets`, `numpy`

For analysis: `scipy` (optional, pure-Python fallback included), `matplotlib`

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


## Cloud deployment (systemd)

A `leaflet.service` file is included for running the app as a persistent background service on a Linux cloud server.

**1. Edit the service file**

```bash
vi leaflet.service
```

At minimum, update:
- `User` / `Group` — your server's deploy user (default: `ubuntu`)
- `WorkingDirectory` — absolute path where the project lives (e.g. `/opt/leaflet_design`)
- `ExecStart` — choose `--hf-repo` or `--folder`, and adjust other flags as needed

If your dataset requires a HuggingFace token, uncomment the `HF_TOKEN` line and set it.

**2. Install and start**

```bash
sudo cp leaflet.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable leaflet
sudo systemctl start leaflet
```

**3. Check status and logs**

```bash
sudo systemctl status leaflet
sudo journalctl -u leaflet -f
```

The service binds to `0.0.0.0:8080` by default — suitable for a server behind a firewall or nginx reverse proxy.

---

## Funding Acknowledgement

![Co-funded by the European Union](eu-funded.png)

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Education and Culture Executive Agency (EACEA). Neither the European Union nor EACEA can be held responsible for them.
