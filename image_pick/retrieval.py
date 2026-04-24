from __future__ import annotations

import logging
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger("retrieval")

DATA_DIR = Path(__file__).parent / "data"


def _slug(name: str) -> str:
    """Turn an arbitrary dataset name into a safe filename stem."""
    return re.sub(r"[^\w\-]+", "_", name).strip("_")


class RetrievalEngine:
    def __init__(self, device: str = "auto"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.dataset: list[Image.Image] = []
        self.image_paths: list[str] = []
        self.image_embeddings: torch.Tensor | None = None
        self._dataset_name: str = "dataset"
        self._model_key: str = ""

    def load_model(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        import open_clip
        log.info(f"Loading CLIP model {model_name} ({pretrained}) on {self.device} …")
        t0 = time.time()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self._model_key = f"{model_name}__{pretrained}"
        log.info(f"Model loaded in {time.time() - t0:.1f}s")

    def load_dataset_from_folder(self, folder: str, max_images: int = 2000):
        folder = Path(folder)
        self._dataset_name = _slug(folder.name)
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in extensions)[:max_images]
        log.info(f"Loading {len(paths)} images from {folder} …")
        self.dataset, self.image_paths = [], []
        for p in paths:
            try:
                self.dataset.append(Image.open(p).convert("RGB"))
                self.image_paths.append(str(p))
            except Exception as e:
                log.warning(f"Skipping {p}: {e}")
        log.info(f"Loaded {len(self.dataset)} images")

    def load_dataset_from_huggingface(self, repo: str, split: str = "train",
                                       image_column: str = "image",
                                       max_images: int = 2000,
                                       hf_config: str | None = None):
        from datasets import load_dataset, get_dataset_split_names
        self._dataset_name = _slug(repo)
        log.info(f"Loading HuggingFace dataset {repo} …")
        t0 = time.time()
        
        # Handle the nlphuji/flickr30k specific case where they named the split 'test'
        if "flickr30k" in repo.lower() and split == "train":
            log.info("Detected Flickr30k: switching default split from 'train' to 'test'")
            split = "test"

        kwargs = {
            "split": split, 
            "streaming": True, 
            "trust_remote_code": True
        }
        if hf_config:
            kwargs["name"] = hf_config
            
        try:
            ds = load_dataset(repo, **kwargs)
        except ValueError as e:
            log.warning(f"Split '{split}' failed. Attempting to find available splits...")
            # Fallback: find any available split and use the first one
            available_splits = get_dataset_split_names(repo, config_name=hf_config, trust_remote_code=True)
            if not available_splits:
                raise RuntimeError(f"No splits found for {repo}")
            new_split = available_splits[0]
            log.info(f"Retrying with split: {new_split}")
            kwargs["split"] = new_split
            ds = load_dataset(repo, **kwargs)
        
        self.dataset = []
        for item in ds:
            if max_images and len(self.dataset) >= max_images:
                break
            
            img_data = item[image_column]
            
            try:
                if isinstance(img_data, Image.Image):
                    self.dataset.append(img_data.convert("RGB"))
                elif isinstance(img_data, dict) and "bytes" in img_data:
                    self.dataset.append(Image.open(BytesIO(img_data["bytes"])).convert("RGB"))
                else:
                    self.dataset.append(Image.open(img_data).convert("RGB"))
            except Exception as e:
                log.warning(f"Could not load image: {e}")
                continue

        log.info(f"Loaded {len(self.dataset)} images in {time.time() - t0:.1f}s")

    def _cache_path(self) -> Path:
        return DATA_DIR / f"{self._dataset_name}.pt"

    @torch.no_grad()
    def embed_images(self, batch_size: int = 64):
        if not self.dataset:
            raise RuntimeError("No dataset loaded")

        cache = self._cache_path()
        if cache.exists():
            try:
                saved = torch.load(cache, map_location="cpu", weights_only=True)
                if (saved.get("model_key") == self._model_key
                        and saved.get("num_images") == len(self.dataset)):
                    self.image_embeddings = saved["embeddings"]
                    log.info(f"Loaded embeddings from cache: {cache} {self.image_embeddings.shape}")
                    return
                else:
                    log.info("Cache exists but is stale (different model or image count) — recomputing")
            except Exception as e:
                log.warning(f"Could not read cache {cache}: {e} — recomputing")

        log.info(f"Embedding {len(self.dataset)} images …")
        t0 = time.time()
        all_embs = []
        for i in range(0, len(self.dataset), batch_size):
            batch = self.dataset[i:i + batch_size]
            tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            embs = F.normalize(self.model.encode_image(tensors), dim=-1)
            all_embs.append(embs.cpu())
            if i % (batch_size * 10) == 0:
                log.info(f"  {i + len(batch)}/{len(self.dataset)}")
        self.image_embeddings = torch.cat(all_embs, dim=0)
        log.info(f"Image embeddings ready: {self.image_embeddings.shape} in {time.time() - t0:.1f}s")

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            "embeddings": self.image_embeddings,
            "num_images": len(self.dataset),
            "model_key": self._model_key,
        }, cache)
        log.info(f"Embeddings saved to {cache}")

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 20) -> dict:
        if self.image_embeddings is None:
            raise RuntimeError("Images not embedded yet")
        tokens = self.tokenizer([query]).to(self.device)
        query_emb = F.normalize(self.model.encode_text(tokens), dim=-1).cpu()
        sims = (query_emb @ self.image_embeddings.T).squeeze(0).numpy()
        ranked = np.argsort(sims)[::-1][:top_k]
        return {
            "indices": ranked.tolist(),
            "similarities": [round(float(sims[i]), 4) for i in ranked],
        }

    def get_image_bytes(self, index: int, max_size: int = 400) -> bytes:
        img = self.dataset[index].copy()
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def dataset_size(self) -> int:
        return len(self.dataset)