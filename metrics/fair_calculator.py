"""
FAIR metric calculation for CLIP retrieval rankings.

FAIR = (1/M) · Σ_{i=1}^{k} [is_curated[i] · (1/(KL(D_r^i‖D*) + 1)) / log₂(i+1)]

where:
  is_curated[i]  1 if rank-i image is campaign-appropriate, 0 if a distractor
  D_r^i          demographic distribution over the CURATED images in the top-i prefix
                 (distractor images do not update demographic counts)
  D*             desired distribution  ← data/desired_distribution.json
  KL(D_r^i‖D*)  Σ_c D_r^i[c] · log(D_r^i[c] / D*[c])   ← Gao et al. (2022) eq. (1)
  M              Σ_{i=1}^{k} 1/log₂(i+1)  (normaliser over all k positions)

Edge cases:
  · No curated images in top-i yet → KL = 0.0 (nothing to measure)
  · A desired category absent from D_r^i → that term is 0·log(0/q) = 0 (no contribution)
  · Images with invalid demographic annotations are excluded from D_r^i counts

D* is NOT hardcoded. Load it from data/desired_distribution.json via
load_desired_distribution(). Null values fall back to uniform with a warning.

Demographic axes:
  gender    – {M.Male, NB, M.Female}          (skip "Cannot determine")
  age       – raw labels → {young, middle, older}  (skip "Cannot determine")
  skin_tone – Fitzpatrick 1-6 → {light, medium, dark}  (skip 0 / "?")
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("fair_calculator")

# ── Axis metadata ─────────────────────────────────────────────────────────────

AXIS_CATEGORIES: dict[str, list[str]] = {
    "gender":    ["M.Male", "NB", "M.Female"],
    "age":       ["young", "middle", "older"],
    "skin_tone": ["light", "medium", "dark"],
}

AXES = list(AXIS_CATEGORIES.keys())

_AGE_MAPPING: dict[str, str] = {
    "Child (0-12)":        "young",
    "Adolescent (13-17)":  "young",
    "Young adult (18-30)": "young",
    "Middle-aged (31-60)": "middle",
    "Older adult (60+)":   "older",
}

_DEFAULT_DIST_PATH = Path(__file__).parent.parent / "data" / "desired_distribution.json"


# ── Desired-distribution loading ──────────────────────────────────────────────

def _uniform(axis: str) -> dict[str, float]:
    cats = AXIS_CATEGORIES[axis]
    return {c: 1.0 / len(cats) for c in cats}


def load_desired_distribution(
    path: str | Path = _DEFAULT_DIST_PATH,
) -> dict[str, dict[str, float]]:
    """
    Load the desired demographic distribution from a JSON file.

    Any axis or category whose value is null falls back to a uniform
    distribution for that axis, and a warning is logged.

    Returns a dict like:
      {"gender": {"M.Male": 0.4, "NB": 0.2, "M.Female": 0.4}, ...}
    """
    path = Path(path)
    result: dict[str, dict[str, float]] = {}

    raw: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        log.warning(
            "Desired distribution file not found: %s — using uniform distributions", path
        )

    for axis, cats in AXIS_CATEGORIES.items():
        axis_raw = {k: v for k, v in raw.get(axis, {}).items() if not k.startswith("_")}
        has_nulls = any(axis_raw.get(c) is None for c in cats) or not axis_raw

        if has_nulls:
            log.warning(
                "Desired distribution for '%s' contains null values — falling back to uniform "
                "(1/%.4f per category). Update %s to set real values.",
                axis, 1.0 / len(cats), path,
            )
            result[axis] = _uniform(axis)
        else:
            result[axis] = {c: float(axis_raw[c]) for c in cats}
            total = sum(result[axis].values())
            if not math.isclose(total, 1.0, abs_tol=1e-3):
                log.warning(
                    "Desired distribution for '%s' sums to %.4f (expected 1.0) — normalising",
                    axis, total,
                )
                result[axis] = {c: v / total for c, v in result[axis].items()}

    return result


# ── Annotation helpers ────────────────────────────────────────────────────────

def is_valid_annotation(value: Any, axis: str) -> bool:
    """Return True if the raw annotation value can be used for this axis."""
    if axis == "gender":
        return value in ("M.Male", "NB", "M.Female")
    if axis == "age":
        return value in _AGE_MAPPING
    if axis == "skin_tone":
        return isinstance(value, int) and 1 <= value <= 6
    return False


def get_category(value: Any, axis: str) -> str | None:
    """Map a raw annotation value to the grouped category for this axis."""
    if axis == "gender":
        return value
    if axis == "age":
        return _AGE_MAPPING.get(value)
    if axis == "skin_tone":
        if value in (1, 2):
            return "light"
        if value in (3, 4):
            return "medium"
        if value in (5, 6):
            return "dark"
    return None


# ── KL divergence ─────────────────────────────────────────────────────────────

def kl_divergence(
    observed: dict[str, float],
    desired:  dict[str, float],
    eps: float = 1e-10,
) -> float:
    """
    KL(D_r^i ‖ D*) = Σ_c D_r^i[c] · log(D_r^i[c] / D*[c])

    Iterates over the observed distribution (Gao et al. 2022, eq. 1).
    When observed[c] = 0 the term is 0·log(…) = 0, so missing categories
    in the curated prefix contribute nothing — no hard cap or eps needed there.
    eps guards only against D*[c] = 0 (unexpected category not in desired).
    Caller is responsible for the observed=None guard (→ KL = 0.0).
    """
    total = 0.0
    for cat, p_obs in observed.items():
        if p_obs > eps:
            p_des = desired.get(cat, eps)
            total += p_obs * math.log(p_obs / p_des)
    return total


# ── Demographic distribution ──────────────────────────────────────────────────

def get_demographic_distribution(
    image_keys: list[str],
    annotations: dict[str, dict],
    axis: str,
) -> tuple[dict[str, float] | None, int]:
    """
    Build the normalised demographic distribution for a list of images.

    Images without a valid annotation for this axis are skipped.
    Returns (distribution_dict, n_valid); distribution_dict is None when
    no images have valid annotations (caller should treat this as KL = 0.0).
    """
    cats = AXIS_CATEGORIES[axis]
    counts: dict[str, int] = {c: 0 for c in cats}
    n_valid = 0

    for key in image_keys:
        ann = annotations.get(key, {})
        raw = ann.get(axis)
        if not is_valid_annotation(raw, axis):
            continue
        cat = get_category(raw, axis)
        if cat in counts:
            counts[cat] += 1
            n_valid += 1

    if n_valid == 0:
        return None, 0

    return {c: counts[c] / n_valid for c in cats}, n_valid


# ── FAIR score ────────────────────────────────────────────────────────────────

def calculate_fair_score(
    ranking:         list[str],
    curation_labels: list[int],
    annotations:     dict[str, dict],
    axis:            str,
    desired:         dict[str, float],
    k:               int = 20,
) -> float:
    """
    FAIR score for one (query, axis) pair.

    ranking:          image keys in model-ranked order
    curation_labels:  1 = curated, 0 = distractor (same length and order as ranking)
    annotations:      {image_key: {"gender": ..., "age": ..., "skin_tone": ...,
                                   "is_curated": 0|1}}
    axis:             'gender' | 'age' | 'skin_tone'
    desired:          target distribution  e.g. {"M.Male": 0.4, "NB": 0.2, "M.Female": 0.4}
    k:                ranking depth to evaluate

    G[i] = curation_labels[i] (1 or 0).
    D_r^i is computed over CURATED images in the top-i prefix only; distractor
    images do not update demographic counts but still occupy a rank position.
    KL(D_r^i‖D*) is used per Gao et al. (2022): iterating over observed means
    absent categories contribute 0 naturally. When no curated images appear in
    top-i yet, D_r^i is undefined and KL = 0.0 (no divergence to measure).
    """
    n = min(k, len(ranking))
    if n == 0:
        return 0.0

    # Normaliser: sum over all k positions regardless of curation status
    M = sum(1.0 / math.log2(j + 2) for j in range(n))

    fair_sum = 0.0
    for j in range(n):
        g_i = float(curation_labels[j])

        # Demographic distribution over curated images in the top-(j+1) prefix
        curated_prefix = [ranking[x] for x in range(j + 1) if curation_labels[x] == 1]
        dist, _ = get_demographic_distribution(curated_prefix, annotations, axis)
        kl = 0.0 if dist is None else kl_divergence(dist, desired)

        fair_sum += g_i * (1.0 / (kl + 1.0)) / math.log2(j + 2)

    return fair_sum / M


# ── Spearman correlation (scipy optional) ─────────────────────────────────────

try:
    from scipy.stats import spearmanr as _scipy_spearman

    def spearman_r(x: list[float], y: list[float]) -> tuple[float, float]:
        """Returns (r, p_value). Requires scipy."""
        result = _scipy_spearman(x, y)
        return float(result.statistic), float(result.pvalue)

except ImportError:
    def spearman_r(x: list[float], y: list[float]) -> tuple[float, float]:  # type: ignore[misc]
        """Spearman r via rank-based Pearson. p-value is NaN without scipy."""
        ax = np.array(x, dtype=float)
        ay = np.array(y, dtype=float)
        rx = np.argsort(np.argsort(ax)).astype(float)
        ry = np.argsort(np.argsort(ay)).astype(float)
        rx -= rx.mean()
        ry -= ry.mean()
        denom = math.sqrt(float(np.sum(rx ** 2)) * float(np.sum(ry ** 2)))
        r = float(np.dot(rx, ry)) / denom if denom > 0 else 0.0
        return r, float("nan")


# ── FAIRCalculator class ──────────────────────────────────────────────────────

class FAIRCalculator:
    """
    Calculate FAIR metrics for a loaded CLIP retrieval engine.

    Parameters
    ----------
    engine:
        A fully-loaded RetrievalEngine (model + dataset + embeddings ready).
        engine.image_paths is used as annotation lookup keys; falls back to
        string indices ("0", "1", …) for HuggingFace datasets.
    annotations:
        {image_key: {"gender": ..., "age": ..., "skin_tone": ..., "is_curated": 0|1}}
        Images missing "is_curated" are treated as distractors (0).
    desired_distributions:
        Either a path to desired_distribution.json, or a pre-loaded dict.
        Defaults to data/desired_distribution.json; falls back to uniform
        for any axis with null values.
    """

    def __init__(
        self,
        engine,
        annotations: dict[str, dict],
        desired_distributions: dict[str, dict[str, float]] | str | Path | None = None,
    ):
        self.engine      = engine
        self.annotations = annotations

        if desired_distributions is None or isinstance(desired_distributions, (str, Path)):
            path = _DEFAULT_DIST_PATH if desired_distributions is None else Path(desired_distributions)
            self.desired_distributions = load_desired_distribution(path)
        else:
            self.desired_distributions = desired_distributions

    def _image_key(self, index: int) -> str:
        paths = self.engine.image_paths
        if index < len(paths):
            return paths[index]
        return str(index)

    def calculate_metrics_for_query(self, query: str, k: int = 20) -> dict:
        """
        Run CLIP retrieval for one query and calculate FAIR for all axes.
        Returns a dict ready to be serialised to JSON.
        """
        raw = self.engine.retrieve(query, top_k=None)
        depth = min(k, len(raw["indices"]))
        indices      = raw["indices"][:depth]
        ranking_keys = [self._image_key(i) for i in indices]

        # Curation labels from annotations; treat missing as distractor (0)
        curation_labels = [
            int(self.annotations.get(key, {}).get("is_curated", 0))
            for key in ranking_keys
        ]
        n_curated = sum(curation_labels)

        result: dict[str, Any] = {
            "query":      query,
            "model_ranking": ranking_keys,
            "n_curated":  n_curated,
            "n_total":    depth,
        }

        for axis in AXES:
            desired = self.desired_distributions[axis]
            # Count valid demographic annotations among curated images only
            n_valid = sum(
                1 for key, c in zip(ranking_keys, curation_labels)
                if c == 1 and is_valid_annotation(self.annotations.get(key, {}).get(axis), axis)
            )
            result[f"valid_annotations_{axis}"] = n_valid
            result[f"FAIR_{axis}"] = round(
                calculate_fair_score(ranking_keys, curation_labels, self.annotations, axis, desired, k), 6
            )
            log.debug(
                "  %s | FAIR_%s=%.4f  (curated=%d/%d, annotated=%d)",
                query[:40], axis, result[f"FAIR_{axis}"], n_curated, depth, n_valid,
            )

        result["calculated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return result

    def calculate_metrics_batch(
        self,
        queries: list[str],
        k: int = 20,
        output_path: str | Path | None = None,
    ) -> dict[str, dict]:
        """Calculate metrics for every query and optionally save to JSON."""
        results: dict[str, dict] = {}
        n = len(queries)
        for idx, query in enumerate(queries, 1):
            log.info("[%d/%d] %s", idx, n, query)
            results[query] = self.calculate_metrics_for_query(query, k=k)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log.info("Saved metrics to %s", output_path)

        return results
