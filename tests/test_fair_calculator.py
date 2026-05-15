"""
Unit tests for metrics/fair_calculator.py

Run with:
    python -m pytest tests/test_fair_calculator.py -v

No CLIP model is required — all tests use synthetic data.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.fair_calculator import (
    AXIS_CATEGORIES,
    calculate_fair_score,
    get_category,
    get_demographic_distribution,
    is_valid_annotation,
    kl_divergence,
)

# Uniform desired distributions used throughout tests
DESIRED_DISTRIBUTIONS = {
    axis: {cat: 1.0 / len(cats) for cat in cats}
    for axis, cats in AXIS_CATEGORIES.items()
}

# ── Fixtures: synthetic image pools ──────────────────────────────────────────

def _make_annotations(gender_list, age_list, skin_list, curated_list=None):
    """Build an annotation dict from parallel lists (one entry per image)."""
    result = {}
    if curated_list is None:
        curated_list = [1] * len(gender_list)
    for i, (g, a, s, c) in enumerate(zip(gender_list, age_list, skin_list, curated_list)):
        result[f"img_{i:03d}"] = {"gender": g, "age": a, "skin_tone": s, "is_curated": c}
    return result


GENDER_VALS   = ["M.Male", "NB", "M.Female"]
AGE_VALS_RAW  = ["Young adult (18-30)", "Middle-aged (31-60)", "Older adult (60+)"]
SKIN_VALS_FST = [1, 3, 5]

# A perfectly balanced pool of 9 curated images (3 of each category per axis)
BALANCED_ANNOTATIONS = _make_annotations(
    gender_list  = GENDER_VALS * 3,
    age_list     = AGE_VALS_RAW * 3,
    skin_list    = SKIN_VALS_FST * 3,
    curated_list = [1] * 9,
)
BALANCED_KEYS       = sorted(BALANCED_ANNOTATIONS.keys())   # img_000 … img_008
BALANCED_CURATIONS  = [1] * 9                               # all curated


# ── 1. is_valid_annotation ────────────────────────────────────────────────────

class TestIsValidAnnotation:
    def test_gender_valid(self):
        for v in ["M.Male", "NB", "M.Female"]:
            assert is_valid_annotation(v, "gender")

    def test_gender_invalid_cannot_determine(self):
        assert not is_valid_annotation("Cannot determine", "gender")

    def test_gender_invalid_none(self):
        assert not is_valid_annotation(None, "gender")

    def test_gender_invalid_empty(self):
        assert not is_valid_annotation("", "gender")

    def test_age_valid(self):
        valid_ages = [
            "Child (0-12)", "Adolescent (13-17)", "Young adult (18-30)",
            "Middle-aged (31-60)", "Older adult (60+)",
        ]
        for v in valid_ages:
            assert is_valid_annotation(v, "age"), f"Expected valid: {v}"

    def test_age_invalid_cannot_determine(self):
        assert not is_valid_annotation("Cannot determine", "age")

    def test_age_invalid_none(self):
        assert not is_valid_annotation(None, "age")

    def test_skin_tone_valid_fst(self):
        for v in range(1, 7):
            assert is_valid_annotation(v, "skin_tone"), f"FST {v} should be valid"

    def test_skin_tone_invalid_zero(self):
        assert not is_valid_annotation(0, "skin_tone")

    def test_skin_tone_invalid_question_mark(self):
        assert not is_valid_annotation("?", "skin_tone")

    def test_skin_tone_invalid_seven(self):
        assert not is_valid_annotation(7, "skin_tone")

    def test_skin_tone_invalid_string_number(self):
        assert not is_valid_annotation("3", "skin_tone")


# ── 2. get_category ───────────────────────────────────────────────────────────

class TestGetCategory:
    def test_gender_passthrough(self):
        assert get_category("M.Male", "gender") == "M.Male"
        assert get_category("NB", "gender") == "NB"
        assert get_category("M.Female", "gender") == "M.Female"

    def test_age_grouping_young(self):
        for raw in ["Child (0-12)", "Adolescent (13-17)", "Young adult (18-30)"]:
            assert get_category(raw, "age") == "young", f"Expected young for: {raw}"

    def test_age_grouping_middle(self):
        assert get_category("Middle-aged (31-60)", "age") == "middle"

    def test_age_grouping_older(self):
        assert get_category("Older adult (60+)", "age") == "older"

    def test_age_cannot_determine_returns_none(self):
        assert get_category("Cannot determine", "age") is None

    def test_skin_tone_grouping_light(self):
        assert get_category(1, "skin_tone") == "light"
        assert get_category(2, "skin_tone") == "light"

    def test_skin_tone_grouping_medium(self):
        assert get_category(3, "skin_tone") == "medium"
        assert get_category(4, "skin_tone") == "medium"

    def test_skin_tone_grouping_dark(self):
        assert get_category(5, "skin_tone") == "dark"
        assert get_category(6, "skin_tone") == "dark"

    def test_skin_tone_zero_returns_none(self):
        assert get_category(0, "skin_tone") is None


# ── 3. kl_divergence ─────────────────────────────────────────────────────────

class TestKLDivergence:
    def test_equal_distributions_give_zero(self):
        d = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
        assert kl_divergence(d, d) == pytest.approx(0.0, abs=1e-9)

    def test_completely_skewed_observed_is_positive_and_finite(self):
        # KL(observed ‖ desired): only A is observed → A contributes log(3), B/C contribute 0
        desired  = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
        observed = {"A": 1.0, "B": 0.0, "C": 0.0}
        kl = kl_divergence(observed, desired)
        assert math.isfinite(kl)
        assert kl > 0
        assert kl == pytest.approx(math.log(3), rel=1e-6)  # 1.0 * log(1.0 / (1/3))

    def test_known_value(self):
        # KL(D_ri ‖ D*) with D_ri={A:0.75, B:0.25}, D*={A:0.5, B:0.5}
        # = 0.75*log(0.75/0.5) + 0.25*log(0.25/0.5)
        desired  = {"A": 0.5, "B": 0.5}
        observed = {"A": 0.75, "B": 0.25}
        expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
        assert kl_divergence(observed, desired) == pytest.approx(expected, rel=1e-6)

    def test_missing_category_in_observed_contributes_zero(self):
        # With KL(observed ‖ desired): observed[C]=0 → term is 0, no penalty
        desired  = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
        observed = {"A": 0.5, "B": 0.5, "C": 0.0}
        kl = kl_divergence(observed, desired)
        expected = 0.5 * math.log(0.5 / (1/3)) + 0.5 * math.log(0.5 / (1/3))
        assert kl == pytest.approx(expected, rel=1e-6)
        assert math.isfinite(kl)

    def test_symmetry_fails(self):
        # Use three categories so the sum of terms can't commute trivially
        desired  = {"A": 0.6, "B": 0.3, "C": 0.1}
        observed = {"A": 0.2, "B": 0.5, "C": 0.3}
        kl_fwd = kl_divergence(observed, desired)
        kl_rev = kl_divergence(desired, observed)
        assert kl_fwd != pytest.approx(kl_rev)

    def test_verification_example(self):
        # Desired: {male: 0.5, female: 0.5}, Observed at rank 5: {male: 0.2, female: 0.8}
        # KL(observed ‖ desired) = 0.2*log(0.2/0.5) + 0.8*log(0.8/0.5) ≈ 0.193
        desired  = {"male": 0.5, "female": 0.5}
        observed = {"male": 0.2, "female": 0.8}
        expected = 0.2 * math.log(0.2 / 0.5) + 0.8 * math.log(0.8 / 0.5)
        assert kl_divergence(observed, desired) == pytest.approx(expected, rel=1e-6)
        assert 0.18 < expected < 0.21


# ── 4. get_demographic_distribution ──────────────────────────────────────────

class TestGetDemographicDistribution:
    def test_balanced_gender_returns_uniform(self):
        keys = ["img_000", "img_001", "img_002"]
        ann  = _make_annotations(["M.Male", "NB", "M.Female"], ["Child (0-12)"] * 3, [1] * 3)
        dist, n = get_demographic_distribution(keys, ann, "gender")
        assert n == 3
        assert dist == pytest.approx({"M.Male": 1 / 3, "NB": 1 / 3, "M.Female": 1 / 3}, abs=1e-9)

    def test_cannot_determine_excluded(self):
        ann = {
            "a": {"gender": "M.Male",            "age": "Cannot determine", "skin_tone": 1},
            "b": {"gender": "Cannot determine",   "age": "Young adult (18-30)", "skin_tone": 2},
            "c": {"gender": "M.Female",           "age": "Older adult (60+)", "skin_tone": 3},
        }
        dist_g, n_g = get_demographic_distribution(["a", "b", "c"], ann, "gender")
        assert n_g == 2
        assert dist_g["M.Male"]   == pytest.approx(0.5)
        assert dist_g["M.Female"] == pytest.approx(0.5)
        assert dist_g["NB"]       == pytest.approx(0.0)

        dist_a, n_a = get_demographic_distribution(["a", "b", "c"], ann, "age")
        assert n_a == 2

    def test_skin_tone_zero_excluded(self):
        ann = {
            "x": {"gender": "M.Male", "age": "Young adult (18-30)", "skin_tone": 0},
            "y": {"gender": "M.Male", "age": "Young adult (18-30)", "skin_tone": 3},
            "z": {"gender": "M.Male", "age": "Young adult (18-30)", "skin_tone": "?"},
        }
        dist, n = get_demographic_distribution(["x", "y", "z"], ann, "skin_tone")
        assert n == 1
        assert dist["medium"] == pytest.approx(1.0)
        assert dist["light"]  == pytest.approx(0.0)
        assert dist["dark"]   == pytest.approx(0.0)

    def test_all_invalid_returns_none(self):
        ann = {
            "a": {"gender": "Cannot determine", "age": "Cannot determine", "skin_tone": 0},
        }
        dist, n = get_demographic_distribution(["a"], ann, "gender")
        assert dist is None
        assert n == 0

    def test_empty_list_returns_none(self):
        dist, n = get_demographic_distribution([], BALANCED_ANNOTATIONS, "gender")
        assert dist is None
        assert n == 0

    def test_age_grouping_applied(self):
        ann = {
            "a": {"gender": "M.Male", "age": "Child (0-12)",        "skin_tone": 1},
            "b": {"gender": "M.Male", "age": "Adolescent (13-17)",  "skin_tone": 1},
            "c": {"gender": "M.Male", "age": "Young adult (18-30)", "skin_tone": 1},
        }
        dist, n = get_demographic_distribution(["a", "b", "c"], ann, "age")
        assert n == 3
        assert dist["young"]  == pytest.approx(1.0)
        assert dist["middle"] == pytest.approx(0.0)
        assert dist["older"]  == pytest.approx(0.0)

    def test_skin_tone_grouping_applied(self):
        ann = {
            "a": {"gender": "M.Male", "age": "Young adult (18-30)", "skin_tone": 1},
            "b": {"gender": "M.Male", "age": "Young adult (18-30)", "skin_tone": 4},
            "c": {"gender": "M.Male", "age": "Young adult (18-30)", "skin_tone": 6},
        }
        dist, n = get_demographic_distribution(["a", "b", "c"], ann, "skin_tone")
        assert n == 3
        for cat in ("light", "medium", "dark"):
            assert dist[cat] == pytest.approx(1 / 3, abs=1e-9)


# ── 5. calculate_fair_score ───────────────────────────────────────────────────

class TestFAIRScore:
    def test_returns_float_in_zero_one(self):
        score = calculate_fair_score(BALANCED_KEYS, BALANCED_CURATIONS, BALANCED_ANNOTATIONS,
                                     "gender", DESIRED_DISTRIBUTIONS["gender"], k=9)
        assert 0.0 <= score <= 1.0

    def test_all_curated_balanced_near_one(self):
        ann = _make_annotations(
            gender_list  = ["M.Male", "NB", "M.Female"] * 3,
            age_list     = AGE_VALS_RAW * 3,
            skin_list    = SKIN_VALS_FST * 3,
            curated_list = [1] * 9,
        )
        keys      = sorted(ann.keys())
        curations = [1] * 9
        score = calculate_fair_score(keys, curations, ann, "gender",
                                     DESIRED_DISTRIBUTIONS["gender"], k=9)
        assert score > 0.5, f"Expected > 0.5 for balanced curated set, got {score}"

    def test_biased_ranking_lower_than_balanced(self):
        # Biased: 6 M.Male, 1 NB, 2 M.Female — all curated
        biased_ann = {}
        for i in range(9):
            g = "M.Male" if i < 6 else ("NB" if i < 7 else "M.Female")
            biased_ann[f"img_{i:03d}"] = {
                "gender": g, "age": "Young adult (18-30)", "skin_tone": 1, "is_curated": 1,
            }
        biased_keys = [f"img_{i:03d}" for i in range(9)]
        curations   = [1] * 9

        balanced_ann = _make_annotations(
            gender_list  = ["M.Male", "NB", "M.Female"] * 3,
            age_list     = AGE_VALS_RAW * 3,
            skin_list    = SKIN_VALS_FST * 3,
            curated_list = [1] * 9,
        )
        desired_g = DESIRED_DISTRIBUTIONS["gender"]
        score_biased   = calculate_fair_score(biased_keys, curations, biased_ann,   "gender", desired_g, k=9)
        score_balanced = calculate_fair_score(BALANCED_KEYS, curations, balanced_ann, "gender", desired_g, k=9)
        assert score_biased < score_balanced, (
            f"Biased ({score_biased:.4f}) should be < balanced ({score_balanced:.4f})"
        )

    def test_all_distractors_gives_zero(self):
        curations = [0] * 9
        score = calculate_fair_score(BALANCED_KEYS, curations, BALANCED_ANNOTATIONS,
                                     "gender", DESIRED_DISTRIBUTIONS["gender"], k=9)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_mixed_curated_distractor(self):
        # Only curated images should contribute a positive G[i]
        curations_all = [1] * 9
        curations_half = [1, 0, 1, 0, 1, 0, 1, 0, 1]  # 5 curated
        score_all  = calculate_fair_score(BALANCED_KEYS, curations_all,  BALANCED_ANNOTATIONS,
                                          "gender", DESIRED_DISTRIBUTIONS["gender"], k=9)
        score_half = calculate_fair_score(BALANCED_KEYS, curations_half, BALANCED_ANNOTATIONS,
                                          "gender", DESIRED_DISTRIBUTIONS["gender"], k=9)
        assert score_half < score_all

    def test_k_larger_than_ranking_handled_gracefully(self):
        score = calculate_fair_score(BALANCED_KEYS, BALANCED_CURATIONS, BALANCED_ANNOTATIONS,
                                     "gender", DESIRED_DISTRIBUTIONS["gender"], k=1000)
        assert 0.0 <= score <= 1.0

    def test_empty_ranking_returns_zero(self):
        score = calculate_fair_score([], [], BALANCED_ANNOTATIONS,
                                     "gender", DESIRED_DISTRIBUTIONS["gender"], k=100)
        assert score == 0.0

    def test_all_cannot_determine_with_curated_gives_one(self):
        # No valid demographic annotations → KL = 0 at every position → FAIR = 1.0
        ann  = {f"img_{i:03d}": {"gender": "Cannot determine", "is_curated": 1}
                for i in range(9)}
        curations = [1] * 9
        keys = [f"img_{i:03d}" for i in range(9)]
        score = calculate_fair_score(keys, curations, ann, "gender",
                                     DESIRED_DISTRIBUTIONS["gender"], k=9)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_distractor_does_not_update_demographic_counts(self):
        # Ranking: distractor(M.Male), curated(M.Female), curated(M.Male)
        # After rank 1: 0 curated → KL=0, G=0 (no contribution)
        # After rank 2: 1 curated F → all-female prefix, but with 3-category desired,
        #               NB and M.Male are missing → KL = penalty
        # After rank 3: 2 curated {F, M} → NB still missing → KL = penalty
        ann = {
            "d": {"gender": "M.Male",   "age": "Young adult (18-30)", "skin_tone": 1, "is_curated": 0},
            "a": {"gender": "M.Female", "age": "Young adult (18-30)", "skin_tone": 1, "is_curated": 1},
            "b": {"gender": "M.Male",   "age": "Young adult (18-30)", "skin_tone": 1, "is_curated": 1},
        }
        ranking   = ["d", "a", "b"]
        curations = [0, 1, 1]
        desired_g = DESIRED_DISTRIBUTIONS["gender"]

        score = calculate_fair_score(ranking, curations, ann, "gender", desired_g, k=3)
        assert math.isfinite(score)
        assert 0.0 <= score <= 1.0

        # Score for distractor-first should be lower than curated-first with same demographics
        ann_curated_first = {
            "a": {"gender": "M.Female", "age": "Young adult (18-30)", "skin_tone": 1, "is_curated": 1},
            "b": {"gender": "M.Male",   "age": "Young adult (18-30)", "skin_tone": 1, "is_curated": 1},
            "d": {"gender": "M.Male",   "age": "Young adult (18-30)", "skin_tone": 1, "is_curated": 0},
        }
        ranking_curated_first   = ["a", "b", "d"]
        curations_curated_first = [1, 1, 0]
        score_curated_first = calculate_fair_score(
            ranking_curated_first, curations_curated_first,
            ann_curated_first, "gender", desired_g, k=3,
        )
        assert score_curated_first >= score


# ── 6. Edge cases across all axes ─────────────────────────────────────────────

class TestEdgeCases:
    def test_mixed_valid_invalid_annotations(self):
        ann = {
            "a": {"gender": "M.Male",          "age": "Young adult (18-30)", "skin_tone": 0},
            "b": {"gender": "Cannot determine", "age": "Middle-aged (31-60)", "skin_tone": 3},
            "c": {"gender": "M.Female",         "age": "Cannot determine",   "skin_tone": 5},
        }
        for axis in ("gender", "age", "skin_tone"):
            dist, n = get_demographic_distribution(["a", "b", "c"], ann, axis)
            assert n == 2, f"Expected 2 valid for {axis}, got {n}"

    def test_skin_tone_fst_all_six_values(self):
        ann = {f"img_{i}": {"skin_tone": i + 1} for i in range(6)}
        keys = list(ann.keys())
        dist, n = get_demographic_distribution(keys, ann, "skin_tone")
        assert n == 6
        assert dist["light"]  == pytest.approx(2 / 6, abs=1e-9)
        assert dist["medium"] == pytest.approx(2 / 6, abs=1e-9)
        assert dist["dark"]   == pytest.approx(2 / 6, abs=1e-9)

    def test_single_curated_image(self):
        ann = {"img_000": {"gender": "M.Male", "age": "Young adult (18-30)",
                           "skin_tone": 1, "is_curated": 1}}
        for axis in ("gender", "age", "skin_tone"):
            desired = DESIRED_DISTRIBUTIONS[axis]
            fair = calculate_fair_score(["img_000"], [1], ann, axis, desired, k=1)
            assert math.isfinite(fair)

    def test_single_distractor_image(self):
        ann = {"img_000": {"gender": "M.Male", "age": "Young adult (18-30)",
                           "skin_tone": 1, "is_curated": 0}}
        for axis in ("gender", "age", "skin_tone"):
            fair = calculate_fair_score(["img_000"], [0], ann, axis,
                                        DESIRED_DISTRIBUTIONS[axis], k=1)
            assert fair == pytest.approx(0.0, abs=1e-9)

    def test_annotations_missing_for_some_images(self):
        # Only img_000 has annotations; rest are unannotated curated images
        ann  = {"img_000": {"gender": "M.Male", "age": "Young adult (18-30)",
                            "skin_tone": 1, "is_curated": 1}}
        keys      = [f"img_{i:03d}" for i in range(5)]
        curations = [1] * 5
        for axis in ("gender", "age", "skin_tone"):
            fair = calculate_fair_score(keys, curations, ann, axis,
                                        DESIRED_DISTRIBUTIONS[axis], k=5)
            assert math.isfinite(fair)
