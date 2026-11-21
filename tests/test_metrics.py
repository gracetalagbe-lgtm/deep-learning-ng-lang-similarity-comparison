"""Tests for the metrics/significance library."""

from __future__ import annotations

import numpy as np
import pytest

from afripara.evaluation.metrics import (
    accuracy_metric,
    bootstrap_ci,
    classification_metrics,
    f1_metric,
    mcnemar_exact,
    paired_bootstrap,
    similarity_metrics,
)


def test_classification_metrics_perfect_predictions():
    y = [0, 1, 0, 1]
    score = [0.1, 0.9, 0.2, 0.8]
    m = classification_metrics(y, score, threshold=0.5)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_classification_metrics_known_values():
    y = [1, 1, 0, 0, 1, 0]
    score = [0.9, 0.4, 0.8, 0.1, 0.3, 0.2]  # threshold 0.5 -> pred [1,0,1,0,0,0]
    m = classification_metrics(y, score, threshold=0.5)
    # tp=1 (first), fp=1 (third), fn=2 (second, fifth)
    assert m["precision"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(2 * (0.5 * (1 / 3)) / (0.5 + 1 / 3))


def test_classification_metrics_single_class_roc():
    m = classification_metrics([1, 1, 1], [0.2, 0.3, 0.4])
    assert np.isnan(m["roc_auc"])


def test_similarity_metrics_direction_and_bounds():
    gold = [0.1, 0.2, 0.3, 0.9]
    perfect = [0.1, 0.2, 0.3, 0.9]
    inverted = [0.9, 0.3, 0.2, 0.1]
    assert similarity_metrics(gold, perfect)["spearman"] == pytest.approx(1.0)
    assert similarity_metrics(gold, inverted)["spearman"] == pytest.approx(-1.0)
    s = similarity_metrics([0.5], [0.5])
    assert np.isnan(s["spearman"])  # degenerate input -> nan, not crash


def test_mcnemar_exact_known_case():
    # A right on 10 that B misses; B right on 2 that A misses -> strong effect
    y = [1] * 12
    pred_a = [1] * 10 + [0, 0]  # correct on 10, wrong on 2
    pred_b = [0] * 10 + [0, 1]  # wrong on 10, correct on 1... adjust below
    # make b=10 discordant (A right, B wrong), c=1 (B right, A wrong)
    y = [1] * 11
    pred_a = [1] * 10 + [0]
    pred_b = [0] * 10 + [1]
    res = mcnemar_exact(y, pred_a, pred_b)
    assert res["b"] == 10 and res["c"] == 1
    assert res["p_value"] < 0.05
    # balanced discordance -> p = 1
    y2 = [1, 1]
    res2 = mcnemar_exact(y2, [1, 0], [0, 1])
    assert res2["p_value"] == pytest.approx(1.0)


def test_mcnemar_no_discordant():
    y = [1, 0, 1]
    res = mcnemar_exact(y, [1, 0, 1], [1, 0, 1])
    assert res["n_discordant"] == 0
    assert res["p_value"] == pytest.approx(1.0)


def test_paired_bootstrap_detects_better_system():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    score_better = np.clip(y * 0.9 + rng.normal(0, 0.2, n), 0, 1)
    score_worse = np.clip(rng.normal(0.5, 0.3, n), 0, 1)
    res = paired_bootstrap(
        y, score_better, score_worse, f1_metric, n_resamples=300, seed=42
    )
    assert res["mean_delta"] > 0
    assert res["p_value"] < 0.05


def test_paired_bootstrap_deterministic_with_seed():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=100)
    a = rng.random(100)
    b = rng.random(100)
    r1 = paired_bootstrap(y, a, b, accuracy_metric, n_resamples=100, seed=7)
    r2 = paired_bootstrap(y, a, b, accuracy_metric, n_resamples=100, seed=7)
    assert r1 == r2


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=300)
    score = np.clip(y * 0.8 + rng.normal(0, 0.25, 300), 0, 1)
    point, lo, hi = bootstrap_ci(y, score, f1_metric, n_resamples=200, seed=3)
    assert lo <= point <= hi
