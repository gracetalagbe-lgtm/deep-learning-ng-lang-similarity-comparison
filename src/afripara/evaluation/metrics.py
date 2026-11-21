"""Evaluation metrics and significance tests.

Classification: accuracy, precision, recall, F1 (positive class), ROC-AUC.
Similarity: Spearman rho (primary for this project - SemRel scores are
Best-Worst-Scaled and rank-based), Pearson r.

Significance testing practice (feasibility report section 8):
  - McNemar's EXACT test for comparing two classifiers on one test set
    (Dietterich 1998; Dror et al. 2018);
  - paired bootstrap on a metric (Berg-Kirkpatrick et al. 2012; Koehn 2004).

All functions take plain numpy arrays so they are model-agnostic.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

DEFAULT_BOOTSTRAP_RESAMPLES = 10000


def classification_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
    zero_division: int = 0,
) -> Dict[str, float]:
    """Binary classification metrics from continuous scores and a threshold."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred, average="binary", zero_division=zero_division
    )
    metrics = {
        "n": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    # ROC-AUC is threshold-free but undefined if only one class is present.
    if len(np.unique(y_true_arr)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def similarity_metrics(y_score_a: Sequence[float], y_score_b: Sequence[float]) -> Dict[str, float]:
    """Spearman rho and Pearson r between two score vectors (e.g. model
    similarity vs gold relatedness)."""
    a = np.asarray(y_score_a, dtype=float)
    b = np.asarray(y_score_b, dtype=float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return {"spearman": float("nan"), "pearson": float("nan"), "n": int(len(a))}
    rho = spearmanr(a, b)
    r = pearsonr(a, b)
    return {
        "n": int(len(a)),
        "spearman": float(rho.statistic if hasattr(rho, "statistic") else rho.correlation),
        "pearson": float(r.statistic if hasattr(r, "statistic") else r.correlation),
    }


def mcnemar_exact(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
) -> Dict[str, float]:
    """Exact McNemar test on discordant pairs.

    b = #examples A gets right and B gets wrong; c = vice versa. Under H0
    (equal accuracy) b ~ Binomial(b+c, 0.5); the test is two-sided
    (Dietterich 1998, "Approximate Statistical Tests...", Neural Comp. 10(7)).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    a_ok = np.asarray(pred_a, dtype=int) == y_true_arr
    b_ok = np.asarray(pred_b, dtype=int) == y_true_arr
    b_count = int(np.sum(a_ok & ~b_ok))
    c_count = int(np.sum(~a_ok & b_ok))
    n = b_count + c_count
    if n == 0:
        return {"b": b_count, "c": c_count, "p_value": 1.0, "n_discordant": 0}
    p_value = binomtest(min(b_count, c_count), n, 0.5).pvalue
    return {
        "b": b_count,
        "c": c_count,
        "n_discordant": n,
        "p_value": float(p_value),
    }


def paired_bootstrap(
    y_true: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
) -> Dict[str, float]:
    """Paired bootstrap for the difference of a metric between two systems.

    Resamples the test set with replacement; reports mean delta, 95% CI,
    and a two-sided p-value approximated as 2 x min proportion of resamples
    on the wrong side of zero (Berg-Kirkpatrick et al. 2012 practice).
    """
    y_arr = np.asarray(y_true, dtype=int)
    a_arr = np.asarray(score_a, dtype=float)
    b_arr = np.asarray(score_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_arr)
    deltas = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ys, as_, bs = y_arr[idx], a_arr[idx], b_arr[idx]
        pred_a = (as_ >= threshold_a).astype(int)
        pred_b = (bs >= threshold_b).astype(int)
        deltas[i] = metric_fn(ys, pred_a) - metric_fn(ys, pred_b)
    mean_delta = float(np.mean(deltas))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_left = float(np.mean(deltas <= 0))
    p_right = float(np.mean(deltas >= 0))
    return {
        "mean_delta": mean_delta,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value": float(2 * min(p_left, p_right)),
        "n_resamples": int(n_resamples),
    }


def bootstrap_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    threshold: float = 0.5,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """CI for a single metric (default: F1 with a decision threshold)."""
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_arr)
    values = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        values[i] = metric_fn(y_arr[idx], (s_arr[idx] >= threshold).astype(int))
    point = float(metric_fn(y_arr, (s_arr >= threshold).astype(int)))
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(values, [100 * alpha, 100 * (1 - alpha)])
    return point, float(lo), float(hi)


# Common metric functions for use with the bootstrap helpers.
def f1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(f1)


def accuracy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))
