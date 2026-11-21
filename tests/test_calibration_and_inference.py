"""Tests for demo calibration and predictor logic (no model downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.calibrate_models import (
    expected_calibration_error,
    fit_temperature,
    probs_to_logits,
)
from afripara.inference.predictor import Prediction


def test_probs_to_logits_roundtrip():
    p = np.array([0.1, 0.5, 0.9])
    z = probs_to_logits(p)
    recovered = 1 / (1 + np.exp(-z))
    assert np.allclose(recovered, p, atol=1e-6)


def test_fit_temperature_reduces_nll_for_overconfident_model():
    """Overconfident = ~90% accurate but ~0.97/0.03 extreme confidence."""
    rng = np.random.default_rng(0)
    n = 500
    labels = rng.integers(0, 2, n)
    correct = rng.random(n) < 0.90
    good_score = np.where(labels == 1, 0.97, 0.03)
    bad_score = np.where(labels == 1, 0.03, 0.97)
    scores = np.where(correct, good_score, bad_score)
    fit = fit_temperature(scores, labels.astype(float))
    assert fit["temperature"] > 1.0          # overconfidence -> soften
    assert fit["dev_nll_after"] < fit["dev_nll_before"]


def test_temperature_one_when_already_calibrated():
    """Genuinely calibrated data: labels drawn as Bernoulli(p) with p = score."""
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.05, 0.95, 4000)
    labels = (rng.random(4000) < scores).astype(float)
    fit = fit_temperature(scores, labels)
    assert 0.8 < fit["temperature"] < 1.3    # near identity
    assert fit["dev_nll_after"] <= fit["dev_nll_before"] + 1e-6


def test_ece_bounds_and_underconfident_case():
    """probs 0.9 with labels all 1 (and 0.1 with labels all 0): each bin is
    10% underconfident -> ECE 0.1, not 0."""
    labels = np.array([1.0] * 50 + [0.0] * 50)
    probs = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])
    assert expected_calibration_error(probs, labels) == pytest.approx(0.1, abs=1e-9)
    # all predictions at 0.7 but labels 50/50 -> ECE = |0.7 - 0.5| = 0.2
    probs2 = np.full(100, 0.7)
    assert expected_calibration_error(probs2, labels) == pytest.approx(0.2, abs=1e-9)


def test_prediction_dataclass_threshold_decision():
    p_hi = Prediction(probability=0.8, label=1, raw_probability=0.9, temperature=1.5, threshold=0.5)
    p_lo = Prediction(probability=0.3, label=0, raw_probability=0.2, temperature=1.5, threshold=0.5)
    assert p_hi.label == 1 and p_lo.label == 0
    assert p_hi.probability >= p_lo.probability
