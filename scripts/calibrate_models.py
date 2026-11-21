#!/usr/bin/env python3
"""Fit temperature scaling for the demo's similarity scores (Phase 7).

Guo et al. (2017): a single temperature T divides the logit before the
sigmoid; it changes confidence, never the argmax, so accuracy and all
rank metrics are unchanged. T is fit on VALIDATION (dev) predictions -
never on test data - by minimising binary NLL.

Inputs:  outputs/colab/<run>/predictions_dev.json  (scores, labels)
Outputs: outputs/calibration/<run>.json  {temperature, dev_nll, dev_ece_bins}

Usage:
    python scripts/calibrate_models.py            # calibrate all completed runs
    python scripts/calibrate_models.py --run afroxlmr_hau_semrel_seed42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("calibrate")

EPS = 1e-6


def probs_to_logits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def nll_for_temperature(logit_t: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """NLL of sigmoid(logits / exp(logit_T)) - parameterised in log space so
    the optimiser cannot propose T <= 0."""
    T = np.exp(logit_t)
    z = logits / T
    # numerical-stable binary NLL: softplus(z) - label * z
    loss = np.logaddexp(0, z) - labels * z
    return float(loss.mean())


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> dict:
    p = np.clip(np.asarray(scores, dtype=float), EPS, 1 - EPS)
    y = np.asarray(labels, dtype=float)
    logits = probs_to_logits(p)
    result = minimize_scalar(nll_for_temperature, bounds=(-3.0, 3.0), method="bounded",
                             args=(logits, y))
    T = float(np.exp(result.x))
    nll_raw = nll_for_temperature(0.0, logits, y)          # T = 1
    nll_fit = nll_for_temperature(result.x, logits, y)
    return {
        "temperature": round(T, 4),
        "dev_nll_before": round(nll_raw, 4),
        "dev_nll_after": round(nll_fit, 4),
        "n_dev": int(len(y)),
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    conf = np.clip(probs, 0, 1)
    pred = (conf >= 0.5).astype(int)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= 1)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(conf[mask].mean() - labels[mask].mean())
    return float(ece)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="calibrate one run (default: all with dev predictions)")
    args = parser.parse_args(argv)

    setup_logging()
    colab = PROJECT_ROOT / "outputs" / "colab"
    out_dir = PROJECT_ROOT / "outputs" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(p for p in colab.iterdir() if p.is_dir() and (p / "predictions_dev.json").exists())
    if args.run:
        run_dirs = [p for p in run_dirs if p.name == args.run]
        if not run_dirs:
            logger.error("Run not found or has no dev predictions: %s", args.run)
            return 1

    for run_dir in run_dirs:
        d = json.loads((run_dir / "predictions_dev.json").read_text(encoding="utf-8"))
        fit = fit_temperature(np.asarray(d["scores"]), np.asarray(d["labels"]))
        # ECE before/after as a sanity readout (not used for fitting)
        p = np.clip(np.asarray(d["scores"], dtype=float), EPS, 1 - EPS)
        y = np.asarray(d["labels"], dtype=float)
        fit["ece_before"] = round(expected_calibration_error(p, y), 4)
        calibrated = 1 / (1 + np.exp(-probs_to_logits(p) / fit["temperature"]))
        fit["ece_after"] = round(expected_calibration_error(calibrated, y), 4)
        out = out_dir / f"{run_dir.name}.json"
        out.write_text(json.dumps(fit, indent=2), encoding="utf-8")
        logger.info("%s: T=%.3f | dev NLL %.4f -> %.4f | ECE %.4f -> %.4f",
                    run_dir.name, fit["temperature"], fit["dev_nll_before"],
                    fit["dev_nll_after"], fit["ece_before"], fit["ece_after"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
