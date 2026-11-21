"""Calibrated paraphrase prediction for the demonstration app.

Loads a fine-tuned cross-encoder checkpoint produced in Phase 5 and serves
calibrated scores: sigmoid(logit / T) with T fit on validation predictions
(scripts/calibrate_models.py). The decision threshold is the dev-tuned
threshold recorded by the Phase 5 collector - the demo uses exactly the
same decision rule as the reported experiments. No hardcoded predictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from afripara.features.normalization import clean_text

logger = logging.getLogger("afripara.inference")

EPS = 1e-6


@dataclass
class Prediction:
    probability: float      # calibrated P(paraphrase)
    label: int              # dev-tuned threshold decision
    raw_probability: float  # uncalibrated sigmoid(logit)
    temperature: float
    threshold: float


class ParaphrasePredictor:
    """Serve one fine-tuned checkpoint with its calibration and threshold."""

    def __init__(self, checkpoint_dir: Path, calibration_dir: Optional[Path] = None,
                 results_path: Optional[Path] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        checkpoint_dir = Path(checkpoint_dir)
        if not (checkpoint_dir / "config.json").exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_dir} (config.json missing)")

        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        if getattr(self.tokenizer, "model_max_length", None) is None or self.tokenizer.model_max_length > 10**6:
            self.tokenizer.model_max_length = 128
        self.model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
        self.model.eval()

        run_name = checkpoint_dir.parent.name
        self.temperature: float = 1.0
        if calibration_dir:
            calib_file = Path(calibration_dir) / f"{run_name}.json"
            if calib_file.exists():
                self.temperature = float(json.loads(calib_file.read_text(encoding="utf-8"))["temperature"])
                logger.info("Loaded temperature %.3f from %s", self.temperature, calib_file)
            else:
                logger.warning("No calibration file for %s - scores will be uncalibrated", run_name)

        self.threshold: float = 0.5
        if results_path and Path(results_path).exists():
            results = json.loads(Path(results_path).read_text(encoding="utf-8"))
            entry = results.get("runs", {}).get(run_name, {})
            self.threshold = float(entry.get("threshold", 0.5))

    def predict(self, sentence1: str, sentence2: str, max_length: int = 128) -> Prediction:
        import torch

        inputs = self.tokenizer(
            clean_text(sentence1), clean_text(sentence2),
            truncation=True, max_length=max_length, return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        raw_logit = float(logits[1] - logits[0])  # binary: log-odds of class 1
        raw_prob = 1.0 / (1.0 + np.exp(-raw_logit))
        prob = 1.0 / (1.0 + np.exp(-raw_logit / self.temperature))
        prob = float(np.clip(prob, EPS, 1 - EPS))
        return Prediction(
            probability=prob,
            label=int(prob >= self.threshold),
            raw_probability=float(raw_prob),
            temperature=self.temperature,
            threshold=self.threshold,
        )
