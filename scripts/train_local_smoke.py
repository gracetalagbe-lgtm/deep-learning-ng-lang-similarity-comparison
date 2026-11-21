#!/usr/bin/env python3
"""Local CPU smoke run of the full training pipeline (Phase 5 validation).

Trains the small cached AfriBERTa checkpoint (83M params, engineering model
- NOT an experiment) on a small slice of real Yoruba constructed pairs for a
single epoch on CPU, then scores predictions with the Phase 4 metrics
library. Purpose: prove the training/prediction/metrics chain works before
spending Colab GPU hours. Produces NO research result.

Usage:
    python scripts/train_local_smoke.py            # ~5-10 min on this Mac
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.config import load_config  # noqa: E402
from afripara.evaluation.metrics import classification_metrics  # noqa: E402
from afripara.training.data import dataframe_to_arrays  # noqa: E402
from afripara.training.trainer import predict_scores, train_cross_encoder  # noqa: E402
from afripara.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("train_local_smoke")

SMOKE_MODEL = "castorini/afriberta_small"  # cached locally; engineering only


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-dev", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config("default")
    processed = PROJECT_ROOT / "data" / "processed"

    df = pd.read_csv(processed / "constructed" / "translate_train_yor.csv")
    train = df[df.split == "train"].head(args.n_train)
    dev = df[df.split == "dev"].head(args.n_dev)
    tr_s1, tr_s2, tr_y = dataframe_to_arrays(train)
    dv_s1, dv_s2, dv_y = dataframe_to_arrays(dev)
    logger.info("Smoke data: %d train / %d dev pairs (real Yoruba constructed pairs)", len(tr_y), len(dv_y))

    out_dir = PROJECT_ROOT / "outputs" / "local_smoke"
    t0 = time.time()
    model, tokenizer, history = train_cross_encoder(
        model_config_name="afroxlmr_base",  # config provenance echo; repo overridden below
        train_s1=tr_s1, train_s2=tr_s2, train_y=tr_y,
        dev_s1=dv_s1, dev_s2=dv_s2, dev_y=dv_y,
        output_dir=out_dir,
        seed=cfg["project"]["seed"],
        training_overrides={
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
        },
        model_repo_override=SMOKE_MODEL,  # engineering-only override (cached, small)
    )
    logger.info("Training stage finished in %.1fs (pipeline validation only)", time.time() - t0)

    scores = predict_scores(model, tokenizer, dv_s1, dv_s2, max_length=cfg["training"]["max_seq_length"], batch_size=8)
    metrics = classification_metrics(dv_y, scores, threshold=0.5)
    payload = {
        "purpose": "engineering smoke run - NOT a research result",
        "model": SMOKE_MODEL,
        "n_train": len(tr_y),
        "n_dev": len(dv_y),
        "dev_metrics": metrics,
        "training_wallclock_s": round(time.time() - t0, 1),
    }
    (out_dir / "smoke_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Smoke metrics (meaningless scientifically): %s", json.dumps(metrics))
    logger.info("=== RESULT: pipeline PASS (see %s) ===", out_dir / "smoke_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
