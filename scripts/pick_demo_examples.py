#!/usr/bin/env python3
"""Pick demo example pairs from real held-out data (Phase 7).

For each language, selects from the evaluation predictions:
  - one correctly-classified paraphrase pair with high calibrated confidence;
  - one correctly-classified non-paraphrase pair with high confidence.
Examples are REAL evaluation items (never training items, never invented),
with their gold labels and the model's verified behaviour - so the demo's
example buttons show pairs whose verdicts are already validated.

Output: src/afripara/app/demo_examples.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from afripara.utils.logging_utils import setup_logging  # noqa: E402
from run_analysis import _pair_token_overlap  # noqa: E402

logger = logging.getLogger("pick_examples")

LANGS = ["yor", "ibo", "hau"]

# Where each language's demo examples come from + which run scored them.
SOURCES = {
    "yor": {
        "csv": "data/processed/constructed/translate_train_yor.csv",
        "filter_col": "split", "filter_val": "test",
        "pred": "outputs/colab/afroxlmr_yor_constructed_seed42/predictions_constructed_test_yor.json",
        "demo_name": "Yoruba",
        "source_note": "constructed silver test pair (label inherited from PAWS via MT)",
    },
    "ibo": {
        "csv": "data/processed/constructed/translate_train_ibo.csv",
        "filter_col": "split", "filter_val": "test",
        "pred": "outputs/colab/afroxlmr_ibo_constructed_seed42/predictions_constructed_test_ibo.json",
        "demo_name": "Igbo",
        "source_note": "constructed silver test pair (label inherited from PAWS via MT)",
    },
    "hau": {
        "csv": "data/processed/semrel_hau_disjoint/test.csv",
        "filter_col": None, "filter_val": None,
        "pred": "outputs/colab/afroxlmr_hau_semrel_seed42/predictions_semrel_disjoint_test.json",
        "demo_name": "Hausa",
        "source_note": "SemRel2024-Hausa test pair (human-annotated 0-1 relatedness)",
    },
}


def _has_mojibake(text: str) -> bool:
    """Detect Latin-1 double-encoding artifacts (valid UTF-8, wrong chars).

    Found in one Igbo test sentence during Phase 7 (Lăpuş -> LÄ puÅ); the
    audit's strict-UTF-8 check cannot catch these because the bytes ARE
    valid UTF-8.
    """
    return any(seq in text for seq in ["Ä", "Å", "Ã", "â€"])


def main() -> int:
    setup_logging()
    out: dict = {}
    for lang, cfg in SOURCES.items():
        df = pd.read_csv(PROJECT_ROOT / cfg["csv"])
        if cfg["filter_col"]:
            df = df[df[cfg["filter_col"]] == cfg["filter_val"]]
        df = df.reset_index(drop=True)
        pred = json.loads((PROJECT_ROOT / cfg["pred"]).read_text())
        scores = pred["scores"]
        if "label" in df.columns:
            labels = df.label.astype(int).tolist()
        else:
            labels = (df.score >= 0.5).astype(int).tolist()
        thr = 0.5
        clean = df.sentence1.map(_has_mojibake) | df.sentence2.map(_has_mojibake)
        overlap_ok = np.array([
            _pair_token_overlap(a, b) < 0.95  # avoid visually-identical demo pairs
            for a, b in zip(df.sentence1, df.sentence2)
        ]) if hasattr(__import__("run_analysis"), "_pair_token_overlap") else np.ones(len(df), bool)
        eligible = [
            i for i in range(len(labels))
            if not clean[i] and overlap_ok[i] and int(scores[i] >= thr) == labels[i]
        ]
        pos = max((i for i in eligible if labels[i] == 1), key=lambda i: abs(scores[i] - 0.5), default=None)
        neg = max((i for i in eligible if labels[i] == 0), key=lambda i: abs(scores[i] - 0.5), default=None)
        out[cfg["demo_name"]] = {
            "source_note": cfg["source_note"],
            "pairs": [
                {
                    "sentence1": df.sentence1[i], "sentence2": df.sentence2[i],
                    "gold": labels[i], "score": round(scores[i], 3),
                }
                for i in (pos, neg) if i is not None
            ],
        }
        logger.info("%s: picked %d verified examples", lang, len(out[cfg["demo_name"]]["pairs"]))
    dest = PROJECT_ROOT / "src" / "afripara" / "app" / "demo_examples.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Demo examples written to %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
