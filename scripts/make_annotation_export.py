#!/usr/bin/env python3
"""Export the Yoruba gold-annotation candidate set (run AFTER Colab MT).

Selects from the constructed silver pairs:
  - ~10 calibration pairs whose inherited PAWS label is known to the
    project (not to the annotator) for the guidelines' qualification check;
  - the self-consistency repeats (~40, shown twice a week apart);
  - the main annotation pool, balanced across inherited labels and
    construction types, up to configs/default.yaml
    construction.gold_annotation_target.yor.

Output: data/processed/annotation/yor_gold_candidates.csv

The annotator never sees the inherited labels; they are kept in a separate
side-file (yor_gold_key.csv) used later for calibration/agreement analysis
and, eventually, for comparing gold vs inherited labels (MT quality study).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.config import load_config, resolve_path  # noqa: E402
from afripara.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("annotation_export")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default=None, help="override: constructed pairs CSV")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config("default")
    processed = resolve_path(cfg, "data_processed")
    target = int(cfg["construction"]["gold_annotation_target"]["yor"])

    pairs_path = Path(args.pairs_csv) if args.pairs_csv else processed / "constructed" / "translate_train_yor.csv"
    if not pairs_path.exists():
        logger.error(
            "Constructed pairs not found at %s. Run the Colab MT job, place outputs in "
            "data/interim/mt/, then: python scripts/build_datasets.py --stage constructed",
            pairs_path,
        )
        return 1
    df = pd.read_csv(pairs_path)

    # Gold pairs are evaluation-only: sample candidates from the dev/test
    # folds of the constructed data so no gold-validated pair can ever be
    # used for training (models must never see gold pairs in training).
    df = df[df.split.isin(["dev", "test"])].reset_index(drop=True)
    if len(df) < target:
        logger.warning("Only %d dev/test constructed pairs available (target %d)", len(df), target)

    # Annotation pool: balanced across inherited labels.
    n_repeat = 40
    n_calibration = 10
    n_main = max(target - n_calibration, target // 2)
    per_class = n_main // 2

    parts = []
    for label in [0, 1]:
        sub = df[df.label == label]
        take = min(per_class, len(sub))
        parts.append(sub.sample(n=take, random_state=cfg["project"]["seed"]))
    pool = pd.concat(parts).reset_index(drop=True)

    calibration = pool.head(n_calibration).copy()
    main_pool = pool.tail(len(pool) - n_calibration).copy()
    repeats = main_pool.head(n_repeat).copy()

    rows = []
    for _, r in calibration.iterrows():
        rows.append({
            "pair_id": f"cal_{r['pair_id']}",
            "sentence1": r.sentence1,
            "sentence2": r.sentence2,
            "language": "yor",
            "english_s1": r.english_s1,
            "english_s2": r.english_s2,
            "source_id": r.pair_id,
            "repeat": False,
        })
    for _, r in main_pool.iterrows():
        rows.append({
            "pair_id": f"ann_{r['pair_id']}",
            "sentence1": r.sentence1,
            "sentence2": r.sentence2,
            "language": "yor",
            "english_s1": "",
            "english_s2": "",
            "source_id": r.pair_id,
            "repeat": bool(r.pair_id in set(repeats.pair_id)),
        })
    out_df = pd.DataFrame(rows)

    out_dir = processed / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "yor_gold_candidates.csv", index=False)

    # Key file (project-only): inherited labels for the exported pairs.
    key = pool[["pair_id", "label", "label_source", "construction", "source_id"]].copy()
    key["export_id"] = [
        ("cal_" if i < n_calibration else "ann_") + pid for i, pid in enumerate(key.pair_id)
    ]
    key.to_csv(out_dir / "yor_gold_key.csv", index=False)

    logger.info(
        "Exported %d candidates (%d calibration + %d main, %d marked repeat) to %s",
        len(out_df), n_calibration, len(main_pool), int(out_df.repeat.sum()), out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
