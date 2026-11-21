#!/usr/bin/env python3
"""Run the TF-IDF + cosine baseline across all evaluation sets (Phase 4).

Protocol per dataset:
  - fit the TF-IDF vectorizer on TRAIN-fold sentences only (anti-leakage);
  - tune the decision threshold on the DEV fold (maximise F1);
  - evaluate on the TEST fold: accuracy/precision/recall/F1/ROC-AUC
    (+ Spearman/Pearson where continuous gold scores exist);
  - report bootstrap 95% CIs for accuracy and F1.

Evaluation sets (all from data/processed/, produced by build_datasets.py):
  - translate-train yor/ibo/hau (constructed; train/dev/test folds)
  - semrel_hau official (paper-faithful; has known sentence leakage) and
    semrel_hau disjoint (component split, leakage-free)
  - afrixnli_binary_{yor,ibo,hau,eng}: zero-shot evaluation of the baseline
    (vectorizer+threshold transferred from the language's translate-train
    train/dev folds) - establishes the lexical-overlap reference on the
    external proxy test.

Outputs: outputs/baseline/results.json and BASELINE_RESULTS.md.

Usage:
    python scripts/run_baseline.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.config import load_config, resolve_path  # noqa: E402
from afripara.evaluation.metrics import (  # noqa: E402
    accuracy_metric,
    bootstrap_ci,
    classification_metrics,
    f1_metric,
    similarity_metrics,
)
from afripara.models.tfidf_baseline import TfidfCosineBaseline  # noqa: E402
from afripara.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("run_baseline")

LANGS = ["yor", "ibo", "hau"]


def eval_translate_train(processed: Path, lang: str, cfg: dict) -> dict:
    d = processed / "constructed" / f"translate_train_{lang}.csv"
    df = pd.read_csv(d)
    train, dev, test = (df[df.split == s] for s in ["train", "dev", "test"])
    model = TfidfCosineBaseline()
    model.fit(train.sentence1.tolist(), train.sentence2.tolist(), dev.sentence1.tolist(), dev.sentence2.tolist(), dev.label.astype(int).tolist())
    scores = model.pair_scores(test.sentence1.tolist(), test.sentence2.tolist())
    y = test.label.astype(int).tolist()
    metrics = classification_metrics(y, scores, threshold=model.threshold)
    acc_pt, acc_lo, acc_hi = bootstrap_ci(y, scores, accuracy_metric, seed=cfg["project"]["seed"], threshold=model.threshold)
    f1_pt, f1_lo, f1_hi = bootstrap_ci(y, scores, f1_metric, seed=cfg["project"]["seed"], threshold=model.threshold)
    metrics["accuracy_ci95"] = [round(acc_lo, 4), round(acc_hi, 4)]
    metrics["f1_ci95"] = [round(f1_lo, 4), round(f1_hi, 4)]
    return {"test_metrics": metrics, "threshold": model.threshold, "n_train": len(train), "n_dev": len(dev),
            "_predictions": {"scores": scores.tolist(), "labels": y}}


def eval_semrel(processed: Path, variant: str, cfg: dict) -> dict:
    """SemRel Hausa: binary (dev-tuned threshold) + similarity (Spearman/Pearson)."""
    base = processed / f"semrel_hau_{variant}"
    train = pd.read_csv(base / "train.csv")
    dev = pd.read_csv(base / "dev.csv")
    test = pd.read_csv(base / "test.csv")
    model = TfidfCosineBaseline()
    model.fit(train.sentence1.tolist(), train.sentence2.tolist(), dev.sentence1.tolist(), dev.sentence2.tolist(), (dev.score >= 0.5).astype(int).tolist())
    scores = model.pair_scores(test.sentence1.tolist(), test.sentence2.tolist())
    y_binary = (test.score >= 0.5).astype(int).tolist()  # threshold documented; tuned tau on dev is 0.5 by design
    metrics = classification_metrics(y_binary, scores, threshold=model.threshold)
    _, f1_lo, f1_hi = bootstrap_ci(y_binary, scores, f1_metric, seed=cfg["project"]["seed"], threshold=model.threshold)
    _, acc_lo, acc_hi = bootstrap_ci(y_binary, scores, accuracy_metric, seed=cfg["project"]["seed"], threshold=model.threshold)
    metrics["f1_ci95"] = [round(f1_lo, 4), round(f1_hi, 4)]
    metrics["accuracy_ci95"] = [round(acc_lo, 4), round(acc_hi, 4)]
    sim = similarity_metrics(scores, test.score.tolist())  # cosine vs human 0-1 relatedness
    return {"test_metrics": metrics, "similarity_vs_gold": sim, "threshold": model.threshold, "n_train": len(train),
            "_predictions": {"scores": scores.tolist(), "labels": y_binary}}


def eval_afrixnli_zeroshot(processed: Path, raw_dir: Path, lang: str, cfg: dict) -> dict:
    """Zero-shot baseline: vectorizer/threshold transferred, eval on AfriXNLI.

    For yor/ibo/hau the transfer source is the language's translate-train
    train/dev folds. For English it is PAWS-Wiki itself (train fit on a
    deterministic subsample, threshold on validation).
    """
    if lang == "eng":
        from afripara.data.construct import load_paws, select_translate_train

        paws_dev = load_paws(raw_dir, "validation")
        paws_train = load_paws(raw_dir, "train")
        train_fold = select_translate_train(paws_train, n_pairs=3000, seed=cfg["project"]["seed"])
        dev_fold = paws_dev.head(2000)
        source_desc = "PAWS-Wiki train/validation"
    else:
        train = pd.read_csv(processed / "constructed" / f"translate_train_{lang}.csv")
        train_fold = train[train.split == "train"]
        dev_fold = train[train.split == "dev"]
        source_desc = "translate-train folds"
    test = pd.read_csv(processed / "eval" / f"afrixnli_binary_{lang}.csv")
    test = test[test.split == "test"]  # dev kept for transformers' model selection
    model = TfidfCosineBaseline()
    model.fit(train_fold.sentence1.tolist(), train_fold.sentence2.tolist(), dev_fold.sentence1.tolist(), dev_fold.sentence2.tolist(), dev_fold.label.astype(int).tolist())
    scores = model.pair_scores(test.sentence1.tolist(), test.sentence2.tolist())
    y = test.label.astype(int).tolist()
    metrics = classification_metrics(y, scores, threshold=model.threshold)
    return {"test_metrics": metrics, "threshold": model.threshold, "n_test": len(test), "transfer_source": source_desc, "setting": "zero-shot",
            "_predictions": {"scores": scores.tolist(), "labels": y}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="output dir (default outputs/baseline)")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config("default")
    processed = resolve_path(cfg, "data_processed")
    raw_dir = resolve_path(cfg, "data_raw")
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"model": "tfidf_cosine", "seed": cfg["project"]["seed"], "sets": {}}

    for lang in LANGS:
        results["sets"][f"translate_train_{lang}"] = eval_translate_train(processed, lang, cfg)
        logger.info("translate-train %s done", lang)
    for variant in ["official", "disjoint"]:
        results["sets"][f"semrel_hau_{variant}"] = eval_semrel(processed, variant, cfg)
        logger.info("semrel %s done", variant)
    for lang in LANGS + ["eng"]:
        results["sets"][f"afrixnli_binary_{lang}"] = eval_afrixnli_zeroshot(processed, raw_dir, lang, cfg)
        logger.info("afrixnli %s done", lang)

    with (out_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)

    # Persist test-fold predictions (scores) for later McNemar comparison
    # against the transformer runs (Phase 5 collector joins on pair index).
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, r in results["sets"].items():
        preds = r.pop("_predictions", None)
        if preds:
            (pred_dir / f"{name}.json").write_text(
                json.dumps(preds), encoding="utf-8"
            )

    # Markdown table
    lines = [
        "# Baseline Results — TF-IDF + cosine (Phase 4)",
        "",
        "Vectorizer fitted on train folds only; threshold tuned on dev folds; "
        f"test-set metrics with bootstrap 95% CIs (seed {cfg['project']['seed']}).",
        "",
        "| Evaluation set | n_test | Acc | Acc 95% CI | P | R | F1 | F1 95% CI | ROC-AUC | Spearman |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results["sets"].items():
        m = r["test_metrics"]
        acc_ci = r["test_metrics"].get("accuracy_ci95", ["-", "-"])
        f1_ci = r["test_metrics"].get("f1_ci95", ["-", "-"])
        spearman = r.get("similarity_vs_gold", {}).get("spearman", "-")
        lines.append(
            f"| {name} | {m['n']} | {m['accuracy']:.3f} | {acc_ci} | {m['precision']:.3f} "
            f"| {m['recall']:.3f} | {m['f1']:.3f} | {f1_ci} | {m['roc_auc']:.3f} "
            f"| {spearman if isinstance(spearman, str) else round(spearman, 3)} |"
        )
    lines += [
        "",
        f"Decision thresholds: " + ", ".join(f"{k}={v['threshold']:.2f}" for k, v in results["sets"].items()),
        "",
        "Similarity vs gold (SemRel Hausa test, cosine vs human 0-1 scores):",
    ]
    for variant in ["official", "disjoint"]:
        sim = results["sets"][f"semrel_hau_{variant}"]["similarity_vs_gold"]
        lines.append(f"- {variant}: Spearman {sim['spearman']:.3f}, Pearson {sim['pearson']:.3f} (n={sim['n']})")
    lines += [
        "",
        "Note: AfriXNLI rows are zero-shot for this baseline (vectorizer and threshold",
        "transferred from the language's translate-train folds). Transformer runs in Phase 5",
        "will be compared against these numbers with McNemar's exact test.",
        "",
    ]
    (out_dir / "BASELINE_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
