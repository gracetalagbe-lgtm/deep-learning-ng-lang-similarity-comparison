#!/usr/bin/env python3
"""E6 diacritics ablation (paired perturbation, Yoruba).

Manipulates ONE factor - Yoruba tone/subdot marks - on an otherwise
identical evaluation set, so any performance drop is attributable to the
manipulation (causal, within-model). Conditions:

    C0: original pair (both sentences fully marked)
    C1: sentence1 stripped of diacritics
    C2: sentence2 stripped of diacritics
    C3: both sentences stripped

Protocol: the decision threshold is tuned ONCE on the unperturbed dev set
(deployment scenario - one does not retune after text corruption) and held
fixed across conditions. The TF-IDF baseline is evaluated under the same
protocol for contrast. A paired bootstrap gives CIs for each condition's
accuracy delta vs C0.

Eval sets: constructed_test_yor (silver, n=287) and the AfriXNLI Yoruba
test rows (proxy, n=400).

Usage:
    python scripts/e6_perturbation.py [--checkpoint outputs/colab/afroxlmr_yor_constructed_seed42/checkpoint]
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

from afripara.config import get_device  # noqa: E402
from afripara.features.normalization import clean_text, strip_diacritics  # noqa: E402
from afripara.models.tfidf_baseline import TfidfCosineBaseline  # noqa: E402
from afripara.training.data import PairDataset, load_tokenizer_for_training  # noqa: E402
from afripara.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("e6_perturbation")


def tune_threshold(y: np.ndarray, s: np.ndarray) -> float:
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (s >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t


def paired_acc_delta(scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray,
                     thr: float, seed: int = 42, n: int = 2000):
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    N = len(labels)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        da = ((scores_a[idx] >= thr).astype(int) == labels[idx]).mean()
        db = ((scores_b[idx] >= thr).astype(int) == labels[idx]).mean()
        deltas[i] = da - db
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def strip_variants(s1: list[str], s2: list[str]):
    """Return the four diacritic conditions of the same pairs."""
    clean1 = [clean_text(t) for t in s1]
    clean2 = [clean_text(t) for t in s2]
    return {
        "C0_original": (clean1, clean2),
        "C1_s1_stripped": ([strip_diacritics(t) for t in clean1], clean2),
        "C2_s2_stripped": (clean1, [strip_diacritics(t) for t in clean2]),
        "C3_both_stripped": ([strip_diacritics(t) for t in clean1], [strip_diacritics(t) for t in clean2]),
    }


def accuracy(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    return float((((scores >= thr).astype(int)) == labels).mean())


def f1_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    pred = (scores >= thr).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/colab/afroxlmr_yor_constructed_seed42/checkpoint")
    args = parser.parse_args(argv)

    setup_logging()
    root = PROJECT_ROOT
    ckpt = root / args.checkpoint
    if not ckpt.exists():
        logger.error("Checkpoint not found: %s - fetch it from Drive first.", ckpt)
        return 1

    import torch
    from transformers import AutoModelForSequenceClassification

    device = get_device()
    logger.info("Loading checkpoint %s on %s", ckpt, device)
    tokenizer = load_tokenizer_for_training(str(ckpt), 128)
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    model.to(device)
    model.eval()

    def model_scores(s1, s2, batch_size: int = 8) -> np.ndarray:
        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding

        ds = PairDataset(s1, s2, [0] * len(s1), tokenizer, 128)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
        out = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
                out.extend(probs.cpu().float().tolist())
        return np.asarray(out, dtype=float)

    report: dict = {"checkpoint": str(ckpt), "threshold_protocol": "tuned once on unperturbed dev, held fixed"}

    # ---------------- constructed silver test ----------------
    train_df = pd.read_csv(root / "data/processed/constructed/translate_train_yor.csv")
    train = train_df[train_df.split == "train"]
    dev_df = train_df[train_df.split == "dev"]
    test_df = train_df[train_df.split == "test"].reset_index(drop=True)
    labels = test_df.label.astype(int).to_numpy()

    logger.info("Tuning threshold on unperturbed dev...")
    dev_variants = strip_variants(dev_df.sentence1.tolist(), dev_df.sentence2.tolist())
    dev_scores_c0 = model_scores(*dev_variants["C0_original"])
    thr_model = tune_threshold(dev_df.label.astype(int).to_numpy(), dev_scores_c0)
    logger.info("Model threshold (dev, C0): %.2f", thr_model)

    tfidf = TfidfCosineBaseline()
    tfidf.fit(train.sentence1.tolist(), train.sentence2.tolist(),
              dev_df.sentence1.tolist(), dev_df.sentence2.tolist(), dev_df.label.astype(int).tolist())
    thr_tfidf = tfidf.threshold

    variants = strip_variants(test_df.sentence1.tolist(), test_df.sentence2.tolist())
    report["constructed_test_yor"] = {"n": len(test_df), "conditions": {}}
    for name, (s1, s2) in variants.items():
        ms = model_scores(s1, s2)
        bs = tfidf.pair_scores(s1, s2)
        report["constructed_test_yor"]["conditions"][name] = {
            "model_acc": round(accuracy(ms, labels, thr_model), 4),
            "model_f1": round(f1_at(ms, labels, thr_model), 4),
            "tfidf_acc": round(accuracy(bs, labels, thr_tfidf), 4),
            "tfidf_f1": round(f1_at(bs, labels, thr_tfidf), 4),
        }
        if name != "C0_original":
            dm, lo, hi = paired_acc_delta(ms, model_scores(*variants["C0_original"]), labels, thr_model)
            report["constructed_test_yor"]["conditions"][name]["model_acc_delta_vs_C0"] = {
                "mean": round(dm, 4), "ci95": [round(lo, 4), round(hi, 4)]
            }
        logger.info("constructed %s: model acc %.3f | tfidf acc %.3f", name,
                    report["constructed_test_yor"]["conditions"][name]["model_acc"],
                    report["constructed_test_yor"]["conditions"][name]["tfidf_acc"])

    # ---------------- AfriXNLI proxy test ----------------
    axn = pd.read_csv(root / "data/processed/eval/afrixnli_binary_yor.csv")
    axn_test = axn[axn.split == "test"].reset_index(drop=True)
    axn_dev = axn[axn.split == "dev"].reset_index(drop=True)
    labels_axn = axn_test.label.astype(int).to_numpy()

    dev_var_axn = strip_variants(axn_dev.sentence1.tolist(), axn_dev.sentence2.tolist())
    thr_model_axn = tune_threshold(axn_dev.label.astype(int).to_numpy(), model_scores(*dev_var_axn["C0_original"]))
    logger.info("AfriXNLI model threshold (dev, C0): %.2f", thr_model_axn)

    variants_axn = strip_variants(axn_test.sentence1.tolist(), axn_test.sentence2.tolist())
    report["afrixnli_test_yor"] = {"n": len(axn_test), "conditions": {}}
    c0_scores = None
    for name, (s1, s2) in variants_axn.items():
        ms = model_scores(s1, s2)
        row = {"model_acc": round(accuracy(ms, labels_axn, thr_model_axn), 4),
               "model_f1": round(f1_at(ms, labels_axn, thr_model_axn), 4)}
        if name == "C0_original":
            c0_scores = ms
        else:
            dm, lo, hi = paired_acc_delta(ms, c0_scores, labels_axn, thr_model_axn)
            row["model_acc_delta_vs_C0"] = {"mean": round(dm, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        report["afrixnli_test_yor"]["conditions"][name] = row
        logger.info("afrixnli %s: model acc %.3f", name, row["model_acc"])

    out = root / "outputs/phase6" / "e6_diacritics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("E6 diacritics report written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
