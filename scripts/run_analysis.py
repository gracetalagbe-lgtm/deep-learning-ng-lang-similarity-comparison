#!/usr/bin/env python3
"""Phase 6 analysis: observational linguistic analyses, error analysis and
figures - computed from the prediction files already produced in Phase 5.

Requires NO model inference (that is scripts/e6_perturbation.py, which uses
a downloaded checkpoint). Analyses:

  A. Lexical-overlap buckets  - baseline vs transformer divergence as word
     overlap rises (the PAWS-style test for semantic vs lexical processing).
  B. Negator-word buckets     - accuracy on pairs where exactly one sentence
     contains a negation marker (heuristic word lists; observational, not a
     perturbation).
  C. Diacritic-density buckets - accuracy on fully-marked vs unmarked text
     (observational; the Menyo/IGBONLP mixed-diacritisation finding).
  D. Length effects           - accuracy by character-length quartile.
  E. Error analysis           - high-confidence errors, confusion counts,
     and readable examples for the thesis discussion.

Outputs: outputs/phase6/ANALYSIS_REPORT.md, outputs/phase6/figures/*.png,
         outputs/phase6/analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.config import load_config  # noqa: E402
from afripara.features.normalization import clean_text, diacritic_density, has_diacritics  # noqa: E402
from afripara.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("phase6_analysis")

LANGS = ["yor", "ibo", "hau"]

# Conservative negation markers (heuristic detection; documented as such).
NEGATORS = {
    "yor": ["kò", "kì í", "má", "kí ò"],
    "ibo": [" anaghị", " adịghị", " bụghị", " ekweghi"],
    "hau": ["bai ", "ba a ", "ba ta ", "ba su ", "kada", "kar ", "ba su da"],
}

SEED = 42
BOOTSTRAP_N = 2000


def _pair_token_overlap(s1: str, s2: str) -> float:
    """Jaccard word overlap after normalisation (diacritics preserved)."""
    t1 = set(clean_text(s1).lower().split())
    t2 = set(clean_text(s2).lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def load_scores(phase5: Path, run_family: str, eval_set: str, seed: int):
    """Load one run's prediction file."""
    p = phase5 / f"{run_family}_seed{seed}" / f"predictions_{eval_set}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return np.asarray(d["scores"], dtype=float), np.asarray(d["labels"], dtype=int)


def bucket_accuracy(scores, labels, bucket_values, n_buckets: int = 4):
    """Accuracy per quantile bucket of ``bucket_values``."""
    qs = np.quantile(bucket_values, np.linspace(0, 1, n_buckets + 1))
    qs[-1] += 1e-9
    out = []
    for i in range(n_buckets):
        mask = (bucket_values >= qs[i]) & (bucket_values < qs[i + 1])
        if mask.sum() == 0:
            out.append({"bucket": f"Q{i+1}", "n": 0, "acc": None, "range": (float(qs[i]), float(qs[i+1]))})
            continue
        acc = float((((scores[mask] >= 0.5).astype(int)) == labels[mask]).mean())
        out.append({"bucket": f"Q{i+1}", "n": int(mask.sum()), "acc": round(acc, 4),
                    "range": (round(float(qs[i]), 3), round(float(qs[i+1]), 3))})
    return out


def bootstrap_acc_delta(a_scores, a_labels, b_scores, b_labels, seed=SEED, n=BOOTSTRAP_N):
    """CI for accuracy(a) - accuracy(b) on paired items."""
    rng = np.random.default_rng(seed)
    n_items = len(a_labels)
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_items, size=n_items)
        da = ((a_scores[idx] >= 0.5).astype(int) == a_labels[idx]).mean()
        db = ((b_scores[idx] >= 0.5).astype(int) == b_labels[idx]).mean()
        deltas[i] = da - db
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def analysis_for_language(lang: str, processed: Path, phase5: Path, baseline_pred: Path, out_dir: Path) -> dict:
    eval_file = processed / "constructed" / f"translate_train_{lang}.csv"
    df = pd.read_csv(eval_file)
    test = df[df.split == "test"].reset_index(drop=True)
    result: dict = {"language": lang, "n_test": len(test)}

    model_scores = load_scores(phase5, f"afroxlmr_{lang}_constructed", f"constructed_test_{lang}", SEED)
    base_file = baseline_pred / f"translate_train_{lang}.json"
    base = json.loads(base_file.read_text())
    base_scores, labels = np.asarray(base["scores"], dtype=float), np.asarray(base["labels"], dtype=int)
    if model_scores is None:
        logger.warning("No model predictions for %s - skipping", lang)
        return result
    m_scores, m_labels = model_scores
    if len(m_scores) != len(test):
        logger.warning("Length mismatch for %s (%d vs %d)", lang, len(m_scores), len(test))
        return result

    # --- A. lexical-overlap buckets ---
    overlap = np.asarray([_pair_token_overlap(a, b) for a, b in zip(test.sentence1, test.sentence2)])
    result["overlap_buckets"] = {
        "model": bucket_accuracy(m_scores, m_labels, overlap),
        "baseline": bucket_accuracy(base_scores, labels, overlap),
        "model_mean_acc_high_overlap": float(
            (((m_scores[overlap >= 0.5] >= 0.5).astype(int)) == m_labels[overlap >= 0.5]).mean()
        ) if (overlap >= 0.5).sum() else None,
        "baseline_mean_acc_high_overlap": float(
            (((base_scores[overlap >= 0.5] >= 0.5).astype(int)) == labels[overlap >= 0.5]).mean()
        ) if (overlap >= 0.5).sum() else None,
    }
    mean_delta, lo, hi = bootstrap_acc_delta(m_scores, m_labels, base_scores, labels)
    result["overall_acc_delta_model_vs_baseline"] = {"mean": round(mean_delta, 4), "ci95": [round(lo, 4), round(hi, 4)]}

    # --- B. negator-word buckets (heuristic marker lists) ---
    negs = NEGATORS[lang]

    def has_negator(text: str) -> bool:
        t = clean_text(text)
        return any(n in t for n in negs)

    one_side = np.asarray(
        [has_negator(a) != has_negator(b) for a, b in zip(test.sentence1, test.sentence2)]
    )
    both = np.asarray([has_negator(a) and has_negator(b) for a, b in zip(test.sentence1, test.sentence2)])
    result["negation"] = {
        "markers": negs,
        "n_one_side": int(one_side.sum()),
        "n_both": int(both.sum()),
        "model_acc_one_side_negation": float(
            (((m_scores[one_side] >= 0.5).astype(int)) == m_labels[one_side]).mean()
        ) if one_side.sum() else None,
        "baseline_acc_one_side_negation": float(
            (((base_scores[one_side] >= 0.5).astype(int)) == labels[one_side]).mean()
        ) if one_side.sum() else None,
        "model_acc_overall": float((((m_scores >= 0.5).astype(int)) == m_labels).mean()),
    }

    # --- C. diacritic-density buckets ---
    density = np.asarray(
        [(diacritic_density(clean_text(a)) + diacritic_density(clean_text(b))) / 2 for a, b in zip(test.sentence1, test.sentence2)]
    )
    marked = density > 0.05
    result["diacritics_observational"] = {
        "n_marked_pairs": int(marked.sum()),
        "n_unmarked_pairs": int((~marked).sum()),
        "model_acc_marked": float((((m_scores[marked] >= 0.5).astype(int)) == m_labels[marked]).mean()) if marked.sum() else None,
        "model_acc_unmarked": float((((m_scores[~marked] >= 0.5).astype(int)) == m_labels[~marked]).mean()) if (~marked).sum() else None,
        "baseline_acc_marked": float((((base_scores[marked] >= 0.5).astype(int)) == labels[marked]).mean()) if marked.sum() else None,
        "baseline_acc_unmarked": float((((base_scores[~marked] >= 0.5).astype(int)) == labels[~marked]).mean()) if (~marked).sum() else None,
    }

    # --- D. length effects ---
    lengths = np.asarray([(len(clean_text(a)) + len(clean_text(b))) / 2 for a, b in zip(test.sentence1, test.sentence2)])
    result["length_buckets"] = {
        "model": bucket_accuracy(m_scores, m_labels, lengths),
        "baseline": bucket_accuracy(base_scores, labels, lengths),
    }

    # --- E. error analysis: high-confidence errors of the model ---
    pred = (m_scores >= 0.5).astype(int)
    err_mask = pred != m_labels
    conf = np.abs(m_scores - 0.5) * 2  # confidence in [0,1]
    order = np.argsort(-conf[err_mask])
    err_idx = np.where(err_mask)[0][order][:10]
    result["errors"] = {
        "n_errors": int(err_mask.sum()),
        "n_false_positives": int(((pred == 1) & (m_labels == 0)).sum()),
        "n_false_negatives": int(((pred == 0) & (m_labels == 1)).sum()),
        "examples": [
            {
                "pair_id": str(test.pair_id[i]),
                "sentence1": test.sentence1[i],
                "sentence2": test.sentence2[i],
                "gold": int(m_labels[i]),
                "pred": int(pred[i]),
                "score": round(float(m_scores[i]), 3),
                "overlap": round(float(overlap[i]), 3),
            }
            for i in err_idx
        ],
    }
    return result


def figures(all_results: dict, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overlap buckets: model vs baseline per language
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (lang, res) in zip(axes, all_results.items()):
        buckets = res.get("overlap_buckets")
        if not buckets:
            continue
        x = np.arange(4)
        ax.bar(x - 0.2, [b["acc"] or 0 for b in buckets["baseline"]], 0.4, label="TF-IDF baseline", color="#9aa5b1")
        ax.bar(x + 0.2, [b["acc"] or 0 for b in buckets["model"]], 0.4, label="AfroXLMR", color="#3b6ea5")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i+1}" for i in range(4)])
        ax.set_title(f"{lang}: accuracy by lexical-overlap quartile")
        ax.set_xlabel("word-overlap quartile (Q1 low ... Q4 high)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(fig_dir / "overlap_buckets.png", dpi=150)
    plt.close(fig)

    # 2. Overall accuracy delta with CI
    fig, ax = plt.subplots(figsize=(6, 4))
    langs = list(all_results.keys())
    deltas = [all_results[l].get("overall_acc_delta_model_vs_baseline", {}).get("mean", 0) for l in langs]
    lows = [all_results[l].get("overall_acc_delta_model_vs_baseline", {}).get("ci95", [0, 0])[0] for l in langs]
    highs = [all_results[l].get("overall_acc_delta_model_vs_baseline", {}).get("ci95", [0, 0])[1] for l in langs]
    ax.bar(langs, deltas, color="#3b6ea5")
    ax.errorbar(langs, deltas, yerr=[np.array(deltas) - np.array(lows), np.array(highs) - np.array(deltas)],
                fmt="none", ecolor="black", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("accuracy(AfroXLMR) - accuracy(TF-IDF)")
    ax.set_title("Model advantage over lexical baseline (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_delta.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    setup_logging()
    cfg = load_config("default")
    processed = PROJECT_ROOT / "data" / "processed"
    phase5 = PROJECT_ROOT / "outputs" / "colab"
    baseline_pred = PROJECT_ROOT / "outputs" / "baseline" / "predictions"
    out_dir = PROJECT_ROOT / "outputs" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for lang in LANGS:
        results[lang] = analysis_for_language(lang, processed, phase5, baseline_pred, out_dir)

    figures(results, out_dir)
    (out_dir / "analysis.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Phase 6 analysis written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
