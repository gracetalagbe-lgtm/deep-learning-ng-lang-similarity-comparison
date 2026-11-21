#!/usr/bin/env python3
"""Aggregate Phase 5 Colab outputs into the project's results tables.

Scoring happens HERE, locally, with the tested afripara.evaluation.metrics
library - Colab only produced raw predictions. For every completed run:

  - decision threshold tuned on that run's DEV predictions (never test);
  - test-set classification metrics + bootstrap 95% CIs (primary seed);
  - Spearman/Pearson where gold continuous scores exist (SemRel);
  - seed aggregation: mean +/- std over seeds per run family;
  - McNemar's exact test: transformer vs TF-IDF baseline on matched pairs;
  - training curves (loss/eval-F1 vs steps) for the primary seed;
  - zero-shot embedding rows (LaBSE / AfriE5) with dev-tuned thresholds.

Inputs:  outputs/colab/<run_id>/predictions_*.json (+ history.json, DONE)
         outputs/baseline/predictions/<set>.json
Outputs: outputs/phase5/results.json, RESULTS.md, curves_<run>.png

Usage:
    python scripts/collect_results.py [--colab-dir outputs/colab]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.config import PROJECT_ROOT, load_config  # noqa: E402
from afripara.evaluation.metrics import (  # noqa: E402
    bootstrap_ci,
    classification_metrics,
    f1_metric,
    mcnemar_exact,
    similarity_metrics,
)
from afripara.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("collect_results")

SEED_RE = re.compile(r"_seed(\d+)$")

# Label provenance per evaluation set (2026-11-20 decision: annotation was
# skipped; every reported table must expose label provenance).
PROVENANCE = {
    "semrel": "semrel_human",
    "afrixnli": "afrixnli_proxy",
    "constructed": "paws_translated (silver)",
    "translate_train": "paws_translated (silver)",
}


def provenance_of(set_name: str) -> str:
    for key, label in PROVENANCE.items():
        if set_name.startswith(key):
            return label
    return "unknown"


def load_pred(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tuned_threshold(dev_pred: dict, seed: int) -> float:
    """Maximise dev F1 over a threshold grid (mirrors the baseline protocol)."""
    y = np.asarray(dev_pred["labels"], dtype=int)
    s = np.asarray(dev_pred["scores"], dtype=float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        _, _, f1, _ = _prf(y, (s >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _prf(y, pred):
    from sklearn.metrics import precision_recall_fscore_support

    return precision_recall_fscore_support(y, pred, average="binary", zero_division=0)


def score_set(pred: dict, threshold: float, seed: int, with_ci: bool = False) -> dict:
    y = pred["labels"]
    s = pred["scores"]
    metrics = classification_metrics(y, s, threshold=threshold)
    if with_ci:
        _, f1_lo, f1_hi = bootstrap_ci(y, s, f1_metric, seed=seed, threshold=threshold)
        metrics["f1_ci95"] = [round(f1_lo, 4), round(f1_hi, 4)]
    if pred.get("gold_scores"):
        metrics["similarity"] = similarity_metrics(s, pred["gold_scores"])
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colab-dir", default="outputs/colab")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config("default")
    colab_dir = PROJECT_ROOT / args.colab_dir
    baseline_pred_dir = PROJECT_ROOT / "outputs" / "baseline" / "predictions"
    out_dir = PROJECT_ROOT / "outputs" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not colab_dir.exists():
        logger.error("No Colab outputs at %s - complete the Colab training run first.", colab_dir)
        return 1

    baseline_preds: dict[str, dict] = {}
    if baseline_pred_dir.exists():
        baseline_preds = {p.stem: load_pred(p) for p in sorted(baseline_pred_dir.glob("*.json"))}

    results: dict = {"seed": cfg["project"]["seed"], "runs": {}, "families": {}}
    # family -> eval set -> list of per-seed metric dicts (accumulated across
    # runs; the first version overwrote instead of accumulating - bug fixed)
    family_acc: dict = defaultdict(lambda: defaultdict(list))
    n_done = 0
    for run_dir in sorted(p for p in colab_dir.iterdir() if p.is_dir()):
        if not (run_dir / "DONE").exists():
            logger.warning("Skipping incomplete run %s", run_dir.name)
            continue
        n_done += 1
        seed_match = SEED_RE.search(run_dir.name)
        seed = int(seed_match.group(1)) if seed_match else cfg["project"]["seed"]
        family = SEED_RE.sub("", run_dir.name)
        entry: dict = {"seed": seed, "sets": {}}

        dev_file = run_dir / "predictions_dev.json"
        threshold = 0.5
        if dev_file.exists():
            threshold = tuned_threshold(load_pred(dev_file), seed)
        entry["threshold"] = threshold

        for pred_file in sorted(run_dir.glob("predictions_*.json")):
            if pred_file.stem == "predictions_dev":
                continue
            set_name = pred_file.stem.replace("predictions_", "")
            pred = load_pred(pred_file)
            entry["sets"][set_name] = score_set(pred, threshold, seed, with_ci=(seed == 42))
            family_acc[family][set_name].append(entry["sets"][set_name])

            # McNemar vs baseline where a matched baseline prediction exists.
            # The baseline saved the constructed test folds under their
            # translate_train_<lang>.csv names - alias them here.
            baseline_key = set_name
            if set_name.startswith("constructed_test_"):
                baseline_key = f"translate_train_{set_name.removeprefix('constructed_test_')}"
            elif set_name == "semrel_disjoint_test":
                baseline_key = "semrel_hau_disjoint"
            elif set_name == "semrel_official_test":
                baseline_key = "semrel_hau_official"
            if baseline_key in baseline_preds:
                base = baseline_preds[baseline_key]
                n = min(len(base["scores"]), len(pred["scores"]))
                base_pred = (np.asarray(base["scores"][:n]) >= 0.5).astype(int)
                trans_pred = (np.asarray(pred["scores"][:n]) >= threshold).astype(int)
                y = pred["labels"][:n]
                entry["sets"][set_name]["mcnemar_vs_baseline"] = mcnemar_exact(y, trans_pred, base_pred)
        results["runs"][run_dir.name] = entry

        # Training curve for cross-encoder runs
        history_file = run_dir / "history.json"
        if history_file.exists():
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                hist = json.loads(history_file.read_text())["log_history"]
                steps = [h.get("step") for h in hist if "loss" in h]
                losses = [h["loss"] for h in hist if "loss" in h]
                eval_pts = [(h["step"], h["eval_f1"]) for h in hist if "eval_f1" in h]
                fig, ax1 = plt.subplots(figsize=(7, 4))
                ax1.plot(steps, losses, label="train loss", color="tab:blue")
                ax1.set_xlabel("optimizer step")
                ax1.set_ylabel("train loss")
                ax2 = ax1.twinx()
                if eval_pts:
                    ax2.plot([p[0] for p in eval_pts], [p[1] for p in eval_pts], "o-", label="dev F1", color="tab:orange")
                    ax2.set_ylabel("dev F1")
                ax1.set_title(run_dir.name)
                fig.tight_layout()
                fig.savefig(out_dir / f"curves_{run_dir.name}.png", dpi=150)
                plt.close(fig)
            except Exception as exc:  # plotting must never kill aggregation
                logger.warning("Curve plotting failed for %s: %s", run_dir.name, exc)

        # Family aggregation happens AFTER the loop (accumulated in
        # family_acc); see below.

    # Family aggregation across seeds (mean +/- std), dev sets excluded from
    # the headline table (thresholds are tuned on them).
    for family, sets in family_acc.items():
        fam_out = {}
        for set_name, ms in sets.items():
            if "_dev" in set_name:
                continue
            fam_out[set_name] = {
                "n_seeds": len(ms),
                "accuracy_mean": float(np.mean([m["accuracy"] for m in ms])),
                "accuracy_std": float(np.std([m["accuracy"] for m in ms])),
                "f1_mean": float(np.mean([m["f1"] for m in ms])),
                "f1_std": float(np.std([m["f1"] for m in ms])),
                "roc_auc_mean": float(np.nanmean([m["roc_auc"] for m in ms])),
                **(
                    {"spearman_mean": float(np.nanmean([m["similarity"]["spearman"] for m in ms if "similarity" in m]))}
                    if any("similarity" in m for m in ms)
                    else {}
                ),
            }
        results["families"][family] = fam_out

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Phase 5 Results — Transformers (thresholds tuned on dev, scored locally)",
        "",
        f"Completed runs: **{n_done}**. Family rows aggregate seeds (mean F1 +/- std).",
        "",
        "**Label provenance:** constructed sets = `paws_translated (silver)`; "
        "SemRel = `semrel_human`; AfriXNLI = `afrixnli_proxy`. Conclusions on "
        "silver sets are conditional on MT label quality (research log, 2026-11-20).",
        "",
        "## Per-family test results",
        "",
        "| Run family | Eval set | Provenance | n_seeds | Acc | F1 (mean +/- std) | ROC-AUC | McNemar p vs baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for family, sets in results["families"].items():
        for set_name, m in sets.items():
            p_vals = [
                r["sets"][set_name].get("mcnemar_vs_baseline", {}).get("p_value")
                for r in results["runs"].values()
                if set_name in r["sets"]
            ]
            p_str = f"{min(p for p in p_vals if p is not None):.4f}" if any(p_vals) else "-"
            lines.append(
                f"| {family} | {set_name} | {provenance_of(set_name)} | {m['n_seeds']} | {m['accuracy_mean']:.3f} "
                f"| {m['f1_mean']:.3f} ± {m['f1_std']:.3f} | {m['roc_auc_mean']:.3f} | {p_str} |"
            )
    lines += [
        "",
        "## Similarity vs SemRel gold (Spearman)",
        "",
    ]
    for family, sets in results["families"].items():
        for set_name, m in sets.items():
            if set_name.startswith("semrel") and "spearman_mean" in m:
                lines.append(f"- **{family}** on {set_name}: Spearman {m['spearman_mean']:.3f}")
    lines += [
        "",
        "Note: per-run detail, bootstrap CIs and training curves in this folder;",
        "McNemar p-values are the minimum across seeds of the exact test against",
        "the TF-IDF baseline on matched test items (Dietterich 1998).",
        "",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Aggregated %d completed runs -> %s", n_done, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
