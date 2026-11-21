#!/usr/bin/env python3
"""Build the processed datasets (Phase 3).

Local stage (run anywhere, no MT needed):
  1. SemRel2024-Hausa: official-split export AND a sentence-disjoint
     re-split (robustness check for the P2 leakage finding).
  2. AfriXNLI: binary evaluation exports per language (entailment->1,
     contradiction->0, neutral->excluded; proxy labels, documented).
  3. MT inputs: PAWS translate-train sentence lists per language, sized to
     the configured targets - consumed by the Colab translation notebook.

MT-dependent stage (runs after the Colab notebook writes outputs):
  4. Constructed translate-train pairs + sentence-disjoint splits.
  5. MT-vs-human candidate positives (Menyo / IGBONLP English sources).

Usage:
    python scripts/build_datasets.py --stage local
    python scripts/build_datasets.py --stage constructed   # after Colab MT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.config import load_config, resolve_path  # noqa: E402
from afripara.data.construct import (  # noqa: E402
    build_translate_train_pairs,
    deduplicate_pairs,
    load_mt_output,
    load_paws,
    pairs_to_dataframe,
    prep_mt_input,
    select_translate_train,
)
from afripara.data.loaders import PairRecord, load_afrixnli, load_menyo, load_parallel, load_semrel  # noqa: E402
from afripara.data.splits import assign_components_disjoint, check_disjointness  # noqa: E402
from afripara.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("build_datasets")

LANGS = ["yor", "ibo", "hau"]


def export_semrel(raw_dir: Path, processed: Path, salt: str, weights: dict) -> dict:
    """Official-split and sentence-disjoint SemRel exports."""
    stats = {}
    out_official = processed / "semrel_hau_official"
    out_disjoint = processed / "semrel_hau_disjoint"
    out_official.mkdir(parents=True, exist_ok=True)
    out_disjoint.mkdir(parents=True, exist_ok=True)

    for split, fname in [("train", "hau_train.csv"), ("dev", "hau_dev_with_labels.csv"), ("test", "hau_test_with_labels.csv")]:
        records, anomalies = load_semrel(raw_dir / "semrel2024_hau" / fname)
        df = pairs_to_dataframe(records)
        df["split"] = split
        df["label_source"] = "semrel_human"
        df["construction"] = "none"
        df = df.drop(columns=["label"])
        df.to_csv(out_official / f"{split}.csv", index=False)
        stats[f"official_{split}"] = len(df)

    # Sentence-disjoint re-split (all official splits pooled, then re-split).
    # Component-based: shared sentences form clusters; each cluster is
    # assigned atomically, so leakage is impossible and no pairs are dropped.
    pooled: list[PairRecord] = []
    for split, fname in [("train", "hau_train.csv"), ("dev", "hau_dev_with_labels.csv"), ("test", "hau_test_with_labels.csv")]:
        records, _ = load_semrel(raw_dir / "semrel2024_hau" / fname)
        pooled.extend(records)
    folds, split_stats = assign_components_disjoint(pooled, weights=weights)
    overlaps = check_disjointness(folds)
    total = sum(len(v) for v in folds.values())
    for fold, group in folds.items():
        df = pairs_to_dataframe(group)
        df["label_source"] = "semrel_human"
        df["construction"] = "none"
        df = df.drop(columns=["label"])
        df.to_csv(out_disjoint / f"{fold}.csv", index=False)
    stats["disjoint"] = {**split_stats, "overlaps_must_be_zero": overlaps}
    return stats


def export_afrixnli_binary(raw_dir: Path, processed: Path) -> dict:
    """AfriXNLI -> binary eval sets. Entailment=1, contradiction=0, neutral excluded."""
    out = processed / "eval"
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for lang in LANGS + ["eng"]:
        rows = []
        for split in ["dev", "test"]:
            records, anomalies = load_afrixnli(raw_dir / "afrixnli" / f"{lang}_{split}.tsv")
            for i, rec in enumerate(records):
                label = {"0": "1", "2": "0"}.get(rec.label)  # 1=entail->1, 2=contradiction->0
                if label is None:
                    continue  # neutral: genuinely ambiguous, excluded (documented)
                rows.append({
                    "pair_id": f"afrixnli_{lang}_{split}_{i}",
                    "sentence1": rec.sentence1.strip(),
                    "sentence2": rec.sentence2.strip(),
                    "label": int(label),
                    "label_source": "afrixnli_proxy",
                    "language": lang,
                    "split": split,
                })
        df = pd.DataFrame(rows)
        df.to_csv(out / f"afrixnli_binary_{lang}.csv", index=False)
        stats[f"afrixnli_{lang}"] = {"kept": len(df), "label_counts": df.label.value_counts().to_dict()}
    return stats


def prep_mt_inputs(raw_dir: Path, interim: Path, cfg: dict) -> dict:
    """Per-language MT inputs: PAWS translate-train + MT-vs-human sources."""
    cons = cfg["construction"]
    paws = load_paws(raw_dir, "train")
    stats = {}
    for lang in LANGS:
        n = cons["paws_train_target_pairs"][lang]
        subset = select_translate_train(
            paws, n_pairs=n, seed=cfg["project"]["seed"], max_pair_chars=cons["max_pair_chars"]
        )
        out = prep_mt_input(subset, interim / "mt" / f"paws_translate_input_{lang}.jsonl")
        stats[f"paws_input_{lang}"] = {"pairs": len(subset), "sentences": 2 * len(subset), "file": str(out)}

    # MT-vs-human inputs: English source sentences from the parallel corpora
    # (subsampled; the target side already exists as the human translation).
    import json as _json

    quota = cons.get("mt_vs_human_target", {})
    quotas = {"yor": quota.get("yor", 1000), "ibo": quota.get("ibo", 1000)}
    for lang in ["yor", "ibo"]:
        if lang == "yor":
            recs = []
            for split in ["train", "dev"]:
                r, _ = load_menyo(raw_dir / "menyo20k_mt" / f"{split}.tsv")
                recs.extend(r)
        else:
            r, _ = load_parallel(raw_dir / "igbonlp_benchmark" / "train.en", raw_dir / "igbonlp_benchmark" / "train.ig")
            recs = r
        # deterministic subsample: take every kth non-empty short English line
        short = [(i, rec) for i, rec in enumerate(recs) if rec.sentence1.strip() and len(rec.sentence1) <= cons["max_pair_chars"]]
        k = max(1, len(short) // quotas[lang]) if quotas[lang] else 0
        chosen = short[::k][: quotas[lang]]
        out = interim / "mt" / f"mt_vs_human_input_{lang}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for idx, rec in chosen:
                fh.write(_json.dumps({"id": f"{'menyo' if lang == 'yor' else 'igbo'}{idx}_mth", "text": rec.sentence1}, ensure_ascii=False) + "\n")
        stats[f"mt_vs_human_input_{lang}"] = {"sentences": len(chosen), "file": str(out)}
    return stats


def build_constructed(raw_dir: Path, interim: Path, processed: Path, cfg: dict) -> dict:
    """MT-dependent construction: translate-train pairs + MT-vs-human positives."""
    cons = cfg["construction"]
    salt = cons["salt"]
    weights = cons["split_weights"]
    mt_dir = interim / "mt"
    out_dir = processed / "constructed"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    # --- translate-train per language ---
    paws = load_paws(raw_dir, "train")
    for lang in LANGS:
        mt_path = mt_dir / f"paws_translated_{lang}.jsonl"
        if not mt_path.exists():
            report[f"translate_train_{lang}"] = "PENDING - run the Colab MT notebook first"
            continue
        n = cons["paws_train_target_pairs"][lang]
        subset = select_translate_train(paws, n_pairs=n, seed=cfg["project"]["seed"], max_pair_chars=cons["max_pair_chars"])
        mt = load_mt_output(mt_path)
        pairs, stats = build_translate_train_pairs(subset, mt, language=lang)
        pairs, dupes = deduplicate_pairs(pairs)
        folds, split_stats = assign_components_disjoint(pairs, weights=weights)
        overlaps = check_disjointness(folds)
        df = pairs_to_dataframe(pairs, split_by_fold=folds)
        df.to_csv(out_dir / f"translate_train_{lang}.csv", index=False)
        report[f"translate_train_{lang}"] = {**stats, "duplicates_removed": dupes, **split_stats, "overlaps_must_be_zero": overlaps}

    # --- MT-vs-human positives (Yoruba: Menyo; Igbo: IGBONLP) ---
    for lang, source, loader in [
        ("yor", "menyo20k_mt", "menyo"),
        ("ibo", "igbonlp_benchmark", "parallel"),
    ]:
        mt_path = mt_dir / f"mt_vs_human_input_{lang}_translated.jsonl"
        if not mt_path.exists():
            report[f"mt_vs_human_{lang}"] = "PENDING - run the Colab MT notebook first"
            continue
        if source == "menyo20k_mt":
            human = []
            for split in ["train", "dev"]:
                recs, _ = load_menyo(raw_dir / source / f"{split}.tsv")
                human.extend(recs)
            human_pairs = [(f"menyo{i}", r.sentence2, r.sentence1) for i, r in enumerate(human)]
        else:
            human = []
            for split in ["train"]:
                recs, _ = load_parallel(raw_dir / source / "train.en", raw_dir / source / "train.ig")
                human.extend(recs)
            human_pairs = [(f"igbo{i}", r.sentence2, r.sentence1) for i, r in enumerate(human)]
        mt = load_mt_output(mt_path)
        pairs, stats = build_mt_vs_human_pairs(mt, human_pairs, language=lang)
        pairs, dupes = deduplicate_pairs(pairs)
        df = pairs_to_dataframe(pairs)
        df.to_csv(out_dir / f"mt_vs_human_{lang}_positives.csv", index=False)
        report[f"mt_vs_human_{lang}"] = {**stats, "duplicates_removed": dupes}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["local", "constructed", "all"], default="local")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config("default")
    raw_dir = resolve_path(cfg, "data_raw")
    processed = resolve_path(cfg, "data_processed")
    interim = resolve_path(cfg, "data_interim")

    if args.stage in ("local", "all"):
        semrel_stats = export_semrel(raw_dir, processed, cfg["construction"]["salt"], cfg["construction"]["split_weights"])
        logger.info("SemRel exports: %s", json.dumps(semrel_stats))
        axnli_stats = export_afrixnli_binary(raw_dir, processed)
        logger.info("AfriXNLI binary exports: %s", json.dumps(axnli_stats))
        mt_stats = prep_mt_inputs(raw_dir, interim, cfg)
        logger.info("MT inputs: %s", json.dumps(mt_stats))
    if args.stage in ("constructed", "all"):
        report = build_constructed(raw_dir, interim, processed, cfg)
        logger.info("Constructed datasets: %s", json.dumps(report, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
