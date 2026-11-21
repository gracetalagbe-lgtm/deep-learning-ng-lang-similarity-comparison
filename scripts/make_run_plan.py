#!/usr/bin/env python3
"""Generate the Phase 5 experiment run plan + Colab data bundle.

Produces (in outputs/colab_bundle/):
  - run_plan.json: every training/evaluation run with dataset paths, model
    repo, seed, and hyperparameter provenance;
  - data_bundle.zip: the small CSVs the Colab script needs (constructed
    translate-train frames, SemRel disjoint frames, AfriXNLI binary evals,
    PAWS English subsample for the transfer arm).

Run grid (manageable, per feasibility report section 8):
  Cross-encoders
    afroxlmr_base   x {yor, ibo, hau-constructed, hau-semrel} x seeds 42/43/44
    afriberta_large x {yor, ibo} seed 42; x {hau-semrel} seeds 42/43/44
    afroxlmr_base   x en-paws (transfer arm, seed 42)
  Embeddings
    labse zero-shot on all eval sets (no training)
    afrie5 zero-shot on all eval sets (query: prefixes)
    labse fine-tune x {yor, ibo, hau-constructed, hau-semrel} seed 42

Each run evaluates on: its own test fold, the language's AfriXNLI test, and
 SemRel test folds for Hausa arms (both splits).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.config import load_config  # noqa: E402
from afripara.data.construct import load_paws, select_translate_train  # noqa: E402
from afripara.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("make_run_plan")

LANGS = ["yor", "ibo", "hau"]
PRIMARY_SEEDS = [42, 43, 44]


def build_plan(cfg: dict) -> list[dict]:
    training_cfg = cfg["training"]
    eval_sets_common = lambda lang: [f"afrixnli_binary_{lang}.csv"]

    runs: list[dict] = []

    def add_cross(run_id, model_repo, model_name, train_csv, train_filter, dev_csv, dev_filter, eval_sets, seeds):
        for seed in seeds:
            runs.append({
                "run_id": f"{run_id}_seed{seed}",
                "kind": "cross_encoder",
                "model_repo": model_repo,
                "model_name": model_name,
                "train": {"file": train_csv, "filter": train_filter},
                "dev": {"file": dev_csv, "filter": dev_filter},
                "eval_sets": eval_sets,
                "seed": seed,
                "hyperparams": {
                    "learning_rate": training_cfg["learning_rate"],
                    "per_device_train_batch_size": training_cfg["per_device_train_batch_size"],
                    "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
                    "num_train_epochs": training_cfg["num_train_epochs"],
                    "warmup_ratio": training_cfg["warmup_ratio"],
                    "weight_decay": training_cfg["weight_decay"],
                    "max_seq_length": training_cfg["max_seq_length"],
                    "early_stopping_patience": training_cfg["early_stopping_patience"],
                    "source_config": "configs/default.yaml:training",
                },
            })

    # --- AfroXLMR on constructed data per language (3 seeds) ---
    # NOTE: file names are FLAT and must match data_bundle.zip arcnames
    # exactly (the Colab script resolves them under /content/bundle/data/).
    for lang in ["yor", "ibo", "hau"]:
        train_csv = f"translate_train_{lang}.csv"
        eval_sets = [f"constructed_test_{lang}.csv"] + eval_sets_common(lang)
        if lang == "hau":
            eval_sets += ["semrel_disjoint_test.csv", "semrel_official_test.csv"]
        add_cross(
            f"afroxlmr_{lang}_constructed", "Davlan/afro-xlmr-base", "afroxlmr_base",
            train_csv, {"split": "train"}, train_csv, {"split": "dev"},
            eval_sets, PRIMARY_SEEDS,
        )

    # --- Hausa arms on SemRel (real human labels), 3 seeds ---
    add_cross(
        "afroxlmr_hau_semrel", "Davlan/afro-xlmr-base", "afroxlmr_base",
        "semrel_disjoint_train.csv", None, "semrel_disjoint_dev.csv", None,
        ["semrel_disjoint_test.csv", "semrel_official_test.csv", "afrixnli_binary_hau.csv"],
        PRIMARY_SEEDS,
    )

    # --- AfriBERTa comparator (African-specific; 1 seed for lang arms, 3 for hau-semrel) ---
    for lang in ["yor", "ibo"]:
        add_cross(
            f"afriberta_{lang}_constructed", "castorini/afriberta_large", "afriberta_large",
            f"translate_train_{lang}.csv", {"split": "train"},
            f"translate_train_{lang}.csv", {"split": "dev"},
            [f"constructed_test_{lang}.csv"] + eval_sets_common(lang), [cfg["project"]["seed"]],
        )
    add_cross(
        "afriberta_hau_semrel", "castorini/afriberta_large", "afriberta_large",
        "semrel_disjoint_train.csv", None, "semrel_disjoint_dev.csv", None,
        ["semrel_disjoint_test.csv", "semrel_official_test.csv", "afrixnli_binary_hau.csv"],
        PRIMARY_SEEDS,
    )

    # --- English PAWS transfer arm (E5): train on EN, zero-shot everywhere ---
    add_cross(
        "afroxlmr_en_paws", "Davlan/afro-xlmr-base", "afroxlmr_base",
        "paws_en_train_subsample.csv", None, "paws_en_validation_head.csv", None,
        [f"afrixnli_binary_{lang}.csv" for lang in LANGS + ["eng"]]
        + [f"constructed_test_{lang}.csv" for lang in LANGS],
        [cfg["project"]["seed"]],
    )

    # --- Embedding track ---
    for lang in ["yor", "ibo", "hau"]:
        runs.append({
            "run_id": f"labse_zeroshot_{lang}",
            "kind": "bi_encoder_zeroshot",
            "model_repo": "sentence-transformers/LaBSE",
            "model_name": "labse",
            "dev_eval": f"constructed_dev_{lang}.csv",
            "eval_sets": [f"constructed_dev_{lang}.csv", f"constructed_test_{lang}.csv"] + eval_sets_common(lang),
            "seed": cfg["project"]["seed"],
        })
        runs.append({
            "run_id": f"afrie5_zeroshot_{lang}",
            "kind": "bi_encoder_zeroshot",
            "model_repo": "McGill-NLP/AfriE5-Large-instruct",
            "model_name": "afrie5_large",
            "prefix": "query:",
            "dev_eval": f"constructed_dev_{lang}.csv",
            "eval_sets": [f"constructed_dev_{lang}.csv", f"constructed_test_{lang}.csv"] + eval_sets_common(lang),
            "seed": cfg["project"]["seed"],
        })
    runs.append({
        "run_id": "labse_zeroshot_semrel",
        "kind": "bi_encoder_zeroshot",
        "model_repo": "sentence-transformers/LaBSE",
        "model_name": "labse",
        "dev_eval": "semrel_disjoint_dev.csv",
        "eval_sets": ["semrel_disjoint_dev.csv", "semrel_disjoint_test.csv", "semrel_official_test.csv"],
        "seed": cfg["project"]["seed"],
    })
    runs.append({
        "run_id": "afrie5_zeroshot_semrel",
        "kind": "bi_encoder_zeroshot",
        "model_repo": "McGill-NLP/AfriE5-Large-instruct",
        "model_name": "afrie5_large",
        "prefix": "query:",
        "dev_eval": "semrel_disjoint_dev.csv",
        "eval_sets": ["semrel_disjoint_dev.csv", "semrel_disjoint_test.csv", "semrel_official_test.csv"],
        "seed": cfg["project"]["seed"],
    })
    # LaBSE fine-tune (cosine loss) - one seed per training source
    for lang in ["yor", "ibo"]:
        runs.append({
            "run_id": f"labse_finetune_{lang}_constructed",
            "kind": "bi_encoder_finetune",
            "model_repo": "sentence-transformers/LaBSE",
            "model_name": "labse",
            "train": {"file": f"translate_train_{lang}.csv", "filter": {"split": "train"}},
            "dev": {"file": f"translate_train_{lang}.csv", "filter": {"split": "dev"}},
            "eval_sets": [f"constructed_test_{lang}.csv"] + eval_sets_common(lang),
            "seed": cfg["project"]["seed"],
        })
    runs.append({
        "run_id": "labse_finetune_hau_semrel",
        "kind": "bi_encoder_finetune",
        "model_repo": "sentence-transformers/LaBSE",
        "model_name": "labse",
        "train": {"file": "semrel_disjoint_train.csv", "filter": None},
        "dev": {"file": "semrel_disjoint_dev.csv", "filter": None},
        "eval_sets": ["semrel_disjoint_test.csv", "semrel_official_test.csv"],
        "seed": cfg["project"]["seed"],
    })
    return runs


def build_bundle(processed: Path, raw_dir: Path, out_dir: Path, cfg: dict) -> list[str]:
    """Test-fold eval CSVs + SemRel folds + PAWS subsample -> data_bundle.zip."""
    export_dir = out_dir / "data"
    export_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    # Constructed test folds as standalone eval files (predictions keyed by row index)
    for lang in LANGS:
        df = pd.read_csv(processed / "constructed" / f"translate_train_{lang}.csv")
        test = df[df.split == "test"].reset_index(drop=True)
        dev = df[df.split == "dev"].reset_index(drop=True)
        p = export_dir / f"constructed_test_{lang}.csv"
        test.to_csv(p, index=False)
        pd_dev = export_dir / f"constructed_dev_{lang}.csv"
        dev.to_csv(pd_dev, index=False)
        files += [p, pd_dev]
        files.append(processed / "constructed" / f"translate_train_{lang}.csv")
        files.append(processed / "eval" / f"afrixnli_binary_{lang}.csv")
    files.append(processed / "eval" / "afrixnli_binary_eng.csv")

    # PAWS English for the transfer arm
    paws_dev = load_paws(raw_dir, "validation").head(2000)
    paws_train = select_translate_train(load_paws(raw_dir, "train"), n_pairs=3000, seed=cfg["project"]["seed"])
    for frame, name in [(paws_train, "paws_en_train_subsample.csv"), (paws_dev, "paws_en_validation_head.csv")]:
        p = export_dir / name
        frame.to_csv(p, index=False)
        files.append(p)

    # SemRel folds under flat plan-consistent names (avoids test.csv collision)
    entries: list[tuple[Path, str]] = [(p, f"data/{p.name}") for p in files]
    for split in ["train", "dev", "test"]:
        entries.append((processed / "semrel_hau_disjoint" / f"{split}.csv", f"data/semrel_disjoint_{split}.csv"))
    entries.append((processed / "semrel_hau_official" / "test.csv", "data/semrel_official_test.csv"))

    bundle = out_dir / "data_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        seen = set()
        for p, arcname in entries:
            if arcname in seen:
                raise ValueError(f"Duplicate bundle entry: {arcname}")
            seen.add(arcname)
            zf.write(p, arcname=arcname)
    logger.info("Bundle: %s (%.1f MB, %d files)", bundle, bundle.stat().st_size / 1e6, len(entries))
    return [name for _, name in entries]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config("default")
    processed = PROJECT_ROOT / "data" / "processed"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    out_dir = PROJECT_ROOT / "outputs" / "colab_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = build_plan(cfg)
    (out_dir / "run_plan.json").write_text(json.dumps({"runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8")
    n_cross = sum(1 for r in runs if r["kind"] == "cross_encoder")
    logger.info("Run plan: %d runs (%d cross-encoder, %d embedding) -> %s",
                len(runs), n_cross, len(runs) - n_cross, out_dir / "run_plan.json")
    bundled = build_bundle(processed, raw_dir, out_dir, cfg)

    # CONSISTENCY CHECK: every file the plan references must exist in the
    # bundle (guards against the plan/bundle path mismatch that wasted the
    # first Colab session on 2026-11-20).
    bundled_flat = {name.removeprefix("data/") for name in bundled}
    missing = []
    for run in runs:
        refs = [run.get("train", {}).get("file"), run.get("dev", {}).get("file"), run.get("dev_eval")]
        refs += run.get("eval_sets", [])
        for ref in filter(None, refs):
            if ref not in bundled_flat:
                missing.append((run["run_id"], ref))
    if missing:
        for run_id, ref in missing:
            logger.error("PLAN/BUNDLE MISMATCH: %s references missing file %s", run_id, ref)
        raise SystemExit(f"{len(missing)} plan/bundle mismatches - fix before uploading to Colab")
    logger.info("Plan/bundle consistency check passed (%d files)", len(bundled_flat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
