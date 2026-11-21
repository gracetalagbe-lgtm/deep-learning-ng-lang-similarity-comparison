"""Colab one-cell training job for Phase 5 (paste whole file into a cell).

Runs the experiment plan (run_plan.json) produced by scripts/make_run_plan.py.
Architecture: Colab TRAINS and computes raw predictions; ALL scientific
scoring happens locally with the tested afripara.evaluation metrics library.

Usage for the researcher:
  1. colab.research.google.com -> New notebook -> Runtime > Change runtime
     type > T4 GPU > Save.
  2. Upload to the Colab file pane (drag & drop):
       outputs/colab_bundle/data_bundle.zip
       outputs/colab_bundle/run_plan.json
  3. Paste this entire file into one cell and run it.
     - Mounts Google Drive (approvals pop up) and saves everything under
       Drive:/afripara_runs/<timestamp>/outputs/colab/
     - Skips runs whose predictions already exist in Drive (RESUMABLE across
       sessions - just run the same cell again).
  4. When finished (or when you stop), copy the Drive folder
     .../outputs/colab/ back into the project's outputs/ directory, then run
     locally:  python scripts/collect_results.py

Expected wall-clock on one free T4: cross-encoder runs ~8-20 min each
(18 runs), embedding runs ~5-15 min each (11). Split across 2-3 sessions if
Colab disconnects; rerunning the cell resumes where it stopped.
"""

import json
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path("/content")
DRIVE_RUNS_ROOT_NAME = "afripara_runs"

# Pinned to match the local environment (requirements.txt).
SUBPROCESS_PIP = [
    "transformers==4.49.0",
    "sentence-transformers==3.4.1",
    "accelerate==1.3.0",
    "datasets==3.2.0",
]


def setup() -> tuple[Path, Path]:
    import torch

    assert torch.cuda.is_available(), (
        "No GPU runtime: use Runtime > Change runtime type > T4 GPU, then rerun."
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for pkg in SUBPROCESS_PIP:
        __import__("subprocess").run(["pip", "install", "-q", pkg], check=True)

    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    # Reuse the latest existing run folder when resuming.
    drive_base = Path("/content/drive/MyDrive") / DRIVE_RUNS_ROOT_NAME
    drive_base.mkdir(parents=True, exist_ok=True)
    existing = sorted(drive_base.glob("*/outputs/colab"))
    if existing:
        colab_dir = existing[-1]
        print(f"Resuming in existing Drive folder: {colab_dir}")
    else:
        colab_dir = drive_base / stamp / "outputs" / "colab"
        colab_dir.mkdir(parents=True, exist_ok=True)
        print(f"New run folder: {colab_dir}")

    bundle = REPO_ROOT / "data_bundle.zip"
    if bundle.exists():
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(REPO_ROOT / "bundle")
    return REPO_ROOT / "bundle" / "data", colab_dir


def load_frame(data_dir: Path, name: str):
    import pandas as pd

    return pd.read_csv(data_dir / name)


def frame_slice(df, spec: dict | None):
    if spec:
        for key, value in spec.items():
            df = df[df[key] == value]
    return df.reset_index(drop=True)


def labels_from(df) -> list[int]:
    if "label" in df.columns:
        return df["label"].astype(int).tolist()
    if "score" in df.columns:
        return (df["score"] >= 0.5).astype(int).tolist()  # documented binarisation
    raise ValueError("Frame has neither label nor score")


def pair_arrays(df):
    return df.sentence1.tolist(), df.sentence2.tolist()


# --------------------------------------------------------------------------- #
# Cross-encoder
# --------------------------------------------------------------------------- #

def run_cross_encoder(run: dict, data_dir: Path, run_dir: Path, cfg: dict) -> None:
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    from afripara_style_pairdata import PairDataset  # defined below

    h = run["hyperparams"]
    seed = run["seed"]
    device = "cuda"

    def seed_all(s: int) -> None:
        import random

        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)

    seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(run["model_repo"])
    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 10**6:
        tokenizer.model_max_length = h["max_seq_length"]

    train_df = frame_slice(load_frame(data_dir, run["train"]["file"]), run["train"].get("filter"))
    dev_df = frame_slice(load_frame(data_dir, run["dev"]["file"]), run["dev"].get("filter"))
    tr_s1, tr_s2 = pair_arrays(train_df)
    dv_s1, dv_s2 = pair_arrays(dev_df)
    tr_y, dv_y = labels_from(train_df), labels_from(dev_df)

    model = AutoModelForSequenceClassification.from_pretrained(run["model_repo"], num_labels=2)
    train_ds = PairDataset(tr_s1, tr_s2, tr_y, tokenizer, h["max_seq_length"])
    dev_ds = PairDataset(dv_s1, dv_s2, dv_y, tokenizer, h["max_seq_length"])

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        }

    args = TrainingArguments(
        output_dir=str(run_dir / "trainer_tmp"),
        learning_rate=h["learning_rate"],
        per_device_train_batch_size=h["per_device_train_batch_size"],
        per_device_eval_batch_size=h["per_device_train_batch_size"] * 2,
        gradient_accumulation_steps=h["gradient_accumulation_steps"],
        num_train_epochs=h["num_train_epochs"],
        warmup_ratio=h["warmup_ratio"],
        weight_decay=h["weight_decay"],
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=20,
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=h["early_stopping_patience"])],
    )
    t0 = time.time()
    trainer.train()
    print(f"  trained in {time.time()-t0:.0f}s | best dev F1: {trainer.state.best_metric}")

    (run_dir / "history.json").write_text(
        json.dumps(
            {
                "log_history": trainer.state.log_history,
                "best_metric": trainer.state.best_metric,
                "model": run["model_repo"],
                "seed": seed,
                "n_train": len(tr_y),
                "hyperparams": h,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    if seed == 42:  # keep only primary-seed checkpoints (Drive space)
        model.save_pretrained(run_dir / "checkpoint")
        tokenizer.save_pretrained(run_dir / "checkpoint")

    # Dev predictions (threshold tuning happens locally, never on test)
    from torch.utils.data import DataLoader

    dev_loader = DataLoader(dev_ds, batch_size=64, collate_fn=DataCollatorWithPadding(tokenizer))
    dev_scores = []
    with torch.no_grad():
        for batch in dev_loader:
            batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            dev_scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().float().tolist())
    (run_dir / "predictions_dev.json").write_text(
        json.dumps({"pair_ids": dev_df.pair_id.astype(str).tolist() if "pair_id" in dev_df.columns else list(range(len(dev_df))),
                    "scores": dev_scores, "labels": dv_y}), encoding="utf-8"
    )

    # Predictions for every configured evaluation set
    model.eval()
    for eval_name in run["eval_sets"]:
        eval_df = load_frame(data_dir, eval_name)
        s1, s2 = pair_arrays(eval_df)
        ds = PairDataset(s1, s2, [0] * len(s1), tokenizer, h["max_seq_length"])
        loader = DataLoader(ds, batch_size=64, collate_fn=DataCollatorWithPadding(tokenizer))
        scores = []
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels")
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().float().tolist())
        payload = {
            "pair_ids": eval_df.pair_id.astype(str).tolist() if "pair_id" in eval_df.columns else list(range(len(eval_df))),
            "scores": scores,
            "labels": labels_from(eval_df),
            "gold_scores": eval_df.score.tolist() if "score" in eval_df.columns else None,
        }
        (run_dir / f"predictions_{eval_name.replace('.csv', '.json')}").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        print(f"  predictions_{eval_name}: n={len(scores)}")


# --------------------------------------------------------------------------- #
# Bi-encoder (zero-shot + fine-tune)
# --------------------------------------------------------------------------- #

def run_bi_encoder(run: dict, data_dir: Path, run_dir: Path) -> None:
    import numpy as np
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    from afripara_style_pairdata import clean_text

    seed = run["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    prefix = run.get("prefix", "")
    model = SentenceTransformer(run["model_repo"])

    def embed_and_score(df):
        s1 = [prefix + clean_text(t) for t in df.sentence1.tolist()]
        s2 = [prefix + clean_text(t) for t in df.sentence2.tolist()]
        # max_length=128 is justified by the pipeline test (pair tokens p50
        # 36-56, max 95); it cuts activation memory ~2-4x vs model defaults
        # (LaBSE 256 / AfriE5 512) - the T4 OOM fix from the first session.
        e1 = model.encode(s1, batch_size=16, max_length=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        e2 = model.encode(s2, batch_size=16, max_length=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.sum(e1 * e2, axis=1)  # cosine of unit vectors

    if run["kind"] == "bi_encoder_finetune":
        train_df = frame_slice(load_frame(data_dir, run["train"]["file"]), run["train"].get("filter"))
        examples = [
            InputExample(texts=[clean_text(a), clean_text(b)], label=float(y))
            for a, b, y in zip(train_df.sentence1, train_df.sentence2, labels_from(train_df))
        ]
        loader = DataLoader(examples, shuffle=True, batch_size=8)  # T4 OOM fix (was 16)
        loss = losses.CosineSimilarityLoss(model)
        t0 = time.time()
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=3,
            warmup_steps=max(10, int(0.1 * len(examples) / 16)),
            show_progress_bar=False,
        )
        print(f"  fine-tuned on {len(examples)} pairs in {time.time()-t0:.0f}s")
        if seed == 42:
            model.save(str(run_dir / "checkpoint"))

    for eval_name in run["eval_sets"]:
        eval_df = load_frame(data_dir, eval_name)
        scores = embed_and_score(eval_df)
        payload = {
            "pair_ids": eval_df.pair_id.astype(str).tolist() if "pair_id" in eval_df.columns else list(range(len(eval_df))),
            "scores": scores.tolist(),
            "labels": labels_from(eval_df),
            "gold_scores": eval_df.score.tolist() if "score" in eval_df.columns else None,
        }
        (run_dir / f"predictions_{eval_name.replace('.csv', '.json')}").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        print(f"  predictions_{eval_name}: n={len(scores)}")


# --------------------------------------------------------------------------- #
# Minimal shims so the script stays self-contained but behaves identically
# to src/afripara (same normalisation, same dataset semantics).
# --------------------------------------------------------------------------- #

def _pairdata_shim() -> None:
    import unicodedata

    import re

    ws = re.compile(r"\s+")
    zw = re.compile("[​‎‏﻿]")

    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = zw.sub("", text)
        return ws.sub(" ", text).strip()

    class PairDataset:
        """Mirrors afripara.training.data.PairDataset."""

        def __init__(self, s1, s2, labels, tokenizer, max_length=128):
            self.enc = tokenizer(
                [clean_text(x) for x in s1],
                [clean_text(x) for x in s2],
                truncation=True,
                max_length=max_length,
            )
            import numpy as np

            self.labels = np.asarray(labels, dtype=np.int64)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            import torch

            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    import sys

    module = type(sys)("afripara_style_pairdata")
    module.PairDataset = PairDataset
    module.clean_text = clean_text
    sys.modules["afripara_style_pairdata"] = module


def main() -> None:
    _pairdata_shim()
    data_dir, colab_dir = setup()
    plan = json.loads((REPO_ROOT / "run_plan.json").read_text(encoding="utf-8"))
    runs = plan["runs"]
    print(f"{len(runs)} runs in plan")

    done = skipped = failed = 0
    for i, run in enumerate(runs, 1):
        run_dir = colab_dir / run["run_id"]
        marker = run_dir / "DONE"
        if marker.exists():
            skipped += 1
            print(f"[{i}/{len(runs)}] {run['run_id']} - already complete, skipping")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(runs)}] {run['run_id']} ({run['kind']})")
        try:
            t0 = time.time()
            if run["kind"] == "cross_encoder":
                run_cross_encoder(run, data_dir, run_dir, plan)
            else:
                run_bi_encoder(run, data_dir, run_dir)
            # Free GPU memory between runs (T4 OOM fix from session 1: the
            # caching allocator holds blocks across the 29 sequential runs).
            import gc

            gc.collect()
            import torch as _torch

            _torch.cuda.empty_cache()
            error_file = run_dir / "ERROR.txt"
            if error_file.exists():
                error_file.unlink()  # cleared on successful retry
            marker.write_text(json.dumps({"finished": time.time(), "seconds": time.time() - t0}))
            done += 1
            print(f"  -> done in {time.time()-t0:.0f}s")
        except Exception as exc:
            failed += 1
            print(f"  !! FAILED: {exc}")
            (run_dir / "ERROR.txt").write_text(str(exc))
    print(f"\nFinished: {done} run(s), {skipped} skipped (already done), {failed} failed.")
    print(f"All outputs are in Drive: {colab_dir}")
    print("Copy that folder into the project's outputs/ directory, then run locally:")
    print("  .venv/bin/python scripts/collect_results.py")


if __name__ == "__main__":
    main()
