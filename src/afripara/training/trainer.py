"""Cross-encoder fine-tuning (the reference implementation).

The Colab script (scripts/colab_train_transformers.py) is a deployment of
this exact configuration: LR 2e-5, effective batch 32 (16 x grad-accum 2),
warmup 10%, weight decay 0.01, max_len 128, <=5 epochs, early stopping on
dev F1 (patience 3), best-checkpoint selection, seeded. Hyperparameters come
from configs/default.yaml; nothing is hardcoded here.

fp16 is enabled only on CUDA (it is a GPU feature; CPU runs fall back).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from afripara.config import get_device, load_config, load_model_config, set_seed
from afripara.training.data import PairDataset, load_tokenizer_for_training

logger = logging.getLogger("afripara.training")


def compute_pair_metrics(eval_pred) -> Dict[str, float]:
    """HF Trainer compute_metrics: F1 (binary, class 1), accuracy, ROC-AUC."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    logits, labels = eval_pred
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    preds = logits.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels, probs[:, 1]))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def train_cross_encoder(
    model_config_name: str,
    train_s1,
    train_s2,
    train_y,
    dev_s1,
    dev_s2,
    dev_y,
    output_dir: Path,
    seed: int = 42,
    config_name: str = "default",
    training_overrides: Optional[dict] = None,
    model_repo_override: Optional[str] = None,
):
    """Fine-tune a cross-encoder; returns (model, tokenizer, history).

    Early stopping monitors dev F1 (patience from config); the best
    checkpoint is loaded at the end (load_best_model_at_end).
    ``model_repo_override`` exists for ENGINEERING smoke runs only (e.g. the
    small cached AfriBERTa checkpoint on CPU) - never for experiments, which
    must use the registered model configs for provenance.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    cfg = load_config(config_name)["training"]
    if training_overrides:
        cfg = {**cfg, **training_overrides}
    model_cfg = load_model_config(model_config_name)["model"]
    device = get_device()

    set_seed(seed)
    repo_id = model_repo_override or model_cfg["hf_repo"]
    tokenizer = load_tokenizer_for_training(repo_id, cfg["max_seq_length"])
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, num_labels=2)

    train_ds = PairDataset(train_s1, train_s2, train_y, tokenizer, cfg["max_seq_length"])
    dev_ds = PairDataset(dev_s1, dev_s2, dev_y, tokenizer, cfg["max_seq_length"])
    logger.info(
        "Run %s seed=%d: train=%d dev=%d device=%s model=%s",
        model_config_name, seed, len(train_ds), len(dev_ds), device, model_cfg["hf_repo"],
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_train_batch_size"] * 2,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        fp16=(device == "cuda"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=20,
        seed=seed,
        report_to=[],
        save_safetensors=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_pair_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )
    trainer.train()

    history = {
        "log_history": trainer.state.log_history,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "model": repo_id,
        "seed": seed,
        "max_seq_length": cfg["max_seq_length"],
        "learning_rate": cfg["learning_rate"],
    }
    return trainer.model, tokenizer, history


def predict_scores(model, tokenizer, s1, s2, max_length: int = 128, batch_size: int = 32, device: str | None = None) -> np.ndarray:
    """Positive-class probability for each pair (for the metrics library)."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    dev = device or get_device()
    model.eval()
    ds = PairDataset(s1, s2, [0] * len(s1), tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    scores = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(dev) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.cpu().float().tolist())
    return np.asarray(scores, dtype=float)
