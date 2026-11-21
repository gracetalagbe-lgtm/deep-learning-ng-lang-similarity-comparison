"""Sentence-pair datasets and tokenization for cross-encoder training."""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from afripara.features.normalization import clean_text

logger = logging.getLogger("afripara.training")


class PairDataset(Dataset):
    """Tokenized sentence-pair dataset for AutoModelForSequenceClassification.

    Text is normalised (Unicode NFC + zero-width removal) but never
    diacritic-stripped; tokenization is model-specific.
    """

    def __init__(
        self,
        sentence1: Sequence[str],
        sentence2: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 128,
    ):
        if not (len(sentence1) == len(sentence2) == len(labels)):
            raise ValueError("sentence1/sentence2/labels must have equal length")
        self.encodings = tokenizer(
            [clean_text(s) for s in sentence1],
            [clean_text(s) for s in sentence2],
            truncation=True,
            max_length=max_length,
        )
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_tokenizer_for_training(repo_id: str, max_length: int):
    """Load a tokenizer, falling back to the slow implementation if needed.

    AfriBERTa checkpoints ship no tokenizer.json; the slow SentencePiece
    tokenizer requires `sentencepiece` (pinned in requirements.txt). The
    model_max_length is normalised (AfriBERTa cards leave it unset).
    """
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
    except Exception as exc:
        logger.warning("Fast tokenizer unavailable for %s (%s); using use_fast=False", repo_id, exc)
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 10**6:
        tokenizer.model_max_length = max_length
    return tokenizer


def dataframe_to_arrays(df, label_col: str = "label", score_threshold: float | None = None):
    """Convert a processed pairs CSV dataframe into (s1, s2, y) arrays.

    For SemRel-style frames (score column, no binary label), pass
    ``score_threshold`` to binarise deterministically.
    """
    if label_col in df.columns:
        y = df[label_col].astype(int).tolist()
    elif score_threshold is not None and "score" in df.columns:
        y = (df["score"] >= score_threshold).astype(int).tolist()
    else:
        raise ValueError("Need a label column or a score column with threshold")
    return df.sentence1.tolist(), df.sentence2.tolist(), y
