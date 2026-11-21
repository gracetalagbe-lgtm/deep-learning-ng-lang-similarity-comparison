"""Traditional baseline: TF-IDF + cosine similarity.

Anti-leakage design: the vectorizer is fitted on TRAINING-fold sentences
only; evaluation sentences are transformed with that fitted vectorizer.
The decision threshold is tuned on the dev fold to maximise F1.

This baseline is the lexical-overlap control: the gap between it and the
transformers is the evidence of semantic understanding beyond word overlap
(the PAWS lesson).
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from afripara.config import load_model_config

logger = logging.getLogger("afripara.baseline")


class TfidfCosineBaseline:
    """TF-IDF cosine scorer with dev-tuned decision threshold."""

    def __init__(self, model_config_name: str = "tfidf_baseline"):
        cfg = load_model_config(model_config_name)["model"]
        vec_cfg = cfg["vectorizer"]
        self.vectorizer = TfidfVectorizer(
            analyzer=vec_cfg["analyzer"],
            ngram_range=tuple(vec_cfg["ngram_range"]),
            min_df=vec_cfg["min_df"],
            sublinear_tf=vec_cfg["sublinear_tf"],
            lowercase=vec_cfg["lowercase"],        # False: diacritics are phonemic
            strip_accents=None,                    # never strip - E6 ablation only
        )
        self.threshold: float = 0.5

    # ------------------------------------------------------------------ #
    def fit_vectorizer(self, sentences: Iterable[str]) -> None:
        """Fit TF-IDF on training-fold sentences ONLY (anti-leakage)."""
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            raise ValueError("No non-empty sentences to fit the vectorizer on")
        self.vectorizer.fit(sentences)
        logger.info("TF-IDF fitted on %d sentences (%d features)",
                    len(sentences), len(self.vectorizer.vocabulary_))

    def pair_scores(self, s1: Sequence[str], s2: Sequence[str]) -> np.ndarray:
        """Cosine similarity per pair using the fitted vectorizer."""
        if not hasattr(self.vectorizer, "vocabulary_") or not self.vectorizer.vocabulary_:
            raise RuntimeError("Vectorizer not fitted - call fit_vectorizer first")
        v1 = normalize(self.vectorizer.transform(s1))
        v2 = normalize(self.vectorizer.transform(s2))
        return np.asarray(np.sum(v1.multiply(v2), axis=1)).ravel()

    def tune_threshold(self, y_true: Sequence[int], scores: Sequence[float]) -> float:
        """Grid-search the decision threshold on dev F1."""
        y = np.asarray(y_true, dtype=int)
        s = np.asarray(scores, dtype=float)
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            pred = (s >= t).astype(int)
            tp = int(np.sum((pred == 1) & (y == 1)))
            fp = int(np.sum((pred == 1) & (y == 0)))
            fn = int(np.sum((pred == 0) & (y == 1)))
            denom = 2 * tp + fp + fn
            f1 = 2 * tp / denom if denom else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.threshold = best_t
        logger.info("Threshold tuned on dev: %.2f (dev F1 %.3f)", best_t, best_f1)
        return best_t

    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_s1: Sequence[str],
        train_s2: Sequence[str],
        dev_s1: Sequence[str],
        dev_s2: Sequence[str],
        dev_y: Sequence[int],
    ) -> Dict[str, float]:
        """Fit vectorizer on train sentences, tune threshold on dev pairs."""
        self.fit_vectorizer(list(train_s1) + list(train_s2))
        dev_scores = self.pair_scores(dev_s1, dev_s2)
        self.tune_threshold(dev_y, dev_scores)
        return {"threshold": self.threshold}
