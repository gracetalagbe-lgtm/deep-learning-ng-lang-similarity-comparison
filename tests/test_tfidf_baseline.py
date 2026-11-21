"""Tests for the TF-IDF + cosine baseline."""

from __future__ import annotations

import numpy as np
import pytest

from afripara.models.tfidf_baseline import TfidfCosineBaseline


def test_scores_higher_for_identical_pair():
    b = TfidfCosineBaseline()
    b.fit_vectorizer(["Mo fẹ́ràn iresi", "Mo kọ́ ẹ̀kọ́", "Ẹ kú àárọ̀"])
    scores = b.pair_scores(["Mo fẹ́ràn iresi"], ["Mo fẹ́ràn iresi"])
    assert scores[0] == pytest.approx(1.0, abs=1e-6)


def test_scores_differentiate_pairs():
    b = TfidfCosineBaseline()
    b.fit_vectorizer(
        ["Mo fẹ́ràn iresi àti ẹ̀wà", "Mo fẹ́ràn iresi", "Ọjọ́ àájọ náà ti dé", "Àjọ náà ti pẹpẹ"]
    )
    same = float(b.pair_scores(["Mo fẹ́ràn iresi àti ẹ̀wà"], ["Mo fẹ́ràn iresi"])[0])
    diff = float(b.pair_scores(["Mo fẹ́ràn iresi àti ẹ̀wà"], ["Ọjọ́ àájọ náà ti dé"])[0])
    assert same > diff
    assert 0.0 < same < 1.0


def test_diacritics_are_preserved_in_features():
    """Stripping diacritics would merge distinct words; features must keep them."""
    b = TfidfCosineBaseline()
    b.fit_vectorizer(["ọmọ", "omo", "ọmọ ńlá"])
    vocab = b.vectorizer.vocabulary_
    assert "ọmọ" in vocab and "omo" in vocab  # distinct features
    scores = b.pair_scores(["ọmọ"], ["omo"])
    assert scores[0] < 0.5  # different words -> low cosine


def test_tune_threshold_finds_perfect_boundary():
    b = TfidfCosineBaseline()
    y = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    t = b.tune_threshold(y, scores)
    assert 0.2 < t <= 0.8
    preds = (np.array(scores) >= t).astype(int)
    assert preds.tolist() == y


def test_fit_requires_train_sentences():
    b = TfidfCosineBaseline()
    with pytest.raises(ValueError):
        b.fit_vectorizer(["", "  "])


def test_scores_before_fit_raises():
    b = TfidfCosineBaseline()
    with pytest.raises(RuntimeError):
        b.pair_scores(["a"], ["b"])
