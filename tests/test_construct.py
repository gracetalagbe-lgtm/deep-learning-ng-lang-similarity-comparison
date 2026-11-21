"""Tests for the pair-construction pipeline (construct.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from afripara.data.construct import (
    build_mt_vs_human_pairs,
    build_translate_train_pairs,
    deduplicate_pairs,
    load_paws,
    select_translate_train,
)
from afripara.data.loaders import PairRecord
from afripara.config import resolve_path, load_config


def test_load_paws_real_file():
    """The real downloaded PAWS train parquet has the documented shape."""
    cfg = load_config("default")
    raw = resolve_path(cfg, "data_raw")
    df = load_paws(raw, "train")
    assert len(df) == 49401
    assert set(df.columns) >= {"id", "sentence1", "sentence2", "label"}
    assert set(df.label.unique()) == {0, 1}


def test_select_translate_train_deterministic_and_filtered():
    df = pd.DataFrame(
        {
            "id": range(100),
            "sentence1": [f"sentence one {i} " + "x" * (i % 50) for i in range(100)],
            "sentence2": [f"sentence two {i}" for i in range(100)],
            "label": [i % 2 for i in range(100)],
        }
    )
    a = select_translate_train(df, n_pairs=20, seed=42, max_pair_chars=200)
    b = select_translate_train(df, n_pairs=20, seed=42, max_pair_chars=200)
    assert a.id.tolist() == b.id.tolist()  # deterministic
    assert len(a) == 20
    assert (a.sentence1.str.len() <= 200).all()


def test_build_translate_train_pairs_filters_and_labels():
    paws = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "sentence1": ["a", "b", "c"],
            "sentence2": ["d", "e", "f"],
            "label": [1, 0, 1],
        }
    )
    mt = {
        "paws1_s1": "ikọ́kọ́", "paws1_s2": "ìkejì",            # fine
        "paws2_s1": "",          "paws2_s2": "igbo keji",      # empty -> drop
        "paws3_s1": "òngbé",     "paws3_s2": "k" * 100,        # len ratio > 3 -> drop
    }
    pairs, stats = build_translate_train_pairs(paws, mt, language="yor")
    assert stats["candidates"] == 3
    assert stats["kept"] == 1
    assert stats["dropped_empty_mt"] == 1
    assert stats["dropped_len_ratio"] == 1
    assert len(pairs) == 1
    assert pairs[0].label == "1"                          # inherited PAWS label
    assert pairs[0].meta["label_source"] == "paws_translated"
    assert pairs[0].meta["language"] == "yor"
    assert pairs[0].meta["english_s1"] == "a"             # provenance kept


def test_build_mt_vs_human_pairs():
    mt = {"menyo0_mth": "Mo fẹ́ràn iresi.", "menyo1_mth": ""}
    human = [("menyo0", "Mo fẹ́ràn láti jẹ iresi.", "I like rice."), ("menyo1", "Ẹ kú àárọ̀", "Good morning")]
    pairs, stats = build_mt_vs_human_pairs(mt, human, language="yor")
    assert stats["kept"] == 1
    assert stats["dropped_empty_mt"] == 1
    assert pairs[0].label == "1"
    assert pairs[0].meta["label_source"] == "mt_vs_human"
    assert pairs[0].sentence2 == "Mo fẹ́ràn láti jẹ iresi."


def test_deduplicate_pairs_order_insensitive():
    pairs = [
        PairRecord("Mo wá.", "Ó ń lo.", label="1"),
        PairRecord("Ó ń lo.", "Mo wá.", label="1"),          # same pair, reversed
        PairRecord("Ndewo", "Daalụ", label="0"),
    ]
    kept, removed = deduplicate_pairs(pairs)
    assert len(kept) == 2
    assert removed == 1
