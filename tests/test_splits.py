"""Tests for leakage-controlled splitting (splits.py)."""

from __future__ import annotations

from afripara.data.loaders import PairRecord
from afripara.data.splits import (
    assign_components_disjoint,
    check_disjointness,
    sentence_fold,
)

W = {"train": 0.8, "dev": 0.1, "test": 0.1}


def _pair(s1: str, s2: str, label: str = "1") -> PairRecord:
    return PairRecord(sentence1=s1, sentence2=s2, label=label)


def test_sentence_fold_deterministic():
    assert sentence_fold("Ẹ kú àárọ̀", "salt1", W) == sentence_fold("Ẹ kú àárọ̀", "salt1", W)
    assert sentence_fold("Mo fẹ́ràn láti jẹ iresi", "salt1", W) == sentence_fold(
        "Mo fẹ́ràn láti jẹ iresi ", "salt1", W  # whitespace-insensitive (cleaned)
    )


def test_sentence_fold_sensitive_to_salt():
    # Different salts must not produce identical assignments for all sentences.
    a = [sentence_fold(f"sentence {i}", "salt1", W) for i in range(200)]
    b = [sentence_fold(f"sentence {i}", "salt2", W) for i in range(200)]
    assert a != b


def test_component_split_disjoint_and_proportional():
    pairs = [_pair(f"s{i}", f"t{i}", label=str(i % 2)) for i in range(1000)]
    # Add shared sentences between some pairs (forms components).
    for i in range(0, 200, 2):
        pairs.append(_pair(f"s{i}", f"t{i+1}", "0"))
    folds, stats = assign_components_disjoint(pairs, weights=W)
    overlaps = check_disjointness(folds)
    assert all(v == 0 for v in overlaps.values())
    total = sum(len(v) for v in folds.values())
    assert total == stats["total"]
    assert stats["dropped_mixed_fold"] == 0
    # Approximate proportions (within a component-size tolerance).
    assert stats["n_train"] / total > 0.7
    assert 0.05 < stats["n_dev"] / total < 0.16
    assert 0.05 < stats["n_test"] / total < 0.16


def test_component_split_shared_sentences_stay_together():
    # One sentence shared by three pairs -> all must land in the same fold.
    shared = "Àwọn ọmọdé ń fọ ajá."
    pairs = [_pair(shared, "other-a"), _pair(shared, "other-b"), _pair("other-a", "other-c")]
    folds, _ = assign_components_disjoint(pairs, weights=W)
    placements = {f for f, group in folds.items() for p in group}
    assert len(placements) == 1  # all three pairs in exactly one fold


def test_component_split_deterministic():
    pairs = [_pair(f"s{i}", f"t{i}") for i in range(500)]
    f1, s1 = assign_components_disjoint(pairs, weights=W)
    f2, s2 = assign_components_disjoint(pairs, weights=W)
    assert [p.sentence1 for p in f1["train"]] == [p.sentence1 for p in f2["train"]]
    assert s1 == s2
