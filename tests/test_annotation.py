"""Tests for the annotation store (gold validation workflow)."""

from __future__ import annotations

from pathlib import Path

import pytest

from afripara.data.annotation import AnnotationStore, Candidate, load_candidates


def _candidates(n: int = 5) -> list[Candidate]:
    return [
        Candidate(
            pair_id=f"p{i}",
            sentence1=f"Ọpọlọpọ ènìyàn ni {i}.",
            sentence2=f"Ọpọlọpọ àwọn ènìyàn ni {i}.",
            language="yor",
            repeat=(i < 2),  # first two are self-consistency repeats
        )
        for i in range(n)
    ]


def _write_candidates_csv(tmp_path: Path, candidates: list[Candidate]) -> Path:
    import pandas as pd

    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "pair_id": c.pair_id,
                "sentence1": c.sentence1,
                "sentence2": c.sentence2,
                "language": c.language,
                "english_s1": c.english_s1,
                "english_s2": c.english_s2,
                "source_id": c.source_id,
            }
            for c in candidates
        ]
    ).to_csv(path, index=False)
    return path


def test_load_candidates_roundtrip(tmp_path):
    path = _write_candidates_csv(tmp_path, _candidates())
    loaded = load_candidates(path)
    assert len(loaded) == 5
    assert loaded[0].pair_id == "p0"
    assert loaded[0].language == "yor"


def test_load_candidates_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,sentence1\n1,x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_candidates(bad)


def test_store_round1_flow(tmp_path):
    store = AnnotationStore(tmp_path / "labels.csv")
    cands = _candidates()
    assert len(store.next_candidates(cands, 1)) == 5
    store.save_label("p0", 1, 1, 3, "", 12.3)
    assert store.labeled_ids(1) == {"p0"}
    remaining = store.next_candidates(cands, 1)
    assert len(remaining) == 4
    assert remaining[0].pair_id == "p1"
    stats = store.stats(1)
    assert stats == {"labelled": 1, "skipped": 0, "total": 1}


def test_store_persists_across_instances_and_rounds(tmp_path):
    path = tmp_path / "labels.csv"
    cands = _candidates()
    store1 = AnnotationStore(path)
    store1.save_label("p0", 1, 0, None, "negation case", 5.0)
    store1.save_label("p1", 1, None, None, "skipped", 4.0)

    store2 = AnnotationStore(path)  # resumed session
    remaining = store2.next_candidates(cands, 1)
    assert [c.pair_id for c in remaining] == ["p2", "p3", "p4"]
    assert store2.stats(1)["skipped"] == 1

    # Round 2 only shows repeat candidates
    remaining_r2 = store2.next_candidates(cands, 2)
    assert {c.pair_id for c in remaining_r2} == {"p0", "p1"}


def test_self_agreement(tmp_path):
    store = AnnotationStore(tmp_path / "labels.csv")
    store.save_label("p0", 1, 1, None, "", 1.0)
    store.save_label("p1", 1, 0, None, "", 1.0)
    store.save_label("p0", 2, 1, None, "", 1.0)
    store.save_label("p1", 2, 1, None, "", 1.0)
    assert store.self_agreement() == pytest.approx(0.5)
