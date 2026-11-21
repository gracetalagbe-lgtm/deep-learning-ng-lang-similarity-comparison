"""Tests for the raw-data loaders, including the quirks found in the P2 audit
(BOM characters, embedded newlines in SemRel, malformed TSV lines)."""

from __future__ import annotations

from pathlib import Path

import pytest

from afripara.data.loaders import (
    load_afrixnli,
    load_menyo,
    load_ntrex,
    load_parallel,
    load_semrel,
)
from afripara.features.normalization import clean_text


def test_load_afrixnli(tmp_path: Path):
    p = tmp_path / "test.tsv"
    p.write_text(
        "premise\thypothesis\tlabel\n"
        "Ó dáa bá a sọ̀rọ̀.\tMi ò tún tíì bá a sọ̀rọ̀.\t2\n"
        "Ó bẹ̀rẹ̀ sí ní sọ̀rọ̀.\tÓ bẹ̀rẹ̀ sí ní sọ̀rọ̀.\t0\n",
        encoding="utf-8",
    )
    records, anomalies = load_afrixnli(p)
    assert len(records) == 2
    assert records[0].label == "2" and records[1].label == "0"
    assert anomalies == []


def test_load_afrixnli_flags_malformed_rows(tmp_path: Path):
    """The ibo/dev quirk: quotation marks break the TSV conversion, leaving
    rows without a premise field (2 fields instead of 3)."""
    p = tmp_path / "dev.tsv"
    p.write_text(
        "premise\thypothesis\tlabel\n"
        "Ó sọ̀rọ̀.\tÓ sọ̀rọ̀.\t0\n"
        '"Ó tún sọ̀rọ̀.\t1\n'   # malformed row: premise lost to the quote break
        "Ó kù.\tÓ kù náà.\t1\n",
        encoding="utf-8",
    )
    records, anomalies = load_afrixnli(p)
    assert len(records) == 2  # malformed row dropped, documented
    assert any("2 tab-separated" in a for a in anomalies)


def test_load_semrel_embedded_newline(tmp_path: Path):
    """SemRel stores both sentences in ONE quoted field with a newline."""
    p = tmp_path / "train.csv"
    p.write_text(
        'PairID,Text,Score\n'
        'hau_train_00001,"Line one sentence.\nLine two sentence.",0.94\n'
        'hau_train_00002,"A.\nB.",0.10\n',
        encoding="utf-8",
    )
    records, anomalies = load_semrel(p)
    assert len(records) == 2
    assert records[0].sentence1 == "Line one sentence."
    assert records[0].sentence2 == "Line two sentence."
    assert records[0].score == pytest.approx(0.94)
    assert anomalies == []


def test_load_semrel_flags_out_of_range_score(tmp_path: Path):
    p = tmp_path / "dev.csv"
    p.write_text('PairID,Text,Score\nhau_dev_00001,"A.\nB.",1.50\n', encoding="utf-8")
    records, anomalies = load_semrel(p)
    assert len(records) == 1
    assert any("outside [0,1]" in a for a in anomalies)


def test_load_menyo_strips_bom(tmp_path: Path):
    """Menyo files start with a UTF-8 BOM and carry zero-width artifacts in
    cells (both found in the P2 audit)."""
    p = tmp_path / "train.tsv"
    payload = "English\tYoruba\nHello\t﻿Ìdá 1: Kín ni?\n".encode("utf-8")
    p.write_bytes(b"\xef\xbb\xbf" + payload)  # file-start BOM, as in the raw file
    records, anomalies = load_menyo(p)
    assert len(records) == 1
    # Header still recognised (file-start BOM stripped)...
    assert anomalies == []
    # ...and zero-width artifacts are removed from cell text by clean_text.
    assert clean_text(records[0].sentence2) == "Ìdá 1: Kín ni?"


def test_load_parallel_detects_length_mismatch(tmp_path: Path):
    en = tmp_path / "train.en"
    ig = tmp_path / "train.ig"
    en.write_text("One.\nTwo.\n", encoding="utf-8")
    ig.write_text("Otu.\n", encoding="utf-8")
    records, anomalies = load_parallel(en, ig)
    assert len(records) == 1  # zipped to shortest
    assert any("mismatch" in a for a in anomalies)


def test_load_ntrex_two_columns(tmp_path: Path):
    p = tmp_path / "hau_Latn.tsv"
    p.write_text(
        "sentence_hau_Latn\tsentence_eng_Latn\n"
        "Welsh AMs ta damu\tWelsh AMs worried\n",
        encoding="utf-8",
    )
    records, anomalies = load_ntrex(p)
    assert len(records) == 1
    assert records[0].sentence1 == "Welsh AMs ta damu"
    assert anomalies == []
