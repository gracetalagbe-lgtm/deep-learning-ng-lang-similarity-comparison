"""Canonical parsers for the raw dataset formats.

Each loader returns a list of records and a list of anomaly strings; the
audit module aggregates anomalies. Parsers are deliberately explicit about
the quirks discovered during the P2 audit (BOM characters, embedded
newlines, header rows) so downstream code never re-discovers them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List


@dataclass
class PairRecord:
    """A single sentence pair in any of our dataset schemas."""

    sentence1: str
    sentence2: str
    label: str | None = None       # afrixnli: "0"/"1"/"2"; constructed: "0"/"1"
    score: float | None = None     # semrel: 0-1 relatedness
    meta: Dict[str, str] = field(default_factory=dict)


def _read_text(path: Path) -> str:
    """Read a file as UTF-8, stripping BOM if present (found in Menyo-20k)."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.decode("utf-8")  # raises on invalid UTF-8: encoding audit


def load_afrixnli(path: Path) -> tuple[List[PairRecord], List[str]]:
    """AfriXNLI TSV: header ``premise<TAB>hypothesis<TAB>label``.

    Labels follow XNLI convention: 0=entailment, 1=neutral, 2=contradiction.
    Rows must have exactly 3 fields; extra tabs are recorded as anomalies.
    """
    anomalies: List[str] = []
    records: List[PairRecord] = []
    lines = _read_text(path).splitlines()
    if not lines or lines[0].split("\t") != ["premise", "hypothesis", "label"]:
        anomalies.append("missing or unexpected header row")
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            anomalies.append(f"line {i}: blank")
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            anomalies.append(f"line {i}: {len(parts)} tab-separated fields (expected 3)")
            continue
        premise, hypothesis, label = parts
        if label not in {"0", "1", "2"}:
            anomalies.append(f"line {i}: unexpected label {label!r}")
            continue
        records.append(PairRecord(premise, hypothesis, label=label))
    return records, anomalies


def load_semrel(path: Path) -> tuple[List[PairRecord], List[str]]:
    """SemRel2024 CSV: ``PairID,Text,Score`` where Text holds BOTH sentences.

    The two sentences live inside one quoted field separated by a newline
    (discovered in the P2 audit) - hence the csv module, not naive splitting.
    """
    anomalies: List[str] = []
    records: List[PairRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["PairID", "Text", "Score"]:
            anomalies.append(f"unexpected columns: {reader.fieldnames}")
        for row in reader:
            pair_id = (row.get("PairID") or "").strip()
            text = row.get("Text") or ""
            score_raw = (row.get("Score") or "").strip()
            parts = text.split("\n")
            if len(parts) != 2:
                anomalies.append(
                    f"{pair_id}: Text splits into {len(parts)} lines (expected 2)"
                )
                if len(parts) < 2:
                    continue
            sentence1, sentence2 = parts[0].strip(), parts[1].strip()
            try:
                score = float(score_raw)
            except ValueError:
                anomalies.append(f"{pair_id}: unparsable score {score_raw!r}")
                continue
            if not (0.0 <= score <= 1.0):
                anomalies.append(f"{pair_id}: score {score} outside [0,1]")
            records.append(PairRecord(sentence1, sentence2, score=score, meta={"pair_id": pair_id}))
    return records, anomalies


def load_parallel(en_path: Path, target_path: Path) -> tuple[List[PairRecord], List[str]]:
    """Line-aligned parallel plain-text files (IGBONLP .en/.ig; no headers).

    The two sentences are translations of each other (NOT paraphrases);
    loaders only aligns lines - pair construction happens in Phase 3.
    """
    anomalies: List[str] = []
    src_lines = _read_text(en_path).splitlines()
    tgt_lines = _read_text(target_path).splitlines()
    if len(src_lines) != len(tgt_lines):
        anomalies.append(
            f"line count mismatch: {en_path.name}={len(src_lines)} vs {target_path.name}={len(tgt_lines)}"
        )
    records = [
        PairRecord(s, t, meta={"source_file": en_path.name})
        for s, t in zip(src_lines, tgt_lines)
    ]
    return records, anomalies


def load_menyo(path: Path) -> tuple[List[PairRecord], List[str]]:
    """Menyo-20k TSV: header ``English<TAB>Yoruba``."""
    anomalies: List[str] = []
    records: List[PairRecord] = []
    lines = _read_text(path).splitlines()
    if not lines or lines[0].split("\t") != ["English", "Yoruba"]:
        anomalies.append("missing or unexpected header row")
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 2:
            anomalies.append(f"line {i}: {len(parts)} tab-separated fields (expected 2)")
            continue
        records.append(PairRecord(parts[0], parts[1], meta={"split_file": path.name}))
    return records, anomalies


def load_ntrex(path: Path) -> tuple[List[PairRecord], List[str]]:
    """NTREX TSV: header ``sentence_<target><TAB>sentence_eng``.

    Returns translation pairs (same caveat as parallel corpora).
    """
    anomalies: List[str] = []
    records: List[PairRecord] = []
    lines = _read_text(path).splitlines()
    if not lines:
        anomalies.append("empty file")
        return records, anomalies
    header = lines[0].split("\t")  # shape recorded; language code varies per file
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 2:
            anomalies.append(f"line {i}: {len(parts)} tab-separated fields (expected 2)")
            continue
        records.append(PairRecord(parts[0], parts[1], meta={"split_file": path.name}))
    return records, anomalies


Loaders: Dict[str, Callable[..., tuple[List[PairRecord], List[str]]]] = {
    "afrixnli": load_afrixnli,
    "semrel": load_semrel,
    "parallel": load_parallel,
    "menyo": load_menyo,
    "ntrex": load_ntrex,
}


def read_text(path: Path) -> str:
    """Public wrapper around the BOM-safe reader (used by audit/splits)."""
    return _read_text(path)
