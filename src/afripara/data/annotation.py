"""Annotation data handling for gold validation (single annotator).

Separates the label-store logic (pure, testable) from the Streamlit
frontend (scripts/annotate_app.py). Labels are appended to a CSV with
timestamps; candidates are never modified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("afripara.annotation")

CANDIDATE_COLUMNS = ["pair_id", "sentence1", "sentence2", "language", "english_s1", "english_s2", "source_id"]
LABEL_COLUMNS = ["pair_id", "round", "label", "score", "comment", "seconds", "timestamp"]


@dataclass
class Candidate:
    pair_id: str
    sentence1: str
    sentence2: str
    language: str
    english_s1: str = ""
    english_s2: str = ""
    source_id: str = ""
    repeat: bool = False  # part of the self-consistency block


def load_candidates(path: Path) -> List[Candidate]:
    """Load candidate pairs from CSV (columns per CANDIDATE_COLUMNS)."""
    df = pd.read_csv(path)
    missing = {"pair_id", "sentence1", "sentence2"} - set(df.columns)
    if missing:
        raise ValueError(f"Candidates CSV missing columns: {sorted(missing)}")
    out: List[Candidate] = []
    for _, row in df.iterrows():
        out.append(
            Candidate(
                pair_id=str(row["pair_id"]),
                sentence1=str(row["sentence1"]),
                sentence2=str(row["sentence2"]),
                language=str(row.get("language", "")),
                english_s1=str(row.get("english_s1", "") or ""),
                english_s2=str(row.get("english_s2", "") or ""),
                source_id=str(row.get("source_id", "") or ""),
            )
        )
    return out


class AnnotationStore:
    """Append-only label store with resume support and two rounds."""

    def __init__(self, labels_path: Path):
        self.labels_path = labels_path
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        if self.labels_path.exists():
            self._labels: pd.DataFrame = pd.read_csv(self.labels_path)
        else:
            self._labels = pd.DataFrame(columns=LABEL_COLUMNS)

    @property
    def labels(self) -> pd.DataFrame:
        return self._labels

    def labeled_ids(self, round_no: int) -> set:
        if self._labels.empty:
            return set()
        return set(self._labels.loc[self._labels["round"] == round_no, "pair_id"].astype(str))

    def next_candidates(self, candidates: List[Candidate], round_no: int) -> List[Candidate]:
        """Candidates still to label in this round (round 2 = repeats only)."""
        done = self.labeled_ids(round_no)
        pool = [c for c in candidates if (round_no == 1 or c.repeat)]
        return [c for c in pool if c.pair_id not in done]

    def save_label(
        self,
        pair_id: str,
        round_no: int,
        label: Optional[int],
        score: Optional[int],
        comment: str,
        seconds: float,
    ) -> None:
        row = {
            "pair_id": pair_id,
            "round": round_no,
            "label": label if label is not None else pd.NA,
            "score": score if score is not None else pd.NA,
            "comment": comment,
            "seconds": round(seconds, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._labels = pd.concat([self._labels, pd.DataFrame([row])], ignore_index=True)
        self._labels.to_csv(self.labels_path, index=False)
        logger.info("Saved label %s (round %d, label=%s)", pair_id, round_no, label)

    def stats(self, round_no: int) -> Dict[str, int]:
        sub = self._labels[self._labels["round"] == round_no] if not self._labels.empty else self._labels
        labelled = sub["label"].notna().sum() if not sub.empty else 0
        skipped = (sub["label"].isna().sum() if not sub.empty else 0)
        return {"labelled": int(labelled), "skipped": int(skipped), "total": int(len(sub))}

    def self_agreement(self) -> Optional[float]:
        """Raw agreement rate between round 1 and round 2 on repeated pairs."""
        if self._labels.empty:
            return None
        r1 = self._labels[(self._labels["round"] == 1) & self._labels["label"].notna()].set_index("pair_id")["label"]
        r2 = self._labels[(self._labels["round"] == 2) & self._labels["label"].notna()].set_index("pair_id")["label"]
        shared = r1.index.intersection(r2.index)
        if len(shared) == 0:
            return None
        return float((r1.loc[shared] == r2.loc[shared]).mean())
