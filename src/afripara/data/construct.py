"""Constructed pair pipeline (Phase 3).

Two construction routes, both honest about label provenance:

1. **translate-train (PAWS route)** - primary source of constructed data.
   An English PAWS-Wiki pair (human-written, adversarially controlled lexical
   overlap, human binary label) has BOTH sentences machine-translated into
   the target language. The target-language pair inherits the label. This
   preserves PAWS's crucial property: high-overlap non-paraphrases and
   low-overlap paraphrases, so models cannot solve the task by lexical
   overlap alone. MT noise is a known limitation: candidates are filtered
   mechanically here, and a native-speaker gold subset is validated by the
   researcher (Yoruba) before evaluation use.

2. **MT-vs-human (parallel-corpus route)** - supplementary positives.
   An English source sentence translated by NLLB is paired with the
   *independent human* translation from Menyo-20k / IGBONLP. Same meaning,
   different (human vs MT) wording. Label = 1 candidate-positive.

Every exported pair carries provenance columns (`construction`, `source_id`,
English text where applicable, QC stats). Constructed pairs are NEVER mixed
silently with human-labelled data: the `label_source` column distinguishes
`paws_translated` (inherited human English label) from `mt_vs_human`
(candidate, requires validation) from `semrel_human` (human 0-1 score).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from afripara.data.loaders import PairRecord
from afripara.features.normalization import clean_text

logger = logging.getLogger("afripara.construct")

PAIR_COLUMNS = [
    "pair_id",
    "sentence1",
    "sentence2",
    "label",           # binary label (1 = paraphrase/near-equivalent)
    "score",           # continuous relatedness where available (semrel)
    "label_source",    # paws_translated | mt_vs_human | semrel_human
    "language",
    "split",
    "construction",    # paws_translate_train | mt_vs_human | none
    "source_id",       # PAWS id / pair id / document id
    "english_s1",      # English source text (provenance; may be empty)
    "english_s2",
    "qc_len_ratio",    # character length ratio max/min of the pair (target lang)
]


# --------------------------------------------------------------------------- #
# PAWS handling
# --------------------------------------------------------------------------- #

def load_paws(raw_dir: Path, split: str) -> pd.DataFrame:
    """Load the PAWS-Wiki labelled parquet (columns: id, sentence1, sentence2, label)."""
    path = raw_dir / "paws_wiki_en" / f"{split}.parquet"
    df = pd.read_parquet(path)
    expected = {"id", "sentence1", "sentence2", "label"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Unexpected PAWS columns: {df.columns.tolist()}")
    return df


def select_translate_train(
    paws: pd.DataFrame,
    n_pairs: int,
    seed: int,
    max_pair_chars: int = 200,
) -> pd.DataFrame:
    """Select pairs for translation: prefer shorter pairs, keep natural label
    balance, deduplicate, deterministic under ``seed``."""
    df = paws.copy()
    df = df[(df.sentence1.str.len() <= max_pair_chars) & (df.sentence2.str.len() <= max_pair_chars)]
    df = df.drop_duplicates(subset=["sentence1", "sentence2"])
    df = df.sort_values(["sentence1", "sentence2"]).reset_index(drop=True)
    if len(df) <= n_pairs:
        return df
    return df.sample(n=n_pairs, random_state=seed).sort_values("id").reset_index(drop=True)


def prep_mt_input(paws_subset: pd.DataFrame, out_path: Path) -> Path:
    """Write the JSONL MT input: one record per sentence, both pair members."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for _, row in paws_subset.iterrows():
            fh.write(json.dumps({"id": f"paws{row['id']}_s1", "text": row["sentence1"], "pair_id": int(row["id"]), "side": "s1"}, ensure_ascii=False) + "\n")
            fh.write(json.dumps({"id": f"paws{row['id']}_s2", "text": row["sentence2"], "pair_id": int(row["id"]), "side": "s2"}, ensure_ascii=False) + "\n")
    logger.info("MT input written: %s (%d sentences)", out_path, 2 * len(paws_subset))
    return out_path


def load_mt_output(path: Path) -> Dict[str, str]:
    """Read a JSONL MT output file -> {sentence_id: translation}."""
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["id"]] = rec.get("mt", "")
    return out


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def _len_ratio(a: str, b: str) -> float:
    la, lb = max(len(a), 1), max(len(b), 1)
    return max(la, lb) / min(la, lb)


def build_translate_train_pairs(
    paws_subset: pd.DataFrame,
    mt: Dict[str, str],
    language: str,
) -> Tuple[List[PairRecord], Dict[str, int]]:
    """Build monolingual target-language pairs from PAWS pairs + MT outputs.

    Drops pairs with missing/empty translations or extreme length ratios
    (quality filter on MT noise; recorded in the returned stats).
    """
    stats = {"candidates": 0, "kept": 0, "dropped_empty_mt": 0, "dropped_len_ratio": 0}
    pairs: List[PairRecord] = []
    for _, row in paws_subset.iterrows():
        stats["candidates"] += 1
        s1 = mt.get(f"paws{row['id']}_s1", "").strip()
        s2 = mt.get(f"paws{row['id']}_s2", "").strip()
        if not s1 or not s2:
            stats["dropped_empty_mt"] += 1
            continue
        ratio = _len_ratio(s1, s2)
        if ratio > 3.0:
            stats["dropped_len_ratio"] += 1
            continue
        pairs.append(
            PairRecord(
                sentence1=clean_text(s1),
                sentence2=clean_text(s2),
                label=str(int(row["label"])),
                meta={
                    "label_source": "paws_translated",
                    "construction": "paws_translate_train",
                    "source_id": str(row["id"]),
                    "english_s1": row["sentence1"],
                    "english_s2": row["sentence2"],
                    "qc_len_ratio": f"{ratio:.2f}",
                    "language": language,
                },
            )
        )
        stats["kept"] += 1
    return pairs, stats


def build_mt_vs_human_pairs(
    mt_by_id: Dict[str, str],
    human_pairs: Iterable[Tuple[str, str, str]],
    language: str,
    max_len_ratio: float = 3.0,
) -> Tuple[List[PairRecord], Dict[str, int]]:
    """Pair an MT rendering with the independent human translation.

    ``human_pairs`` yields (source_id, human_target_sentence, english_source).
    The MT record id convention is ``f"{source_id}_mth"``.
    """
    stats = {"candidates": 0, "kept": 0, "dropped_empty_mt": 0, "dropped_len_ratio": 0}
    pairs: List[PairRecord] = []
    for source_id, human, english in human_pairs:
        stats["candidates"] += 1
        machine = mt_by_id.get(f"{source_id}_mth", "").strip()
        human = clean_text(human)
        if not machine or not human:
            stats["dropped_empty_mt"] += 1
            continue
        ratio = _len_ratio(machine, human)
        if ratio > max_len_ratio:
            stats["dropped_len_ratio"] += 1
            continue
        pairs.append(
            PairRecord(
                sentence1=clean_text(machine),
                sentence2=human,
                label="1",
                meta={
                    "label_source": "mt_vs_human",
                    "construction": "mt_vs_human",
                    "source_id": source_id,
                    "english_s1": english,
                    "english_s2": "",
                    "qc_len_ratio": f"{ratio:.2f}",
                    "language": language,
                },
            )
        )
        stats["kept"] += 1
    return pairs, stats


def deduplicate_pairs(pairs: List[PairRecord]) -> Tuple[List[PairRecord], int]:
    """Drop exact duplicate pairs (same cleaned sentence pair, either order)."""
    seen = set()
    kept: List[PairRecord] = []
    for p in pairs:
        key = tuple(sorted((clean_text(p.sentence1), clean_text(p.sentence2))))
        if key in seen:
            continue
        seen.add(key)
        kept.append(p)
    return kept, len(pairs) - len(kept)


def pairs_to_dataframe(pairs: List[PairRecord], split_by_fold: Dict[str, List[PairRecord]] | None = None) -> pd.DataFrame:
    """Export PairRecords (optionally with fold assignment) to a tidy frame."""
    rows = []
    source = split_by_fold or {"": pairs}
    for fold, group in source.items():
        for p in group:
            source_id = p.meta.get("source_id") or stable_pair_id(p.sentence1, p.sentence2, salt="afripara")
            rows.append({
                "pair_id": f"{p.meta.get('label_source', 'pair')}_{source_id}",
                "sentence1": p.sentence1,
                "sentence2": p.sentence2,
                "label": p.label,
                "score": p.score,
                "label_source": p.meta.get("label_source", ""),
                "language": p.meta.get("language", ""),
                "split": fold,
                "construction": p.meta.get("construction", ""),
                "source_id": p.meta.get("source_id", ""),
                "english_s1": p.meta.get("english_s1", ""),
                "english_s2": p.meta.get("english_s2", ""),
                "qc_len_ratio": p.meta.get("qc_len_ratio", ""),
            })
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def stable_pair_id(sentence1: str, sentence2: str, salt: str) -> str:
    """Deterministic pair id independent of pair order."""
    a, b = sorted((clean_text(sentence1), clean_text(sentence2)))
    return hashlib.md5(f"{salt}\x1e{a}\x1e{b}".encode("utf-8")).hexdigest()[:16]
