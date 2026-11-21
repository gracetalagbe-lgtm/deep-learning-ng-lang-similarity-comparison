"""Data-quality audit for the raw datasets (Phase 2).

Produces machine-readable reports (data/interim/audit/report.json) and a
human-readable summary (data/interim/audit/AUDIT_REPORT.md). The audit
NEVER modifies raw files.

Checks per dataset:
  - parse anomalies (from loaders): field counts, headers, unparsable values
  - encoding: strict UTF-8; BOM presence; U+FFFD replacement characters
  - missing/empty sentences
  - duplicates: identical sentence-pairs and identical single sentences
  - labels/scores: distribution and range
  - length statistics (characters and whitespace tokens)
  - orthography signals: language-specific characters + diacritic density
    (heuristic metadata validation; model-based LID runs later on
    constructed data, where language metadata is not already trusted)

Usage:
    python -m afripara.data.audit            # audit everything
    python -m afripara.data.audit --list     # list audited datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from afripara.config import PROJECT_ROOT, resolve_path
from afripara.data.loaders import (
    PairRecord,
    load_afrixnli,
    load_menyo,
    load_ntrex,
    load_parallel,
    load_semrel,
)
from afripara.features.normalization import clean_text, diacritic_density
from afripara.utils.logging_utils import setup_logging

logger = logging.getLogger("afripara.audit")

# Language-specific orthography signals (heuristic, not proof of language).
ORTHOGAPHY_MARKERS = {
    "yor": ["ẹ", "ọ", "ṣ", "ǹ", "ń", "à", "è", "ó", "ú"],
    "ibo": ["ị", "ọ", "ụ", "ṅ"],
    "hau": ["ƙ", "ɓ", "ɗ", "ƴ", "ʼ", "'"],  # hooked letters + glottal apostrophe
}


def _stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    values_sorted = sorted(values)
    n = len(values_sorted)
    return {
        "n": n,
        "min": values_sorted[0],
        "p50": values_sorted[n // 2],
        "p95": values_sorted[int(n * 0.95)],
        "max": values_sorted[-1],
        "mean": round(sum(values_sorted) / n, 2),
    }


def audit_pairs(
    name: str,
    records: List[PairRecord],
    anomalies: List[str],
    expected_language: str | None = None,
) -> Dict:
    """Run the shared audit battery over parsed pair records."""
    findings: Dict = {"name": name, "n_records": len(records), "anomalies": {}}

    # --- anomalies from parsing ---
    anomaly_counts = Counter(anomalies)
    findings["anomalies"] = dict(anomaly_counts.most_common())

    # --- encoding / text quality (from raw bytes level issues, plus sentinels) ---
    replacement = sum(rec.sentence1.count("�") + rec.sentence2.count("�") for rec in records)
    findings["replacement_chars"] = replacement

    # --- empty fields ---
    empty = sum(
        1
        for rec in records
        if not rec.sentence1.strip() or not rec.sentence2.strip()
    )
    findings["empty_sentences"] = empty

    # --- duplicates ---
    pair_counter = Counter(
        (clean_text(r.sentence1), clean_text(r.sentence2)) for r in records
    )
    findings["duplicate_pairs"] = sum(c - 1 for c in pair_counter.values() if c > 1)
    s1_counter = Counter(clean_text(r.sentence1) for r in records)
    s2_counter = Counter(clean_text(r.sentence2) for r in records)
    findings["duplicate_sentence1"] = sum(c - 1 for c in s1_counter.values() if c > 1)
    findings["duplicate_sentence2"] = sum(c - 1 for c in s2_counter.values() if c > 1)
    # Cross-side duplicates: same sentence used as both s1 somewhere and s2
    # somewhere (relevant for leakage planning in P3).
    cross = set(s1_counter) & set(s2_counter)
    findings["cross_side_duplicate_sentences"] = len(cross)

    # --- labels / scores ---
    if records and records[0].label is not None:
        findings["label_distribution"] = dict(Counter(r.label for r in records))
    if records and records[0].score is not None:
        scores = [r.score for r in records if r.score is not None]
        findings["score_stats"] = _stats([int(s * 100) for s in scores])
        findings["score_histogram"] = dict(
            sorted(Counter(round(s, 1) for s in scores).items())
        )

    # --- length statistics (characters and whitespace tokens, cleaned text) ---
    all_text = [clean_text(r.sentence1) for r in records] + [clean_text(r.sentence2) for r in records]
    findings["char_lengths"] = _stats([len(t) for t in all_text])
    findings["token_lengths"] = _stats([len(t.split()) for t in all_text])
    findings["over_128_chars"] = sum(1 for t in all_text if len(t) > 128)
    findings["over_64_words"] = sum(1 for t in all_text if len(t.split()) > 64)

    # --- orthography signals (heuristic language metadata validation) ---
    sample = all_text
    findings["diacritic_density_mean"] = round(
        sum(diacritic_density(t) for t in sample) / max(len(sample), 1), 4
    )
    if expected_language:
        markers = ORTHOGAPHY_MARKERS.get(expected_language, [])
        if markers:
            hits = sum(1 for t in sample if any(m in t for m in markers))
            findings["marker_coverage"] = round(hits / max(len(sample), 1), 4)
        findings["expected_language"] = expected_language

    # --- flag summary: conditions that need a decision before P3 ---
    flags = []
    if empty:
        flags.append(f"{empty} empty sentences")
    if findings.get("replacement_chars"):
        flags.append(f"{findings['replacement_chars']} U+FFFD replacement chars")
    if findings.get("duplicate_pairs"):
        flags.append(f"{findings['duplicate_pairs']} duplicate pairs")
    if expected_language and findings.get("marker_coverage", 1.0) < 0.3:
        flags.append(
            f"orthography markers in only {findings.get('marker_coverage', 0):.0%} "
            f"of texts (expected for {expected_language})"
        )
    findings["flags"] = flags
    return findings


def audit_cross_split(raw_dir: Path, report: Dict) -> Dict:
    """Sentence-level overlap across official splits of each corpus.

    Leakage check required before any of these splits are reused in P3/P5:
    if the same sentence appears in train and test, models can memorise
    rather than generalise. Returns per-corpus shared-sentence counts.
    """
    out: Dict = {}

    def sent_set(records: List[PairRecord], side: str) -> set:
        return {clean_text(getattr(r, side)) for r in records if clean_text(getattr(r, side))}

    # AfriXNLI: dev vs test, per language (both pair members)
    for lang in ["hau", "ibo", "yor", "eng"]:
        dev, _ = load_afrixnli(raw_dir / "afrixnli" / f"{lang}_dev.tsv")
        test, _ = load_afrixnli(raw_dir / "afrixnli" / f"{lang}_test.tsv")
        dev_sents = sent_set(dev, "sentence1") | sent_set(dev, "sentence2")
        test_sents = sent_set(test, "sentence1") | sent_set(test, "sentence2")
        out[f"afrixnli/{lang}"] = {
            "dev_sentences": len(dev_sents),
            "test_sentences": len(test_sents),
            "shared": len(dev_sents & test_sents),
        }

    # SemRel Hausa: train vs dev vs test
    splits = {}
    for split, fname in [("train", "hau_train.csv"), ("dev", "hau_dev_with_labels.csv"), ("test", "hau_test_with_labels.csv")]:
        splits[split], _ = load_semrel(raw_dir / "semrel2024_hau" / fname)
    sets = {k: sent_set(v, "sentence1") | sent_set(v, "sentence2") for k, v in splits.items()}
    out["semrel2024_hau"] = {
        "sizes": {k: len(v) for k, v in sets.items()},
        "shared_train_dev": len(sets["train"] & sets["dev"]),
        "shared_train_test": len(sets["train"] & sets["test"]),
        "shared_dev_test": len(sets["dev"] & sets["test"]),
    }

    # Menyo (translation corpus: check BOTH sides)
    menyo_sets: Dict[str, Dict[str, set]] = {}
    for split in ["train", "dev", "test"]:
        recs, _ = load_menyo(raw_dir / "menyo20k_mt" / f"{split}.tsv")
        menyo_sets[split] = {
            "eng": sent_set(recs, "sentence1"),
            "yor": sent_set(recs, "sentence2"),
        }
    out["menyo20k_mt"] = {
        "shared_eng_train_test": len(menyo_sets["train"]["eng"] & menyo_sets["test"]["eng"]),
        "shared_yor_train_test": len(menyo_sets["train"]["yor"] & menyo_sets["test"]["yor"]),
        "shared_eng_train_dev": len(menyo_sets["train"]["eng"] & menyo_sets["dev"]["eng"]),
        "shared_yor_train_dev": len(menyo_sets["train"]["yor"] & menyo_sets["dev"]["yor"]),
    }

    # IGBONLP
    igbo_sets: Dict[str, Dict[str, set]] = {}
    for split in ["train", "val", "test"]:
        recs, _ = load_parallel(
            raw_dir / "igbonlp_benchmark" / f"{split}.en",
            raw_dir / "igbonlp_benchmark" / f"{split}.ig",
        )
        igbo_sets[split] = {
            "eng": sent_set(recs, "sentence1"),
            "ibo": sent_set(recs, "sentence2"),
        }
    out["igbonlp_benchmark"] = {
        "shared_eng_train_val": len(igbo_sets["train"]["eng"] & igbo_sets["val"]["eng"]),
        "shared_eng_train_test": len(igbo_sets["train"]["eng"] & igbo_sets["test"]["eng"]),
        "shared_ibo_train_val": len(igbo_sets["train"]["ibo"] & igbo_sets["val"]["ibo"]),
        "shared_ibo_train_test": len(igbo_sets["train"]["ibo"] & igbo_sets["test"]["ibo"]),
    }
    return out


def audit_all(raw_dir: Path, out_dir: Path) -> Dict:
    """Audit every registered dataset; returns the full report dict."""
    report: Dict = {"generated_for": "afripara P2", "datasets": {}}

    # --- AfriXNLI (per language x split) ---
    for lang in ["hau", "ibo", "yor", "eng"]:
        for split in ["dev", "test"]:
            path = raw_dir / "afrixnli" / f"{lang}_{split}.tsv"
            records, anomalies = load_afrixnli(path)
            key = f"afrixnli/{lang}/{split}"
            report["datasets"][key] = audit_pairs(
                key, records, anomalies, expected_language=lang if lang != "eng" else None
            )

    # --- SemRel2024 Hausa ---
    for split, fname in [
        ("train", "hau_train.csv"),
        ("dev", "hau_dev_with_labels.csv"),
        ("test", "hau_test_with_labels.csv"),
    ]:
        path = raw_dir / "semrel2024_hau" / fname
        records, anomalies = load_semrel(path)
        key = f"semrel2024_hau/{split}"
        report["datasets"][key] = audit_pairs(key, records, anomalies, expected_language="hau")

    # --- Menyo-20k ---
    for split in ["train", "dev", "test"]:
        path = raw_dir / "menyo20k_mt" / f"{split}.tsv"
        records, anomalies = load_menyo(path)
        key = f"menyo20k_mt/{split}"
        report["datasets"][key] = audit_pairs(key, records, anomalies, expected_language="yor")

    # --- IGBONLP benchmark (parallel line-aligned) ---
    for split in ["train", "val", "test"]:
        records, anomalies = load_parallel(
            raw_dir / "igbonlp_benchmark" / f"{split}.en",
            raw_dir / "igbonlp_benchmark" / f"{split}.ig",
        )
        key = f"igbonlp_benchmark/{split}"
        report["datasets"][key] = audit_pairs(key, records, anomalies, expected_language="ibo")

    # --- NTREX-african ---
    for lang in ["hau", "ibo", "yor"]:
        path = raw_dir / "ntrex_african" / f"{lang}_Latn.tsv"
        records, anomalies = load_ntrex(path)
        key = f"ntrex_african/{lang}"
        report["datasets"][key] = audit_pairs(key, records, anomalies, expected_language=lang)

    # --- BOM check across raw text files (found in Menyo) ---
    bom_files = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix in {".tsv", ".csv", ".en", ".ig"}:
            if path.read_bytes().startswith(b"\xef\xbb\xbf"):
                bom_files.append(str(path.relative_to(raw_dir)))
    report["files_with_bom"] = bom_files
    report["cross_split_overlap"] = audit_cross_split(raw_dir, report)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report


def write_markdown_summary(report: Dict, out_dir: Path) -> None:
    """Human-readable audit summary for the research log / thesis appendix."""
    lines = [
        "# Data Audit Report (Phase 2)",
        "",
        f"Datasets audited: **{len(report['datasets'])}**. "
        "Raw originals in `data/raw/` were not modified. Full detail: `report.json`.",
        "",
        "| Dataset | Records | Dup. pairs | Empty | Anomalies | Flags |",
        "|---|---|---|---|---|---|",
    ]
    for key, f in report["datasets"].items():
        n_anom = sum(f["anomalies"].values())
        flags = "; ".join(f["flags"]) if f["flags"] else "-"
        lines.append(
            f"| {key} | {f['n_records']} | {f.get('duplicate_pairs', 0)} "
            f"| {f.get('empty_sentences', 0)} | {n_anom} | {flags} |"
        )
    lines += [
        "",
        "## Label and score distributions",
        "",
    ]
    for key, f in report["datasets"].items():
        if "label_distribution" in f:
            lines.append(f"- **{key}**: labels {f['label_distribution']}")
        if "score_stats" in f:
            lines.append(
                f"- **{key}**: score mean={f['score_stats'].get('mean', 0) / 100:.3f}, "
                f"p50={f['score_stats'].get('p50', 0) / 100:.2f}, "
                f"min={f['score_stats'].get('min', 0) / 100:.2f}, "
                f"max={f['score_stats'].get('max', 0) / 100:.2f}"
            )
    lines += [
        "",
        "## Orthography signals (heuristic language-metadata validation)",
        "",
    ]
    for key, f in report["datasets"].items():
        if "expected_language" in f:
            lines.append(
                f"- **{key}** [{f['expected_language']}]: diacritic density "
                f"{f.get('diacritic_density_mean', 0):.3f}, marker coverage "
                f"{f.get('marker_coverage', 0):.1%}"
            )
    if report.get("files_with_bom"):
        lines += ["", f"Files starting with a UTF-8 BOM: {', '.join(report['files_with_bom'])}"]
    lines += [
        "",
        "## Known limits of this audit",
        "",
        "- Orthography markers are a *heuristic* metadata validation, not proof of language.",
        "- Model-based language identification (AfroLID) is deferred to Phase 3, where it "
        "matters most: constructed pairs from multi-source pools.",
        "- Semantic annotation quality (are labels correct?) cannot be audited automatically; "
        "it is addressed by construction design + human gold validation (Phase 3).",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    setup_logging()
    raw_dir = PROJECT_ROOT / "data" / "raw"
    out_dir = PROJECT_ROOT / "data" / "interim" / "audit"
    report = audit_all(raw_dir, out_dir)
    write_markdown_summary(report, out_dir)
    n_flags = sum(len(f.get("flags", [])) for f in report["datasets"].values())
    logger.info("Audited %d datasets; %d flag(s); reports in %s", len(report["datasets"]), n_flags, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
