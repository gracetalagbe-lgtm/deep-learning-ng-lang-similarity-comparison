#!/usr/bin/env python3
"""Streamlit annotation tool for gold validation (single annotator).

Run:
    .venv/bin/streamlit run scripts/annotate_app.py

Candidate and label file paths are set in the sidebar (defaults point to
data/processed/annotation/). If the constructed candidates file does not
exist yet, a preview set from real AfriXNLI Yoruba dev data is offered so
the workflow can be tested before MT output arrives.

Every action saves to the labels CSV immediately, so progress survives
interruptions. No labelling logic lives here - that is
afripara.data.annotation (unit-tested).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.data.annotation import AnnotationStore, Candidate, load_candidates  # noqa: E402
from afripara.data.loaders import load_afrixnli  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
DEFAULT_CANDIDATES = PROJECT_ROOT / "data/processed/annotation/yor_gold_candidates.csv"
DEFAULT_LABELS = PROJECT_ROOT / "data/processed/annotation/yor_gold_labels.csv"


def fallback_candidates() -> list[Candidate]:
    """Preview set from real AfriXNLI Yoruba dev data (existing data only)."""
    records, _ = load_afrixnli(RAW / "afrixnli" / "yor_dev.tsv")
    return [
        Candidate(
            pair_id=f"preview_{i}",
            sentence1=rec.sentence1,
            sentence2=rec.sentence2,
            language="yor",
            english_s1="(preview pair from AfriXNLI dev — real data)",
            source_id=f"afrixnli_dev_{i}",
        )
        for i, rec in enumerate(records[:20])
    ]


def main() -> None:
    st.set_page_config(page_title="Yorùbá Gold Annotation", layout="centered")
    st.title("Yorùbá Gold Annotation")
    st.markdown(
        "Guidelines: `docs/annotation_guidelines.md` — judge **meaning**, not "
        "fluency or diacritics. When torn, choose **Not paraphrase** (strict)."
    )

    with st.sidebar:
        st.header("Settings")
        candidates_path = Path(st.text_input("Candidates CSV", value=str(DEFAULT_CANDIDATES)))
        labels_path = Path(st.text_input("Labels CSV (appended)", value=str(DEFAULT_LABELS)))
        round_no = st.radio("Round", [1, 2], format_func=lambda r: {1: "Round 1 (main)", 2: "Round 2 (self-consistency repeats)"}[r])

    if candidates_path.exists():
        candidates = load_candidates(candidates_path)
        st.sidebar.success(f"Loaded {len(candidates)} candidates")
    else:
        candidates = fallback_candidates()
        st.sidebar.warning("Constructed candidates not found — using a 20-pair preview from AfriXNLI (real data).")

    store = AnnotationStore(labels_path)
    remaining = store.next_candidates(candidates, round_no)
    stats = store.stats(round_no)

    st.sidebar.metric("Labeled (this round)", stats["labelled"])
    st.sidebar.metric("Skipped", stats["skipped"])
    st.sidebar.metric("Remaining", len(remaining))
    if round_no == 2:
        agreement = store.self_agreement()
        if agreement is not None:
            st.sidebar.metric("Self-agreement so far", f"{agreement:.0%}")

    if not remaining:
        st.success("All candidates labelled in this round. 🎉")
        if round_no == 1:
            st.info(
                "Switch to Round 2 in the sidebar after at least a week to "
                "label the self-consistency repeats (guidelines §4)."
            )
        return

    current = remaining[0]
    st.progress(1 - len(remaining) / max(len(candidates), 1))
    st.caption(f"pair `{current.pair_id}` · {len(remaining)} remaining")

    if current.english_s1:
        with st.expander("English context (calibration only — judge the Yorùbá)"):
            st.write(current.english_s1)
            if current.english_s2:
                st.write(current.english_s2)

    st.markdown(f"### Sentence A\n{current.sentence1}")
    st.markdown(f"### Sentence B\n{current.sentence2}")

    score = st.slider("Relatedness score (optional, 0–4)", min_value=0, max_value=4, step=1, help="Optional; guidelines §1")
    comment = st.text_input("Comment (optional)")

    t0 = st.session_state.get("t0")
    if st.session_state.get("pair_id") != current.pair_id:
        st.session_state["pair_id"] = current.pair_id
        st.session_state["t0"] = time.time()
        t0 = st.session_state["t0"]

    col1, col2, col3 = st.columns(3)
    if col1.button("✅ 1 — Paraphrase", type="primary", use_container_width=True):
        store.save_label(current.pair_id, round_no, 1, score, comment, time.time() - (t0 or time.time()))
        st.rerun()
    if col2.button("❌ 0 — Not paraphrase", use_container_width=True):
        store.save_label(current.pair_id, round_no, 0, score, comment, time.time() - (t0 or time.time()))
        st.rerun()
    if col3.button("⏭ Skip (unjudgeable)", use_container_width=True):
        store.save_label(current.pair_id, round_no, None, score, comment or "skipped", time.time() - (t0 or time.time()))
        st.rerun()


if __name__ == "__main__":
    main()
