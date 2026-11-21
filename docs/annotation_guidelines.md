# Annotation Guidelines — Gold Validation of Constructed Yoruba Pairs

**Annotator:** the researcher (native speaker of Yorùbá).
**Purpose:** transform *candidate* constructed pairs into *gold* labelled
data. Every other constructed pair remains silver-quality and is reported as
such in all results.

---

## 1. What you are labelling

You will see sentence pairs in Yorùbá (and, for a calibration block, some
pairs translated from English PAWS - the original English text is shown for
context but you must judge **the Yorùbá text itself**, not the English).

Give each pair:
1. **A binary paraphrase label** (the primary gold label):
   - **1 — Paraphrase / near-equivalent**
   - **0 — Not paraphrase**
2. **An optional relatedness score 0–4** (secondary, for similarity evaluation):
   | Score | Meaning |
   |---|---|
   | 4 | Equivalent — same information, could replace each other |
   | 3 | Near-equivalent — a small detail differs but the core message is the same |
   | 2 | Related — same topic, clearly different information |
   | 1 | Slightly related — topical overlap only |
   | 0 | Unrelated |
3. **Optional comment** (free text) — especially for strange machine-translated text.

There is also a **Skip** option for pairs you cannot judge at all (e.g.,
meaningless output). Skipped pairs are excluded from gold data and counted in
the MT quality report.

## 2. Decision rule for the binary label

Label **1** when a reader who trusted sentence A could rely on it for the
key information of sentence B, and vice versa. Wording, word order, and
grammar quality **do not matter** — meaning does.

Label **0** when any of these holds:
- The two sentences are about the same topic but convey different facts.
- **Negation or reversal**: one sentence states what the other denies
  (e.g., "ó dé" he arrived vs "kò dé" he did not arrive → 0).
- Different participants, places, times, or quantities.
- One sentence adds or removes key information (asymmetric).
- A name/number refers to different entities or values.

When genuinely torn between 1 and 0, prefer **0** (be strict — this makes the
gold set a conservative test) and note the pair in a comment.

## 3. Special situations in this data

- **Translationese / stilted grammar**: machine-translated sentences may
  sound awkward. Judge *meaning*, not fluency. "Awon omo nso pe..." is fine
  if the meaning is recoverable and matches.
- **Undiacritised text**: some sentences lack tone marks. Read them the same
  way you would read ordinary undiacritised Yoruba writing; missing marks do
  not by themselves make a pair a non-paraphrase.
- **English calibration block**: the first ~20 pairs show the original
  English for orientation only. After that, judge blind.
- **Truly meaningless MT output** (unrecoverable) → Skip, with comment.

## 4. Practical plan

- Target: **400 pairs** (see `configs/default.yaml → construction.gold_annotation_target`).
- Estimated effort: 45–60 seconds per pair ≈ 5–7 hours total; the tool saves
  each label immediately, so you can work in sessions of any length.
- **Self-consistency block**: 40 pairs are shown twice, at least a week
  apart (the tool queues them automatically). Self-agreement (Cohen's κ
  against yourself) is reported — this is our only agreement statistic,
  since we have a single annotator; it is reported as a limitation.
- Qualification check: the first 10 pairs are pre-labelled by construction
  intent (PAWS English originals with known labels) — aim for ≥ 8/10
  agreement before continuing.

## 5. What happens to your labels

- Labels are stored in `data/processed/annotation/yor_gold_labels.csv` with
  timestamps; raw candidates are never overwritten.
- Gold pairs form the **Yoruba gold test/eval set** (E1–E4, E6). Silver pairs
  may be used for *training*; gold pairs are reserved for evaluation.
- The annotation time span, guidelines version, and self-κ will be reported
  in the thesis (annotation provenance).
