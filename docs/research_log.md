# Research Log

Running record of decisions, data provenance, experimental settings and
challenges for the MSc project *Transformer-Based Semantic Similarity and
Paraphrase Detection in Yorùbá, Igbo and Hausa*.

---

## 2026-11-20 — DECISION: gold annotation skipped (researcher)

The researcher elected to skip the remaining Phase 3 human-annotation work
(the planned 390-pair Yoruba gold validation) and the optional MT-vs-human
supplementary run, to prioritise completing the transformer experiments.

**Consequences, recorded explicitly:**
- Dataset construction itself is unaffected (constructed sets were already
  built and split-verified before this decision).
- All constructed-data results (yor, ibo, hau translate-train) are
  **silver-quality**: labels are inherited from human PAWS annotations via
  machine translation (translate-train practice, precedent: PAWS-X,
  Yang et al. 2019). No native-speaker verification of constructed pairs
  was performed; MT meaning-drift was observed in the pilot and is a known,
  documented risk.
- **Every results table carries a label_provenance column**
  (`semrel_human` > `afrixnli_proxy` > `paws_translated`), and headline
  conclusions about Yoruba/Igbo are stated as conditional on constructed-
  label quality. Hausa claims rest on SemRel2024 human annotations (real
  gold data, official + leakage-free splits).
- The E6 linguistic perturbation experiments run on the silver test sets
  (they require consistency, not gold labels).
- The annotation tooling (`scripts/annotate_app.py`,
  `scripts/make_annotation_export.py`, `docs/annotation_guidelines.md`,
  `data/processed/annotation/yor_gold_candidates.csv` + key) is retained
  unchanged: if time permits later, the gold round can still be run and the
  gold rows added to the evaluation tables.
- Offered alternative (not taken, may be revisited): a ~50-100 pair manual
  spot-check to quantify agreement between inherited and human labels.



---

## 2026-11-18 — Phase 0: Technical feasibility assessment

**Method.** All claims verified against primary sources on the date above:
Hugging Face Hub API (models/datasets), GitHub repositories and APIs, ACL
Anthology records, arXiv full text. Full report: `docs/feasibility_report.md`.

**Key findings (summary):**
- No paraphrase-identification or STS corpus exists for Yoruba, Igbo or
  Hausa (HF datasets API search; GitHub search; NaijaNLP survey,
  arXiv:2502.19784 records none). AfriMTEB (arXiv:2510.23896) omits STS for
  the same reason. → Dataset construction is a core contribution of this
  project.
- Hausa-only: SemRel2024 (SemEval-2024 Task 1) provides human-annotated
  sentence pairs with 0–1 relatedness scores. Licence: **unstated** — no
  LICENSE file in the repository.
- Model corrections: mBERT excludes Igbo and Hausa from its 104-language
  pretraining (canonical list at github.com/google-research/bert) → control
  role only. "Native-Bird" exists on HF but is unlicensed/unpublished →
  rejected. Verified primary encoder: AfroXLMR (Davlan, MIT, COLING 2022).
  Verified embedding models with all three languages: LaBSE (ACL 2022,
  Apache-2.0) and AfriE5 (MIT, arXiv:2510.23896).
- Licence red flag: JW300 withdrawn for copyright breach; JW300-derived
  folders exist inside IGBONLP (`masakhane_JW300_enig/`) and
  masakhane-wazobia (`Data/jw300/`) — excluded from this project.
- Compute: local machine is a 2-core Intel Mac (no GPU); transformer
  fine-tuning planned on Google Colab free T4. Fine-tuning defaults justified
  in feasibility report §8 (LR 2e-5, effective batch 32, warmup 10%, maxlen
  128, 3–5 epochs, early stopping on dev F1, ≥3 seeds).

---

## 2026-11-19 — Phase 0 decisions (researcher approved)

| Decision | Choice | Consequence |
|---|---|---|
| Annotation capacity | Researcher cannot recruit 500–1,000 pairs/language; is a **native speaker of Yoruba** and will personally validate | Gold-standard human validation is limited to **Yoruba** constructed pairs (~300–500). Hausa uses SemRel2024 human labels. Igbo uses silver-quality constructed data only — **reported as a limitation**; per-language results will be stratified by label provenance. |
| SemRel2024 licence | Do **not** contact authors; proceed with publicly available resources | SemRel2024 used as published research data; licence recorded as *unstated — research use only; no redistribution of the data or derivatives in any release*. |
| Language priority | Hausa → Yoruba → Igbo (approved) | Hausa has the strongest data; Igbo weakest (unstated licence on IGBONLP; no gold validation). |
| Compute | Google Colab free tier | Researcher needs setup instructions (provided in README + P5 notebooks). |
| Scope | Model list (§7) and three-tier data strategy (§6) approved | P1 implementation begins. |

---

## 2026-11-19 — Phase 1: Environment & project skeleton (completed)

**Actions.**
- Project folder renamed by researcher to `Nigeria Language` (feasibility
  report moved to `docs/`). All work now happens there.
- Package `afripara` created (src layout), modular per brief §12: `data`,
  `features`, `models`, `training`, `evaluation`, `inference`, `app`.
- Config system: `configs/default.yaml` + per-model configs with licence,
  coverage, citation and role metadata; `afripara.config` provides loading,
  deep-merge overrides, seeding and device detection.
- Normalisation module `afripara.features.normalization`: NFC Unicode
  canonicalisation, whitespace collapse, diacritic preservation by default,
  `strip_diacritics` restricted to the E6 ablation (with explicit mapping for
  Yoruba subdot/tone letters, Igbo subdots, Hausa hooked letters).
- Engineering fixtures (`afripara.utils.samples`): short greeting phrases per
  language — used for tokenizer/stack checks only, never as research data.
- Test suite (`tests/`): config loading/merge/seed determinism, model config
  provenance (licence must be present), diacritic preservation and ablation
  behaviour, stack import checks.
- Environment: project virtualenv `.venv` (Python 3.11), full ML stack
  installed; exact versions pinned in `requirements.txt`.
- Smoke test `scripts/smoke_test.py`: downloads `castorini/afriberta_small`
  (small CPU-friendly model covering all three languages; engineering check
  only), verifies tokenizer round-trips preserve diacritics, runs CPU forward
  passes.

**Completion (verified 2026-11-19 08:45).**
- 30/30 unit tests pass (`pytest`).
- Full smoke test **PASS** on `castorini/afriberta_small` (83.1M params):
  tokenizer round-trips preserve every diacritic for all three languages
  (e.g. `Ẹ kú àárọ̀` → `Ẹ kú àárọ̀`; `Ọ dị mma ịhụ gị` → identical); CPU
  forward passes return finite 768-dim hidden states in <0.1 s per batch.
- Stack: torch 2.2.2 / transformers 4.49.0 / sentence-transformers 3.4.1 /
  datasets 3.2.0 / numpy 1.26.4 / sklearn 1.9.1 / Python 3.11.7 (venv).

**Engineering findings (recorded for reproducibility).**
1. Platform pin: PyTorch publishes no macOS x86_64 wheels beyond 2.2.2 →
   local env pinned to torch 2.2.2 + numpy 1.26.4 + pyarrow 15.0.2 +
   transformers 4.49.0 (transformers v5 needs torch ≥ 2.6). Colab notebooks
   will pin their own coherent set; architecture/tokenizer identity is what
   matters for comparability, and the training config is shared via
   `configs/default.yaml`.
2. AfriBERTa checkpoints ship **no `tokenizer.json`** — the fast-tokenizer
   auto-conversion fails. `scripts/smoke_test.py` falls back to
   `use_fast=False`; `sentencepiece==0.1.99` pinned (newer releases have no
   macOS x86_64 wheel). This would have broken `afriberta_large` training in
   Phase 5; caught in Phase 1 by the smoke test.
3. Network measured at ~5–6 MB/s (83M-param model in 73 s). Consequence:
   multi-GB model downloads (LaBSE, AfriE5, afro-xlmr-base) are slow
   locally; policy — **fine-tuning checkpoints live on Colab/Google Drive;
   only the final demo checkpoint is pulled locally**, and the local HF
   cache stays limited to small models.
4. `Davlan/afro-xlmr-mini` verified live on HF (MIT) — candidate local
   development model (XLM-R fast tokenizer, no sentencepiece needed).

**Notes / deviations.**
- The P1 deliverable "smoke test on a 200-pair sample" was deferred to
  Phase 2: no dataset had been downloaded yet, and fabricating a sample was
  not acceptable. Completed in Phase 2 (see below).

---

## 2026-11-19 — Phase 2: Data acquisition & audit (completed)

**Acquisition.** All five registered sources downloaded to `data/raw/` with
sha256 + provenance in `data/raw/provenance.json` (`afripara.data.sources`
records URL, licence, citation for every file). 0 download failures.
FLORES-200 deferred to P3 (canonical mirror gated; only needed for pair
construction).

**Audit results** (`data/interim/audit/AUDIT_REPORT.md`, 20 dataset slices):

1. **SemRel2024-Hausa count discrepancy RESOLVED.** Proper CSV parsing gives
   **2,551 pairs = 1,736 train / 212 dev / 603 test** — exactly the number in
   the SemRel2024 paper. The earlier 5,105 figure counted physical CSV lines;
   every pair's two sentences are stored in one quoted field with an embedded
   newline (1,737 continuation lines in train). Consequence: Hausa supervised
   training data is ~1.7k pairs (MRPC-scale, still viable), not ~3.5k.
   Score distribution: mean 0.495, p50 0.50, range [0,1] — balanced.
2. **Cross-split sentence leakage in SemRel official splits**: 644 sentences
   shared between train and test (62% of unique test sentences), 244
   train∩dev, 101 dev∩test — the corpus samples pairs from shared documents.
   **P3 decision:** evaluate on the official split (comparability with
   published numbers) AND construct a sentence-disjoint re-split as a
   robustness check; never treat the official test split as leakage-free.
   AfriXNLI dev/test share **0** sentences; Menyo/IGBONLP overlaps are
   negligible (0–18 sentences).
3. **Menyo-20k**: 167 duplicate pairs (77/45/45 across train/dev/test) and
   one malformed row (literal tab inside a quoted field, dev line 700);
   ~50% of sentences carry no diacritics at all (mixed diacritisation —
   relevant to E6 and model robustness). Loaders strip the BOM found in the
   Yoruba column.
4. **IGBONLP benchmark**: 452 duplicate pairs, 11 empty lines; only ~47–49%
   of lines contain Igbo subdot letters (substantial undiacritised portion).
5. **AfriXNLI ibo/dev**: 10 malformed lines (quotation marks broke the TSV
   conversion) → 445 usable pairs instead of 450 (1.1% loss, documented and
   dropped). Label balance is otherwise perfect (150/150/150 per class on
   dev; 200/200/200 on test, all languages).
6. **Orthography profile** (heuristic metadata validation; model-based
   LID/AfroLID deferred to P3 constructed data): AfriXNLI yor diacritic
   density 0.40 (fully marked); Hausa texts use hooked letters/glottal marks
   in ~half of sentences (boko orthography often omits them); Igbo sources
   ~48–63% marker coverage.

**Pipeline test on real data** (`data/interim/pipeline_test/report.json`):
200 labelled pairs (AfriXNLI yor/ibo/hau test + SemRel dev, 50 each) through
loading → normalisation → pair tokenisation → CPU forward pass: **PASS**.
Pair-encoded token lengths p50 36–56, max 95 → `max_seq_length=128` is
adequate (0 truncations). CPU inference ≈ 0.15 s/pair (fine for evaluation
and demo; training stays on Colab).

**Engineering findings.**
- protobuf pinned to 3.20.3: sentencepiece 0.1.99 ships protoc-3.x generated
  code incompatible with protobuf≥4 ("Descriptors cannot be created
  directly"). Streamlit declares protobuf≥5.26 — conflict deferred to P7,
  where the demo uses fast-tokenizer models.
- `python -m afripara.data.download --all` is idempotent (sha256-checked)
  and refuses to overwrite changed raw files.

**Decisions for P3** (carried from these findings):
- Drop exact duplicate pairs; deduplicate sentences within constructed data;
  build splits grouped by source document AND sentence-disjoint where
  feasible (SemRel lesson).
- Menyo/IGBONLP diacritic inconsistency: record per-source diacritic density
  as metadata; E6 diacritic ablation gains a second interpretation
  (robustness to marked vs unmarked text, not just synthetic stripping).

---

## 2026-11-19 — Phase 3: Dataset construction & annotation (in progress)

**Infrastructure built (52/52 tests passing).**
- `afripara.data.sources`: PAWS-Wiki English registered (licence: freely
  usable for any purpose, Google notice). Original GCS bucket returns 403
  for all clients; using the official HF mirror's `refs/convert/parquet`
  files (documented in the source entry). Verified: train 49,401 pairs
  (44.2% paraphrase), matching the literature.
- `afripara.data.splits`: two leakage-controlled splitters. (1) hash-based
  sentence-disjoint (dropped mixed-fold pairs); (2) **connected-component
  splitting (preferred)**: shared sentences form clusters (union-find);
  whole clusters are assigned atomically by greedy quota filling → zero
  leakage, **zero dropped pairs**, deterministic. On SemRel Hausa the
  component splitter yields exactly **2,041/255/255 (80/10/10)** across 403
  components (2,775 sentences) — vs the hash splitter's unusable 24/22
  dev/test, showing why component awareness matters when sentences are
  heavily shared.
- `afripara.data.construct`: translate-train builder (PAWS route, inherits
  adversarially-controlled lexical overlap), MT-vs-human builder
  (supplementary positives from Menyo/IGBONLP), dedup, QC filters (empty
  MT, length ratio > 3), full provenance columns (`label_source`,
  `english_s1/s2`, `qc_len_ratio`) so silver and gold data are never mixed
  silently.
- `scripts/build_datasets.py --stage local`: DONE. Exports
  `data/processed/semrel_hau_official/` (1,736/212/603),
  `data/processed/semrel_hau_disjoint/` (2,041/255/255, zero overlap),
  `data/processed/eval/afrixnli_binary_{yor,ibo,hau,eng}.csv`
  (700/697/700/700 balanced; entailment→1, contradiction→0, neutral
  excluded), and MT inputs: 15,000 PAWS sentences + 2,000 MT-vs-human
  English sources.
- `afripara.data.annotation` + `scripts/annotate_app.py`: gold annotation
  store (append-only labels CSV, two rounds, self-agreement) and Streamlit
  tool (verified serving, headless boot HTTP 200). Preview mode works on
  real AfriXNLI data before constructed pairs exist.
- `docs/annotation_guidelines.md`: binary decision rule (strict-0 policy),
  0–4 relatedness scale, translationese/negation/diacritics rules,
  calibration block, self-consistency plan, workload estimate.
- `scripts/colab_translate_pairs.py`: self-contained Colab cell for the MT
  scale-up (T4, fp16, batch 32, copy-through guard).
- `scripts/make_annotation_export.py`: builds the 400-pair Yoruba gold
  candidate CSV + a project-only key file with inherited labels (annotator
  never sees them).

**NLLB pilot (local, 24 sentences → Yoruba) — a bug caught and fixed.**
- First run produced VERBATIM ENGLISH copies, not translations: the wrapper
  passed the SOURCE language code to `forced_bos_token_id`. Correct NLLB
  usage: `tokenizer.src_lang = src` + `forced_bos_token_id = id(tgt)`.
  Fixed; a copy-through guard (first-batch check, aborts if >50% outputs
  equal input) now runs in both local and Colab paths.
- Post-fix translations are genuine Yoruba with full diacritics; 0/24 empty,
  0/24 copies. Observed quality varies: some pairs render faithfully, others
  show meaning drift ("results are high" → "results will be very good") —
  precisely the MT-noise risk SemRel2024 warned about; native-speaker gold
  validation is mandatory before constructed labels are trusted for
  evaluation.
- Measured throughput: ~6.7 s/sentence on this 2-core CPU → full scale
  (17,000 sentences) ≈ 31 h locally → **Colab T4 is required** (projected
  10–20 min there).

**Constructed stage (completed 2026-11-19, after the researcher's Colab MT
run).** Colab outputs verified: 15,000/15,000 PAWS sentences translated,
0 empty, 0 verbatim copies. The 2,000 MT-vs-human inputs were generated
*after* that Colab run (sequencing slip, recorded here) - they are a
pending 5-minute follow-up and are supplementary positives only; nothing
downstream is blocked by them.

Final processed inventory (`data/processed/`):

| File | Pairs | Notes |
|---|---|---|
| constructed/translate_train_yor.csv | 2,873 | train 2,299 / dev 287 / test 287; 43.6% positive |
| constructed/translate_train_ibo.csv | 2,948 | train 2,358 / dev 295 / test 295 |
| constructed/translate_train_hau.csv | 1,477 | train 1,181 / dev 148 / test 148 |
| semrel_hau_official/ | 2,551 | 1,736/212/603 (paper-faithful split; has sentence leakage) |
| semrel_hau_disjoint/ | 2,551 | 2,041/255/255 component split, **0 leakage, 0 drops** |
| eval/afrixnli_binary_*.csv | 700 ×4 | balanced proxy eval (neutral excluded) |
| annotation/yor_gold_candidates.csv | 390 | 10 calibration + 380 main, 40 repeats; **0 drawn from train fold** |
| annotation/yor_gold_key.csv | 390 | project-only inherited labels (annotator never sees) |

Construction QC: 151 pairs dropped for length ratio > 3, 51 duplicates
removed, 0 empty translations. All constructed splits verified
sentence-disjoint (overlap counts all 0).

Design decision recorded: gold-annotation candidates are sampled ONLY from
the dev/test folds, so the native-speaker-validated pairs form a gold
evaluation set that no model will ever train on. Silver dev/test pairs
remain available for model selection; gold replaces them for headline
results.

**Status / next steps.**
1. Researcher runs `scripts/colab_translate_pairs.py` in Colab (uploads the
   5 input JSONLs; downloads 5 outputs back into `data/interim/mt/`).
2. Then locally: `build_datasets.py --stage constructed` +
   `make_annotation_export.py`.
3. Researcher annotates the Yoruba gold candidates with the Streamlit tool
   (~400 pairs; guidelines ready).

---

## 2026-11-19 — Phase 4: Baseline & evaluation harness (completed)

**Metrics library** (`afripara.evaluation.metrics`; 67/67 tests passing).
Classification (accuracy, P/R/F1 binary, ROC-AUC), similarity (Spearman ρ
primary — SemRel scores are Best-Worst-Scaled — plus Pearson), **McNemar's
exact test** (binomial on discordant pairs; Dietterich 1998, Dror et al.
2018) and **paired bootstrap** (10k resamples; Berg-Kirkpatrick et al. 2012,
Koehn 2004) with deterministic seeding. Unit tests verify McNemar against a
hand-computed discordant case and bootstrap determinism.

**TF-IDF + cosine baseline** (`afripara.models.tfidf_baseline`).
Anti-leakage by construction: vectorizer fitted on train-fold sentences
only, threshold tuned on dev F1, evaluation on test folds. Diacritics are
preserved as distinct features (verified by test: `ọmọ` and `omo` are
separate vocabulary items with low cosine).

**First results** (`outputs/baseline/BASELINE_RESULTS.md`; seed 42; all CIs
are bootstrap 95% at the tuned threshold):

| Set | n | Acc | F1 | ROC-AUC | Spearman |
|---|---|---|---|---|---|
| translate-train yor | 287 | 0.519 | 0.621 | 0.579 | – |
| translate-train ibo | 295 | 0.563 | 0.647 | 0.658 | – |
| translate-train hau | 148 | 0.547 | 0.660 | 0.600 | – |
| semrel_hau official | 603 | 0.627 | 0.735 | 0.701 | 0.454 |
| semrel_hau disjoint | 255 | 0.651 | 0.757 | 0.717 | 0.453 |
| afrixnli zero-shot yor/ibo/hau/eng | 400 | 0.50–0.58 | 0.09–0.41 | 0.57–0.64 | – |

**Interpretation (working hypotheses to be tested in Phase 5, not
conclusions):**
1. The lexical baseline **collapses on the constructed (PAWS-style) test
   sets** — accuracy barely above chance. This is the intended consequence
   of PAWS's adversarial overlap control: high-overlap non-paraphrases
   defeat word-overlap matching. The transformers' headroom above this
   floor is the quantitative evidence of semantic processing beyond lexical
   overlap (RQ1).
2. On SemRel *relatedness*, lexical overlap is genuinely informative
   (Spearman 0.45 vs SemRel2024 paper's Dice baseline 0.31 on their
   reported subset) but well below the fine-tuned LaBSE 0.69 the SemRel
   paper reports — a realistic target for our embedding track.
3. The leakage-free disjoint split scores slightly HIGHER than the leaky
   official split for the lexical baseline — evidence that sentence
   leakage mostly inflates *neural* (memorisation) models, supporting our
   decision to report both splits.
4. AfriXNLI zero-shot is hostile to transferred lexical thresholds
   (recall 0.05–0.29): the proxy task's label semantics differ from
   translate-train. Baseline context for the E5 transfer arm.

**Bug found and fixed during this phase:** accuracy bootstrap CIs were
initially computed at the default 0.5 threshold instead of the dev-tuned
threshold — detected because a CI failed to bracket its own point estimate
in the first table (semrel_hau_official). Corrected; verification discipline
(CIs must bracket points) noted for Phase 5 result tables.

---

## 2026-11-19 — Phase 5: Transformer training pipeline (built; Colab run pending)

**Architecture decision.** Colab TRAINS and dumps raw predictions + training
history; ALL scientific scoring happens locally with the tested
`afripara.evaluation.metrics` library (thresholds tuned on dev predictions,
McNemar vs the saved baseline predictions, seed aggregation, curves). This
keeps evaluation logic in one audited place — no duplicated scoring code in
the notebook.

**Built (67/67 tests still passing).**
- `afripara.training` (data + trainer): PairDataset (NFC-normalising,
  diacritic-preserving tokenisation; AfriBERTa slow-tokenizer fallback),
  HF Trainer with the config defaults (LR 2e-5, eff. batch 32 = 16×2
  accumulation, warmup 10%, wd 0.01, maxlen 128, ≤5 epochs, early stopping
  on dev F1 patience 3, best-checkpoint selection, seeded; fp16 on CUDA
  only). `model_repo_override` exists for engineering smoke runs only.
- **Local CPU smoke run PASS** (real Yoruba constructed pairs, 120/40,
  small cached AfriBERTa checkpoint, 1 epoch, 76 s): the full chain
  data → tokenisation → Trainer → prediction → metrics works. Metrics from
  this run are meaningless by design and are labelled as such.
- `scripts/make_run_plan.py`: generates `outputs/colab_bundle/run_plan.json`
  (29 runs) + `data_bundle.zip` (2.0 MB, 19 files, collision-checked).
  **Run grid:** AfroXLMR-base × {yor, ibo, hau-constructed, hau-semrel}
  × 3 seeds (12); AfriBERTa-large × {yor, ibo} ×1 + hau-semrel ×3 (5);
  English-PAWS transfer arm ×1; LaBSE zero-shot (4) + fine-tune (4);
  AfriE5 zero-shot (4, with mandatory `query:` prefixes). Estimated ~5.7 h
  total on one T4, resumable across sessions via DONE markers in Drive.
- `scripts/colab_train_transformers.py`: self-contained Colab cell (pinned
  package versions matching requirements.txt, Drive mount, unzip, resume,
  per-run predictions for every configured eval set, dev predictions for
  local threshold tuning, checkpoints saved to Drive for seed-42 runs only
  — Drive space guard).
- `scripts/collect_results.py`: local aggregation — dev-tuned thresholds,
  test metrics + bootstrap CIs, McNemar vs baseline predictions (saved per
  set by run_baseline.py), Spearman vs SemRel gold, per-family seed
  aggregation (mean ± std), matplotlib training curves. Gracefully refuses
  to run without Colab outputs.
- Baseline re-run to persist its test-fold predictions
  (`outputs/baseline/predictions/*.json`) for the McNemar pairing.

**Design points recorded.**
- Dev-tuned thresholds are computed locally from `predictions_dev.json` —
  the test folds are never touched for tuning (consistent with the baseline
  protocol).
- Checkpoint retention policy: seed-42 runs only (≈8 GB Drive); seeds 43/44
  contribute metrics/predictions but not stored weights. Any of them can be
  re-trained if a checkpoint is later needed.
- The Colab script mirrors `afripara` normalisation and dataset semantics
  through a minimal shim (`_pairdata_shim`) whose behaviour is identical to
  the tested local module; the local module remains the reference.

**Pending.** (1) Researcher runs the Colab cell (~5.7 h GPU, resumable) and
copies `outputs/colab/` back; (2) then `collect_results.py` produces the
Phase 5 tables; (3) Yoruba gold annotation (P3) still pending — gold rows
will be added to the tables once available.

---

## 2026-11-20 — Colab attempt 1 FAILED (my bug); fixed with a consistency gate

The first Colab session produced **zero usable runs**: 5 cross-encoder runs
failed at data load with `FileNotFoundError` and the session ended before
the embedding arms ran. Root cause: `run_plan.json` referenced constructed
training files as `constructed/translate_train_<lang>.csv` while
`data_bundle.zip` stores files flat (`translate_train_<lang>.csv`) — a
plan/bundle path mismatch introduced when naming conventions diverged
between `build_plan()` and `build_bundle()`. The failed attempt is archived
locally as `outputs/afripara_runs_attempt1_failed/` (for the record).

**Fixes:**
1. All plan file references made flat to match the bundle.
2. **Plan/bundle consistency gate added to `make_run_plan.py`**: after
   generation, every file referenced by every run must resolve to a bundle
   entry or generation aborts. The gate immediately caught 4 additional
   mismatches (LaBSE fine-tune arms) that manual inspection had missed.
   Independent post-check: 29/29 runs resolve, 0 unresolved references.
3. Colab script now deletes stale `ERROR.txt` on successful retry.

**Rerun instructions to the researcher:** re-upload BOTH files (the bundle
was regenerated too), run the same cell; the script resumes into the
existing Drive folder and retries every failed run (none has a DONE
marker). Expected: full 29-run grid, ~5.7 h, resumable.

---

## 2026-11-20 — Colab attempt 2: 19/29 runs DONE; results fetched via Drive link

**Transfer workflow.** The researcher shared the Drive folder as a public
link; results were fetched directly with `gdown` (list-without-download,
then selective fetch of the 130 small files, skipping ~9 GB of checkpoints).
One file hit a transient gdown rate limit and was re-fetched on retry.

**Outcome: 19/29 runs completed** — all 12 primary AfroXLMR runs (yor, ibo,
hau-constructed, hau-semrel × 3 seeds), all 5 AfriBERTa runs, the English
PAWS transfer arm, and 4 embedding zero-shots (LaBSE/AfriE5 × yor/ibo).

**7 embedding runs failed with CUDA OOM (T4, 16 GB).** Diagnosis: embedding
encodes ran at batch 64 with the models' full sequence lengths (LaBSE 256 /
AfriE5 512 tokens), and the GPU caching allocator held blocks across all 29
sequential runs. Fixes applied to the Colab script: encode batch 64→16 with
max_length=128 (justified by the pipeline test: pair tokens p50 36–56, max
95), LaBSE fine-tune batch 16→8, and `gc.collect() + cuda.empty_cache()`
between runs. The remaining 7 runs need a short Colab rerun (minutes each).

**Early training signals (from history.json, before full local scoring):**
- `afroxlmr_hau_semrel` (real human labels): dev F1 **0.861** — strong,
  healthy learning; will be compared against SemRel-published LaBSE 0.69
  Spearman context.
- `afroxlmr_en_paws` (English transfer source): dev F1 **0.756** — matches
  published XLM-R/PAWS fine-tuning levels, confirming the training pipeline
  is correct end-to-end.
- `afroxlmr_yor_constructed` (MT-inherited labels): dev F1 plateaus ≈
  **0.65**, test accuracy barely above the TF-IDF floor — consistent with
  MT-induced label noise in translate-train pairs. This is the documented
  risk of the 2026-11-20 decision to skip native-speaker validation, now
  with quantitative evidence: clean human labels → strong learning;
  MT-inherited labels → weak learning. Reported per provenance tier.

---

## 2026-11-20 — Phase 5 results (22/29 runs scored; 7 embedding runs pending rerun)

Full tables: `outputs/phase5/RESULTS.md` (+ per-run JSON, bootstrap CIs,
training-curve PNGs). Thresholds tuned on dev predictions; family rows are
seed-averaged (mean ± std, 3 seeds for all primary arms); McNemar exact
tests vs the saved TF-IDF baseline predictions on matched items.

**F1. Human-label training is the headline result (RQ1).**
AfroXLMR trained on SemRel Hausa (human 0–1 scores) reaches on the
**leakage-free** test split: **accuracy 0.783, F1 0.840 ± 0.008, ROC-AUC
0.863, Spearman 0.697** (3 seeds, tight variance). On the official split:
0.845 / 0.864 / 0.913 / 0.735. Both far exceed the TF-IDF baseline
(0.651 / 0.757 / 0.717 / 0.453) and match-or-beat the SemRel2024 paper's
fine-tuned LaBSE (Spearman 0.69). Seed variance is tiny → the result is
stable, not a lucky seed.

**F2. Sentence leakage inflates neural models — demonstrated (split-design
validation).** For models trained on SemRel's train fold, the leaky official
split scores higher than the clean split: AfroXLMR +6.2 accuracy points
(0.845 vs 0.783); AfriBERTa +6.5 points (0.733 vs 0.668) with Spearman
collapsing 0.671 → 0.426 on the clean split. For models trained on
*constructed* data (no SemRel exposure), the pattern reverses
(AfroXLMR-hau-constructed: 0.548 official vs 0.641 disjoint) — exactly as
the memorisation account predicts: leakage can only help models that have
seen the leaked sentences. TF-IDF shows the same reversal. The dual-split
reporting design is validated empirically.

**F3. Label provenance dominates (silver vs human).** The AfroXLMR model
trained on *constructed* Hausa pairs is at **chance on real relatedness**
(Spearman 0.153 on the disjoint split) while the human-label model reaches
0.697 — a 0.54 Spearman gap attributable to training-label quality.
In-language silver training yields only modest gains over the lexical
baseline (accuracy: yor 0.562 vs 0.519; ibo 0.676 vs 0.563; hau 0.626 vs
0.547), with ROC-AUC 0.64–0.78 showing real but limited ranking signal.

**F4. Cross-lingual zero-shot transfer beats in-language silver training
(RQ3b).** The English-PAWS-trained model transfers to the silver test sets
at accuracy **0.811 (hau), 0.675 (ibo), 0.599 (yor)** — beating the
in-language silver-trained models on hau and yor and matching on ibo
(0.676). Interpretation (stated as interpretation): clean-label English
features transfer better than noisy translated-label fine-tuning teaches.
Caveat: all silver test pairs are NLLB translationese, shared across both
conditions.

**F5. AfroXLMR ≫ AfriBERTa on this task (RQ2).** AfriBERTa-large fails to
learn from silver pairs (accuracy < 0.5 on yor/ibo constructed tests,
ROC-AUC ≈ 0.49–0.55, predict-all behaviour) and trails AfroXLMR on SemRel
(disjoint: 0.668 acc / 0.728 AUC vs 0.783 / 0.863). The African-adaptive-
fine-tuning advantage claimed in the AfroXLMR paper is confirmed on a pair
task. AfriBERTa's 126M/news-only pretraining appears insufficient here.

**F6. Embedding track (partial).** LaBSE/AfriE5 zero-shot on silver test
sets is near-chance to weak (ROC-AUC 0.50–0.67) — consistent with
SemRel2024's observation that AfroXLMR embeddings *hurt* unsupervised
performance; the SemRel zero-shot embedding runs (the key comparison vs the
published 0.69) are among the 7 OOM retries pending.

**F7. McNemar significance with direction (RQ1 significance answered).**
McNemar tests whether error patterns differ; it must be read with the
direction of the accuracy delta (a degenerate model can be "significantly
different" while being worse). Direction-aware verdicts vs the TF-IDF
baseline:
- **AfroXLMR in-language silver models are significantly better on all
  three languages** on their own test sets: yor +4.3 pts (p=0.0049), ibo
  +11.3 (p<0.0001), hau +7.9 (p<0.0001).
- **English→target transfer is significantly better on all three silver
  sets**: hau +26.4 (p<0.0001), ibo +11.2 (p<0.0001), yor +8.0 (p=0.0049).
- AfriBERTa silver models and LaBSE/AfriE5 zero-shot are significantly
  **worse** than the baseline on silver sets (differences without
  superiority) — consistent with their degenerate/lexical-mismatch
  behaviour.
- LaBSE zero-shot is significantly **better** than the baseline on the
  AfriXNLI proxy (ibo +6.5, yor +2.1, p=0.012) but worse on silver sets —
  natural-translated text (LaBSE's home distribution) vs NLLB
  translationese with adversarial overlap.
- SemRel-trained models vs baseline on AfriXNLI: no significant difference
  (p=0.17) — expected across task boundaries.
- **On SemRel (human labels), the headline comparison is significant**: the
  human-label AfroXLMR beats the lexical baseline by +13.2 points on the
  leakage-free split (0.783 vs 0.651, p<0.0001) and +21.8 on the official
  split (p<0.0001); AfriBERTa is also significantly better (+1.7 /
  +10.6). The silver-trained model is significantly **worse** than the
  baseline on both SemRel splits (−1.0 disjoint, −7.8 official) — silver
  fine-tuning actively degrades real-relatedness performance relative to
  doing nothing beyond lexical overlap. F3's provenance contrast is now
  significance-backed in both directions.

**Bugs found in the collector before reporting (fixed):** family
aggregation overwrote instead of accumulating across seeds (family rows
showed n_seeds=1); dev-set rows leaked into the headline table; McNemar
alias missing for constructed_test_* ↔ translate_train_* name mapping.

**Pending.** 7 embedding reruns (OOM-fixed) → then LaBSE/AfriE5 SemRel
Spearman comparison. Phase 6 next: E5 transfer interpretation, E6 linguistic
perturbation study on silver test sets, error analysis, model-comparison
figures.

---

## 2026-11-21 — Phase 6: Analysis (E6 + error analysis + figures)

All 29 Colab runs completed after the OOM fixes (researcher rerun; results
fetched via the public Drive link with gdown, checkpoints excluded). Full
report: `outputs/phase6/ANALYSIS_REPORT.md`; data in `outputs/phase6/*.json`;
figures in `outputs/phase6/figures/`.

**E6 diacritics ablation (paired, Yorùbá, seed-42 checkpoint).**
- Accuracy is insensitive to tone-mark stripping (all deltas within CIs of
  zero) — no catastrophic diacritic dependence.
- The signal is in F1/calibration: mixed-marking pairs (one side stripped)
  shift the model toward fewer positive predictions (F1 0.616 → ~0.45),
  recovering when both sides are stripped (0.580) — the model uses
  diacritic *consistency* as a surface cue.
- TF-IDF collapses under mixed marking (F1 0.621 → 0.186, −70%); the
  neural model degrades far less (−26% relative) — subword robustness
  demonstrated, and the empirical justification for diacritic-preserving
  preprocessing.
- AfriXNLI arm uninformative (degenerate constant-positive predictions of
  the silver model); reported for transparency.

**Observational analyses.** Igbo/Hausa: transformer beats baseline in
*every* lexical-overlap bucket including Q4-high-overlap (CIs exclude zero:
+0.164 / +0.128) — advantage does not vanish as overlap rises. Yoruba: no
reliable advantage at seed 42 (CI includes zero; 3-seed McNemar +4.3
significant — seed variance reported, not hidden). Negation: no systematic
weakness — the strong SemRel model classifies one-side-negated pairs
*better* than average (0.838 vs 0.788, n=37). Diacritic-density buckets
largely vacuous (NLLB output is consistently marked; Hausa boko mostly
unmarked by design).

**Error analysis.** Two classes quantified: (1) silver-label artifacts
(MT erasure; rare outright — 11 identical-text Yoruba pairs, 2 labelled
non-paraphrase — but the broader mechanism appears as a strong
false-positive skew, yor 101 FP vs 29 FN); (2) genuine adversarial
entity/role-swap failures (Hausa: 44.8% of FPs are ≥0.9-overlap pairs) —
the residual capability gap that adversarial PAWS training targets,
consistent with the English-transfer arm's strength (F4).

**Scope decision recorded:** word-order perturbation NOT attempted (no safe
automatic reorderer for these languages; fabricating ungrammatical
manipulations would produce unsupported conclusions).

**Collector/archival notes.** The definitive 29-run `RESULTS.md` pass was
still computing bootstrap CIs at the time of writing (the 22-run version is
the current file; embedding family rows will appear when it completes).
Hausa seed-42 checkpoint fetched locally for the Phase 7 demo (disk now
54 GB free).

**Collector/archival notes.** The definitive 29-run `RESULTS.md` pass was
still computing bootstrap CIs at the time of writing (the 22-run version is
the current file; embedding family rows will appear when it completes).
Hausa seed-42 checkpoint fetched locally for the Phase 7 demo (disk now
54 GB free).

---

## 2026-11-21 — Phase 7: Demonstration application & documentation (project complete)

**Built.**
- `scripts/calibrate_models.py`: temperature scaling per model on
  validation predictions (Guo et al. 2017). Headline model
  (`afroxlmr_hau_semrel_seed42`): T=1.66, dev ECE 0.110 → **0.049**
  (well-calibrated). Yoruba silver model: T=3.46, ECE 0.210 → 0.130 —
  heavily overconfident even after scaling, itself a fingerprint of
  silver-label noise (ties to finding F3/R4).
- `src/afripara/inference/predictor.py`: `ParaphrasePredictor` serving a
  Phase 5 checkpoint with its dev-tuned threshold and fitted temperature —
  the demo uses exactly the decision rule of the reported experiments.
- `scripts/pick_demo_examples.py`: demo examples selected from real
  held-out items with verified verdicts (never invented, never training
  items). Filters: mojibake detection + near-identical-pair exclusion.
  **New data-quality finding:** one Igbo test sentence contains Latin-1
  double-encoding (`Lăpuş` → `LÄ puÅ`) that passed the strict-UTF-8 audit
  because the bytes ARE valid UTF-8 — recorded as a known limitation of
  encoding audits.
- `src/afripara/app/demo.py`: Streamlit app — language selector (Hausa
  recommended; Yorùbá/Igbo served with displayed silver-provenance
  warnings), model information panel (checkpoint, training data, label
  quality, held-out performance), two-sentence input, paraphrase verdict
  with calibrated score, real held-out example pairs with verified gold
  labels.
- `docs/RESULTS_SUMMARY.md`: thesis-facing one-page map of all findings
  (R1–R9) with evidence locations, the honest limitations list, and a
  reproduction map.
- README finalised (phase table, demo instructions, research principles).

**Verification.** 72/72 tests pass (new: temperature fitting on
synthetic calibrated/overconfident data, ECE bounds, prediction threshold
logic — three initial test constructions were wrong and fixed, not the
code). Hausa predictor loads in 10.5 s; both verified examples classified
correctly (0.927 vs 0.136 calibrated — well separated). Demo boots headless
(HTTP 200). Igbo checkpoint fetch prepared (not yet pulled — demo enables
Igbo automatically when its checkpoint is present).

**Known loose ends.** No Igbo checkpoint locally (demo hides Igbo until
fetched); 29-run RESULTS.md now definitive (embedding families included);
`outputs/colab` holds ~3.3 GB of checkpoints locally (yor + hau) — Igbo
adds ~1.1 GB if fetched.
