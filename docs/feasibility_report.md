# Technical Feasibility Report

## Transformer-Based Semantic Similarity and Paraphrase Detection in Yorùbá, Igbo, and Hausa

**Project:** MSc Computer Science and Artificial Intelligence research project
**Date:** 18 November 2026
**Status:** Feasibility phase — awaiting researcher approval before implementation
**Verification method:** All model and dataset claims below were checked against live Hugging Face Hub metadata, GitHub repositories, ACL Anthology records, or arXiv full text on 18 November 2026. Items that could **not** be verified are explicitly flagged in Appendix A. No results in this report are invented.

---

## 1. Executive Summary

The project is **feasible**, but not in the form the original brief implicitly assumed. The decisive finding is:

> **No paraphrase-identification dataset exists for Yoruba, Igbo, or Hausa.** This was verified against the Hugging Face datasets API (full `paraphrase` search), GitHub repository search, and the 2025 *NaijaNLP* survey of Nigerian-language NLP (arXiv:2502.19784), which records zero paraphrase or sentence-similarity corpora for these languages. The benchmark AfriMTEB — which *does* cover Hausa, Yoruba and Igbo — **excludes semantic similarity entirely**, for lack of data (arXiv:2510.23896).

Consequences for the design:

1. **Dataset construction becomes a core research contribution**, not a preprocessing step. The project must build its own sentence-pair dataset under clear licences, with a human-verified gold subset. This is also the project's novelty claim: a first paraphrase/similarity evaluation covering all three languages.
2. **Hausa has one real human-annotated resource** — SemRel2024 (SemEval-2024 Task 1), 5,105 Hausa sentence pairs with continuous 0–1 relatedness scores — usable immediately for supervised training and evaluation, subject to a licence clarification.
3. **Two models from the original brief need correction:** mBERT is **not a valid model for Igbo or Hausa** (neither language is in its 104-language pretraining set — verified against the canonical BERT multilingual list), and "Native-Bird" exists on Hugging Face but is **unlicensed, unpublished, and unused** (~19 downloads/month) — unusable for an academic project. Verified replacements exist: **AfroXLMR** (COLING 2022, MIT) as the primary encoder, and **LaBSE** (ACL 2022, Apache-2.0) / **AfriE5** (2025, MIT) as verified multilingual sentence-embedding models covering all three languages.
4. **Compute:** your local machine is an Intel MacBook with 2 CPU cores and no GPU — transformer fine-tuning must run on **Google Colab (free T4, 16 GB)**, which comfortably handles the recommended models. Fine-tuning a base-size cross-encoder for 3 epochs is projected at under 2 hours per run (projection — to be measured in the first run).

**Recommended scope (manageable):** one traditional baseline (TF-IDF + cosine), **two fine-tuned cross-encoders** (AfroXLMR-base, AfriBERTa-large), **two zero-shot/fine-tuned sentence-embedding models** (LaBSE, AfriE5), one negative control (mBERT), evaluated on a three-tier data strategy across monolingual and English→target cross-lingual settings.

---

## 2. Feasibility Verdict

| Dimension | Verdict | Notes |
|---|---|---|
| Research gap / novelty | **Strong** | Verified: no paraphrase or STS corpus for yo/ig/ha exists; AfriMTEB explicitly omits STS. A constructed, documented, human-verified dataset + evaluation is publishable as an MSc contribution. |
| Models | **Feasible after corrections** | AfroXLMR, AfriBERTa, LaBSE, AfriE5 verified live on HF Hub with permissive or stated licences; mBERT restricted to a control role; "Native-Bird" rejected. |
| Data | **Feasible but requires construction + your annotation input** | SemRel2024 (Hausa), AfriXNLI (all three), licensed parallel corpora (Menyo-20k, IGBONLP, FLORES-200, NTREX) provide raw material; a gold subset must be human-verified. |
| Compute | **Feasible via free cloud GPU** | Local Mac is CPU-only (2 cores); fine-tuning on Colab T4 is a proven configuration (SemEval-2024 winner and other public model cards use it). |
| Timeline | **Feasible in one semester** | ~12–14 weeks with the phased plan in §11, contingent on early dataset-construction decisions (§13). |
| Main risk | **Annotation capacity & licence clarity** | Both require your action in the first week (see §13). |

---

## 3. Research Questions, Operationalised

The objectives translate into answerable experimental questions:

- **RQ1 (Effectiveness):** How accurately can transformer cross-encoders classify paraphrase pairs in Yoruba, Igbo and Hausa, compared with a TF-IDF + cosine baseline? → E1, E2
- **RQ2 (African-specific vs multilingual):** Do African-language-adapted encoders (AfroXLMR, AfriBERTa) outperform general multilingual encoders (XLM-R-lineage models, LaBSE, AfriE5, mBERT-control) on this task? → E2, E3
- **RQ3 (Transfer learning):** (a) What is the effect of fine-tuning relative to zero-shot embedding use? (b) How far does English→target zero-shot transfer go (English PAWS training data → yo/ig/ha evaluation)? → E4, E5
- **RQ4 (Linguistic factors):** How do diacritics/tone marks, negation, word-order variation and lexical overlap affect performance? → E6 (controlled perturbation experiments)
- **RQ5 (Application):** Can the best model serve in a usable demo producing a paraphrase decision and a calibrated similarity score? → E7

Note on framing: the primary task is **binary paraphrase classification** as specified in the brief. The Hausa SemRel scores additionally support a **secondary similarity-scoring task** (Spearman/Pearson). For Yoruba and Igbo, continuous similarity is reported only for embedding-based models evaluated on constructed pairs — labelled as such, not conflated with the classification task.

---

## 4. Model Audit (verified 2026-11-18)

### 4.1 Summary

| Model | HF repo | Params | yo | ig | ha | Licence | Role in project |
|---|---|---|---|---|---|---|---|
| **AfroXLMR-base** | `Davlan/afro-xlmr-base` | 278M | ✅ | ✅ | ✅ | MIT | **Primary cross-encoder** |
| AfroXLMR-large | `Davlan/afro-xlmr-large` | 560M | ✅ | ✅ | ✅ | MIT | Optional stretch (if Colab budget allows) |
| **AfriBERTa-large** | `castorini/afriberta_large` | 126M | ✅ | ✅ | ✅ | MIT | African-specific encoder baseline |
| AfriBERTa-base | `castorini/afriberta_base` | 111M | ✅ | ✅ | ✅ | **none declared** | Avoid (no licence) — use large instead |
| mBERT | `google-bert/bert-base-multilingual-cased` | 178M | ✅ | ❌ | ❌ | Apache-2.0 | **Negative control** (see §4.2) |
| **LaBSE** | `sentence-transformers/LaBSE` | ~471M (768-dim) | ✅ | ✅ | ✅ | Apache-2.0 | Embedding track (cosine similarity) |
| **AfriE5-Large-instruct** | `McGill-NLP/AfriE5-Large-instruct` | 560M (1024-dim) | ✅ | ✅ | ✅ | MIT | Embedding track (African-specialised) |
| "Native-Bird" | `Modularcomputing/Native-Bird` | 471M | ✅ | ✅ | ✅ | **none declared** | **Rejected** — no licence, no paper, 19 downloads/month |
| NaijaBERT | `Davlan/naija-bert-base` | ~110M | ✅ | ✅ | ✅ | Apache-2.0 | Optional small encoder (no paper; low usage) |
| paraphrase-multilingual-mpnet-v2 | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 355M | ❌ | ❌ | ❌ | Apache-2.0 | **Rejected** — 50-language list excludes all three |
| paraphrase-multilingual-MiniLM-v2 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | ❌ | ❌ | ❌ | Apache-2.0 | **Rejected** — same exclusion |
| distiluse-multilingual-v2 | `sentence-transformers/distiluse-base-multilingual-cased-v2` | 135M | ? | ? | ? | Apache-2.0 | Not recommended — coverage unverifiable |

✅ = language verified in the model's own pretraining/coverage documentation (primary sources: BERT `multilingual.md` language list; LaBSE paper Table 10; AfroXLMR/AfriBERTa/AfriE5 model cards and papers).

### 4.2 Corrections to the original brief — and why they matter

1. **mBERT cannot serve as the multilingual baseline for this task.** The canonical Google BERT multilingual language list contains **Yoruba but neither Igbo nor Hausa**. Fine-tuning mBERT on Igbo/Hausa pair data would measure tokenisation-out-of-vocabulary behaviour, not semantic competence. Scientifically sound uses: (a) Yoruba-only comparison point; (b) documented negative control demonstrating why African-adapted models are needed. This is itself a citable finding consistent with prior work (MasakhaNER, TACL 2021, reports weaker transfer to Niger-Congo than Afro-Asiatic languages).
2. **"Native-Bird" is not usable.** It exists (`Modularcomputing/Native-Bird`, a LaBSE fine-tune for native-query→English retrieval) but has **no licence** (legally unusable without author permission), **no peer-reviewed paper** (only a `@misc` citation), and ~19 downloads/month. Its successor Native-Bird2 is a CLIP-style *image-text* model — different task entirely. Recommendation: mention in the thesis as an existing but unverifiable artifact; build on LaBSE (its verified, licensed base) instead.
3. **"AfriE5" exists** — confirmed: `McGill-NLP/AfriE5-Large-instruct` (MIT), XLM-R-large backbone, contrastively trained for 9–10 African languages including yo/ig/ha, published as Uemura, Zhang & Adelani, *AfriMTEB and AfriE5* (arXiv:2510.23896, Oct 2025; EACL 2026). Usage note: it requires `query:`/`passage:` input prefixes at inference.
4. **AfriBERTa's repo id is `castorini/...`** (not "Castelniko"). Its training corpus is BBC News + Common Crawl (<1 GB) per its own cards and paper (Ogueji, Zhu & Lin, MRL@EMNLP 2021) — **JW300 is not part of AfriBERTa** (a claim sometimes seen in secondary sources; unverified and contradicted by primary sources).

### 4.3 Prior evidence these models work on this task family

- The **SemEval-2024 Task 1 winning system (AAdaM)** fine-tuned **AfroXLMR as a cross-encoder** with task-adaptive pretraining and MT-augmentation, reaching average **Spearman 0.800** on Semantic Textual Relatedness (arXiv:2403.18933). This is the closest published recipe to our primary architecture.
- SemRel2024's Hausa baselines give realistic expectations for **zero-shot embedding** approaches: unsupervised XLM-R 0.04, mBERT 0.16, lexical-overlap Dice 0.31, fine-tuned LaBSE **0.69** Spearman (arXiv:2402.08638). Notably, AfroXLMR *hurt* performance in the unsupervised embedding setting — a reason to keep both cross-encoder and embedding tracks and report both.
- Cross-encoder vs bi-encoder: Reimers & Gurevych (EMNLP 2019) report ~2.7 points Spearman advantage for cross-encoders on STS-Benchmark (88.77 vs 86.10, large models); follow-up work (e.g. AugSBERT, arXiv:2010.08240) confirms bi-encoders trade accuracy for scalable retrieval. Since our evaluation is a fixed labelled test set (not 10k×10k retrieval), **the cross-encoder is the correct primary architecture**, with the bi-encoder track retained for (a) the calibrated similarity score in the demo and (b) scalability discussion.

---

## 5. Dataset Landscape and Gaps

### 5.1 What exists (verified)

| Resource | Languages | Size | Labels | Licence | Usable for |
|---|---|---|---|---|---|
| **SemRel2024 / SemEval-2024 Task 1** | **Hausa** (of 14 langs) | **5,105 pairs** (3,473 train / 425 dev / 1,207 test — counted from released CSVs) | Continuous 0–1 relatedness (Best-Worst Scaling) | **Unstated** ⚠️ | Hausa supervised training + similarity evaluation |
| **AfriXNLI** (Masakhane / MTEB mirror) | **yo, ig, ha** (+15) | 1,050 pairs/lang (450 dev + 600 test) | 3-way NLI | **Apache-2.0** (masakhane) / CC-BY-4.0 (mteb) | Cross-language evaluation harness; hard negatives |
| **Menyo-20k** | yo–en | 20,100 pairs | parallel text | CC-BY-NC-4.0 | Yoruba pair construction (research-only) |
| **IGBONLP** (Ezeani et al., LREC 2020) | ig–en | ~10,752 pairs (benchmark subset) | parallel text | **Unstated** ⚠️ | Igbo pair construction (⚠️ avoid its JW300-derived subfolder) |
| **FLORES-200** (NLLB, Meta) | yo, ig, ha (+197) | 997 dev + 1,012 devtest sents/lang | parallel text | **CC-BY-SA-4.0** | Cross-language pair construction (share-alike applies) |
| **NTREX-african** (Masakhane) | yo, ig, ha (+29) | ~1,997 sents/lang | parallel text | CC-BY-SA-4.0 | Same as FLORES |
| **MAFAND** | ha, ig (+others) | 1K–10K | parallel text | CC-BY-NC-4.0 | Hausa/Igbo pair construction (research-only) |
| **AfroLID** | 517 African langs | — | LID tool | Apache-2.0 | Language-metadata validation |
| **AfriSenti / NaijaSenti** | yo, ig, ha, pcm | ~4–5k tweets/lang | sentiment (not pair data) | CC-BY-NC-SA | Sentence pool for negative pairs (⚠️ Twitter register) |
| **Belebele** | yo, ig, ha (+119) | ~1k questions/lang | MRC | CC-BY-SA-4.0 | Clean sentence pool (⚠️ topically coherent within passages) |
| **MasakhaNEWS** | yo, ig, ha (+7) | news articles | topic | AFL-3.0 | Sentence pool (news register) |
| **PAWS-X** (English portion) | en (+6 non-African) | 49,401 human-written train pairs | binary paraphrase | — per repo | **English→target zero-shot transfer training data** |

### 5.2 Confirmed negatives (checked, not assumed)

- **No paraphrase dataset for yo/ig/ha** on Hugging Face or GitHub (full API search executed).
- **No graded STS dataset for Igbo or Yoruba.**
- **"AfriSTS" does not exist** as a released artifact — any citation of it should be checked; the real resource is SemRel2024/SemEval-2024 Task 1.
- **"NaijaLID" does not exist** — use AfroLID (Apache-2.0, verified).
- **JW300 is legally unusable** — withdrawn from OPUS at the publishers' request for copyright breach (documented in mtdata issue #77 and the TACL Survey of Low-Resource MT). JW300-derived files still circulate inside otherwise-useful repos (IGBONLP `masakhane_JW300_enig/`, masakhane-wazobia `Data/jw300/`) — these directories must be excluded, and this should be documented in the research log as a data-governance decision.

### 5.3 Key validity cautions from the literature

- **Translated pairs are not paraphrases** (as the brief anticipates). More specifically for our languages: **SemRel2024's authors explicitly rejected MT-pivot/back-translation paraphrase construction for African languages**, citing unreliable MT (arXiv:2402.08638). Any constructed positive pairs must therefore be **human-verified on a gold subset** before being treated as labels, and construction method must be reported transparently.
- **NLI is a proxy, not paraphrase.** Entailment pairs are *inferrable*, not rewordings; "neutral" pairs are topic-related but not equivalent (their binary label is ambiguous). If AfriXNLI is relabelled (entailment→positive, contradiction→negative, **neutral discarded**), this proxy bias must be stated wherever those labels are used.
- **Register mismatch.** AfriSenti is Twitter text (code-mixing, hashtags); Menyo-20k/FLORES are news/formal. Models trained on one register and tested on another will show domain shift — a factor to control in the experimental design, not to ignore.
- **Relatedness ≠ paraphrase.** SemRel scores measure *relatedness* (0–1); thresholding into a binary label produces a *near-paraphrase/near-equivalent* task. This distinction must be stated in the thesis.
- **SemRel size discrepancy to resolve at download time:** the SemRel2024 paper reports 2,551 Hausa pairs; the released Hausa CSVs count 5,105 rows (including 425 dev + 1,207 test). This is likely a paper-vs-revision difference; it will be recorded precisely in the data audit.

---

## 6. Proposed Data Strategy (three tiers)

**Tier 1 — Real human labels (Hausa, immediate).** SemRel2024 Hausa: train cross-encoders and embedding models on the 3,473 train pairs (scores thresholded for the binary task; threshold τ tuned on the 425 dev pairs), evaluate on the 1,207 held-out test pairs, and report Spearman/Pearson on the continuous scores. *Precondition: licence clarification from the SemRel Hausa contacts (your action — see §13). Fallback if no licence is granted: use it as a private evaluation set, or drop Hausa supervised training and rely on Tiers 2–3.*

**Tier 2 — Licensed cross-language evaluation harness (all three languages).** AfriXNLI (Apache-2.0) relabelled: entailment→1, contradiction→0, neutral→excluded (recorded in the research log). ~700 usable pairs/language. Too small for training; used as (a) a common evaluation set across all three languages, (b) few-shot/low-resource training probes, and (c) a validation set for constructed labels.

**Tier 3 — Constructed paraphrase dataset (the contribution; all three languages).**
- *Positives:* for English source sentences shared across FLORES-200 / NTREX / Menyo-20k / IGBONLP, generate two renderings in the target language (human reference vs MT, or two MT systems) → same meaning, different wording.
- *Negatives:* sentence pairs from different source documents, length-matched, cross-checked with AfroLID.
- *Quality control:* lexical-overlap filtering to avoid trivial pairs (the PAWS lesson: paraphrase data must contain high-overlap non-paraphrases and low-overlap paraphrases, or models learn lexical heuristics); automatic pseudo-labels from an embedding model **only as candidate filtering**, never as gold labels.
- *Gold standard:* **human verification of ~500–1,000 pairs per language** (binary label + optional 0–4 similarity score). This requires annotator capacity — see §13. Inter-annotator agreement (Cohen's κ) reported on a doubly-annotated subset.

**Split policy (all tiers):** grouped splits by source document — no sentence from a document appears in two splits, preventing the leakage the brief demands; fixed seed (documented in config); per-language stratification preserved.

**Volume estimate (defensible target):** ~8–15k pairs/language total, of which ~500–1,000/lang are human-verified gold; Hausa additionally has the 5,105-pair SemRel set. This is in the range where cross-encoder fine-tuning is effective (MRPC-scale, 5.8k pairs, is a standard BERT fine-tuning task) but small enough for MSc-scale annotation.

---

## 7. Recommended Architectures and Baseline

**A. Traditional baseline (required by brief):** TF-IDF (word 1–2 grams, diacritics **preserved**) + cosine similarity, threshold tuned on validation. Serves as the lexical-overlap control — essential given PAWS's finding that paraphrase models can collapse onto lexical overlap.

**B. Cross-encoder track (primary):** sentence-pair classification head on
1. **AfroXLMR-base** (primary, MIT, COLING 2022, strongest African-adapted encoder — also the architecture of the SemEval-2024 Task 1 winner);
2. **AfriBERTa-large** (African-specific comparator; different pretraining corpus and tokenizer — gives RQ2 its contrast).

**C. Sentence-embedding track (secondary + demo scoring):**
3. **LaBSE** — evaluated zero-shot (cosine) and fine-tuned with cosine/contrastive loss (Siamese);
4. **AfriE5-Large-instruct** — evaluated zero-shot (with mandatory `query:` prefixes); fine-tuning optional depending on Colab budget (560M params).

**D. Controls:** mBERT (Yoruba-only point + negative control for ig/ha); English PAWS-X cross-encoder for the zero-shot transfer experiment (E5).

**Explicitly rejected:** Native-Bird (licence/paper), paraphrase-multilingual-mpnet & MiniLM (no language coverage), distiluse (unverifiable coverage), AfriBERTa-base (no licence), any JW300-derived data.

**Not recommended at MSc scale:** contrastive pretraining from scratch, or fine-tuning AfroXLMR-large everywhere — the former lacks justification (verified strong checkpoints exist), the latter doubles Colab time for an expected ~1–2 point gain (AfroXLMR paper: 81.8→83.4 NER F1 base→large).

---

## 8. Experimental Design

| # | Experiment | Data | Models | Answers |
|---|---|---|---|---|
| E1 | Baseline vs transformer | Tier 3 (+Tier 1 Hausa) | TF-IDF+cosine; AfroXLMR-base | RQ1 |
| E2 | African-adapted vs general multilingual | Tiers 1+3 | AfroXLMR-base vs AfriBERTa-large vs LaBSE vs AfriE5 vs mBERT(-control) | RQ1, RQ2 |
| E3 | Zero-shot embeddings (no fine-tuning) | Tiers 2+3 eval sets | LaBSE, AfriE5, TF-IDF | RQ2, RQ3a |
| E4 | Effect of fine-tuning (transfer learning) | Tiers 1+3 | LaBSE zero-shot vs fine-tuned; AfroXLMR zero-shot vs fine-tuned | RQ3a |
| E5 | Cross-lingual transfer: English→target | Train: English PAWS-X; Eval: Tiers 2+3 | PAWS fine-tuned AfroXLMR-base, zero-shot on yo/ig/ha; translate-test optional baseline | RQ3b |
| E6 | Linguistic factors (controlled) | Perturbation sets built from Tier 3 gold | best model + baseline | RQ4 |
| E7 | Demo application | — | best cross-encoder + embedding score | RQ5 |

**Training configuration (config-driven, not hardcoded):** learning rate 2e-5 (grid {1e-5, 2e-5, 3e-5} on validation only), effective batch 32 (per-device 16 × gradient accumulation 2), linear warmup 10% of steps, max sequence length 128, weight decay 0.01, fp16, **3–5 epochs with early stopping on validation F1 (patience 2–3)**, best-checkpoint selection (`load_best_model_at_end`), **≥3 seeds reported as mean ± std** (canonical justification: the BERT README's warning that small MRPC-scale sets have high dev variance). These defaults follow Devlin et al. (2019), XLM-R/XTREME practice, and the HF Trainer documentation.

**Monolingual vs cross-lingual separation:** every result table has a `setting` column (`monolingual-{lang}` / `zero-shot-en→{lang}` / `cross-lingual-pivot`); no averaging across settings.

**Evaluation protocol:**
- Classification: Accuracy, Precision, Recall, F1 (binary), ROC-AUC — per language **and** aggregate; confusion matrices; classification reports; training curves.
- Similarity: **Spearman ρ (primary)** and Pearson r — Spearman primary because SemRel's Best-Worst-Scaled scores are ordinal-leaning, and this matches SemEval-2024 Task 1 / AfriMTEB practice.
- **Significance testing:** McNemar's exact test (Dietterich 1998; Dror et al. 2018) for classifier comparisons vs the TF-IDF baseline; paired bootstrap (Berg-Kirkpatrick et al. 2012; Koehn 2004) on F1/Accuracy as secondary.
- Score calibration for the demo: sigmoid of the cross-encoder logit with a single temperature T fit on validation (Guo et al. 2017); `(cos+1)/2` for embedding and TF-IDF scores so all systems share a [0,1] scale (comparable to SemRel's 0–1). Reported metrics are invariant to these monotone transforms — calibration affects only the UI number.
- Error analysis: confusion-pair sampling from the gold test set, per-register breakdown, length effects, lexical-overlap buckets (ties directly into E6).

---

## 9. Linguistic Analysis Plan (E6)

Controlled perturbation experiments on the human-verified gold test set (each perturbation applied to one sentence of a pair, labels unchanged, so any accuracy drop is attributable to the perturbation):

| Factor | Manipulation | Known constraint |
|---|---|---|
| Diacritics/tone marks | Yorùbá undermarking (díacritics stripped) vs fully marked | Stripping is an *ablation condition*, never the default preprocessing — Yoruba tone is phonemic (e.g. *ọmọ* 'child' vs *omo* can differ); Igbo vowel harmony and Hausa apostrophes/glottals preserved |
| Negation | Insert/replace negators (kò, ìkọ́ / ọ bụ / bà … ba) | Negation flips meaning → these pairs become non-paraphrases; run as label-flip detection test |
| Word order | SVO reordering where grammatical | Limited by grammar; documented per case |
| Morphology | Plural/aspectual affix substitution | Igbo agglutination; Hausa plurals |
| Synonymy | Lexical substitution using the target-language lexicons (Naija-Lex etc. where licensed) | Coverage may be thin — reported honestly |
| Spelling variation | Documented variant orthographies (e.g. Hausa ẖ/'h conventions) | Only variants attested in sources |

No causal linguistic claims will be made beyond what these controlled experiments show; observed deltas are reported as measurements with confidence intervals, and hypotheses about causes are labelled as hypotheses.

---

## 10. Computing Resources

**Local machine (verified):** Intel i7-5557U, **2 cores**, 16 GB RAM, **no GPU, no Apple-Silicon MPS** (PyTorch MPS requires Metal-capable hardware). Verdict: **local CPU fine-tuning of base-size transformers is impractical** (order-of-magnitude > 1 day/epoch); feasible locally only for TF-IDF, embedding inference on small sets, evaluation, tests, and the Streamlit demo (single-pair CPU inference is fine).

**Training host: Google Colab free tier (T4, 16 GB).** Feasibility is proven by published precedent — the FiliSenti model card documents fine-tuning **xlm-roberta-large (355M)** at batch 16, fp16, seq-len 128 on a Colab T4; our largest training model (AfroXLMR-base, 278M) is smaller. Projection (to be measured on first run): ~15k pairs → ~940 steps/epoch at batch 16 → **roughly 15–35 min/epoch, < 2 h for 3 epochs**; well inside Colab's 12-hour session cap. Required workflow: checkpoint to Google Drive, `save_total_limit`, resume-on-disconnect logic. Kaggle (2× T4, 30 h/week) is the backup.

**Estimated total GPU budget (all experiments, 3 seeds, hyperparameter sweep on 1 seed):** order of 15–25 T4-hours → comfortably within free tiers across 2–3 weeks of part-time runs.

**Storage:** all checkpoints ≤ 1.1 GB each (base models, fp16); final artefacts (best checkpoints + metrics) < 10 GB. Local disk has ~14 GB free — checkpoints live in Google Drive; only best models are pulled locally for the demo.

---

## 11. Phased Roadmap (with approval gates)

| Phase | Weeks | Deliverables | Gate |
|---|---|---|---|
| **P0. Feasibility** (this report) | 1 | This report; decisions in §13 | **Your approval** |
| **P1. Environment & skeleton** | 1 | Repo structure, `requirements.txt`, config system (YAML), logging, research log, CI-runnable tests, local CPU smoke-test on a 200-pair sample | Review |
| **P2. Data acquisition & audit** | 2 | Downloaded raw sources (originals immutable), provenance + licence register per source, data-quality report (encoding, duplicates, LID validation via AfroLID, language metadata checks) | Review + **licence emails sent** |
| **P3. Dataset construction & annotation** | 3–4 | Pair-construction pipeline (positives/negatives/length-matching/overlap filtering), pseudo-label + filtering pipeline, annotation guidelines document, gold annotation export, IAA report, leakage-safe splits (grouped by source, seeded) | Review + **your annotation capacity confirmed** |
| **P4. Baseline & evaluation harness** | 1–2 | TF-IDF baseline results, metrics library (classification + correlation + McNemar/bootstrap), evaluation CLI, first results tables | Review |
| **P5. Transformer training (Colab)** | 3–4 | Colab notebooks + identical local pipeline, trained AfroXLMR/AfriBERTa (+ LaBSE fine-tune), checkpoints on Drive, training curves, seed-averaged results | Review |
| **P6. Analysis** | 2 | Per-language result tables, significance tests, E5 transfer results, E6 linguistic perturbation study, error-analysis report, model-comparison charts | Review |
| **P7. Application & write-up support** | 2 | Streamlit demo (language selector, model selector, paraphrase decision, calibrated score), README, tests, documentation | Final review |

Realistic total: **12–14 weeks**, with the critical path running through P3 (annotation) — hence the §13 questions.

---

## 12. Risks, Limitations, Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Annotation capacity insufficient for 500–1,000 pairs × 3 languages | Medium | **High** — gold data underpins all claims | Reduce gold to ~300/lang (still reportable); prioritise Hausa (SemRel exists) + one other; use Tier-2 AfriXNLI as secondary eval; your input needed (§13) |
| R2 | SemRel licence not granted | Medium | Medium (Hausa training only) | Fallback: SemRel as private eval only, or Hausa trained on Tier-3 constructed data like the others |
| R3 | Constructed positives are semantically shallow (MT-rendering artifacts) | Medium-High | High | Human verification mandatory; report label provenance per subset; ablation with/without constructed data (E4 design already supports this) |
| R4 | Models learn lexical-overlap shortcut | Medium | Medium | PAWS-style overlap filtering in construction; overlap-bucketed error analysis; TF-IDF baseline contextualises the gap |
| R5 | Register mismatch (news/Twitter/formal) | High | Medium | Register recorded as metadata; per-register reporting; stated as limitation |
| R6 | Colab quota/session limits | Medium | Low-Medium | Checkpoint-to-Drive + resume; Kaggle fallback; large-model experiments (AfroXLMR-large, AfriE5 fine-tune) marked optional |
| R7 | Small eval sets → wide confidence intervals | High | Medium | Report bootstrap CIs; ≥3 seeds; avoid overclaiming small deltas; McNemar before claiming significance |
| R8 | mBERT Igbo/Hausa results misread as "mBERT is bad at semantics" | — | Medium (integrity) | Framed strictly as tokenisation/coverage control, with the canonical language list cited |
| R9 | Scope creep (all architectures, all models) | Medium | High (timeline) | Fixed model list (§7); stretch items explicitly optional |

**Honest limitations to state in the thesis regardless of outcomes:** no pre-existing paraphrase data (so all conclusions are conditional on our construction and annotation choices); single-annotator risk if R1 bites; Hausa advantage (only language with external human-annotated data); results may not generalise beyond news/formal register.

---

## 13. Decisions Required From You (before P1 starts)

1. **Annotation capacity** — can you (and any native-speaker colleagues you can recruit) annotate ~500–1,000 sentence pairs per language? Options: (a) you annotate all three; (b) you + 2–3 native speakers (enables proper IAA); (c) reduced gold (~300/lang); (d) LLM-assisted candidate labelling with smaller human verification (documented as such). Are you a native/proficient speaker of any of the three languages?
2. **SemRel licence** — are you willing to email the SemRel2024 Hausa contacts (named in their repo README) for redistribution/usage permission? (I can draft the email.)
3. **Language priority** — if capacity is constrained, which language is prioritised? (Recommendation: Hausa first — it has real annotated data — then Yoruba, then Igbo, since Igbo's largest corpus has an unstated licence.)
4. **Compute** — confirm you have a Google account able to run Colab (and ideally 15–25 h of GPU over the semester); Kaggle as fallback.
5. **Report scope** — approve the model list in §7 and the three-tier data strategy in §6, or request adjustments (e.g. adding NaijaBERT as a small-model point, or the optional AfroXLMR-large run).

---

## 14. References (all verified to exist; see Appendix A for caveats)

1. Ogueji, K., Zhu, Y., & Lin, J. (2021). *Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-resourced Languages.* MRL @ EMNLP 2021. https://aclanthology.org/2021.mrl-1.11/ — AfriBERTa.
2. Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). *Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning.* COLING 2022. https://aclanthology.org/2022.coling-1.382/ — AfroXLMR.
3. Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). *Language-agnostic BERT Sentence Embedding (LaBSE).* ACL 2022. https://aclanthology.org/2022.acl-long.62/
4. Uemura, K., Zhang, M., & Adelani, D. I. (2025). *AfriMTEB and AfriE5: Benchmarking and Adapting Text Embedding Models for African Languages.* arXiv:2510.23896 (EACL 2026).
5. Ousidhoum, N., Muhammad, S. H., et al. (2024). *SemRel2024: A Collection of Semantic Textual Relatedness Datasets for 13 Languages.* Findings of ACL 2024 / arXiv:2402.08638.
6. Ousidhoum, N., et al. (2024). *SemEval-2024 Task 1: Semantic Textual Relatedness for African and Asian Languages.* SemEval-2024. https://aclanthology.org/2024.semeval-1.272/ (arXiv:2403.18933).
7. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. arXiv:1908.10084.
8. Yang, Y., Zhang, Y., Tar, C., & Baldridge, D. (2019). *PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification.* EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1382/
9. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019. arXiv:1810.04805.
10. Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R).* ACL 2020. arXiv:1911.02116.
11. Adelani, D. I., et al. (2021). *MasakhaNER: Named Entity Recognition for African Languages.* TACL 2021. (Per-language transfer analysis.)
12. Ezeani, I., et al. (2020). *Igbo-English Machine Translation: An Evaluation Benchmark.* LREC 2020. arXiv:2004.00648 — IGBONLP corpus.
13. Adebara, A., et al. (2024). *Cheetah: Efficient and Flexible LLMs for 517 African Languages.* arXiv:2401.01053. (Paraphrase *generation* context.)
14. Nwafor, E., et al. (2025). *NaijaNLP: A Survey of Nigerian Low-Resource Languages and Technologies.* arXiv:2502.19784. (Documents the paraphrase-data gap.)
15. Dolan, W. B., & Brockett, C. (2005). *Automatically Constructing a Corpus of Sentential Paraphrases.* IWP 2005. https://aclanthology.org/I05-5002/ — MRPC (methodological precedent: automated extraction, human labels).
16. Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.* Neural Computation 10(7).
17. Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). *The Hitchhiker's Guide to Testing Statistical Significance in NLP.* ACL 2018. https://aclanthology.org/P18-1128/
18. Berg-Kirkpatrick, T., Burkett, D., & Klein, D. (2012). *An Empirical Investigation of Statistical Significance in NLP.* EMNLP-CoNLL 2012. https://aclanthology.org/D12-1091/
19. Koehn, P. (2004). *Statistical Significance Tests for Machine Translation Evaluation.* EMNLP 2004.
20. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML 2017. arXiv:1706.04599.
21. Agirre, E., et al. (2012). *SemEval-2012 Task 6: A Pilot on Semantic Textual Similarity.* SemEval 2012. (Pearson convention for STS.)

---

## Appendix A — Items that could NOT be verified (do not cite without checking)

| Item | Status |
|---|---|
| BERT paper Appendix A.3 exact hyperparameter wording | README verified verbatim; appendix text not fetched — treat "batch 16/32, LR {5e-5…2e-5}, 2–4 epochs" as standard practice, not a verified quote |
| XLM-R appendix exact fine-tuning hyperparameters (LR 2e-5, 10k warmup) | From secondary summaries; verify against Appendix A before quoting in thesis |
| Artetxe et al. (2019) translate-test results on PAWS-X | Paper identified, PAWS-X claim not confirmed by direct fetch |
| T4 wall-clock for XLM-R-base fine-tuning | **Projection** (§10) — measure on first run |
| AfriMTEB `SemRel24STS` language list | Table not fetched; confirm from PDF before citing which languages it covers |
| SemRel2024 paper (2,551) vs released CSVs (5,105) Hausa pair counts | **RESOLVED in P2 (2026-11-19):** proper CSV parsing yields 2,551 pairs (1,736/212/603) — the paper is correct; the 5,105 figure was physical lines (each pair stores both sentences in one quoted field with an embedded newline) |
| "Jesujoba Ojo" = "Jesujoba O. Alabi" (name change) | Plausible, unverified — cite as Alabi et al. |
| `castorini/afriberta_base` licence | No licence declared — use `afriberta_large` (MIT) |
| mBERT vocabulary size (110,000 vs 119,548) | Conflicting sources; irrelevant to design decisions |
| Adegoke-Elijah et al. (2024) Yoruba path-based similarity | Exists (minor venue); cite as evidence of the gap, not as a baseline |
| AfriBERTa tokenizer `model_max_length` must be set manually (512) | Stated on model card; verify empirically in P1 smoke test |
