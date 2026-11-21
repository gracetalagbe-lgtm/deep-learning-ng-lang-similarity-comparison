# Low-Resource Paraphrase Detection for Yorùbá, Igbo and Hausa: Dataset Construction, Cross-Lingual Transfer, and the Effects of Label Provenance and Split Design

[AUTHORS]
MSc Candidate, Computer Science & Artificial Intelligence

[AFFILIATION]
MSc research project, Deep Learning. System and evaluation code: `afripara` v0.1.0 (Python 3.11). All artifacts reproducible from the project repository.

[ABSTRACT]
**Context.** Semantic textual relatedness and paraphrase identification are core NLP capabilities, yet no paraphrase-identification dataset exists for Yorùbá, Igbo, or Hausa — three of Africa's most widely spoken languages — a gap verified against the Hugging Face and GitHub APIs, the AfriMTEB benchmark (which excludes semantic similarity entirely), and the 2025 NaijaNLP survey. **Problem.** Without in-language pair data, neither supervised paraphrase detection nor trustworthy cross-lingual evaluation is possible, and it is unknown how transformer models behave when the only available training signal is machine-translation-inherited ("silver") or when evaluation splits silently share sentences. **Methodology.** We build a complete train-and-evaluate stack: a provenance-audited data pipeline (31 files, SHA-256 verified); a SemRel2024-Hausa sentence-disjoint re-split (2,041/255/255; zero leakage) alongside the official split (which we show shares 644 sentences between train and test); NLLB-200-translate-train construction of silver pair sets (Yorùbá 2,873; Igbo 2,948; Hausa 1,477) with adversarial PAWS-Wiki overlap control and per-pair label provenance; and 29 fine-tuning/zero-shot runs (AfroXLMR-base, AfriBERTa-large, LaBSE, AfriE5, English-PAWS transfer) on a Colab T4 with 3-seed replication, dev-tuned thresholds, exact McNemar tests, paired bootstrap CIs, and temperature calibration. **Key quantitative results.** The Hausa cross-encoder fine-tuned on human labels reaches 0.783 accuracy (F1 0.840 ± 0.008 across three seeds), ROC-AUC 0.863, and Spearman ρ=0.697 on the leakage-free split — matching the SemRel2024 paper's fine-tuned LaBSE (0.69) with a different architecture. Official-split scores are inflated by ≈+6 accuracy points for leakage-exposed models, and the inflation can invert model selection: LaBSE-FT beats the cross-encoder on the leaky split (ρ 0.767 vs 0.735) but loses on the clean split (0.604 vs 0.697). Label provenance dominates outcomes: a Hausa model trained on silver labels is at chance on real relatedness (ρ=0.153 vs 0.697), and English-PAWS zero-shot transfer beats in-language silver fine-tuning on Hausa (0.811 vs 0.626 accuracy; exact McNemar min-p=2.5×10⁻⁷). A diacritics-stripping ablation shows accuracy is tone-mark insensitive but mixed-marking pairs shift calibration (F1 0.616→0.442), while the lexical baseline collapses (−70%). **Conclusion.** For low-resource paraphrase work, split design and label provenance are first-order experimental variables, not hygiene details; we release the full audited pipeline, splits, run artifacts, and a serving demo to support reproducible evaluation.

[KEYWORDS]
paraphrase identification; low-resource languages; Yorùbá; Igbo; Hausa; transformer cross-encoders; label provenance; leakage-controlled evaluation

## 1. Introduction

### 1.1 Background and Domain Context

Yorùbá, Igbo, and Hausa are spoken by well over 100 million people across West Africa, yet remain low-resource languages for sentence-level semantics. African-language NLP has concentrated on morphosyntactic and token-level tasks — named entity recognition (MasakhaNER; Adelani et al., 2021), sentiment (AfriSenti), and machine translation (Menyo-20k; IGBONLP) — supported by increasingly capable multilingual encoders such as AfriBERTa (Ogueji et al., 2021), AfroXLMR (Alabi et al., 2022), LaBSE (Feng et al., 2022), and AfriE5 (Uemura et al., 2025). Sentence-pair semantics has lagged: the SemRel2024 effort (Ousidhoum et al., 2024) supplied graded semantic-textual-relatedness (STR) data for Hausa but no paraphrase labels for any of the three languages, and the African embedding benchmark AfriMTEB explicitly excludes semantic similarity for lack of data (Uemura et al., 2025). A survey of the Hugging Face datasets API, GitHub, and the NaijaNLP survey (Nwafor et al., 2025) confirms that no paraphrase-identification corpus exists for Yorùbá, Igbo, or Hausa.

The methodological backdrop is equally consequential. Transformer cross-encoders — which attend jointly over a concatenated sentence pair — consistently outperform bi-encoder similarity for pair tasks (Reimers & Gurevych, 2019), but they require supervised pair data. Where such data does not exist, two workarounds dominate: inherit labels through machine translation (translate-train, as popularised by PAWS-X; Yang et al., 2019), or transfer zero-shot from English. Both paths interact with label noise, MT quality, and evaluation-split integrity in ways that the low-resource NLP literature rarely quantifies.

### 1.2 Problem Statement

This project addresses four concrete problems:

1. **No training or evaluation data.** No paraphrase-identification dataset exists for Yorùbá, Igbo, or Hausa; even graded STR exists only for Hausa (SemRel2024). Any claim about paraphrase capability in these languages must therefore begin with dataset construction — and constructed labels inherit the quality of the machine-translation system used to build them.
2. **Silent evaluation leakage.** The official SemRel2024-Hausa split shares sentences across folds: we measured 644 sentences common to train and test (62% of the 1,032 unique test sentences), 244 shared between train and dev, and 101 between dev and test, because the corpus samples pairs from shared documents. Models evaluated on such splits can score well by memorisation, and published comparisons on the official split may not measure generalisation.
3. **Unknown behaviour of silver labels.** Translate-train labels are inherited from human annotations of the *English* source. If translation erases the semantic distinction that the label encodes (e.g., PAWS-style entity swaps), silver training can teach the wrong decision boundary. The magnitude of this effect on real relatedness data had not been measured for these languages.
4. **No working system.** Beyond offline metrics, the project required a deployed inference path — calibrated scores, fixed dev-tuned thresholds, and displayed label provenance — so that model behaviour can be inspected directly rather than only through aggregate tables.

### 1.3 Proposed Solution and Core Contributions

We answer these problems with an end-to-end research system (Figure 1) comprising a provenance-audited data pipeline, a leakage-controlled split, an MT-based dataset construction route with per-pair provenance, a 29-run transformer benchmark, a statistical evaluation harness, and a serving demo. The contributions are:

- **C1 — A verified-gap analysis.** Systematic confirmation (HF/GitHub API searches, AfriMTEB, NaijaNLP survey) that no paraphrase dataset exists for the three languages, making dataset construction a research contribution rather than preprocessing (feasibility report, 2026-11-18).
- **C2 — Provenance-audited data acquisition.** Six sources downloaded with SHA-256 manifests (31/31 hashes re-verified), licence register, and a 20-slice audit that quantified duplicates (Menyo-20k 167; IGBONLP 453), embedded zero-width characters (U+200B/U+200E/U+200F/U+FEFF), empty lines, and split leakage.
- **C3 — A sentence-disjoint re-split of SemRel2024-Hausa** via connected-component splitting (union-find over shared sentences, atomic component assignment, greedy quota filling): 2,041/255/255 pairs with zero cross-fold sentence overlap and zero dropped pairs, released alongside the official split for comparability.
- **C4 — Silver pair sets for three languages** (translate-train over adversarial PAWS-Wiki pairs through NLLB-200-distilled-600M: Yorùbá 2,873, Igbo 2,948, Hausa 1,477), with per-pair provenance columns (`label_source`, `english_s1/s2`, `qc_len_ratio`), copy-through guards, and QC filters, so silver and gold data are never silently mixed.
- **C5 — A 29-run benchmark** (AfroXLMR-base, AfriBERTa-large, LaBSE, AfriE5, English-PAWS transfer; 3 seeds where compute allowed) scored by a single audited evaluation module: dev-tuned thresholds, bootstrap CIs, exact McNemar tests, and temperature calibration.
- **C6 — Findings on split design and label provenance** (Section 4): leakage inflates leakage-exposed neural scores by ≈+6 accuracy points and can invert model selection; silver-trained models collapse on real relatedness (ρ 0.153 vs 0.697) while English-PAWS zero-shot transfer beats in-language silver fine-tuning on Hausa (+18.5 points, min-p=2.5×10⁻⁷).
- **C7 — A deployed, calibrated demo** (Streamlit) serving the fine-tuned checkpoints with temperature-scaled scores, dev-tuned thresholds, and on-screen provenance warnings.

## 2. Related Work and Theoretical Foundation

**Paraphrase and STR datasets.** Paraphrase identification was established by MRPC (Dolan & Brockett, 2005) and made adversarial by PAWS (Zhang et al., 2019), whose high-overlap pairs defeat lexical baselines. PAWS-X (Yang et al., 2019) extended it to six languages via professional translation and validated translate-train as a low-resource strategy — the precedent our construction route follows. SemRel2024 (Ousidhoum et al., 2024) provided graded STR for 14 languages including Hausa, collected by Best-Worst Scaling; its authors explicitly rejected MT-pivot construction for African languages, citing unreliable MT — a caution our silver-label findings quantify. AfriMTEB/AfriE5 (Uemura et al., 2025) benchmark African embeddings and note that semantic similarity remains uncovered for these languages.

**Multilingual encoders.** XLM-R (Conneau et al., 2020) established large-scale multilingual pretraining; mBERT (Devlin et al., 2019) covers 104 languages but includes neither Igbo nor Hausa, which is why we exclude it as a baseline and retain it only as a documented negative control. AfroXLMR (Alabi et al., 2022) applies multilingual adaptive fine-tuning to 17 African languages (plus en/fr/ar) and outperforms AfriBERTa on MasakhaNER (reported 81.8 vs 76.8 F1). LaBSE (Feng et al., 2022) provides 109-language sentence embeddings with verified Yorùbá/Igbo/Hausa coverage; AfriE5 (Uemura et al., 2025) is African-specialised and requires `query:`/`passage:` input prefixes. Bi-encoders trade accuracy for speed: cross-encoders gained ≈2.7 Spearman points over bi-encoders on STS-B (Reimers & Gurevych, 2019), motivating our two-track design.

**Published Hausa STR baselines.** On SemRel2024-Hausa, the paper reports unsupervised XLM-R at ρ=0.04, mBERT at 0.16, a lexical-overlap Dice baseline at 0.31, and fine-tuned LaBSE at 0.69; the SemEval-2024 Task 1 winning system (AAdaM), an AfroXLMR-family cross-encoder, averaged ρ=0.800 across languages. These anchor our comparisons in Section 4.

**Evaluation methodology.** We adopt McNemar's exact test (Dietterich, 1998; Dror et al., 2018), paired bootstrap resampling (Berg-Kirkpatrick et al., 2012; Koehn, 2004), Spearman ρ as the primary ranking metric for BWS-derived ordinal scores (Agirre et al., 2012), and post-hoc temperature scaling on validation data (Guo et al., 2017) for the serving path.

## 3. System Architecture and Methodology

### 3.1 End-to-End System Architecture

Figure 1 shows the six-stage architecture. Stages 1–3 (data acquisition, audit/normalisation, split/construction) run locally on CPU; stage 4 (training) runs on a Google Colab T4 GPU; stages 5–6 (scoring, calibration, serving) return to local CPU. The design principle is that **training only ever produces raw predictions and training history**; all scientific scoring happens locally in one audited module (`afripara.evaluation.metrics`, unit-tested), so no duplicated scoring code exists in the training notebook.

[FIG:fig1_architecture.png|Figure 1: End-to-end system architecture. Training (stage 4) is the only GPU stage; every scientific number is computed locally by the audited evaluation module from saved predictions.]

The serving path (`afripara.inference.predictor`) loads a seed-42 checkpoint, applies the same `clean_text` normalisation as training, computes the positive-class log-odds, rescales them by the validation-fitted temperature, and applies the dev-tuned threshold:

[EQ] z = log(p₁) − log(p₀);  p̂ = σ(z / T);  ŷ = 1 [p̂ ≥ τ_dev ] [/EQ]

where T is fitted by minimising dev negative log-likelihood (Section 3.4) and τ_dev maximises dev F1 over a 99-point grid. The Streamlit demo (stage 6) wraps this predictor with a model-provenance panel.

### 3.2 Dataset Acquisition and Preprocessing Pipeline

**Sources.** Table I summarises the acquired sources; every file carries a SHA-256 entry in `data/raw/provenance.json` (31/31 re-verified). JW300-derived folders inside the IGBONLP repository were excluded as legally unusable (withdrawn from OPUS for copyright); SemRel2024 and IGBONLP licences are unstated, so both are used as published research data without redistribution.

[TCAP:Table I: Data sources acquired in Phase 2 (sizes are on-disk parsed counts).]
| Source | Language | Size | Labels | Licence | Role |
|---|---|---|---|---|---|
| SemRel2024-Hausa | Hausa | 2,551 pairs (1,736/212/603) | graded 0–1 (BWS) | unstated — research use only | primary human-labelled train/eval |
| AfriXNLI | yo/ig/ha/en | 1,050 pairs/lang raw | 3-way NLI | Apache-2.0 | proxy binary eval harness |
| PAWS-Wiki (en) | English | 49,401 train (44.2% positive), 8,000 val, 8,000 test | binary paraphrase | Google free-use | construction source; transfer arm |
| Menyo-20k | yo–en | 10,070/3,396/6,633 (train/dev/test) | parallel | CC-BY-NC-4.0 | MT-vs-human candidates (unused) |
| IGBONLP benchmark | ig–en | 10,000/193/597 | parallel | unstated | MT-vs-human candidates (unused) |
| NTREX-african | yo/ig/ha | 1,997 sents/lang | parallel | CC-BY-SA-4.0 | acquired; unused downstream |

**Audit findings.** The 20-slice audit quantified: 167 duplicate pairs in Menyo-20k and 453 in IGBONLP (duplicates removed downstream by order-insensitive keys); 11 empty IGBONLP lines; one malformed Menyo row; zero-width artifacts (U+200B most frequent: 138 occurrences in the Menyo train split; U+FEFF, U+200E, U+200F also present) — motivating a `clean_text` that applies Unicode NFC, strips exactly the zero-width class, and collapses whitespace while preserving all diacritics (Yorùbá tone marks are phonemic). A Latin-1 double-encoding case in one Igbo test sentence ("Lăpuş" → mojibake) passed the strict-UTF-8 audit because the bytes are valid UTF-8 — a recorded limit of encoding audits.

**Split integrity.** The official SemRel2024-Hausa split is sentence-leaky: 644 sentences shared between train and test (62% of 1,032 unique test sentences). We therefore built a second, leakage-free split with a connected-component splitter: sentences are union-find components (a pair's two sentences always share a component); whole components are assigned atomically to folds by greedy quota filling (weights 0.8/0.1/0.1). The resulting split is 2,041/255/255 pairs with zero cross-fold sentence overlap and zero dropped pairs (the simpler hash splitter drops every pair whose sentences straddle folds — unusable here). Dev and test folds both drew 65.1% positive labels vs 54.1% for train, a skew of the re-split that we report rather than hide.

**Construction.** Silver pairs are built by translate-train: 15,000 PAWS-Wiki English sentences (6,000/6,000/3,000 for yo/ig/ha) were translated on a Colab T4 with NLLB-200-distilled-600M (fp16, batch 32, `forced_bos_token_id` = target-language id), then paired so both members inherit the human PAWS label. Guards: a copy-through check aborts if >50% of the first batch outputs equal their inputs (an initial run with the wrong `forced_bos_token_id` produced verbatim English copies and was fixed); pairs are dropped for empty translations or length ratio >3.0 (151 dropped) and deduplicated (51 removed; 0 empty). Targets were 3,000/3,000/1,500 pairs; deduplication yielded 2,873/2,948/1,477. Every row carries `label_source=paws_translated` plus the English source sentences, so silver data is structurally distinguishable from human-labelled data. A planned MT-vs-human positive route and a 390-pair native-speaker gold validation were designed and tooling built, but skipped by researcher decision (2026-11-20) to prioritise the transformer benchmark; consequences are analysed in Sections 4 and 6.

**Tokenisation statistics (recomputed under the deployed tokenizer).** An earlier engineering estimate (p50 36–56 tokens, max 95) was based on a 200-pair sample under AfriBERTa-small and does **not** hold for the deployed AfroXLMR tokenizer. Full-data pair-encoded lengths are: Yorùbá p50=122 (max 1,534), Igbo p50=101 (max 1,524), Hausa p50=73 (max 762); SemRel-Hausa pairs fit easily (p50≈70, max 104). Consequently, at the configured max_seq_length=128, **44.8% of Yorùbá, 23.6% of Igbo, and 0.9% of Hausa constructed pairs are truncated**, as are 15.1%/6.3% of the Yorùbá/Igbo AfriXNLI eval sets, while all SemRel sets fit. A repetition screen also found NLLB degeneration loops in 2.3% of Yorùbá, 0.5% of Hausa, and 0.1% of Igbo constructed pairs (repeated-word runs whose length ratio remains ≈1 and thus pass QC). These facts are carried into the limitations analysis (Section 6).

[TCAP:Table II: Processed datasets used for training and evaluation (binary label derivation: SemRel score ≥ 0.5; constructed labels inherited from PAWS).]
| Dataset | Train | Dev | Test | Positive % | label_source |
|---|---|---|---|---|---|
| semrel_hau_official | 1,736 | 212 | 603 | 56.3 | semrel_human |
| semrel_hau_disjoint | 2,041 | 255 | 255 | 54.1 / 65.1 / 65.1 | semrel_human |
| translate_train_yor | 2,299 | 287 | 287 | 45.1 | paws_translated (silver) |
| translate_train_ibo | 2,358 | 295 | 295 | 44.7 | paws_translated (silver) |
| translate_train_hau | 1,181 | 148 | 148 | 43.9 | paws_translated (silver) |
| afrixnli_binary_{yor,ibo,hau,eng} | — | 300/297/300/300 | 400 each | 50.0 | afrixnli_proxy |

AfriXNLI is converted to binary by entailment→1, contradiction→0, neutral dropped (perfectly balanced); the Igbo dev set lost 5 pairs to 10 malformed TSV lines (445 usable dev pairs, 1.1% loss), so Igbo eval totals 697 rather than 700.

### 3.3 Model Selection and Technical Justification

Table III records the selection rationale. AfroXLMR-base is the primary cross-encoder: XLM-R-base architecture (12 layers, hidden 768, 12 heads, 278,045,186 parameters — counted exactly from the deployed checkpoints' safetensors headers), adaptively fine-tuned on 17 African languages including all three targets, MIT-licensed, and the architecture family of the SemEval-2024 Task 1 winning system. AfriBERTa-large (≈126M, MIT) is the African-pretrained baseline: trained on <1 GB of news/web text, it tests whether African corpus exposure without adaptive fine-tuning suffices. LaBSE (471M, Apache-2.0) and AfriE5 (560M, MIT) are the embedding track — the only two families with verified yo/ig/ha coverage; AfriE5 requires `query:`/`passage:` prefixes. mBERT is excluded from benchmarking because Igbo and Hausa are absent from its 104-language pretraining set: fine-tuning it would measure tokenisation out-of-vocabulary behaviour, not semantics; it is retained as a documented negative control, consistent with MasakhaNER's transfer findings (Adelani et al., 2021). Proprietary-unlicensed candidates ("Native-Bird", AfriBERTa-base) and models without verified coverage (multilingual MPNet/MiniLM) were rejected during feasibility.

[TCAP:Table III: Candidate models and selection decisions (coverage verified against primary sources during feasibility).]
| Model | Params | yo/ig/ha coverage | Licence | Decision & role |
|---|---|---|---|---|
| AfroXLMR-base (Davlan/afro-xlmr-base) | 278,045,186 (counted) | yes/yes/yes | MIT | primary cross-encoder |
| AfriBERTa-large (castorini/afriberta_large) | ≈126M | yes/yes/yes | MIT | African-specific baseline (no tokenizer.json → slow SentencePiece) |
| LaBSE (sentence-transformers/LaBSE) | ≈471M | yes/yes/yes (paper Table 10) | Apache-2.0 | embedding track; fine-tune arm |
| AfriE5-Large-instruct (McGill-NLP) | ≈560M | yes/yes/yes | MIT | embedding track, zero-shot (prefix-mandatory) |
| mBERT | ≈178M | yes/no/no | Apache-2.0 | negative control only (pretraining exclusion) |
| "Native-Bird"; AfriBERTa-base | — | — | none declared | rejected (unlicensed) |
| multilingual MPNet/MiniLM | — | no/no/no | Apache-2.0 | rejected (no coverage) |

### 3.4 Training Configuration and Implementation Details

All 29 runs execute one self-contained Colab cell driven by a generated run plan (`run_plan.json`) with a plan/bundle consistency gate that aborts if any run references a file absent from the data bundle — a gate added after a first failed attempt (path mismatch, zero usable runs) and which immediately caught four further mismatches. Table IV lists the exact configuration. Optimiser and schedule are Hugging Face defaults (AdamW, linear decay with 10% warmup); early stopping monitors dev F1 with patience 3; the best epoch is restored before prediction. Seeds 42/43/44 replicate every AfroXLMR cross-encoder arm (12 runs); AfriBERTa-constructed arms, the English-PAWS transfer arm, and all bi-encoder arms run seed 42 only. Checkpoints (≈1.06 GB each) are retained for seed 42. Between runs the notebook garbage-collects and empties the CUDA cache; seven embedding runs hit transient T4 OOM and were re-run successfully after reducing encode batch 64→16 and fine-tune batch 16→8 (all seven nevertheless wrote complete predictions in the failed pass — flagged for comparability in Section 6).

[TCAP:Table IV: Training configuration and environment (identical for every cross-encoder run; bi-encoder arms are zero-shot or 3-epoch cosine-loss fine-tunes without early stopping).]
| Item | Value |
|---|---|
| Hardware | Google Colab T4 (14.6 GB usable), fp16; local Intel i7-5557U CPU-only for scoring/serving |
| Software | Python 3.11; torch 2.2.2 (last macOS x86_64 wheel — local pin); transformers 4.49.0; sentence-transformers 3.4.1; datasets 3.2.0; accelerate 1.3.0; sentencepiece 0.1.99; scikit-learn 1.9.1; numpy 1.26.4; scipy 1.17.1; pandas; matplotlib 3.11.2; streamlit 1.67.0 |
| Optimiser / schedule | AdamW, linear decay, warmup ratio 0.1 (HF defaults) |
| Learning rate | 2e-5 (a {1e-5, 2e-5, 3e-5} validation grid was declared but never consumed — only 2e-5 ran) |
| Batch | 16 per device × 2 gradient accumulation = effective 32 |
| Epochs / early stopping | ≤5; patience 3 on dev F1; best checkpoint restored (`load_best_model_at_end`) |
| Max sequence length | 128 tokens (truncation rates in Section 3.2) |
| Weight decay / label smoothing | 0.01 / none (cross-entropy on 2 classes) |
| Seeds | 42/43/44 (cross-encoder arms); 42 only otherwise |
| Metrics during training | accuracy, F1 (argmax of logits); ROC-AUC computed at scoring time |
| Loss (bi-encoder fine-tune) | CosineSimilarityLoss on unit-normalised embeddings, 3 epochs, batch 8 |
| Total GPU budget | 29 runs ≈ 5.7 h on one T4 (per-run 263–1,147 s for cross-encoders) |

**Evaluation protocol.** Every run dumps raw dev and test predictions (softmax positive-class scores) to Drive; scoring happens locally. Decision thresholds are tuned on dev only (grid 0.01–0.99, 99 points) and then applied unchanged to test — never tuned on test. Per-run F1 CIs use percentile bootstrap (10,000 resamples, seed 42); phase-6 paired Δ-accuracy CIs use 2,000 resamples; McNemar's exact test compares each system against the saved TF-IDF baseline predictions on matched items (b, c discordant counts; two-sided exact binomial). Continuous ranking uses Spearman ρ (primary; BWS scores are ordinal-leaning) and Pearson r. Similarity metrics are undefined (NaN) on silver sets, which carry binary labels only — we report this as "undefined", never as zero.

## 4. Experimental Results and Quantitative Evaluation

### 4.1 Evaluation Metrics

For a threshold τ and positive-class probability ŷ:

[EQ] Accuracy = (TP+TN)/n;  Precision = TP/(TP+FP);  Recall = TP/(TP+FN);  F1 = 2PR/(P+R) [/EQ]

[EQ] ROC-AUC = P[score⁺ > score⁻];  ρ = corr(rank(y), rank(ŷ)) [/EQ]

McNemar's exact test on two systems' discordant predictions (b = A right/B wrong, c = A wrong/B right) assumes b ~ Binomial(b+c, 0.5) under the null; the two-sided p-value is computed exactly by `scipy.stats.binomtest` (verified against a hand-computed case, b=10/c=1 → p<0.05). Bootstrap CIs are percentile CIs (2.5th/97.5th) over resampled test items. Expected Calibration Error uses 10 equal-width bins:

[EQ] ECE = Σ_b (n_b/N) · |acc(b) − conf(b)| [/EQ]

Temperature scaling (Guo et al., 2017) fits T on dev by minimising NLL = logaddexp(0, z) − y·z with T searched in log-space over [e⁻³, e³].

### 4.2 Training Dynamics and Convergence Analysis

Figure 2 plots the seed-42 AfroXLMR runs. The human-labelled Hausa run converges smoothly (train loss 0.69→0.35; dev F1 rising monotonically to 0.861 at epoch 5, which was selected as best). Both silver runs show the same characteristic shape: dev F1 climbs from a near-degenerate first epoch (0.275 Yorùbá, 0.356 Igbo — the model initially collapses toward one class) to a plateau (0.650 at epoch 4 for Yorùbá, 0.703 for Igbo) while **dev loss rises after epoch 1–2** (Yorùbá dev loss 0.67→0.77 by epoch 3) — the model fits the training distribution while its fit to the held-out fold deteriorates, a classic mild-overfit signature on small noisy sets, and the reason best-checkpoint selection (dev F1) rather than final-epoch weights is used. No run triggered early stopping; all ran the full 5 epochs.

[FIG:fig2_training_curves.png|Figure 2: Seed-42 training dynamics (top: train/dev cross-entropy; bottom: dev F1 per epoch, star = selected best). Silver runs (b, c) overfit after epoch 1–2; best-checkpoint selection guards the reported numbers.]

### 4.3 Benchmark Performance on Human-Labelled Hausa (SemRel2024)

Table V reports the SemRel-Hausa results on both split designs; Figure 4 ranks systems by Spearman ρ on the leakage-free test.

[TCAP:Table V: SemRel2024-Hausa results (3-seed mean ± std for AfroXLMR/AfriBERTa cross-encoders; single run otherwise; best per split in bold). Official = published leaky split; Disjoint = our sentence-disjoint re-split.]
| System | Split | Acc | F1 | ROC-AUC | Spearman ρ |
|---|---|---|---|---|---|
| AfroXLMR-FT (human labels) | official | 0.845 ± 0.009 | 0.864 ± 0.008 | 0.913 | 0.735 |
| AfroXLMR-FT (human labels) | disjoint | 0.783 ± 0.012 | 0.840 ± 0.008 | **0.863** | 0.697 |
| **LaBSE-FT** | official | **0.871** | **0.883** | **0.946** | **0.767** |
| LaBSE-FT | disjoint | 0.765 | 0.815 | 0.846 | 0.604 |
| AfriBERTa-FT | official | 0.733 ± 0.032 | 0.799 ± 0.014 | 0.900 | 0.671 |
| AfriBERTa-FT | disjoint | 0.668 ± 0.021 | 0.777 ± 0.004 | 0.728 | 0.426 |
| AfriE5 (zero-shot) | official | 0.556 | 0.714 | 0.788 | 0.598 |
| AfriE5 (zero-shot) | disjoint | 0.651 | 0.789 | 0.790 | 0.590 |
| LaBSE (zero-shot) | official | 0.660 | 0.704 | 0.719 | 0.477 |
| LaBSE (zero-shot) | disjoint | 0.671 | 0.742 | 0.750 | 0.476 |
| AfroXLMR-FT (silver labels) | official | 0.548 ± 0.005 | 0.701 ± 0.011 | 0.562 | 0.146 |
| AfroXLMR-FT (silver labels) | disjoint | 0.641 ± 0.010 | 0.771 ± 0.015 | 0.539 | 0.153 |
| TF-IDF + cosine | official | 0.627 | 0.735 | 0.701 | 0.454 |
| TF-IDF + cosine | disjoint | 0.651 | 0.757 | 0.717 | 0.453 |

*Per-run detail, thresholds, and CIs: `outputs/phase5/results.json` (+ `run_plan.json` for model metadata). Seed-42 headline run: disjoint acc 0.776, F1 0.836 [0.792, 0.876], Spearman 0.698. For significance under the tuned-vs-tuned protocol used throughout this paper, see Table VII.*

The fine-tuned AfroXLMR cross-encoder trained on human labels is the strongest clean-split classifier (0.783 acc / 0.863 AUC) and matches the SemRel2024 paper's fine-tuned LaBSE (ρ=0.69 published) at ρ=0.697 with a different architecture; it remains below the SemEval-2024 winning average (0.800, cross-encoder family — consistent with that system's use of larger data). Fine-tuning dominates zero-shot embedding use on ranking (AfroXLMR-FT 0.697 vs LaBSE-zs 0.476 / AfriE5-zs 0.590), and the African-specialised AfriE5 beats LaBSE zero-shot — an RQ2 data point. AfriBERTa-FT ranks poorly on the clean split (0.426 vs 0.671 on the leaky split — a preview of the leakage analysis below) and behaves degenerately on silver data (Section 4.4).

[FIG:fig4_spearman_ranking.png|Figure 4: System ranking by Spearman ρ against SemRel-Hausa gold scores on the leakage-free disjoint test (n=255). Dashed line: the SemRel2024 paper's fine-tuned LaBSE (0.69). The silver-trained model (bottom) is at chance.]

### 4.4 Constructed-Language Results (Silver Labels)

Table VI summarises the constructed test sets. In-language silver fine-tuning beats the lexical baseline by mean +4.3/+11.3/+7.9 accuracy points (Yorùbá/Igbo/Hausa); under exact McNemar tests the Igbo and Hausa gains are significant (min-p = 2.8×10⁻⁵ and 2.0×10⁻⁴) while the Yorùbá gain is not (min-p = 0.066 across seeds — seed variance on the smallest silver set, consistent with the observational analysis in Section 4.7). Every silver track is weak in absolute terms (0.54–0.68 accuracy), with a systematic false-positive skew in Yorùbá (101 FP vs 29 FN; Section 4.7). The English-PAWS transfer arm — trained on 3,000 human-labelled English pairs, no target-language data — is the strongest silver-track system on Hausa (0.811) and Yorùbá (0.599) and statistically ties in-language fine-tuning on Igbo (0.675 vs 0.676, min-p=0.215): clean-label features travel better than noisy-label fine-tuning teaches (Hausa +18.5 points, min-p=2.5×10⁻⁷, Table VIII; the Yorùbá +3.7-point mean advantage does not reach significance, min-p=0.068). AfriBERTa fails outright on silver data (predict-all behaviour: recall 1.0, accuracy ≈0.48, ROC-AUC 0.49), and LaBSE/AfriE5 zero-shot are near-chance on silver sets (AUC 0.50–0.67). On the AfriXNLI proxy harness all silver-trained systems are at or near chance (0.48–0.55 accuracy; several tuned thresholds collapse to 0.01–0.13, producing degenerate recall-1.0 behaviour) — in-language relatedness skill does not transfer to the NLI-derived proxy task. We caution that AfriXNLI rows for Hausa/Igbo/Yorùbá are 15–25% truncated at 128 tokens (Section 3.2).

[TCAP:Table VI: Constructed (silver) test sets — seed-mean ± std where 3 seeds ran; single seed otherwise. label_source = paws_translated.]
| System | Yorùbá (n=287) | Igbo (n=295) | Hausa (n=148) |
|---|---|---|---|
| AfroXLMR-FT in-language (silver) | 0.562 ± 0.016 / 0.622 / 0.639 | 0.676 ± 0.026 / 0.721 / 0.779 | 0.626 ± 0.049 / 0.696 / 0.731 |
| English-PAWS transfer (zero-shot) | **0.599** / 0.681 / 0.697 | **0.675** / **0.738** / **0.828** | **0.811** / **0.833** / **0.913** |
| LaBSE fine-tune (silver) | 0.624 / 0.671 / 0.719 | 0.654 / 0.723 / 0.829 | — |
| LaBSE (zero-shot) | 0.456 / 0.627 / 0.504 | 0.481 / 0.650 / 0.632 | 0.486 / 0.655 / 0.610 |
| AfriE5 (zero-shot) | 0.463 / 0.633 / 0.549 | 0.481 / 0.650 / 0.668 | 0.486 / 0.655 / 0.652 |
| AfriBERTa-FT (silver) | 0.463 / 0.615 / 0.547 | 0.481 / 0.650 / 0.486 | — |
| TF-IDF + cosine | 0.519 / 0.621 / 0.579 | 0.563 / 0.647 / 0.658 | 0.547 / 0.660 / 0.600 |

*Cells: accuracy / F1 / ROC-AUC on each language's constructed test fold. All conclusions on this table are conditional on MT label quality (silver labels; no native-speaker verification was performed).*

### 4.5 Statistical Significance

Table VII summarises the McNemar tests (exact, vs the saved TF-IDF baseline predictions on matched items; p-values for multi-seed systems are the minimum across seeds, following the project's reporting convention). On human-labelled Hausa the cross-encoder beats the lexical baseline decisively on both splits; silver fine-tuning wins significantly on Igbo and Hausa but not on Yorùbá (seed variance); the silver-trained Hausa model is significantly *worse* than the baseline on the official split and statistically indistinguishable on the clean split — the label-provenance result in its sharpest form.

[TCAP:Table VII: Exact McNemar tests (two-sided binomial on discordant pairs). Both systems use their own dev-tuned thresholds; multi-seed rows report the mean accuracy delta and the minimum p across seeds (b/c from that seed). Recomputed for this paper from the saved prediction files; see the protocol note below.]
| Comparison | Evaluation set | Δacc (pts) | b / c | min p |
|---|---|---|---|---|
| AfroXLMR-FT (human) vs TF-IDF baseline | SemRel disjoint | +13.2 | 55 / 17 | 8.1×10⁻⁶ |
| AfroXLMR-FT (human) vs TF-IDF baseline | SemRel official | +21.8 | 166 / 32 | 5.1×10⁻²³ |
| AfriBERTa-FT (human) vs TF-IDF baseline | SemRel disjoint | +1.7 | 50 / 38 | 0.241 (n.s.) |
| AfriBERTa-FT (human) vs TF-IDF baseline | SemRel official | +10.6 | 139 / 56 | 2.5×10⁻⁹ |
| AfroXLMR-FT (silver) vs TF-IDF baseline | SemRel disjoint | −1.0 | 37 / 43 | 0.576 (n.s.) |
| AfroXLMR-FT (silver) vs TF-IDF baseline | SemRel official | −7.9 | 24 / 69 | 3.3×10⁻⁶ (silver worse) |
| AfroXLMR-FT (silver) vs TF-IDF baseline | constructed Yorùbá | +4.3 | 52 / 34 | 0.066 (n.s.) |
| AfroXLMR-FT (silver) vs TF-IDF baseline | constructed Igbo | +11.3 | 76 / 32 | 2.8×10⁻⁵ |
| AfroXLMR-FT (silver) vs TF-IDF baseline | constructed Hausa | +7.9 | 28 / 6 | 2.0×10⁻⁴ |
| EN-PAWS transfer vs in-language silver | constructed Hausa | +18.5 | 38 / 5 | 2.5×10⁻⁷ |
| EN-PAWS transfer vs in-language silver | constructed Igbo | −0.1 | 27 / 38 | 0.215 (tie) |
| EN-PAWS transfer vs in-language silver | constructed Yorùbá | +3.7 | 42 / 26 | 0.068 (n.s.) |

*Protocol note: this table compares each system at its own dev-tuned threshold, so the Δacc values match the accuracies reported in Tables V/VI. The project's machine-generated `RESULTS.md` instead fixed the baseline at the untuned 0.5 cut, which yields larger discordant counts and smaller p-values (e.g. seed-42 Hausa disjoint b=114/c=35, p=5.8×10⁻¹¹); that protocol is biased against the baseline. All values here are recomputed from the raw prediction files by `paper/compute_significance.py`.*



### 4.6 Split Design: Leakage Inflation and the Model-Selection Flip

Figure 6 quantifies the leakage effect. SemRel-exposed models gain ≈+6.2/+6.5 accuracy points from the leaky official split (AfroXLMR 0.845 vs 0.783; AfriBERTa 0.733 vs 0.668, with Spearman collapsing 0.671→0.426 on the clean split); the silver-trained model (no SemRel exposure) shows the reverse (0.548 official vs 0.641 disjoint), exactly as the memorisation account predicts; the lexical baseline is essentially split-insensitive (−0.6/+2.4). The most consequential result is the **model-selection flip**: on the official split, LaBSE-FT appears stronger than the AfroXLMR cross-encoder (ρ 0.767 vs 0.735); on the clean split the ordering inverts (0.604 vs 0.697). A team selecting its headline model on the published split would have deployed the wrong system — the empirical case for reporting leakage-controlled splits alongside official ones.

[FIG:fig6_leakage_flip.png|Figure 6: Official (leaky) vs sentence-disjoint (clean) split. (a) accuracy, 3-seed means where available; (b) Spearman ρ on both splits for the SemRel-evaluable systems. SemRel-exposed models gain ≈+6 points from leakage; the silver model and the lexical baseline do not.]

### 4.7 Ablation, Perturbation, and Error Analysis

**Diacritics (controlled, paired).** One factor — Yorùbá tone/subdot marks — is stripped from one or both sides of each of the 287 constructed test pairs, with the threshold tuned once on unperturbed dev and held fixed (deployment scenario). Table VIII: accuracy is insensitive to stripping (all Δacc CIs include zero) but F1 reveals an asymmetry: mixed-marking pairs (one side stripped) drop the neural F1 0.616→0.442 (−28.2% relative) while accuracy *rises* — a calibration shift, not an accuracy loss — and uniformly-stripped pairs recover (0.580). The lexical baseline collapses under mixed marking (F1 0.621→0.186, −70%), because TF-IDF treats stripped and marked word forms as unrelated types. This is the empirical justification for the project's diacritic-preserving preprocessing (RQ4/R8). The AfriXNLI arm of this experiment was uninformative (the silver model predicts one class there regardless of perturbation).

[TCAP:Table VIII: Diacritics ablation on constructed_test_yor (n=287; AfroXLMR silver seed-42 checkpoint; threshold fixed from unperturbed dev).]
| Condition | Model acc | Model F1 | TF-IDF acc | TF-IDF F1 |
|---|---|---|---|---|
| C0 original | 0.544 | 0.616 | 0.519 | 0.621 |
| C1 s1 stripped | 0.578 | 0.442 | 0.544 | 0.186 |
| C2 s2 stripped | 0.585 | 0.457 | 0.551 | 0.189 |
| C3 both stripped | 0.540 | 0.580 | 0.474 | 0.618 |

**Lexical-overlap buckets (observational).** In Igbo and Hausa the transformer beats the baseline in *every* Jaccard-overlap quartile including the highest (Δacc +0.164 [0.098, 0.227] Igbo; +0.128 [0.034, 0.223] Hausa) — evidence of processing beyond word matching. In Yorùbá the seed-42 advantage is −0.001 [−0.066, +0.059], i.e., not reliable at seed 42; the +4.3-point significance in Table VII comes from the other seeds — seed variance on the smallest silver set is reported, not hidden.

**Negation.** On human-labelled Hausa (recomputed from saved predictions: n=37 one-side-negated pairs), the strong model classifies one-side-negated pairs at 0.838 vs 0.788 overall, and those pairs carry lower mean gold relatedness (0.496 vs 0.570) which the model reflects rather than ignores — no systematic negation weakness (RQ4/R9). Marker lists are conservative heuristics; silver/proxy bucket analyses were uninformative (small n, weak models).

**Word-order perturbation was not attempted**: no safe automatic reorderer exists for these languages, and fabricating ungrammatical manipulations would produce unsupported conclusions (documented scope decision).

**Errors.** Table IX: errors split into two classes. (i) *Silver-label artifacts*: 11 identical-text Yorùbá pairs labelled non-paraphrase because the inherited English label refers to a distinction the translation erased — the model is semantically right and the label wrong; the broader mechanism is the Yorùbá false-positive skew (101 FP vs 29 FN). (ii) *Adversarial swaps preserved by translation* (entity/role swaps à la PAWS): in Hausa these dominate — 44.8% of false positives are ≥0.9-overlap pairs, missed at up to 0.98 confidence. Figure 3 shows the corresponding confusion structure at each model's tuned threshold: the Hausa human-labelled model is balanced (FP 36 / FN 21 at τ=0.37), while the silver models skew toward over-prediction of the positive class.

[TCAP:Table IX: Error analysis on each language's constructed test set (tuned thresholds).]
| Language | Errors | FP | FN | FP with overlap ≥ 0.9 | Identical-text pairs (mislabeled) |
|---|---|---|---|---|---|
| Yorùbá (n=287) | 130 | 101 | 29 | 17.8% | 11 (2) |
| Igbo (n=295) | 89 | 58 | 31 | 15.5% | 15 (0) |
| Hausa (n=148) | 50 | 29 | 21 | 44.8% | 7 (0) |

[FIG:fig3_confusion.png|Figure 3: Confusion matrices at each seed-42 model's dev-tuned threshold (counts and row rates). Hausa (human labels) is balanced; silver models over-predict the positive class.]

### 4.8 Calibration

Temperature scaling is fit on dev per run (Table X; Figure 5). The human-labelled Hausa model calibrates well (T=1.66, ECE 0.110→0.049, NLL 0.501→0.450). The silver models need much stronger cooling — Yorùbá T=3.46 (ECE 0.210→0.130, still the worst after calibration), Igbo T=1.30, and the Yorùbá AfriBERTa run T=6.54 — a quantified fingerprint of silver-label noise: the models are overconfident because their training labels conflict with the dev distribution. Ranking metrics are invariant to the monotone temperature transform; calibration matters for the serving path (Section 5), where scores are displayed.

[TCAP:Table X: Temperature scaling on dev predictions (selected runs; full set in outputs/calibration/).]
| Run | T | NLL before→after | ECE before→after |
|---|---|---|---|
| afroxlmr_hau_semrel_seed42 | 1.66 | 0.501 → 0.450 | 0.110 → 0.049 |
| afroxlmr_ibo_constructed_seed42 | 1.30 | 0.599 → 0.592 | 0.085 → 0.061 |
| afroxlmr_yor_constructed_seed42 | 3.46 | 0.768 → 0.673 | 0.210 → 0.130 |
| afroxlmr_hau_constructed_seed42 | 1.79 | 0.675 → 0.655 | 0.096 → 0.047 |
| afroxlmr_en_paws_seed42 | 1.52 | 0.493 → 0.459 | 0.096 → 0.050 |
| afriberta_yor_constructed_seed42 | 6.54 | 0.747 → 0.691 | 0.179 → 0.098 |

[FIG:fig5_reliability.png|Figure 5: Reliability diagrams (10 equal-width bins, dev). Blue = after temperature scaling. Human labels (a) calibrate near the diagonal; silver labels (b) remain overconfident after scaling.]

## 5. System Interface, Deployment, and Practical Demonstration

The serving stack is deliberately the same mathematics as the evaluation: `ParaphrasePredictor` loads the seed-42 checkpoint, applies the identical `clean_text`, computes z = log(p₁/p₀), rescales by the run's fitted temperature, and thresholds at the run's dev-tuned τ (Equation in Section 3.1). The Streamlit demo exposes a language selector (each model's checkpoint, provenance, label quality, and held-out metrics are displayed in the sidebar; Yorùbá/Igbo carry silver-label warnings, and the Yorùbá overconfidence is stated explicitly), an example picker drawn from real held-out pairs, and a verdict panel with the calibrated probability, threshold, and temperature. The demo runs locally on CPU (checkpoint load ≈10 s; inference <1 s per pair); predictions are never hardcoded.

Table XI walks twelve real held-out pairs — the strongest true positive, strongest false positive, strongest false negative, and weakest true negative per language — through the deployed path, recording exactly what the system outputs. The examples show the calibration in action (identical raw scores map to different calibrated probabilities under different temperatures) and illustrate the two error classes of Section 4.7: the Hausa false positive is a ≥0.9-overlap adversarial swap; the Yorùbá false positive reflects the silver skew. (The Igbo true-negative row is the documented Latin-1 mojibake sentence, shown verbatim.)

[TCAP:Table XI: Input→output case study — twelve real held-out pairs through the deployed predictor (raw = pre-calibration sigmoid score; cal = displayed calibrated score; verdict from the dev-tuned threshold τ). Sentences abbreviated for width; full text in paper/case_study.json.]
[CASESTUDY_TABLE]

*Hausa gold values in parentheses are the continuous BWS scores (binarised at 0.5); note the FN row sits exactly at the 0.5 boundary — the binarisation, not the model, flips it. Yorùbá/Igbo silver sets carry binary labels only.*

## 6. Discussion, Limitations, and Future Work

**Answering the research questions.** RQ1 (effectiveness): yes with one caveat — the human-labelled Hausa cross-encoder beats the lexical baseline decisively (+13.2 clean-split accuracy, min-p=8.1×10⁻⁶), silver fine-tuning wins significantly on Igbo and Hausa (p ≤ 2.8×10⁻⁵), and the Yorùbá silver gain (+4.3 mean) does not reach significance (min-p=0.066; Table VII). RQ2 (African-specialised vs general): AfroXLMR ≫ AfriBERTa for pair tasks (degenerate on silver data; 0.426 vs 0.697 clean-split Spearman), and African-specialised AfriE5 > LaBSE zero-shot — adaptive fine-tuning is the decisive factor, not African corpus exposure alone. RQ3 (transfer): fine-tuning ≫ zero-shot embedding use on ranking (0.697 vs 0.590/0.476); English-PAWS zero-shot beats in-language silver fine-tuning on Hausa (0.811 vs 0.626 accuracy, min-p=2.5×10⁻⁷) and ties it on Igbo. RQ4 (linguistic factors): no catastrophic diacritic dependence; mixed marking shifts calibration; the lexical system collapses on mixed marking (−70% F1); no negation weakness detected; word-order manipulation skipped as unsound. RQ5 (application): the calibrated demo serves all three checkpoints with provenance display (Section 5).

**What the title's two claims mean.** *Label provenance* dominates: the same architecture is 0.783-accurate on human labels and near-chance on real relatedness when trained on silver labels (ρ 0.153 vs 0.697) — a 0.54-ρ gap attributable to training-label quality, plus a measurable overconfidence fingerprint (T up to 3.46 vs 1.66). *Split design* changes conclusions: leakage inflates SemRel-exposed models by ≈+6 points and inverts the LaBSE-FT vs cross-encoder ranking. Both effects are larger than any architecture choice in our grid — the data-centric lesson of the study.

**Limitations (stated plainly).**
1. **Silver labels are unverified.** Gold validation (390-pair candidate set, annotation app, guidelines) was built but skipped by researcher decision; all constructed-set conclusions are conditional on MT label quality. The offered-but-untaken 50–100-pair spot-check remains the cheapest path to strengthening them.
2. **Truncation is material for two languages.** At max_seq_length=128, 44.8% of Yorùbá and 23.6% of Igbo constructed pairs are truncated (Section 3.2); Yorùbá/Igbo results are partly truncation-limited, and the earlier "128 is adequate" estimate was based on a different tokenizer and a small sample.
3. **MT artifacts exist.** 2.3% of Yorùbá constructed pairs contain NLLB degeneration loops; 11 identical-text Yorùbá pairs carry contradictory labels. MT-vs-human positives were never built (inputs generated, translation run dropped).
4. **Small silver test sets** (148–295 pairs) widen CIs; seed variance is visible (Hausa silver ±0.049 accuracy).
5. **Register mismatch** between news/formal silver data and the proxy eval sets; AfriXNLI-derived rows are additionally 6–25% truncated.
6. **Bi-encoder comparability.** Seven embedding runs initially hit T4 OOM and were re-run with reduced batch sizes; their fine-tune arm has no dev-based model selection (fixed 3 epochs).
7. **SemRel2024 licence is unstated** (research use only, no redistribution); Hausa is the only language with external human labels, so cross-language comparisons of *absolute* quality partly reflect label quality, not language difficulty.
8. **Single annotation path** — no second annotator, so no inter-annotator agreement exists by design (the planned self-agreement κ was part of the skipped validation).
9. **Evaluation-vs-training tokenizer discrepancy** discovered post hoc (Section 3.2) — an honest-accounting cost of an early engineering estimate; corrected everywhere in this paper.

**Future work.** (i) Run the built gold-validation tooling (390 candidates, ≈5–7 h) to convert the strongest silver conclusions into verified ones. (ii) Re-train Yorùbá/Igbo at max_seq_length 256–320 to remove the truncation confound and quantify its cost. (iii) Filter degeneration loops by repetition screening before training. (iv) Add a native data-acquisition route (translation of PAWS by professional/native speakers, as PAWS-X did) to bypass MT labels entirely. (v) Evaluate distilled/student variants for on-device serving, and extend the demo to batch mode with per-pair provenance in the output.

## 7. Conclusion

We built and studied the first paraphrase-detection evaluation stack for Yorùbá, Igbo, and Hausa: audited data acquisition, a leakage-free re-split of the only human-labelled resource, MT-based silver construction with per-pair provenance, a 29-run transformer benchmark, exact significance testing, calibration, and a deployed demo. The strongest human-labelled system (AfroXLMR cross-encoder, Hausa) reaches 0.783 accuracy / 0.863 ROC-AUC / ρ=0.697 on a leakage-free split — matching published fine-tuned LaBSE with a different architecture — and its scores calibrate well after temperature scaling (ECE 0.049). The study's central empirical results are methodological: sentence leakage in the official split inflates neural scores by ≈+6 points and would have caused the wrong model to be selected; and label provenance outweighs architecture — silver-label training collapses real-relatedness ranking (ρ 0.153) and produces measurably overconfident models, while zero-shot transfer from clean English labels outperforms in-language silver fine-tuning. For low-resource languages, where every label is expensive, these results argue that split integrity and label provenance belong in the headline claims of any evaluation — as they do in this paper's title.

## References

Adelani, D. I., Alpha, A., Egwuchukwu, P., et al. (2021). MasakhaNER: Named entity recognition for African languages. *Transactions of the Association for Computational Linguistics, 9*, 1116–1131.

Agirre, E., Cer, D., Diab, M., Gonzalez-Agirre, A., & Guo, W. (2013). *SEM 2013 shared task: Semantic textual similarity.* Second Joint Conference on Lexical and Computational Semantics.

Agirre, E., Cer, D., Diab, M., & Gonzalez-Agirre, A. (2012). SemEval-2012 Task 6: A pilot on semantic textual similarity. *Proceedings of SemEval 2012.*

Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. *Proceedings of COLING 2022*, 2570–2584.

Adebara, A., et al. (2024). Cheetah: Efficient and flexible large language models for 517 African languages. *arXiv preprint* arXiv:2401.01053.

Berg-Kirkpatrick, T., Burkett, D., & Klein, D. (2012). An empirical investigation of statistical significance in NLP. *Proceedings of EMNLP-CoNLL 2012*, 995–1005.

Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised cross-lingual representation learning at scale. *Proceedings of ACL 2020*, 8440–8451.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923.

Dolan, W. B., & Brockett, C. (2005). Automatically constructing a corpus of sentential paraphrases. *Proceedings of the Third International Workshop on Paraphrasing (IWP 2005).*

Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). The hitchhiker's guide to testing statistical significance in natural language processing. *Proceedings of ACL 2018*, 1383–1392.

Ezeani, I., Dionesia, E., Nweke, G., et al. (2020). Igbo-English machine translation: An evaluation benchmark. *Proceedings of LREC 2020.* (arXiv:2004.00648)

Federmann, C., Kuzman, T., et al. (2022). NTREX-128: News test reference corpus for MT evaluation. *arXiv preprint.*

Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). Language-agnostic BERT sentence embedding (LaBSE). *Proceedings of ACL 2022*, 878–891.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of ICML 2017*, 1321–1330.

Koehn, P. (2004). Statistical significance tests for machine translation evaluation. *Proceedings of EMNLP 2004*, 388–395.

Nwafor, E., et al. (2025). NaijaNLP: A survey of Nigerian low-resource languages and technologies. *arXiv preprint* arXiv:2502.19784.

Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. *Proceedings of the 1st Workshop on Multilingual Representation Learning (EMNLP 2021).*

Ousidhoum, N., Muhammad, S. H., et al. (2024). SemRel2024: A collection of semantic textual relatedness datasets for 13 languages. *Findings of ACL 2024.* (arXiv:2402.08638)

Ousidhoum, N., et al. (2024). SemEval-2024 Task 1: Semantic textual relatedness for African and Asian languages. *Proceedings of SemEval-2024.* (arXiv:2403.18933)

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP 2019*, 3982–3992.

Uemura, K., Zhang, M., & Adelani, D. I. (2025). AfriMTEB and AfriE5: Benchmarking and adapting text embedding models for African languages. *arXiv preprint* arXiv:2510.23896.

Yang, Y., Zhang, Y., Tar, C., & Baldridge, D. (2019). PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. *Proceedings of EMNLP-IJCNLP 2019*, 3685–3690.

Zhang, Y., Baldridge, J., & He, L. (2019). PAWS: Paraphrase adversaries from word scrambling. *Proceedings of NAACL 2019*, 1298–1308.

Zhang, J., et al. (2021). *Menyo-20k: A multilingual Yorùbá-English corpus for machine translation.* Zenodo. DOI 10.5281/zenodo.4297448.
