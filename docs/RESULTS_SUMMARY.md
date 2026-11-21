# Results Summary — for thesis writing

One-page map of every finding in the project, with its evidence location.
Provenance of labels is shown as 🥇 human-annotated · 🥈 MT-inherited
(silver) · 🔁 proxy (NLI-derived). Full detail: `outputs/phase5/RESULTS.md`,
`outputs/phase6/ANALYSIS_REPORT.md`, `docs/research_log.md`.

---

## Headline results

| # | Finding | Evidence |
|---|---|---|
| **R1** | **Fine-tuned AfroXLMR cross-encoders significantly beat the TF-IDF lexical baseline on all three languages** — on human-labelled Hausa data by +13.2 accuracy points (p < 0.0001, leakage-free split), and on silver-constructed sets by +4.3 / +11.3 / +7.9 points (yor/ibo/hau, all p ≤ 0.005). | `outputs/phase5/RESULTS.md` (McNemar) |
| **R2** | **The strongest Hausa system reaches accuracy 0.783, ROC-AUC 0.863, Spearman 0.697** (3 seeds, F1 std 0.008) — matching or exceeding the SemRel paper's fine-tuned LaBSE (0.69 Spearman) with a different architecture. | Phase 5 families |
| **R3** | **Sentence-level leakage in the official SemRel split inflates neural scores by ~6 points and can invert model rankings**: LaBSE-FT beats the cross-encoder on the leaky split (0.767 vs 0.735 Spearman) but loses on the clean split (0.604 vs 0.697). Model selection without a leakage-controlled split would have been wrong. | Phase 5 addendum |
| **R4** | **Label provenance dominates outcomes**: the silver-trained Hausa model is at chance on real relatedness (Spearman 0.153 vs 0.697) and significantly *worse* than the lexical baseline; Yoruba silver fine-tuning shows a systematic false-positive skew (101 FP vs 29 FN). All silver conclusions are conditional on MT label quality (2026-11-20 decision: no native-speaker verification). | Phase 5 + Phase 6 error analysis |
| **R5** | **English→target zero-shot transfer beats in-language silver fine-tuning** (Hausa 0.811 vs 0.626 accuracy, p < 0.0001) — clean-label features travel better than noisy-label fine-tuning teaches. | Phase 5, E5 arm |
| **R6** | **AfroXLMR ≫ AfriBERTa for pair tasks** (AfriBERTa degenerate on silver data; 0.426 vs 0.697 Spearman on SemRel) — African adaptive fine-tuning confirmed as the decisive factor. | Phase 5 |
| **R7** | **Zero-shot embeddings rank fine-tuning ≫ zero-shot**: LaBSE 0.476 / AfriE5 0.590 vs fine-tuned 0.697 Spearman; the African-specialised AfriE5 beats LaBSE zero-shot. | Phase 5 embedding runs |
| **R8** | **Diacritics**: no catastrophic tone-mark dependence (accuracy unchanged under stripping), but mixed-marking pairs shift model calibration (F1 0.616 → ~0.45, recovering when both sides match). A lexical system collapses (−70% F1); the neural model degrades −26% — empirical justification for diacritic-preserving preprocessing. | `outputs/phase6/e6_diacritics.json` |
| **R9** | **Negation**: no systematic weakness detected (strong model classifies one-side-negated pairs at 0.838 vs 0.788 overall, n = 37, SemRel). Exploratory bucket analyses on weaker models were uninformative. | `outputs/phase6/analysis.json` |

## Experimental scope (all runs, 29/29 completed)

- **Cross-encoders**: AfroXLMR-base (yor, ibo, hau-constructed, hau-semrel × 3 seeds), AfriBERTa-large (yor, ibo × 1; hau-semrel × 3), English-PAWS transfer × 1.
- **Embeddings**: LaBSE zero-shot ×4 + fine-tuned ×4; AfriE5 zero-shot ×4 (with mandatory `query:` prefixes).
- **Baselines**: TF-IDF + cosine (train-fold fit, dev-tuned threshold).
- Training: LR 2e-5, effective batch 32, warmup 10%, maxlen 128, ≤5 epochs, early stopping on dev F1, ≥3 seeds on primary arms; Colab T4; locally verified pipeline (smoke run).
- Statistics: McNemar exact tests (Dietterich 1998; Dror et al. 2018), paired bootstrap (Berg-Kirkpatrick 2012; Koehn 2004), bootstrap CIs, ≥3-seed mean ± std.

## Data assets created by this project

- Constructed silver sentence-pair sets (yor 2,873 / ibo 2,948 / hau 1,477) with full provenance columns and PAWS-style overlap control.
- SemRel-Hausa **sentence-disjoint re-split** (2,041 / 255 / 255; zero cross-split sentence overlap; zero pairs dropped) alongside the official split.
- AfriXNLI binary evaluation harness (yor/ibo/hau/eng, ~700 pairs each).
- Annotation tooling (Streamlit app + guidelines) retained for future gold validation.

## Honest limitations (state these in the thesis)

1. Constructed-pair labels are MT-inherited (silver): results on silver sets are conditional on MT quality. Measured consequences: chance-level real-relatedness transfer for silver-trained Hausa models (R4), false-positive skew in Yoruba.
2. Gold validation was skipped (researcher decision, 2026-11-20); a 390-pair candidate set and the annotation tooling remain ready if revisited.
3. SemRel2024 licence is unstated — research use only, no redistribution; Hausa-only human-labelled coverage.
4. Igbo has no native-speaker validation and its largest corpus has an unstated licence.
5. Register effects exist (news/formal silver vs Twitter/proxy sets); word-order perturbation was not attempted (no safe reorderer).
6. One architecture family for the headline (XLM-R lineage) plus AfriBERTa; no LLM-scale models were trained (out of MSc compute scope, documented).
7. The evaluation test sets for silver tracks are small (148–295 pairs per language); bootstrap CIs are reported throughout.

## Reproduction map

| Artefact | Where |
|---|---|
| Run plan + data bundle | `outputs/colab_bundle/` |
| All 29 run outputs | `outputs/colab/` (fetched from Drive) |
| Definitive tables | `outputs/phase5/RESULTS.md` + `results.json` |
| Baseline | `outputs/baseline/BASELINE_RESULTS.md` |
| Analysis + figures | `outputs/phase6/` |
| Provenance/licence register | `data/raw/provenance.json` |
| Decision log | `docs/research_log.md` |
| Feasibility evidence | `docs/feasibility_report.md` |
| Demo | `src/afripara/app/demo.py` |
