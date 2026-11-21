# Transformer-Based Semantic Similarity and Paraphrase Detection in Yorùbá, Igbo and Hausa

MSc Computer Science & AI research project — binary paraphrase classification
(primary) and semantic similarity scoring (secondary) for three Nigerian
languages, with a traditional TF-IDF + cosine baseline.

**Status: COMPLETE (all 7 phases, 2026-11-21).** Key results:
[`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md). Design and evidence:
[`docs/feasibility_report.md`](docs/feasibility_report.md). Decisions and
provenance: [`docs/research_log.md`](docs/research_log.md).

| Phase | Outcome |
|---|---|
| P0 Feasibility | Verified model/dataset landscape; no paraphrase data exists for these languages → construction became a contribution |
| P1–P2 Environment, data acquisition & audit | 5 sources with SHA-256 provenance; SemRel leakage + Menyo/IGBONLP quality findings |
| P3 Dataset construction | Silver translate-train sets (yor 2,873 / ibo 2,948 / hau 1,477) + SemRel leakage-free re-split (2,041/255/255) + AfriXNLI eval harness |
| P4 Baseline & evaluation harness | Metrics library with McNemar/bootstrap; TF-IDF baseline |
| P5 Transformer training | 29/29 runs on Colab T4; cross-encoders + embeddings; all findings significance-tested |
| P6 Analysis | Diacritics ablation, negation/overlap/length buckets, error analysis, figures |
| P7 Demo & documentation | Streamlit app on real checkpoints, results summary |

## Run the demonstration app

```bash
.venv/bin/streamlit run src/afripara/app/demo.py
```

Select a language (Hausa = strongest model, human-annotated training data;
Yorùbá/Igbo = silver-trained models with displayed provenance warnings),
enter two sentences, and get a paraphrase verdict with a temperature-
calibrated similarity score from the real trained checkpoint — same
thresholds and calibration as the reported experiments. Example pairs are
real held-out items with verified verdicts.

## Project structure

```
.
├── configs/
│   ├── default.yaml          # global settings: seeds, paths, training defaults
│   └── models/               # per-model configs (HF repo, licence, citation, role)
├── data/
│   ├── raw/                  # immutable originals as downloaded (P2)
│   ├── external/             # third-party resources not downloaded by us
│   ├── interim/              # audit reports, caches
│   └── processed/            # final modelling-ready datasets + splits (P3)
├── docs/                     # feasibility report, research log
├── scripts/                  # CLI entry points (smoke test, ...)
├── src/afripara/
│   ├── config.py             # YAML config loading, seeding, device
│   ├── features/             # normalisation (diacritics-preserving)
│   ├── data/                 # dataset download/audit/construction/splits (P2-P3)
│   ├── models/               # TF-IDF baseline, cross/bi-encoders (P4-P5)
│   ├── training/             # training loops, early stopping (P5)
│   ├── evaluation/           # metrics, significance tests, plots (P4, P6)
│   ├── inference/            # calibrated prediction for the demo (P7)
│   └── app/                  # Streamlit demonstration (P7)
└── tests/                    # pytest suite
```

## Setup (macOS / Linux)

```bash
cd "<project folder>"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

## Verification

```bash
# Unit tests (fast, CPU)
.venv/bin/python -m pytest

# Engineering smoke test: downloads castorini/afriberta_small (~400 MB) and
# runs tokenizer + model forward passes on fixture sentences (CPU).
.venv/bin/python scripts/smoke_test.py           # full
.venv/bin/python scripts/smoke_test.py --no-model  # tokenizer only
```

## Research principles

- **No fabricated data, labels, citations or results.** Every dataset and
  model is verified and recorded with provenance and licence.
- **Diacritics are phonemic** in these languages: the default pipeline
  preserves them (empirically validated in Phase 6); stripping exists only
  as the E6 linguistic ablation.
- **Label provenance is always reported**: human-annotated (SemRel) >
  proxy (AfriXNLI) > MT-inherited silver (constructed). Conclusions on
  silver sets are stated as conditional on MT label quality.
- **Monolingual vs cross-lingual results are never mixed**; per-language
  results are always reported alongside aggregates.
- **Hyperparameters live in `configs/`**, never hardcoded in source.
- Training ran on Google Colab T4 with the identical, locally-validated
  pipeline; all scientific scoring happened locally with the tested
  metrics library.
