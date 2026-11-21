# Data Audit Report (Phase 2)

Datasets audited: **20**. Raw originals in `data/raw/` were not modified. Full detail: `report.json`.

| Dataset | Records | Dup. pairs | Empty | Anomalies | Flags |
|---|---|---|---|---|---|
| afrixnli/hau/dev | 450 | 0 | 0 | 0 | - |
| afrixnli/hau/test | 600 | 0 | 0 | 0 | - |
| afrixnli/ibo/dev | 445 | 0 | 0 | 10 | - |
| afrixnli/ibo/test | 600 | 0 | 0 | 0 | - |
| afrixnli/yor/dev | 450 | 0 | 0 | 0 | - |
| afrixnli/yor/test | 600 | 0 | 0 | 0 | - |
| afrixnli/eng/dev | 450 | 0 | 0 | 0 | - |
| afrixnli/eng/test | 600 | 0 | 0 | 0 | - |
| semrel2024_hau/train | 1736 | 0 | 0 | 0 | - |
| semrel2024_hau/dev | 212 | 0 | 0 | 0 | - |
| semrel2024_hau/test | 603 | 0 | 0 | 0 | - |
| menyo20k_mt/train | 10070 | 77 | 0 | 0 | 77 duplicate pairs |
| menyo20k_mt/dev | 3396 | 45 | 0 | 1 | 45 duplicate pairs |
| menyo20k_mt/test | 6633 | 45 | 0 | 0 | 45 duplicate pairs |
| igbonlp_benchmark/train | 10000 | 249 | 1 | 0 | 1 empty sentences; 249 duplicate pairs |
| igbonlp_benchmark/val | 193 | 4 | 0 | 0 | 4 duplicate pairs |
| igbonlp_benchmark/test | 597 | 200 | 10 | 0 | 10 empty sentences; 200 duplicate pairs |
| ntrex_african/hau | 1997 | 0 | 0 | 0 | - |
| ntrex_african/ibo | 1997 | 0 | 0 | 0 | - |
| ntrex_african/yor | 1997 | 0 | 0 | 0 | - |

## Label and score distributions

- **afrixnli/hau/dev**: labels {'1': 150, '2': 150, '0': 150}
- **afrixnli/hau/test**: labels {'2': 200, '0': 200, '1': 200}
- **afrixnli/ibo/dev**: labels {'1': 148, '2': 149, '0': 148}
- **afrixnli/ibo/test**: labels {'2': 200, '0': 200, '1': 200}
- **afrixnli/yor/dev**: labels {'1': 150, '2': 150, '0': 150}
- **afrixnli/yor/test**: labels {'2': 200, '0': 200, '1': 200}
- **afrixnli/eng/dev**: labels {'1': 150, '2': 150, '0': 150}
- **afrixnli/eng/test**: labels {'2': 200, '0': 200, '1': 200}
- **semrel2024_hau/train**: score mean=0.495, p50=0.50, min=0.00, max=1.00
- **semrel2024_hau/dev**: score mean=0.494, p50=0.50, min=0.00, max=1.00
- **semrel2024_hau/test**: score mean=0.496, p50=0.50, min=0.00, max=1.00

## Orthography signals (heuristic language-metadata validation)

- **afrixnli/hau/dev** [hau]: diacritic density 0.011, marker coverage 48.4%
- **afrixnli/hau/test** [hau]: diacritic density 0.012, marker coverage 49.2%
- **afrixnli/ibo/dev** [ibo]: diacritic density 0.148, marker coverage 99.1%
- **afrixnli/ibo/test** [ibo]: diacritic density 0.147, marker coverage 98.7%
- **afrixnli/yor/dev** [yor]: diacritic density 0.399, marker coverage 99.8%
- **afrixnli/yor/test** [yor]: diacritic density 0.402, marker coverage 99.9%
- **semrel2024_hau/train** [hau]: diacritic density 0.012, marker coverage 47.9%
- **semrel2024_hau/dev** [hau]: diacritic density 0.013, marker coverage 51.4%
- **semrel2024_hau/test** [hau]: diacritic density 0.012, marker coverage 46.9%
- **menyo20k_mt/train** [yor]: diacritic density 0.204, marker coverage 50.6%
- **menyo20k_mt/dev** [yor]: diacritic density 0.209, marker coverage 49.9%
- **menyo20k_mt/test** [yor]: diacritic density 0.206, marker coverage 50.4%
- **igbonlp_benchmark/train** [ibo]: diacritic density 0.061, marker coverage 47.2%
- **igbonlp_benchmark/val** [ibo]: diacritic density 0.062, marker coverage 48.4%
- **igbonlp_benchmark/test** [ibo]: diacritic density 0.064, marker coverage 48.6%
- **ntrex_african/hau** [hau]: diacritic density 0.004, marker coverage 41.4%
- **ntrex_african/ibo** [ibo]: diacritic density 0.058, marker coverage 49.4%
- **ntrex_african/yor** [yor]: diacritic density 0.164, marker coverage 49.6%

## Known limits of this audit

- Orthography markers are a *heuristic* metadata validation, not proof of language.
- Model-based language identification (AfroLID) is deferred to Phase 3, where it matters most: constructed pairs from multi-source pools.
- Semantic annotation quality (are labels correct?) cannot be audited automatically; it is addressed by construction design + human gold validation (Phase 3).
