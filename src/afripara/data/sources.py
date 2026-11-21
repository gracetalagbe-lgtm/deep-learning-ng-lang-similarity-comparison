"""Registry of data sources with exact download URLs and provenance metadata.

Every file downloaded for this project must be registered here: URL, licence,
citation and notes are recorded BEFORE downloading (see research log).
`data/raw/` holds immutable originals; anything derived goes to
`data/interim/` or `data/processed/`.

Licence red flag honoured here: JW300 is legally withdrawn (copyright
breach). The IGBONLP source below uses ONLY the human-translated
`benchmark_dataset/` files and deliberately excludes the JW300-derived
`masakhane_JW300_enig/` folder of the same repository.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# (url, relative path under data/raw/)
FileSpec = Tuple[str, str]

HF = "https://huggingface.co/datasets"
GH_RAW = "https://raw.githubusercontent.com"

SOURCES: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------ #
    "afrixnli": {
        "title": "AfriXNLI (Masakhane mirror)",
        "description": (
            "XNLI dev/test subset translated into 16 African languages; "
            "1,050 premise-hypothesis pairs per language (450 dev + 600 test), "
            "3-way NLI labels. Role in project: cross-language evaluation "
            "harness (entailment->1, contradiction->0, neutral excluded - "
            "proxy labels, bias documented in feasibility report section 5.3)."
        ),
        "languages": ["hau", "ibo", "yor", "eng"],
        "license": "Apache-2.0",
        "license_note": "Permits derived datasets with attribution.",
        "citation": (
            "Adelani, D. I. et al. AfriXNLI (Masakhane), IrokoBench family. "
            "https://huggingface.co/datasets/masakhane/afrixnli (Apache-2.0); "
            "CC-BY-4.0 mirror: https://huggingface.co/datasets/mteb/AfriXNLI"
        ),
        "files": [
            (f"{HF}/masakhane/afrixnli/resolve/main/README.md", "afrixnli/README.md"),
            (f"{HF}/masakhane/afrixnli/resolve/main/data/hau/dev.tsv", "afrixnli/hau_dev.tsv"),
            (f"{HF}/masakhane/afrixnli/resolve/main/data/hau/test.tsv", "afrixnli/hau_test.tsv"),
            (f"{HF}/masakhane/afrixnli/resolve/main/data/ibo/dev.tsv", "afrixnli/ibo_dev.tsv"),
            (f"{HF}/masakhane/afrixnli/resolve/main/data/ibo/test.tsv", "afrixnli/ibo_test.tsv"),
            (f"{HF}/masakhane/afrixnli/resolve/main/data/yor/dev.tsv", "afrixnli/yor_dev.tsv"),
            (f"{HF}/masakhane/afrixnli/resolve/main/data/yor/test.tsv", "afrixnli/yor_test.tsv"),
            (f"{HF}/masakhane/afrixnli/resolve/main/data/eng/dev.tsv", "afrixnli/eng_dev.tsv"),
            (f"{HF}/masakhane/afrixnli/resolve/main/data/eng/test.tsv", "afrixnli/eng_test.tsv"),
        ],
    },
    # ------------------------------------------------------------------ #
    "semrel2024_hau": {
        "title": "SemRel2024 Hausa (SemEval-2024 Task 1, Track A)",
        "description": (
            "Human-annotated semantic textual relatedness pairs for Hausa: "
            "3,473 train / 425 dev / 1,207 test with continuous 0-1 scores "
            "(Best-Worst Scaling). Role: Hausa supervised training + "
            "similarity evaluation; binary label via validation-tuned "
            "threshold."
        ),
        "languages": ["hau"],
        "license": "UNSTATED",
        "license_note": (
            "No LICENSE file in the source repository. Per researcher decision "
            "(2026-11-19): use as published research data WITHOUT contacting "
            "authors; no redistribution of this data or derivatives in any "
            "release. Cite the SemEval-2024 Task 1 paper."
        ),
        "citation": (
            "Ousidhoum, N., Muhammad, S. H., et al. (2024). SemRel2024: A "
            "Collection of Semantic Textual Relatedness Datasets for 13 "
            "Languages (Findings of ACL 2024, arXiv:2402.08638); SemEval-2024 "
            "Task 1 (arXiv:2403.18933). Data: github.com/"
            "semantic-textual-relatedness/Semantic_Relatedness_SemEval2024"
        ),
        "files": [
            (
                f"{GH_RAW}/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/main/Track%20A/hau/hau_train.csv",
                "semrel2024_hau/hau_train.csv",
            ),
            (
                f"{GH_RAW}/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/main/Track%20A/hau/hau_dev_with_labels.csv",
                "semrel2024_hau/hau_dev_with_labels.csv",
            ),
            (
                f"{GH_RAW}/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/main/Track%20A/hau/hau_test_with_labels.csv",
                "semrel2024_hau/hau_test_with_labels.csv",
            ),
        ],
    },
    # ------------------------------------------------------------------ #
    "menyo20k_mt": {
        "title": "Menyo-20k (Yoruba-English parallel corpus)",
        "description": (
            "20,100 en-y0 parallel pairs (news + broadcast + web text; human "
            "translations). Role: Yoruba raw material for constructed "
            "paraphrase pairs and register analysis."
        ),
        "languages": ["yor", "eng"],
        "license": "CC-BY-NC-4.0",
        "license_note": "Non-commercial; derivatives restricted to research use.",
        "citation": (
            "Alabi et al. Menyo-20k Multilingual Yoruba-English corpus. "
            "DOI 10.5281/zenodo.4297448; github.com/uds-lsv/menyo-20k_MT "
            "(CC-BY-NC-4.0). Author list to be confirmed against the README "
            "during the audit step."
        ),
        "files": [
            (f"{GH_RAW}/uds-lsv/menyo-20k_MT/master/data/train.tsv", "menyo20k_mt/train.tsv"),
            (f"{GH_RAW}/uds-lsv/menyo-20k_MT/master/data/dev.tsv", "menyo20k_mt/dev.tsv"),
            (f"{GH_RAW}/uds-lsv/menyo-20k_MT/master/data/test.tsv", "menyo20k_mt/test.tsv"),
            (f"{HF}/UdS-LSV/menyo20k_mt/resolve/main/README.md", "menyo20k_mt/README.md"),
        ],
    },
    # ------------------------------------------------------------------ #
    "igbonlp_benchmark": {
        "title": "IGBONLP Igbo-English MT benchmark subset",
        "description": (
            "Human-translated Igbo-English parallel benchmark (~10,752 pairs: "
            "10,000 train / 200 val / 552 test). Role: Igbo raw material for "
            "constructed pairs. JW300-derived folders of the same repository "
            "are EXCLUDED (withdrawn corpus; licence red flag)."
        ),
        "languages": ["ibo", "eng"],
        "license": "UNSTATED",
        "license_note": (
            "No LICENSE file on the GitHub repository or HF mirror. Research "
            "use only; no redistribution of data or derivatives without "
            "author permission (not requested, per researcher decision)."
        ),
        "citation": (
            "Ezeani, I., et al. (2020). Igbo-English Machine Translation: An "
            "Evaluation Benchmark. LREC 2020. arXiv:2004.00648. Data: "
            "github.com/IgnatiusEzeani/IGBONLP ig_en_mt/benchmark_dataset/"
        ),
        "files": [
            (f"{GH_RAW}/IgnatiusEzeani/IGBONLP/master/ig_en_mt/benchmark_dataset/train.en", "igbonlp_benchmark/train.en"),
            (f"{GH_RAW}/IgnatiusEzeani/IGBONLP/master/ig_en_mt/benchmark_dataset/train.ig", "igbonlp_benchmark/train.ig"),
            (f"{GH_RAW}/IgnatiusEzeani/IGBONLP/master/ig_en_mt/benchmark_dataset/val.en", "igbonlp_benchmark/val.en"),
            (f"{GH_RAW}/IgnatiusEzeani/IGBONLP/master/ig_en_mt/benchmark_dataset/val.ig", "igbonlp_benchmark/val.ig"),
            (f"{GH_RAW}/IgnatiusEzeani/IGBONLP/master/ig_en_mt/benchmark_dataset/test.en", "igbonlp_benchmark/test.en"),
            (f"{GH_RAW}/IgnatiusEzeani/IGBONLP/master/ig_en_mt/benchmark_dataset/test.ig", "igbonlp_benchmark/test.ig"),
        ],
    },
    # ------------------------------------------------------------------ #
    "ntrex_african": {
        "title": "NTREX-african (Masakhane; NTREX-128 news subset)",
        "description": (
            "~2,000 professionally written news sentences per language, "
            "parallel across languages. Role: additional sentence pool for "
            "constructed pairs (CC-BY-SA share-alike applies to any "
            "redistributed text)."
        ),
        "languages": ["hau", "ibo", "yor"],
        "license": "CC-BY-SA-4.0",
        "license_note": "Share-alike: derivatives containing this text must be CC-BY-SA.",
        "citation": (
            "Masakhane, NTREX-african (NTREX-128 news test set subset). "
            "https://huggingface.co/datasets/masakhane/ntrex_african (CC-BY-SA-4.0); "
            "original: Federmann, Kuzman et al., NTREX-128."
        ),
        "files": [
            (f"{HF}/masakhane/ntrex_african/resolve/main/README.md", "ntrex_african/README.md"),
            (f"{HF}/masakhane/ntrex_african/resolve/main/ntrex_african/hau_Latn.tsv", "ntrex_african/hau_Latn.tsv"),
            (f"{HF}/masakhane/ntrex_african/resolve/main/ntrex_african/ibo_Latn.tsv", "ntrex_african/ibo_Latn.tsv"),
            (f"{HF}/masakhane/ntrex_african/resolve/main/ntrex_african/yor_Latn.tsv", "ntrex_african/yor_Latn.tsv"),
        ],
    },
}

# Deferred (not downloaded in P2):
#   FLORES-200 (CC-BY-SA-4.0) - canonical HF mirror facebook/flores is gated;
#   GitHub distribution is a multi-hundred-MB tarball for 200 languages when
#   we need only yor/ibo/hau/eng. Needed only for pair construction (P3);
#   a targeted download strategy will be decided in P3.

SOURCES["paws_wiki_en"] = {
    "title": "PAWS-Wiki Labeled Final (English)",
    "description": (
        "Human-written paraphrase/non-paraphrase pairs from Wikipedia with "
        "strong lexical-overlap control (49,401 train / ~2k dev / ~2k test, "
        "binary labels). Role in project: SOURCE for constructed "
        "target-language pairs via machine translation of BOTH pair members "
        "(translate-train; PAWS-style overlap control is inherited by the "
        "translated pairs). English pairs are also the training source for "
        "the E5 zero-shot transfer experiment."
    ),
    "languages": ["eng"],
    "license": "Freely usable for any purpose (Google LLC dataset notice)",
    "license_note": (
        "May be freely used for any purpose; acknowledgement of Google LLC "
        "appreciated; provided AS IS. Redistributing DERIVATIVES built from "
        "it is permitted by this notice."
    ),
    "citation": (
        "Zhang, Y., Baldridge, J., & He, L. (2019). PAWS: Paraphrase "
        "Adversaries from Word Scrambling. NAACL 2019; Yang, Y., Zhang, Y., "
        "Tar, C., & Baldridge, D. (2019). PAWS-X. EMNLP 2019. "
        "github.com/google-research-datasets/paws"
    ),
    "files": [
        # Original bucket storage.googleapis.com/paws/english/ returns 403
        # for all clients (checked 2026-11-19); we use the official
        # google-research-datasets/paws HF mirror's auto-converted parquet
        # (refs/convert/parquet branch) instead.
        (
            "https://huggingface.co/datasets/google-research-datasets/paws/resolve/refs%2Fconvert%2Fparquet/labeled_final/train/0000.parquet",
            "paws_wiki_en/train.parquet",
        ),
        (
            "https://huggingface.co/datasets/google-research-datasets/paws/resolve/refs%2Fconvert%2Fparquet/labeled_final/validation/0000.parquet",
            "paws_wiki_en/validation.parquet",
        ),
        (
            "https://huggingface.co/datasets/google-research-datasets/paws/resolve/refs%2Fconvert%2Fparquet/labeled_final/test/0000.parquet",
            "paws_wiki_en/test.parquet",
        ),
        (
            "https://raw.githubusercontent.com/google-research-datasets/paws/master/README.md",
            "paws_wiki_en/README.md",
        ),
        (
            "https://raw.githubusercontent.com/google-research-datasets/paws/master/LICENSE",
            "paws_wiki_en/LICENSE",
        ),
    ],
}


def get_source(name: str) -> Dict[str, Any]:
    if name not in SOURCES:
        raise KeyError(f"Unknown source '{name}'. Available: {sorted(SOURCES)}")
    return SOURCES[name]
