#!/usr/bin/env python3
"""200-pair pipeline test on REAL data (deferred from Phase 1).

Runs 50 labelled sentence pairs from each of four real evaluation sets
through the full local pipeline: BOM-safe loading -> normalisation ->
cross-encoder pair tokenisation (afriberta_small) -> a CPU forward pass.
Produces token-length statistics that inform the max_seq_length=128
decision, and confirms the stack end-to-end on real data.

This is an ENGINEERING check - no modelling result is produced here.

Usage:
    python scripts/pipeline_test.py [--pairs-per-set 50]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.config import PROJECT_ROOT, load_config, load_model_config, set_seed  # noqa: E402
from afripara.data.loaders import load_afrixnli, load_semrel  # noqa: E402
from afripara.features.normalization import clean_text  # noqa: E402
from afripara.utils.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("pipeline_test")

RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "interim" / "pipeline_test"


def collect_pairs(n: int) -> dict[str, list]:
    """n pairs per real evaluation set (deterministic first-n, seed-independent)."""
    sets: dict[str, list] = {}
    for lang in ["yor", "ibo", "hau"]:
        recs, _ = load_afrixnli(RAW / "afrixnli" / f"{lang}_test.tsv")
        sets[f"afrixnli/{lang}/test"] = recs[:n]
    recs, _ = load_semrel(RAW / "semrel2024_hau" / "hau_dev_with_labels.csv")
    sets["semrel2024_hau/dev"] = recs[:n]
    return sets


def run(sets: dict[str, list], repo_id: str, max_len_cfg: int) -> bool:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 10**6:
        tokenizer.model_max_length = 512
    model = AutoModel.from_pretrained(repo_id)
    model.eval()

    summary: dict = {}
    ok = True
    for name, records in sets.items():
        pairs = [
            (
                clean_text(r.sentence1),
                clean_text(r.sentence2),
            )
            for r in records
        ]
        t0 = time.time()
        encoded = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=max_len_cfg,
            return_tensors="pt",
        )
        n_truncated = int((encoded["input_ids"] == tokenizer.sep_token_id).any(dim=1).sum()) if tokenizer.sep_token_id is not None else -1
        lengths = (encoded["attention_mask"].sum(dim=1)).tolist()
        with torch.no_grad():
            out = model(**encoded)
        finite = bool(torch.isfinite(out.last_hidden_state).all())
        summary[name] = {
            "pairs": len(pairs),
            "pair_token_lengths": {
                "min": min(lengths), "p50": sorted(lengths)[len(lengths) // 2],
                "mean": round(sum(lengths) / len(lengths), 1), "max": max(lengths),
            },
            "hit_cfg_limit": sum(1 for L in lengths if L >= max_len_cfg),
            "forward_pass_finite": finite,
            "seconds": round(time.time() - t0, 2),
        }
        ok = ok and finite
        logger.info(
            "%-24s pairs=%d tokens(p50=%.0f, mean=%.1f, max=%d) at_limit=%d finite=%s %.2fs",
            name, len(pairs), summary[name]["pair_token_lengths"]["p50"],
            summary[name]["pair_token_lengths"]["mean"], summary[name]["pair_token_lengths"]["max"],
            summary[name]["hit_cfg_limit"], finite, summary[name]["seconds"],
        )

    OUT.mkdir(parents=True, exist_ok=True)
    import json

    with (OUT / "report.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {"model": repo_id, "max_seq_length_cfg": max_len_cfg, "summary": summary},
            fh, indent=2, ensure_ascii=False,
        )
    logger.info("Report written to %s", OUT / "report.json")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-set", type=int, default=50)
    parser.add_argument("--model", default="castorini/afriberta_small")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config("default")
    set_seed(cfg["project"]["seed"])
    primary = load_model_config("afroxlmr_base")  # provenance echo only

    logger.info(
        "=== 200-pair pipeline test (engineering check; primary model in experiments: %s) ===",
        primary["model"]["hf_repo"],
    )
    sets = collect_pairs(args.pairs_per_set)
    total = sum(len(v) for v in sets.values())
    logger.info("Collected %d real labelled pairs across %d sets", total, len(sets))

    try:
        ok = run(sets, args.model, cfg["training"]["max_seq_length"])
    except Exception as exc:
        logger.error("Pipeline test failed: %s", exc)
        return 1
    logger.info("=== RESULT: %s ===", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
