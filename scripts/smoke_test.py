#!/usr/bin/env python3
"""End-to-end engineering smoke test (CPU-only, local machine).

Verifies the full stack before any real experiment is attempted:
  1. configuration loads and seed is applied;
  2. a small African-language pretrained model's tokenizer downloads and
     encodes Yoruba/Igbo/Hausa fixture sentences (diacritics survive);
  3. (optional, default ON) the model runs a forward pass on CPU.

The test model is `castorini/afriberta_small` (~97M params, covers all three
languages) chosen purely for its small size on a 2-core CPU - it is an
engineering check, NOT an experiment, and carries no research result.

Usage:
    python scripts/smoke_test.py [--no-model]     # --no-model: tokenizer only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.config import get_device, load_config, load_model_config, set_seed  # noqa: E402
from afripara.features.normalization import clean_text, has_diacritics  # noqa: E402
from afripara.utils.logging_utils import setup_logging  # noqa: E402
from afripara.utils.samples import SAMPLE_SENTENCES  # noqa: E402

import logging

logger = logging.getLogger("smoke_test")


def load_tokenizer(repo_id: str):
    """Load a tokenizer, falling back to the slow implementation if needed.

    AfriBERTa checkpoints ship only a slow SentencePiece tokenizer (no
    tokenizer.json); the automatic fast conversion fails for them.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(repo_id)
    except Exception as exc:
        logger.warning("Fast tokenizer unavailable for %s (%s); retrying with use_fast=False", repo_id, exc)
        return AutoTokenizer.from_pretrained(repo_id, use_fast=False)


def check_tokenizer(repo_id: str) -> bool:
    """Encode fixture sentences; verify diacritics survive the round trip."""
    t0 = time.time()
    logger.info("Downloading/loading tokenizer: %s", repo_id)
    tokenizer = load_tokenizer(repo_id)
    # AfriBERTa model cards require model_max_length to be set explicitly.
    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 10**6:
        tokenizer.model_max_length = 512
        logger.info("Set tokenizer.model_max_length=512 (AfriBERTa requirement)")

    all_ok = True
    for lang, sentences in SAMPLE_SENTENCES.items():
        for sentence in sentences:
            cleaned = clean_text(sentence)
            encoded = tokenizer(cleaned, truncation=True, max_length=64)
            decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
            # SentencePiece/BERT decoders may adjust spacing/characters, but a
            # losing round trip must at least not empty the text.
            ok = len(decoded.strip()) > 0
            diacritic_loss = has_diacritics(cleaned) and not has_diacritics(decoded)
            status = "ok" if ok and not diacritic_loss else "CHECK"
            if not ok or diacritic_loss:
                all_ok = False
            logger.info(
                "[%s] %-6s in=%-3d %r -> %r (%s)%s",
                lang, status, len(encoded["input_ids"]), cleaned, decoded,
                f"ids={len(encoded['input_ids'])}",
                " [DIACRITIC LOSS]" if diacritic_loss else "",
            )
    logger.info("Tokenizer stage finished in %.1fs", time.time() - t0)
    return all_ok


def check_model(repo_id: str) -> bool:
    """Run one forward pass per language on CPU."""
    import torch
    from transformers import AutoModel

    device = get_device()
    logger.info("Device: %s | torch %s", device, torch.__version__)
    t0 = time.time()
    logger.info("Downloading/loading model: %s (this is the slow step)", repo_id)
    model = AutoModel.from_pretrained(repo_id)
    model.to(device)
    model.eval()
    logger.info("Model loaded in %.1fs (%.1fM params)", time.time() - t0, model.num_parameters() / 1e6)

    tokenizer = load_tokenizer(repo_id)
    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 10**6:
        tokenizer.model_max_length = 512

    ok = True
    for lang, sentences in SAMPLE_SENTENCES.items():
        batch = [clean_text(s) for s in sentences]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        shape = tuple(outputs.last_hidden_state.shape)
        finite = bool(torch.isfinite(outputs.last_hidden_state).all())
        logger.info("[%s] forward pass -> last_hidden_state %s, finite=%s", lang, shape, finite)
        ok = ok and finite and shape[-1] > 0
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-model", action="store_true", help="skip model download/forward pass")
    parser.add_argument("--model", default=None, help="override HF repo id (default: afriberta_small)")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    cfg = load_config("default")
    set_seed(cfg["project"]["seed"])

    model_cfg = load_model_config("afriberta_large")  # for provenance metadata only
    repo_id = args.model or "castorini/afriberta_small"

    logger.info("=== afripara engineering smoke test (fixture sentences only; NOT research data) ===")
    logger.info("Config seed=%s | smoke model=%s | primary model=%s",
                cfg["project"]["seed"], repo_id, model_cfg["model"]["hf_repo"])

    try:
        tok_ok = check_tokenizer(repo_id)
    except Exception as exc:
        logger.error("Tokenizer stage failed: %s", exc)
        return 1

    if args.no_model:
        logger.info("Model stage skipped (--no-model)")
        model_ok = True
    else:
        try:
            model_ok = check_model(repo_id)
        except Exception as exc:
            logger.error("Model stage failed: %s", exc)
            return 1

    logger.info("=== RESULT: %s ===", "PASS" if (tok_ok and model_ok) else "FAIL")
    return 0 if (tok_ok and model_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
