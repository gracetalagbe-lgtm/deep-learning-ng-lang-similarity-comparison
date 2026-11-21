"""Machine-translation wrapper (NLLB-200) used ONLY for dataset construction.

Why MT, and why this is honest: constructed positive pairs are *candidate*
paraphrases - MT renderings of an English source are paired with an
independent human rendering (Menyo/IGBONLP) or with a second MT rendering
(PAWS translate-train). SemRel2024's authors warned that MT for African
languages is unreliable, so:
  - all constructed pairs carry provenance metadata (never silent);
  - a human-verified gold subset (native-speaker validation) is mandatory
    before constructed pairs are treated as evaluation labels;
  - translation quality statistics are reported (length ratios, drop rates).

CLI (local pilot / Colab scale-up share this exact code):
    python -m afripara.data.mt --task file --input sentences.txt \
        --output out.jsonl --src eng_Latn --tgt yor_Latn [--limit 100]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

from afripara.config import get_device, set_seed
from afripara.utils.logging_utils import setup_logging

logger = logging.getLogger("afripara.mt")

NLLB_REPO = "facebook/nllb-200-distilled-600M"
# FLORES-200 / NLLB language codes for our three languages.
NLLB_CODES = {
    "eng": "eng_Latn",
    "yor": "yor_Latn",
    "ibo": "ibo_Latn",
    "hau": "hau_Latn",
}


class NllbTranslator:
    """Batched NLLB translator; the same code runs on CPU (pilot) and GPU."""

    def __init__(self, repo_id: str = NLLB_REPO, device: str | None = None, max_new_tokens: int = 256):
        import torch  # local import: heavy
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device or get_device()
        logger.info("Loading %s on %s ...", repo_id, self.device)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        if self.device == "cpu":
            torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))
        logger.info("Model loaded in %.1fs", time.time() - t0)

    def translate(self, sentences: Sequence[str], src: str, tgt: str, batch_size: int = 16) -> List[str]:
        """Translate a sequence of sentences; src/tgt are NLLB codes.

        Correct NLLB usage (verified against HF docs): the source language is
        set on the TOKENIZER (``src_lang``), while the TARGET language id is
        the decoder's forced BOS token. Getting these swapped makes the model
        copy the input verbatim (caught by the pilot on 2026-11-19 and by the
        copy-through guard below).
        """
        outputs: List[str] = []
        t0 = time.time()
        n_batches = (len(sentences) + batch_size - 1) // batch_size
        forced_bos = self.tokenizer.convert_tokens_to_ids(tgt)
        self.tokenizer.src_lang = src
        for i in range(0, len(sentences), batch_size):
            batch = list(sentences[i : i + batch_size])
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=200
            ).to(self.device)
            with self.torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=self.max_new_tokens,
                )
            batch_out = [o.strip() for o in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]
            outputs.extend(batch_out)
            done = (i // batch_size) + 1
            if done == 1:
                copied = sum(
                    1 for inp, out in zip(batch, batch_out) if out.strip() == inp.strip()
                )
                if copied / len(batch) > 0.5:
                    logger.warning(
                        "COPY-THROUGH DETECTED: %d/%d first-batch outputs identical to input - "
                        "check src/tgt language codes!", copied, len(batch)
                    )
            if done % 10 == 0 or done == n_batches:
                rate = (i + len(batch)) / max(time.time() - t0, 1e-9)
                logger.info("batch %d/%d | %.1f sentences/s | %d/%d", done, n_batches, rate, i + len(batch), len(sentences))
        return outputs


def translate_file(
    input_path: Path,
    output_path: Path,
    src_code: str,
    tgt_code: str,
    limit: int | None = None,
    batch_size: int = 16,
    seed: int = 42,
) -> None:
    """Translate a JSONL file of {"id": ..., "text": ...} records, adding "mt".

    Existing records with a non-empty "mt" are skipped (idempotent/resumable).
    """
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    todo = [r for r in records if not r.get("mt")]
    if limit is not None:
        todo = todo[:limit]
    logger.info("%d records (%d to translate)", len(records), len(todo))
    if not todo:
        logger.info("Nothing to do.")
        return

    translator = NllbTranslator(device=None)
    t0 = time.time()
    translations = translator.translate([r["text"] for r in todo], src_code, tgt_code, batch_size=batch_size)
    elapsed = time.time() - t0
    by_id = {r["id"]: t for r, t in zip(todo, translations)}
    for r in records:
        r["mt"] = by_id.get(r["id"], r.get("mt", ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(
        "Translated %d sentences in %.1fs (%.2f s/sentence) -> %s",
        len(todo), elapsed, elapsed / max(len(todo), 1), output_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["file"], default="file")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with {id, text}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--src", required=True, help="NLLB source code (e.g. eng_Latn)")
    parser.add_argument("--tgt", required=True, help="NLLB target code (e.g. yor_Latn)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    setup_logging()
    set_seed(args.seed)
    translate_file(args.input, args.output, args.src, args.tgt, args.limit, args.batch_size, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
