"""Colab one-cell MT job: translate the prepared PAWS / MT-vs-human inputs.

Self-contained (no afripara imports) so it runs in a fresh Colab notebook.

Usage for the researcher:
  1. colab.research.google.com -> New notebook
  2. Runtime > Change runtime type > T4 GPU
  3. Upload the input files from data/interim/mt/ to the Colab file pane
     (drag & drop):
       paws_translate_input_yor.jsonl   (6,000 sentences)
       paws_translate_input_ibo.jsonl   (6,000 sentences)
       paws_translate_input_hau.jsonl   (3,000 sentences)
       mt_vs_human_input_yor.jsonl      (1,000 sentences)
       mt_vs_human_input_ibo.jsonl      (1,000 sentences)
  4. Paste this whole file into the notebook cell and run it.
  5. Download the produced *_translated.jsonl files back into
     data/interim/mt/ on your computer, then run locally:
       python scripts/build_datasets.py --stage constructed

Expected runtime on a free T4: ~10-20 minutes total for all five files.
Progress and throughput are logged; a copy-through guard aborts a file if
the model starts echoing input instead of translating (a bug caught locally
in the Phase 3 pilot).
"""

import json
import time
from pathlib import Path

INPUT_DIR = Path("/content")  # where Colab drag-and-drop uploads land
OUTPUT_DIR = Path("/content/translated")
MODEL_REPO = "facebook/nllb-200-distilled-600M"

JOBS = [
    # (input file, output file, src, tgt)
    ("paws_translate_input_yor.jsonl", "paws_translated_yor.jsonl", "eng_Latn", "yor_Latn"),
    ("paws_translate_input_ibo.jsonl", "paws_translated_ibo.jsonl", "eng_Latn", "ibo_Latn"),
    ("paws_translate_input_hau.jsonl", "paws_translated_hau.jsonl", "eng_Latn", "hau_Latn"),
    ("mt_vs_human_input_yor.jsonl", "mt_vs_human_input_yor_translated.jsonl", "eng_Latn", "yor_Latn"),
    ("mt_vs_human_input_ibo.jsonl", "mt_vs_human_input_ibo_translated.jsonl", "eng_Latn", "ibo_Latn"),
]


def main() -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    assert torch.cuda.is_available(), "No GPU: set Runtime > Change runtime type > T4 GPU"
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading", MODEL_REPO)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_REPO, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_name, out_name, src, tgt in JOBS:
        in_path = INPUT_DIR / in_name
        if not in_path.exists():
            print(f"!! skipping {in_name} - not uploaded yet")
            continue
        records = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n=== {in_name}: {len(records)} sentences  ({src} -> {tgt}) ===")
        t0 = time.time()
        outputs = []
        batch_size = 32
        forced_bos = tokenizer.convert_tokens_to_ids(tgt)
        tokenizer.src_lang = src
        for i in range(0, len(records), batch_size):
            batch = [r["text"] for r in records[i : i + batch_size]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, forced_bos_token_id=forced_bos, max_new_tokens=256)
            outputs.extend([o.strip() for o in tokenizer.batch_decode(gen, skip_special_tokens=True)])
            done = i + len(batch)
            if (i // batch_size) % 5 == 0 or done >= len(records):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(records)} | {rate:.1f} sent/s | {time.time()-t0:.0f}s elapsed")
            if i == 0:
                copied = sum(1 for inp, out in zip(batch, outputs) if out == inp.strip())
                if copied / len(batch) > 0.5:
                    raise RuntimeError("COPY-THROUGH DETECTED - wrong language codes? Aborting.")

        out_path = OUTPUT_DIR / out_name
        with out_path.open("w", encoding="utf-8") as fh:
            for r, mt in zip(records, outputs):
                r["mt"] = mt
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {out_path} ({time.time()-t0:.0f}s)")

    print("\nAll done. Download every file in", OUTPUT_DIR, "back to your computer:")
    print("  place them in  data/interim/mt/  keeping the same file names, then run locally:")
    print("  .venv/bin/python scripts/build_datasets.py --stage constructed")
    try:
        from google.colab import files  # type: ignore

        for _, out_name, _, _ in JOBS:
            p = OUTPUT_DIR / out_name
            if p.exists():
                files.download(str(p))
    except Exception as exc:  # not in Colab, or download blocked
        print("(auto-download skipped:", exc, ")")


if __name__ == "__main__":
    main()
