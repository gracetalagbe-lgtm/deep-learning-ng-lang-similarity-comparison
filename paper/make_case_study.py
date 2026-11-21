"""Input-to-output case study for Section 5 of the paper.

Selects deterministic, real held-out pairs per language (top-TP, top-FP, top-FN,
bottom-TN by saved seed-42 scores on each language's primary held-out test set),
then runs them through the deployed inference path (ParaphrasePredictor:
clean_text -> tokenizer -> softmax log-odds -> temperature calibration ->
dev-tuned threshold) to record exactly what the system outputs.

Deterministic: selection depends only on saved predictions + processed CSVs.
Output: paper/case_study.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from afripara.inference.predictor import ParaphrasePredictor  # noqa: E402

# language -> (run, predictions file, processed CSV holding the sentences)
CONFIG = {
    "hau": (
        "afroxlmr_hau_semrel_seed42",
        "predictions_semrel_disjoint_test.json",
        "data/processed/semrel_hau_disjoint/test.csv",
    ),
    "yor": (
        "afroxlmr_yor_constructed_seed42",
        "predictions_constructed_test_yor.json",
        "data/processed/constructed/translate_train_yor.csv",
    ),
    "ibo": (
        "afroxlmr_ibo_constructed_seed42",
        "predictions_constructed_test_ibo.json",
        "data/processed/constructed/translate_train_ibo.csv",
    ),
}


def select_cases(pred: dict, df: pd.DataFrame) -> list[dict]:
    """Pick 4 deterministic cases: strongest TP, strongest FP, strongest FN, weakest TN."""
    scores = pd.Series(pred["scores"], dtype=float)
    labels = pd.Series(pred["labels"], dtype=int)
    pair_ids = pd.Series(pred["pair_ids"])
    # Dev-tuned threshold from results.json is applied by the predictor; the
    # saved confusion cells in the paper use it, so select with it too.
    thr = json.loads((ROOT / "outputs/phase5/results.json").read_text())["runs"][  # noqa: E501
        RUN_NAME
    ]["threshold"]
    pred_lbl = (scores >= thr).astype(int)
    correct = pred_lbl == labels
    merged = pd.DataFrame(
        {"pair_id": pair_ids, "score": scores, "gold": labels, "pred": pred_lbl, "ok": correct}
    ).merge(df[["pair_id", "sentence1", "sentence2"]], on="pair_id", how="left")

    cases: list[dict] = []
    tp = merged[(merged.gold == 1) & merged.ok].sort_values("score", ascending=False).iloc[0]
    fp = merged[(merged.gold == 0) & ~merged.ok].sort_values("score", ascending=False).iloc[0]
    fn = merged[(merged.gold == 1) & ~merged.ok].sort_values("score", ascending=True).iloc[0]
    tn = merged[(merged.gold == 0) & merged.ok].sort_values("score", ascending=True).iloc[0]
    for kind, row in (("true_positive", tp), ("false_positive", fp),
                      ("false_negative", fn), ("true_negative", tn)):
        cases.append(
            {
                "kind": kind,
                "pair_id": row.pair_id,
                "sentence1": str(row.sentence1),
                "sentence2": str(row.sentence2),
                "gold": int(row.gold),
                "saved_score": round(float(row.score), 6),
            }
        )
    return cases


def main() -> None:
    out: dict = {}
    for lang, (run, pred_file, csv_path) in CONFIG.items():
        global RUN_NAME
        RUN_NAME = run
        pred = json.loads((ROOT / "outputs/colab" / run / pred_file).read_text())
        df = pd.read_csv(ROOT / csv_path)
        cases = select_cases(pred, df)
        predictor = ParaphrasePredictor(
            ROOT / "outputs/colab" / run / "checkpoint",
            calibration_dir=ROOT / "outputs/calibration",
            results_path=ROOT / "outputs/phase5/results.json",
        )
        for c in cases:
            p = predictor.predict(c["sentence1"], c["sentence2"])
            c["calibrated_prob"] = round(p.probability, 6)
            c["raw_prob"] = round(p.raw_probability, 6)
            c["threshold"] = round(p.threshold, 4)
            c["temperature"] = round(p.temperature, 4)
            c["predicted_label"] = int(p.label)
        out[lang] = {"run": run, "threshold": cases[0]["threshold"],
                     "temperature": cases[0]["temperature"], "cases": cases}
        print(f"{lang}: {len(cases)} cases through deployed predictor")

    dest = ROOT / "paper" / "case_study.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
