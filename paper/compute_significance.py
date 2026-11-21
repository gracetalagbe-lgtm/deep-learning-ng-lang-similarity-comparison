"""Recompute every significance number for the paper's Table VII directly from
saved predictions (per-seed) and the baseline's saved predictions.

Replicates the project's own comparison protocol:
- model labels: saved score >= run's dev-tuned threshold (outputs/phase5/results.json)
- baseline labels: saved score >= baseline's dev-tuned threshold (outputs/baseline/results.json)
- pairs matched on pair_id; McNemar exact two-sided binomial on discordant pairs
- multi-seed systems: accuracy = mean over seeds; p = min across seeds (reported b/c from that seed)

Output: paper/significance.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "outputs/phase5/results.json").read_text())
BASELINE = json.loads((ROOT / "outputs/baseline/results.json").read_text())

SET_PRED = {  # eval set -> (model prediction filename per run dir, baseline prediction file)
    "constructed_test_yor": ("predictions_constructed_test_yor.json", "translate_train_yor.json"),
    "constructed_test_ibo": ("predictions_constructed_test_ibo.json", "translate_train_ibo.json"),
    "constructed_test_hau": ("predictions_constructed_test_hau.json", "translate_train_hau.json"),
    "semrel_disjoint_test": ("predictions_semrel_disjoint_test.json", "semrel_hau_disjoint.json"),
    "semrel_official_test": ("predictions_semrel_official_test.json", "semrel_hau_official.json"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def mcnemar(scores_a, labels_a, thr_a, scores_b, labels_b, thr_b) -> dict:
    ids = None
    a = np.array(scores_a, dtype=float)
    ya = np.array(labels_a, dtype=int)
    b = np.array(scores_b, dtype=float)
    yb = np.array(labels_b, dtype=int)
    assert len(a) == len(b)
    pa = (a >= thr_a).astype(int)
    pb = (b >= thr_b).astype(int)
    acc_a = float((pa == ya).mean())
    acc_b = float((pb == yb).mean())
    a_right = pa == ya
    b_right = pb == yb
    n_b = int((a_right & ~b_right).sum())  # model right, baseline wrong
    n_c = int((~a_right & b_right).sum())  # model wrong, baseline right
    if n_b + n_c == 0:
        p = 1.0
    else:
        p = float(binomtest(min(n_b, n_c), n_b + n_c, 0.5).pvalue)
    return {"acc_model": round(acc_a, 4), "acc_base": round(acc_b, 4),
            "delta_pts": round(100 * (acc_a - acc_b), 1), "b": n_b, "c": n_c, "p": p}


def compare_family(family: str, set_name: str) -> dict | None:
    pred_file, base_file = SET_PRED[set_name]
    base = load(ROOT / "outputs/baseline/predictions" / base_file)
    base_thr = BASELINE["sets"][set_name.replace("constructed_test", "translate_train")
                                 if set_name.startswith("constructed") else
                                 {"semrel_disjoint_test": "semrel_hau_disjoint",
                                  "semrel_official_test": "semrel_hau_official"}[set_name]]["threshold"]
    runs = [r for r in RESULTS["runs"] if r.startswith(family + "_seed")]
    rows = []
    for run_id in runs:
        thr = RESULTS["runs"][run_id]["threshold"]
        pred = load(ROOT / "outputs/colab" / run_id / pred_file)
        # baseline predictions are positional over the same CSV; verify via gold labels
        assert list(pred["labels"]) == list(base["labels"]), f"label order mismatch for {run_id} {set_name}"
        rows.append(mcnemar(pred["scores"], pred["labels"], thr, base["scores"], base["labels"], base_thr))
    best = min(rows, key=lambda r: r["p"])
    return {"family": family, "set": set_name, "n_seeds": len(rows),
            "acc_model_mean": round(float(np.mean([r["acc_model"] for r in rows])), 4),
            "acc_base": rows[0]["acc_base"],
            "delta_pts": round(100 * (float(np.mean([r["acc_model"] for r in rows])) - rows[0]["acc_base"]), 1),
            "b": best["b"], "c": best["c"], "p_min": best["p"],
            "per_seed_p": [r["p"] for r in rows]}


def compare_transfer(transfer_run: str, family: str, set_name: str) -> dict:
    """EN-PAWS transfer (single seed) vs each seed of the in-language silver family; min p."""
    pred_file, _ = SET_PRED[set_name]
    t = load(ROOT / "outputs/colab" / transfer_run / pred_file)
    thr_t = RESULTS["runs"][transfer_run]["threshold"]
    rows = []
    for run_id in [r for r in RESULTS["runs"] if r.startswith(family + "_seed")]:
        thr = RESULTS["runs"][run_id]["threshold"]
        m = load(ROOT / "outputs/colab" / run_id / pred_file)
        rows.append(mcnemar(t["scores"], t["labels"], thr_t, m["scores"], m["labels"], thr))
    best = min(rows, key=lambda r: r["p"])
    acc_t = float((np.array(t["scores"]) >= thr_t).__eq__(np.array(t["labels"])).mean())
    fam_acc = float(np.mean([r["acc_base"] for r in rows]))
    return {"transfer_run": transfer_run, "vs_family": family, "set": set_name,
            "acc_transfer": round(acc_t, 4),
            "acc_family_mean": round(fam_acc, 4),
            "delta_pts": round(100 * (acc_t - fam_acc), 1),
            "b": best["b"], "c": best["c"], "p_min": best["p"],
            "per_seed_p": [r["p"] for r in rows]}


def main() -> None:
    out = {"vs_baseline": [], "transfer_vs_silver": []}
    for family, sets in [
        ("afroxlmr_hau_semrel", ["semrel_disjoint_test", "semrel_official_test"]),
        ("afriberta_hau_semrel", ["semrel_disjoint_test", "semrel_official_test"]),
        ("afroxlmr_hau_constructed", ["semrel_disjoint_test", "semrel_official_test"]),
        ("afroxlmr_yor_constructed", ["constructed_test_yor"]),
        ("afroxlmr_ibo_constructed", ["constructed_test_ibo"]),
        ("afroxlmr_hau_constructed", ["constructed_test_hau"]),
    ]:
        for s in sets:
            out["vs_baseline"].append(compare_family(family, s))
    for set_name, fam in [("constructed_test_hau", "afroxlmr_hau_constructed"),
                          ("constructed_test_ibo", "afroxlmr_ibo_constructed"),
                          ("constructed_test_yor", "afroxlmr_yor_constructed")]:
        out["transfer_vs_silver"].append(compare_transfer("afroxlmr_en_paws_seed42", fam, set_name))
    dest = Path(__file__).resolve().parent / "significance.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
