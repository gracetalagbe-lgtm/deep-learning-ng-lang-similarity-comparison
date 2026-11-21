"""Streamlit demonstration application (Phase 7).

Run:
    .venv/bin/streamlit run src/afripara/app/demo.py

Serves the ACTUAL fine-tuned checkpoints from Phase 5 with their dev-tuned
thresholds and validation-fitted temperature calibration. Language and model
provenance are always displayed; no predictions are hardcoded. Igbo appears
only when its checkpoint is present locally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # demo.py sits at <root>/src/afripara/app/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afripara.inference.predictor import ParaphrasePredictor  # noqa: E402

COLAB = PROJECT_ROOT / "outputs" / "colab"
CALIB = PROJECT_ROOT / "outputs" / "calibration"
RESULTS = PROJECT_ROOT / "outputs" / "phase5" / "results.json"

# Demo model registry: only models whose checkpoints exist locally are
# selectable. Provenance is displayed because it conditions interpretation.
MODELS = {
    "Hausa": {
        "run": "afroxlmr_hau_semrel_seed42",
        "provenance": "trained on SemRel2024-Hausa (human-annotated relatedness scores, 0-1)",
        "quality": "gold-standard labels (research use; no redistribution)",
        "test_acc": "0.783 accuracy / 0.863 ROC-AUC on the leakage-free SemRel test split",
        "note": "",
    },
    "Yoruba": {
        "run": "afroxlmr_yor_constructed_seed42",
        "provenance": "trained on constructed pairs (labels inherited from English PAWS via machine translation)",
        "quality": "silver labels (NOT native-speaker verified - scores are conditional on MT quality)",
        "test_acc": "0.562 accuracy on the constructed Yoruba test split",
        "note": "This model is overconfident (calibration temperature 3.46); treat its scores as indicative.",
    },
    "Igbo": {
        "run": "afroxlmr_ibo_constructed_seed42",
        "provenance": "trained on constructed pairs (labels inherited from English PAWS via machine translation)",
        "quality": "silver labels (NOT native-speaker verified - scores are conditional on MT quality)",
        "test_acc": "0.676 accuracy on the constructed Igbo test split",
        "note": "Igbo results carry the weakest label provenance in the project.",
    },
}

EXAMPLES_FILE = Path(__file__).resolve().parent / "demo_examples.json"


def load_examples() -> dict:
    """Verified example pairs from real held-out data (see scripts/pick_demo_examples.py)."""
    if EXAMPLES_FILE.exists():
        data = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
        return {
            lang: [(p["sentence1"], p["sentence2"]) for p in cfg["pairs"]]
            for lang, cfg in data.items()
        }
    return {}


@st.cache_resource(show_spinner="Loading model...")
def load_predictor(run: str) -> ParaphrasePredictor:
    return ParaphrasePredictor(COLAB / run / "checkpoint", calibration_dir=CALIB, results_path=RESULTS)


def main() -> None:
    st.set_page_config(page_title="Yorùbá · Igbo · Hausa Paraphrase Demo", layout="centered")
    st.title("Semantic Similarity & Paraphrase Detection")
    st.caption(
        "MSc research project - transformer cross-encoders fine-tuned for "
        "Yorùbá, Igbo and Hausa. Scores come from real trained checkpoints."
    )

    available = {lang: cfg for lang, cfg in MODELS.items() if (COLAB / cfg["run"] / "checkpoint" / "config.json").exists()}
    if not available:
        st.error("No trained checkpoints found under outputs/colab/. Complete Phase 5 first.")
        return

    with st.sidebar:
        lang = st.selectbox("Language", list(available.keys()),
                            index=0 if "Hausa" in available else 0,
                            help="Hausa uses the strongest model (human-annotated training data).")
        cfg = available[lang]
        st.subheader("Model information")
        st.write(f"**Checkpoint:** `{cfg['run']}`")
        st.write(f"**Training data:** {cfg['provenance']}")
        st.write(f"**Label quality:** {cfg['quality']}")
        st.write(f"**Held-out performance:** {cfg['test_acc']}")
        if cfg["note"]:
            st.warning(cfg["note"])
        st.caption(
            "Scores are temperature-calibrated on validation data; the "
            "paraphrase decision uses the validation-tuned threshold. "
            "Accuracy and ranking metrics are unaffected by calibration."
        )

    st.markdown(f"### Enter two sentences in **{lang}**")
    col1, col2 = st.columns(2)
    examples = load_examples().get(lang, [])
    example_idx = st.selectbox(
        "Example pairs (real held-out data, verified verdicts)",
        options=list(range(len(examples))),
        format_func=lambda i: f"Example {i+1}: gold = {'paraphrase' if examples[i][2] == 1 else 'not paraphrase'}"
        if len(examples[i]) > 2 else f"Example {i+1}",
    ) if examples else None
    if examples:
        s1_default, s2_default = examples[example_idx][0], examples[example_idx][1]
    else:
        s1_default, s2_default = "", ""
    s1 = col1.text_area("Sentence A", value=s1_default, height=110)
    s2 = col2.text_area("Sentence B", value=s2_default, height=110)

    if st.button("Compare sentences", type="primary", use_container_width=True):
        if not s1.strip() or not s2.strip():
            st.warning("Please provide both sentences.")
            return
        predictor = load_predictor(cfg["run"])
        with st.spinner("Running the fine-tuned model locally (CPU)..."):
            pred = predictor.predict(s1, s2)
        st.subheader("Verdict")
        if pred.label == 1:
            st.success(f"**Paraphrase / near-equivalent** (calibrated probability {pred.probability:.3f})")
        else:
            st.error(f"**Not a paraphrase** (calibrated probability {pred.probability:.3f})")
        st.progress(pred.probability, text=f"calibrated similarity score: {pred.probability:.3f}")
        st.caption(
            f"Decision threshold {pred.threshold:.2f} (validation-tuned) · temperature "
            f"{pred.temperature:.2f} · calibrated score is a monotone transform of the "
            "model score; ranking metrics are unaffected."
        )
        st.info(
            "Interpretation guidance: Hausa scores rest on human-annotated training "
            "data; Yorùbá and Igbo scores come from models trained on machine-"
            "translation-inherited labels and are research-indicative only."
        )


if __name__ == "__main__":
    main()
