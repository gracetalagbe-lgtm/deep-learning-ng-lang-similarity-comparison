"""afripara - Transformer-based semantic similarity and paraphrase detection
for Yoruba (yor), Igbo (ibo) and Hausa (hau).

MSc research project. Package layout:

    afripara.config        YAML configuration loading, seeding, device
    afripara.utils         logging, engineering fixtures
    afripara.features      text normalisation and feature extraction
    afripara.data          dataset download, audit, construction, splits (P2-P3)
    afripara.models        TF-IDF baseline, cross-encoder, bi-encoder (P4-P5)
    afripara.training      training loops, early stopping, checkpoints (P5)
    afripara.evaluation    metrics, significance tests, plots (P4, P6)
    afripara.inference     calibrated prediction for the demo (P7)
    afripara.app           Streamlit demonstration application (P7)
"""

__version__ = "0.1.0"
