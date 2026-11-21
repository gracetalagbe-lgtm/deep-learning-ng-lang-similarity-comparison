"""Tests for configuration loading, merging and reproducibility."""

from __future__ import annotations

import random

import pytest

from afripara.config import (
    deep_merge,
    get_device,
    load_config,
    load_model_config,
    resolve_path,
    set_seed,
)


def test_default_config_loads():
    cfg = load_config("default")
    assert cfg["project"]["name"] == "afripara"
    assert cfg["languages"]["codes"] == ["yor", "ibo", "hau"]
    # Diacritic preservation must be the default, never stripping.
    assert cfg["preprocessing"]["strip_diacritics"] is False


def test_default_training_defaults_match_report():
    cfg = load_config("default")
    t = cfg["training"]
    assert t["learning_rate"] == 2.0e-5
    assert t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] == 32
    assert t["warmup_ratio"] == 0.1
    assert t["metric_for_best_model"] == "eval_f1"
    assert t["n_seeds"] >= 3


def test_deep_merge_nested_override():
    base = {"training": {"learning_rate": 2.0e-5, "fp16": True}, "project": {"seed": 42}}
    merged = deep_merge(base, {"training": {"learning_rate": 1.0e-5}})
    assert merged["training"]["learning_rate"] == 1.0e-5
    assert merged["training"]["fp16"] is True          # untouched
    assert merged["project"]["seed"] == 42             # untouched
    assert base["training"]["learning_rate"] == 2.0e-5  # base not mutated


@pytest.mark.parametrize(
    "name,repo,role",
    [
        ("afroxlmr_base", "Davlan/afro-xlmr-base", "primary_cross_encoder"),
        ("afriberta_large", "castorini/afriberta_large", "african_specific_baseline"),
        ("labse", "sentence-transformers/LaBSE", "embedding_track"),
        ("afrie5_large", "McGill-NLP/AfriE5-Large-instruct", "embedding_track"),
        ("mbert", "google-bert/bert-base-multilingual-cased", "negative_control"),
        ("tfidf_baseline", None, "traditional_baseline"),
    ],
)
def test_model_configs_load(name, repo, role):
    cfg = load_model_config(name)
    assert cfg["model"]["role"] == role
    assert cfg["model"].get("license")  # every model must record its licence
    if repo is not None:
        assert cfg["model"]["hf_repo"] == repo


def test_mbert_config_documents_control_status():
    """mBERT must never be configurable as a primary model for ibo/hau."""
    cfg = load_model_config("mbert")
    assert cfg["model"]["role"] == "negative_control"
    assert "EXCLUDING" in cfg["model"]["languages_covered"]


def test_resolve_path_relative_to_project_root():
    cfg = load_config("default")
    raw_dir = resolve_path(cfg, "data_raw")
    assert raw_dir.is_absolute()
    assert raw_dir.name == "raw"


def test_set_seed_deterministic():
    set_seed(42)
    first = [random.random() for _ in range(5)]
    set_seed(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_get_device_returns_valid_string():
    assert get_device() in {"cpu", "cuda", "mps"}
