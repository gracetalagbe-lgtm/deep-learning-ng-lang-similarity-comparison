"""Smoke tests: the installed stack imports and versions are sane.

Runs against the project virtual environment; heavy imports are skipped
gracefully if a package is missing so the suite still runs pre-install.
"""

from __future__ import annotations

import importlib

import pytest


def _import_or_skip(module: str):
    try:
        return importlib.import_module(module)
    except ImportError:
        pytest.skip(f"{module} not installed")


def test_afripara_imports():
    import afripara

    assert afripara.__version__


def test_torch_imports_and_cpu_available():
    torch = _import_or_skip("torch")
    assert torch.tensor([1.0, 2.0]).sum().item() == pytest.approx(3.0)


def test_transformers_imports():
    transformers = _import_or_skip("transformers")
    assert transformers.__version__


def test_sentence_transformers_imports():
    sentence_transformers = _import_or_skip("sentence_transformers")
    assert sentence_transformers.__version__


def test_datasets_imports():
    datasets = _import_or_skip("datasets")
    assert datasets.__version__


def test_sklearn_metrics_available():
    sklearn = _import_or_skip("sklearn")
    from sklearn.metrics import f1_score, roc_auc_score  # noqa: F401
