"""Leakage-controlled, deterministic dataset splitting.

Motivation (P2 audit): SemRel2024's official Hausa splits share 644 sentences
between train and test (62% of unique test sentences). Any split we control
must be *sentence-disjoint*: the same sentence must never appear in two
different splits, so a test pair always contains unseen sentences.

Design: a sentence is hashed (md5 of salt + normalised text) into a fold;
a pair is assigned to a fold only when BOTH sentences hash to the same fold,
otherwise the pair is dropped. This is deterministic across runs and
machines (no RNG), auditable, and reports its drop statistics honestly.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from afripara.data.loaders import PairRecord
from afripara.features.normalization import clean_text

logger = logging.getLogger("afripara.splits")

DEFAULT_FOLDS: Tuple[str, ...] = ("train", "dev", "test")
DEFAULT_WEIGHTS: Dict[str, float] = {"train": 0.8, "dev": 0.1, "test": 0.1}


def _cumulative(weights: Dict[str, float], folds: Sequence[str]) -> List[Tuple[str, float]]:
    total = sum(weights.get(f, 0.0) for f in folds)
    out, acc = [], 0.0
    for f in folds:
        acc += weights.get(f, 0.0) / total
        out.append((f, acc))
    return out


def sentence_fold(sentence: str, salt: str, weights: Dict[str, float] | None = None) -> str:
    """Deterministically assign a sentence to a fold via a salted hash.

    Same sentence -> same fold, always (no RNG, no run-to-run variation).
    """
    weights = weights or DEFAULT_WEIGHTS
    digest = hashlib.md5((salt + "\x1f" + clean_text(sentence)).encode("utf-8")).hexdigest()
    x = int(digest[:12], 16) / float(16**12)  # uniform-ish in [0, 1)
    for fold, cum in _cumulative(weights, DEFAULT_FOLDS):
        if x < cum:
            return fold
    return DEFAULT_FOLDS[-1]


def assign_pairs_sentence_disjoint(
    pairs: List[PairRecord],
    salt: str,
    weights: Dict[str, float] | None = None,
    side_a: str = "sentence1",
    side_b: str = "sentence2",
) -> Tuple[Dict[str, List[PairRecord]], Dict[str, int]]:
    """Assign pairs to folds; drop pairs whose sentences land in different folds.

    Returns (fold -> pairs, stats). Stats include per-fold counts, dropped
    pairs, and per-fold label distributions.
    """
    weights = weights or DEFAULT_WEIGHTS
    folds: Dict[str, List[PairRecord]] = {f: [] for f in DEFAULT_FOLDS}
    dropped = 0
    for pair in pairs:
        fa = sentence_fold(getattr(pair, side_a), salt, weights)
        fb = sentence_fold(getattr(pair, side_b), salt, weights)
        if fa == fb:
            folds[fa].append(pair)
        else:
            dropped += 1
    stats: Dict[str, int] = {"total": len(pairs), "dropped_mixed_fold": dropped}
    for fold in DEFAULT_FOLDS:
        stats[f"n_{fold}"] = len(folds[fold])
        for label, count in Counter(p.label for p in folds[fold] if p.label is not None).items():
            stats[f"n_{fold}_label_{label}"] = count
    logger.info("Split stats: %s", stats)
    return folds, stats


def check_disjointness(pairs_by_fold: Dict[str, List[PairRecord]]) -> Dict[str, int]:
    """Verify no sentence appears in two folds; returns overlap counts (must be 0)."""
    sentence_sets = {
        fold: {clean_text(p.sentence1) for p in pairs} | {clean_text(p.sentence2) for p in pairs}
        for fold, pairs in pairs_by_fold.items()
    }
    overlaps: Dict[str, int] = {}
    fold_names = sorted(sentence_sets)
    for i, a in enumerate(fold_names):
        for b in fold_names[i + 1 :]:
            overlaps[f"{a}_vs_{b}"] = len(sentence_sets[a] & sentence_sets[b])
    return overlaps


# --------------------------------------------------------------------------- #
# Connected-component split (preferred: zero drops, controlled proportions)
# --------------------------------------------------------------------------- #

def _sentence_components(pairs: List[PairRecord]) -> Tuple[List[List[PairRecord]], Dict[str, int]]:
    """Group pairs into connected components of shared sentences (union-find).

    Two pairs belong to the same component when they share a sentence (after
    normalisation). All pairs of a component must land in the same fold to
    keep the split sentence-disjoint without dropping anything.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # deterministic root choice: lexicographically smaller hash wins
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for p in pairs:
        union(clean_text(p.sentence1), clean_text(p.sentence2))

    components: Dict[str, List[PairRecord]] = {}
    for p in pairs:
        components.setdefault(find(clean_text(p.sentence1)), []).append(p)

    # Deterministic ordering: by size desc, then by component root id.
    ordered = sorted(components.values(), key=lambda g: (-len(g), find(g[0].sentence1)))
    return ordered, {"n_sentences": len(parent), "n_components": len(components)}


def assign_components_disjoint(
    pairs: List[PairRecord],
    weights: Dict[str, float] | None = None,
) -> Tuple[Dict[str, List[PairRecord]], Dict[str, int]]:
    """Assign whole sentence-components to folds via greedy quota filling.

    Guarantees:
      - zero leakage: no sentence appears in two folds (components are
        assigned atomically and are disjoint by construction);
      - zero dropped pairs (unlike hash-based assignment);
      - fold proportions approximate ``weights`` in PAIR counts;
      - deterministic: component order is size-desc, tie-broken by root id.

    Returns (fold -> pairs, stats) with per-fold counts and label
    distributions.
    """
    weights = weights or DEFAULT_WEIGHTS
    components, comp_stats = _sentence_components(pairs)
    total = len(pairs)
    remaining = {f: weights.get(f, 0.0) * total for f in DEFAULT_FOLDS}
    folds: Dict[str, List[PairRecord]] = {f: [] for f in DEFAULT_FOLDS}

    for component in components:
        # Largest remaining quota wins; ties broken by fold order (deterministic).
        fold = max(DEFAULT_FOLDS, key=lambda f: (remaining.get(f, 0.0), -DEFAULT_FOLDS.index(f)))
        folds[fold].extend(component)
        remaining[fold] -= len(component)

    stats: Dict[str, int] = {**comp_stats, "total": total, "dropped_mixed_fold": 0}
    for fold in DEFAULT_FOLDS:
        stats[f"n_{fold}"] = len(folds[fold])
        for label, count in Counter(p.label for p in folds[fold] if p.label is not None).items():
            stats[f"n_{fold}_label_{label}"] = count
    logger.info("Component split stats: %s", stats)
    return folds, stats
