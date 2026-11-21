"""Text normalisation for Yoruba, Igbo and Hausa.

Design constraint (from docs/feasibility_report.md, section 6/9): diacritics
are PHONEMIC in Yoruba (tone marks distinguish words) and in Igbo/Hausa
(subdot and hooked letters distinguish words). The default pipeline therefore
preserves them exactly; ``strip_diacritics`` exists only for the E6 linguistic
ablation and must never be part of a default preprocessing chain.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict

# Precomposed letters whose diacritic is NOT a combining mark and therefore
# survives NFKD decomposition (dot-below letters, hooked letters, etc.).
# Used ONLY by strip_diacritics().
_PRECOMPOSED_MAP: Dict[str, str] = {
    # Yoruba subdot letters (+ Hausa hooked letters)
    "ọ": "o", "ẹ": "e", "ṣ": "s", "ṇ": "n", "ḍ": "d", "ṛ": "r",
    "Ọ": "O", "Ẹ": "E", "Ṣ": "S", "Ṇ": "N", "Ḍ": "D", "Ṛ": "R",
    # Igbo subdot letters
    "ị": "i", "ụ": "u", "Ị": "I", "Ụ": "U", "Ṅ": "N",
    # Hausa hooked letters and glottal markers
    "ɓ": "b", "ɗ": "d", "ƙ": "k", "ƴ": "y",
    "Ɓ": "B", "Ɗ": "D", "Ƙ": "K", "Ƴ": "Y",
    "ʼ": "'", "ʽ": "'",
}

_WHITESPACE_RE = re.compile(r"\s+")

# Zero-width characters that are encoding/export artifacts, not linguistic
# content (found embedded inside Menyo-20k cells in the P2 audit). Removing
# them never touches diacritics or letters.
_ZERO_WIDTH_RE = re.compile(r"[​‎‏﻿]")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Return text in a canonical Unicode normal form.

    NFC (the project default, see configs/default.yaml) composes base
    letters with their combining tone marks into single code points where
    possible - it PRESERVES diacritics, unlike aggressive NFKD stripping.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize_unicode expects str, got {type(text).__name__}")
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"Unsupported Unicode form: {form!r}")
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str, form: str = "NFC", lowercase: bool = False) -> str:
    """Default cleaning pipeline: canonical Unicode, zero-width artifact
    removal, whitespace collapse.

    Deliberately does NOT lowercase and does NOT touch diacritics unless
    explicitly requested - Yorùbá tone is phonemic.
    """
    cleaned = normalize_unicode(text, form=form)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = normalize_whitespace(cleaned)
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned


def strip_diacritics(text: str) -> str:
    """Remove diacritics/tone marks. E6 ABLATION USE ONLY.

    Two-step removal because West African orthographies use BOTH combining
    marks (tone accents) and precomposed letters (subdots, hooked letters):
      1. NFKD-decompose and drop combining marks (category ``Mn``);
      2. map remaining precomposed letters (ọ, ị, ɓ, ...) via an explicit
         table.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return "".join(_PRECOMPOSED_MAP.get(ch, ch) for ch in without_marks)


def has_diacritics(text: str) -> bool:
    """True if the text contains combining marks or mapped special letters."""
    if any(unicodedata.category(ch) == "Mn" for ch in unicodedata.normalize("NFD", text)):
        return True
    return any(ch in _PRECOMPOSED_MAP and _PRECOMPOSED_MAP[ch] != ch for ch in text)


def diacritic_density(text: str) -> float:
    """Fraction of alphabetic characters carrying diacritics (audit metric)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    marked = sum(1 for ch in letters if has_diacritics(ch))
    return marked / len(letters)
