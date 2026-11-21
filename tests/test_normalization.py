"""Tests for diacritic-preserving normalisation.

These guard the single most linguistically sensitive behaviour of the
project: the default pipeline must never damage Yoruba tone marks or the
subdot/hooked letters of Igbo and Hausa.
"""

from __future__ import annotations

import pytest

from afripara.features.normalization import (
    clean_text,
    diacritic_density,
    has_diacritics,
    normalize_unicode,
    normalize_whitespace,
    strip_diacritics,
)
from afripara.utils.samples import SAMPLE_SENTENCES


def test_clean_text_preserves_yoruba_diacritics():
    text = "Mo fẹ́ràn láti jẹ iresi àti ẹ̀wà"
    assert clean_text(text) == normalize_whitespace(text)
    assert "ẹ́" in clean_text(text) and "à" in clean_text(text)


def test_clean_text_preserves_igbo_subdots():
    text = "Ọ dị mma ịhụ gị"
    cleaned = clean_text(text)
    assert "Ọ" in cleaned and "ị" in cleaned and "ụ" in cleaned


def test_clean_text_is_idempotent():
    text = "Ẹ kú àárọ̀"
    once = clean_text(text)
    assert clean_text(once) == once


def test_normalize_unicode_idempotent_and_nfc():
    text = "Ẹ kú àárọ̀"
    nfc = normalize_unicode(text, "NFC")
    assert normalize_unicode(nfc, "NFC") == nfc
    with pytest.raises(ValueError):
        normalize_unicode(text, form="NFCX")
    with pytest.raises(TypeError):
        normalize_unicode(None)


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("  Sannu \t da \n zuwa  ") == "Sannu da zuwa"


def test_strip_diacritics_yoruba():
    assert strip_diacritics("Ọjọ́") == "Ojo"
    assert strip_diacritics("Ẹ kú àárọ̀") == "E ku aaro"
    assert strip_diacritics("ṣíṣe") == "sise"


def test_strip_diacritics_igbo_and_hausa():
    assert strip_diacritics("ịhụ") == "ihu"
    assert strip_diacritics("ƙarfi") == "karfi"
    assert strip_diacritics("ɓarawo") == "barawo"


def test_strip_diacritics_leaves_plain_ascii_untouched():
    assert strip_diacritics("Sannu da zuwa") == "Sannu da zuwa"


def test_has_diacritics_detection():
    assert has_diacritics("àárọ̀") is True
    assert has_diacritics("aaro") is False
    assert has_diacritics("ƙarfi") is True
    assert has_diacritics("karfi") is False


def test_diacritic_density_bounds():
    assert diacritic_density("") == 0.0
    assert diacritic_density("123 !!!") == 0.0
    assert 0.0 < diacritic_density("Ẹ kú àárọ̀") < 1.0


def test_all_sample_sentences_survive_cleaning():
    for lang, sentences in SAMPLE_SENTENCES.items():
        for sentence in sentences:
            cleaned = clean_text(sentence)
            assert cleaned, f"{lang} sample became empty"
            assert cleaned == sentence or not has_diacritics(sentence)
