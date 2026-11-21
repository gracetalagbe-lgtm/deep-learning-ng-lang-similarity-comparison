"""Engineering fixtures: short sample sentences per target language.

These sentences exist ONLY to exercise the engineering pipeline (tokenizer
round-trips, diacritic handling, smoke tests). They are NOT research data,
are not derived from any dataset, and must never be used in experiments,
training or evaluation.
"""

from __future__ import annotations

# Short, diacritic-bearing phrases per language (common greetings/phrases).
SAMPLE_SENTENCES: dict[str, list[str]] = {
    "yor": [
        "Ẹ kú àárọ̀",                       # diacritics: Ẹ ̀, ú, à, à, ọ̀
        "Èdè Yorùbá ni mo ń kọ́",           # tone marks on nearly every word
        "Mo fẹ́ràn láti jẹ iresi àti ẹ̀wà",
    ],
    "ibo": [
        "Ndewo",
        "Ọ dị mma ịhụ gị",                 # subdots: Ọ, ị, ụ
        "Daalụ",
    ],
    "hau": [
        "Sannu da zuwa",
        "Ina kwana",
        "Na gode",
    ],
}

# Single characters exercising the special letters of each orthography:
# Yoruba subdot + tone letters, Igbo subdot letters, Hausa hooked letters.
SPECIAL_LETTERS: dict[str, list[str]] = {
    "yor": ["ọ", "ẹ", "ṣ", "ń", "ǹ", "ń", "à", "ó", "ù", "ḍ"],
    "ibo": ["ị", "ọ", "ụ", "ṅ"],
    "hau": ["ɓ", "ɗ", "ƙ", "ƴ", "ʼ"],
}
