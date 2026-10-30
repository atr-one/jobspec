"""Minimum years of experience required."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..text import Document

_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "fifteen": 15,
}
_NUMBER = r"(?P<n>\d{1,2}|" + "|".join(_WORDS) + r")"
_UNIT = r"(?:years?|yrs?|year's|jahre?n?|a[ñn]os?|лет|года?|saal)"

# "or more" after the number or after the unit: "4 or more years", "5 years or more", "3 o más años"
_OR_MORE = r"(?:or|and|oder|o|ou)\s+(?:more|above|over|mehr|m[áa]s|mais)"
# "5+ years", "3-5 years", "3 to 5 yrs", "4 or more years", "minimum 4 years", "mindestens 3 Jahre"
_YEARS = re.compile(
    r"(?<![\d.,])"
    + _NUMBER
    + r"\s*(?:\+|plus)?\s*(?:"
    + _OR_MORE
    + r"\s+)?(?:(?:-|to|a|bis|до)\s*\d{1,2}\s*\+?\s*)?"
    + _UNIT
    + r"(?![^\W\d_])(?:\s+"
    + _OR_MORE
    + r"\b)?",
    re.IGNORECASE,
)
_PLUS = re.compile(r"\+|\bplus\b|" + _OR_MORE, re.IGNORECASE)
# Russian puts the number after the word: "опыт работы от 3 лет", "опыт от 1 года"
_RU = re.compile(
    r"опыт\w*[^.\n]{0,40}?(?:от|не\s+менее|более|свыше)\s*(?P<n>\d{1,2})\s*(?:-|до)?\s*(?:\d{1,2}\s*)?(?:лет|года?)\b",
    re.IGNORECASE,
)
_CONTEXT = re.compile(
    r"experience|exp\b|expérience|experiencia|experiência|erfahrung|berufserfahrung|опыт|"
    r"\bat\s+least\b|\bminimum\b|\bmin\.|\bmindestens\b|\bal\s+menos\b|\bm[íi]nimo\b|\bmore\s+than\b|"
    r"\bover\b|\bот\b|\bне\s+менее\b|\bworking\b|\bprofessional\b|\bhands[- ]on\b|\bbuilding\b|"
    r"\bin\s+(?:a\s+)?similar\b",
    re.IGNORECASE,
)
_NOT_EXPERIENCE = re.compile(
    r"founded|been\s+around|in\s+business|history|ago|anniversary|old\b|company\s+for|we\s+have\s+been|"
    r"warranty|contract\s+of|vesting|cliff|for\s+the\s+(?:first|next)|probation",
    re.IGNORECASE,
)


def _value(token: str) -> Optional[int]:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _WORDS.get(token)


def extract_experience(doc: Document, title: Optional[str] = None) -> Optional[Tuple[int, float]]:
    found: List[int] = []
    text = doc.text
    for m in _RU.finditer(text):
        v = _value(m.group("n"))
        if v is not None:
            found.append(v)
    for m in _YEARS.finditer(text):
        v = _value(m.group("n"))
        if v is None or v > 40:
            continue
        before = text[max(0, m.start() - 50) : m.start()]
        after = text[m.end() : m.end() + 50]
        if _NOT_EXPERIENCE.search(before[-25:]) or _NOT_EXPERIENCE.search(after[:25]):
            continue
        plus = bool(_PLUS.search(m.group(0)))
        in_title = title is not None and m.group(0) in title
        if plus or in_title or _CONTEXT.search(before) or _CONTEXT.search(after):
            found.append(v)
    if not found:
        return None
    # The overall requirement is usually the largest one ("5+ years ..., 2+ years with Kubernetes").
    return max(found), 0.8
