"""Job title: a labeled line, the HTML heading, or the first line; cleaned of salary,
work-mode, location and gender-marker suffixes."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from ..text import Document
from ..vocab import CITIES, COUNTRIES

_LABEL = re.compile(
    r"^\W*(?:job\s+title|title|position|role|puesto|vacante|stelle|stellenbezeichnung|"
    r"должность|вакансия|vaga|cargo)\s*:\s*(?P<value>.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_GENDER = re.compile(
    r"\s*\(\s*(?:[mwfdxhgn]\s*/\s*){1,3}[mwfdxhgn]\s*\)|\s*\(\s*all\s+genders?\s*\)|\s*\(gn\)|\s*\*in\b",
    re.IGNORECASE,
)
_MONEY = re.compile(r"[$€£₹₽¥]|\d\s*k\b|\d{2,3}[,. ]\d{3}|\blpa\b|\bsalary\b", re.IGNORECASE)
_MODE = re.compile(
    r"\b(?:remote|hybrid|on[- ]?site|wfh|wfo|remot[oa]|h[ií]brid[oa]|presencial)\b|удал[её]нн?\w*|гибрид\w*",
    re.IGNORECASE,
)
_YEARS = re.compile(
    r"\d+\s*\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?|a[ñn]os|jahre|лет|года?|months?|weeks?|meses|monate?)\b",
    re.IGNORECASE,
)
_TERMS = re.compile(
    r"\b(?:contract|contractor|freelance|full[- ]?time|part[- ]?time|permanent|temporary|fixed[- ]term)\b",
    re.IGNORECASE,
)
_PLACE_NAMES = {name.lower() for name in CITIES} | set(COUNTRIES)


def _is_noise(segment: str) -> bool:
    seg = segment.strip(" ()[],-|/").lower()
    if not seg:
        return True
    if _MONEY.search(seg) or _MODE.search(seg) or _YEARS.search(seg) or _TERMS.search(seg):
        return True
    parts = [p.strip() for p in re.split(r"[,/]| or ", seg) if p.strip()]
    return bool(parts) and all(p in _PLACE_NAMES for p in parts)


def clean_title(title: str) -> str:
    """Strip markup, gender markers and trailing salary / location / work-mode segments."""
    t = re.sub(r"^[#*>\-\s]+|[*\s]+$", "", title)
    t = _GENDER.sub("", t)
    money = re.search(r"\s+(?:-|\||·|:)\s+(?=(?:[A-Z]{3}\s?|[A-Z]{0,2}[$€£₹₽¥])?\s?\d)(?P<rest>.*)$", t)
    if money and _MONEY.search(money.group("rest")):
        t = t[: money.start()]
    for _ in range(4):
        before = t
        m = re.search(r"\s*[(\[]([^()\[\]]*)[)\]]\s*$", t)
        if m and _is_noise(m.group(1)):
            t = t[: m.start()]
        cuts = list(re.finditer(r"\s+(?:-|\||·|@)\s+|,\s+", t))
        if cuts and _is_noise(t[cuts[-1].end() :]):
            t = t[: cuts[-1].start()]
        t = t.rstrip(" -|,:")
        if t == before:
            break
    return " ".join(t.split())


def extract_title(doc: Document) -> Optional[Tuple[str, float]]:
    m = _LABEL.search(doc.text)
    if m:
        value = clean_title(m.group("value"))
        if value:
            return value, 0.95
    if doc.html_title:
        value = clean_title(doc.html_title)
        if value:
            return value, 0.85
    for line in doc.lines[:3]:
        if line.endswith(":") or len(line) > 120 or len(line.split()) > 14:
            continue
        if line.lower().startswith(("about ", "company:", "we are", "who we are")):
            continue
        value = clean_title(line)
        if value:
            return value, 0.7
    return None
