"""Seniority from the title, then from explicit statements in the body."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..text import Document

_NL = r"(?<![^\W\d_])"  # no letter before
_NR = r"(?![^\W\d_])"  # no letter after

# Checked in this order: the first band that matches wins ("Senior Engineering Manager" is a lead,
# "Junior Data Analyst Intern" an intern).
_TITLE_BANDS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "intern",
        re.compile(
            _NL + r"(?:interns?|internship|praktikant(?:in)?|praktikum|werkstudent(?:in)?|working\s+student|"
            r"apprentice(?:ship)?|estagi[áa]ri[oa]|est[áa]gio|pasante|becari[oa]|practicante|pr[áa]cticas|"
            r"стаж[её]р\w*|стажировк\w*)" + _NR,
            re.IGNORECASE,
        ),
    ),
    (
        "junior",
        re.compile(
            _NL + r"(?:junior|jr\.?|jnr|graduate|new\s+grad|entry[- ]level|младш\w+|джуниор|джун|j[úu]nior)" + _NR,
            re.IGNORECASE,
        ),
    ),
    (
        "head",
        re.compile(
            _NL + r"(?:head\s+of|head,|vp|svp|vice\s+president|director|directora?|chief|cto|cio|ciso|cpo|"
            r"руководител\w+|директор\w*|начальник\w*|jefe|jefa|diretor[a]?|leiter(?:in)?)" + _NR,
            re.IGNORECASE,
        ),
    ),
    (
        "lead",
        re.compile(
            _NL + r"(?:(?:team|tech|technical|engineering|development|qa|design)\s+lead(?:er)?|lead(?!\s+generation)"
            r"|leader|тимлид|техлид|ведущ\w+|l[ií]der|(?:engineering|development|software|technical|it|qa|"
            r"data|platform|infrastructure|devops)\s+manager|manager,\s+(?:engineering|software|data|platform))" + _NR,
            re.IGNORECASE,
        ),
    ),
    ("principal", re.compile(_NL + r"(?:principal|distinguished|fellow|главн\w+)" + _NR, re.IGNORECASE)),
    ("staff", re.compile(_NL + r"(?:staff)(?!ing)" + _NR, re.IGNORECASE)),
    (
        "senior",
        re.compile(
            _NL + r"(?<!semi[ -])(?<!semi)(?:senior|sr\.?|snr|s[êe]nior|старш\w+|сеньор|синьор)" + _NR,
            re.IGNORECASE,
        ),
    ),
    (
        "mid",
        re.compile(
            _NL + r"(?:middle|mid[- ]level|mid|intermediate|semi[- ]?senior|ssr|pleno|мидл|миддл|ii|iii)" + _NR,
            re.IGNORECASE,
        ),
    ),
]

_BODY = re.compile(
    r"(?:seniority(?:\s+level)?|experience\s+level|level|nivel|n[íi]vel|уровень|grade)\s*:\s*(?P<value>[^\n,;]+)|"
    r"\bthis\s+is\s+an?\s+(?P<value2>junior|mid[- ]level|senior|entry[- ]level|lead|principal|staff)\s+"
    r"(?:role|position|opening)",
    re.IGNORECASE,
)


def seniority_from_title(title: str) -> Optional[str]:
    for band, pattern in _TITLE_BANDS:
        if pattern.search(title):
            return band
    return None


def extract_seniority(doc: Document, title: Optional[str]) -> Optional[Tuple[str, float]]:
    if title:
        band = seniority_from_title(title)
        if band:
            return band, 0.9
    for m in _BODY.finditer(doc.text):
        value = m.group("value") or m.group("value2") or ""
        band = seniority_from_title(value)
        if band:
            return band, 0.6
    return None
