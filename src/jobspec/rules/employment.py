"""Employment type: full-time, part-time, contract or internship."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..text import Document

_LABEL = re.compile(
    r"^\W*(?:employment(?:\s+type)?|job\s+type|contract(?:\s+type)?|type\s+of\s+(?:employment|contract)|"
    r"engagement|tipo\s+de\s+(?:contrato|empleo|vaga|jornada)|jornada|anstellungsart|besch[äa]ftigungsart|"
    r"vertragsart|arbeitszeit|regime(?:\s+de\s+contrata[cç][aã]o)?|тип\s+занятости|занятость|"
    r"формат\s+сотрудничества)\s*:\s*(?P<value>.+)$",
    re.IGNORECASE | re.MULTILINE,
)

_TYPES: List[Tuple[str, re.Pattern[str]]] = [
    (
        "internship",
        re.compile(
            r"\binternship\b|\bintern\b|\bpraktikum\b|\bpasant[ií]a\b|\bpr[áa]cticas\b|\best[áa]gio\b|"
            r"\bстажировк\w*",
            re.IGNORECASE,
        ),
    ),
    (
        "contract",
        re.compile(
            r"\bcontract(?:or)?\b(?!\s+type)|\bfreelance\w*|\bb2b\b|\bfixed[- ]term\b|\btemporary\b|\bcontrato\s+pj\b|"
            r"\bpj\b|\bcontratista\b|\bwerkvertrag\b|\bгпх\b|\bдоговор\s+подряда\b|\bсамозанят\w*|\bконтракт\b",
            re.IGNORECASE,
        ),
    ),
    (
        "part_time",
        re.compile(
            r"\bpart[- ]?time\b|\bteilzeit\b|\bmedio\s+tiempo\b|\btiempo\s+parcial\b|\bmedia\s+jornada\b|"
            r"\bmeio\s+per[íi]odo\b|\bmeio\s+expediente\b|\bчастичн\w+\s+занятост\w*|\bнеполн\w+\s+(?:рабоч\w+\s+)?(?:день|занятост\w*)",
            re.IGNORECASE,
        ),
    ),
    (
        "full_time",
        re.compile(
            r"^\s*completa\b|\bfull[- ]?time\b|\bpermanent\b|\bvollzeit\b|\bfestanstellung\b|\btiempo\s+completo\b|"
            r"\bjornada\s+completa\b|\bindefinido\b|\btempo\s+integral\b|\bclt\b|\bполн\w+\s+занятост\w*|"
            r"\bполный\s+(?:рабочий\s+)?день\b|\bштат\b|\bfte\b",
            re.IGNORECASE,
        ),
    ),
]

# Statements in the body strong enough to override a "full-time" mention elsewhere.
_STRONG_CONTRACT = re.compile(
    r"\b(?:\d+|six|three|twelve)[- ](?:month|week)s?\s+contract\b|\bcontract\s+(?:role|position|basis|engagement)\b|"
    r"\b(?:as\s+an?\s+|independent\s+|full[- ]time\s+)contractor\b|\bon\s+a\s+contract\b|\bb2b\b|"
    r"\bfreelance\b|\bday\s+rate\b|\boutside\s+ir35\b|\binside\s+ir35\b",
    re.IGNORECASE,
)


def _classify(value: str) -> Optional[str]:
    for name, pattern in _TYPES:
        if pattern.search(value):
            return name
    return None


def extract_employment_type(doc: Document, title: Optional[str]) -> Optional[Tuple[str, float]]:
    for m in _LABEL.finditer(doc.text):
        kind = _classify(m.group("value"))
        if kind:
            return kind, 0.95
    if title:
        kind = _classify(title)
        if kind:
            return kind, 0.9
    text = doc.text
    if _STRONG_CONTRACT.search(text):
        return "contract", 0.8
    found = {name: bool(pattern.search(text)) for name, pattern in _TYPES}
    if found["part_time"] and not found["full_time"]:
        return "part_time", 0.75
    if found["full_time"]:
        return "full_time", 0.75
    if found["internship"]:
        return "internship", 0.6
    if found["part_time"]:
        return "part_time", 0.6
    return None
