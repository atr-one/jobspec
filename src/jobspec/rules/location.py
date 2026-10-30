"""Work locations from "Location:" style lines, or an office city named in prose."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..text import Document
from ..vocab import CITIES

_LABEL = re.compile(
    r"^\W*(?:locations?|job\s+location|office(?:\s+location)?|standort|arbeitsort|ort|"
    r"ubicaci[oó]n|localizaci[oó]n|lugar(?:\s+de\s+trabajo)?|localiza[cç][aã]o|local(?:\s+de\s+trabalho)?|"
    r"cidade|город|локация|местоположение|место\s+работы)\s*:\s*(?P<value>.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_SPLIT = re.compile(r"\s*(?:;|\s/\s|\||\s+or\s+|\s+oder\s+|\s+ou\s+|\s+o\s+|\s+или\s+)\s*", re.IGNORECASE)
_MODE_WORDS = re.compile(
    r"\b(?:remote|hybrid|on[- ]?site|in[- ]office|wfh|wfo|remot[oa]|h[ií]brid[oa]|presencial|"
    r"teletrabajo|fully|100\s*%|anywhere|worldwide)\b|удал[её]нн?\w*|гибрид\w*|в\s+офисе|офис",
    re.IGNORECASE,
)
_PARENS = re.compile(r"\s*[(\[][^)\]]*[)\]]")
_CITY_ALT = "|".join(re.escape(c) for c in sorted(CITIES, key=len, reverse=True))
_PROSE = re.compile(
    r"(?:\b(?:office|offices|hq|headquarters|based|located|work\s+from\s+our|join\s+us)\s+(?:in|at)\s+"
    r"(?:our\s+)?(?:\w+\s+)?|\b(?:at|in)\s+our\s+)(?P<city>" + _CITY_ALT + r")(?![^\W\d_])"
    r"(?P<rest>,\s*[A-Z][\w ]{1,20}|,\s*[A-Z]{2}\b)?",
    re.IGNORECASE,
)


def _clean(item: str) -> Optional[str]:
    had_mode = bool(_MODE_WORDS.search(item))
    stripped = _PARENS.sub("", item)
    if _MODE_WORDS.search(stripped):
        return None  # "Remote (EU)", "Remote - US": a remote statement, not a place
    stripped = stripped.strip(" -,.:")
    if not stripped or (had_mode and not re.search(r"[^\W\d_]{2}", stripped)):
        return None
    return stripped


_REMOTE_WORD = re.compile(r"\bremot[eoa]\b|\bwfh\b|удал[её]нн?\w*|\banywhere\b|\bworldwide\b", re.IGNORECASE)


def extract_locations(doc: Document) -> Optional[Tuple[List[str], float]]:
    for m in _LABEL.finditer(doc.text):
        if _REMOTE_WORD.search(_PARENS.sub("", m.group("value"))):
            return None  # "Remote - US or Canada": where the hire may live, not an office
        items = [_clean(part) for part in _SPLIT.split(m.group("value"))]
        places = [item for item in items if item]
        if places:
            return places, 0.85
        # a label that only says "Remote" is an answer too: no office location
        if any(_MODE_WORDS.search(part) for part in _SPLIT.split(m.group("value"))):
            return None
    m2 = _PROSE.search(doc.text)
    if m2:
        place = m2.group("city") + (m2.group("rest") or "")
        return [place.strip()], 0.6
    return None
