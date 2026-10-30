"""Work locations from "Location:" style lines, or an office city named in prose."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..text import Document
from ..vocab import CITIES

_LABEL = re.compile(
    r"^\W*(?:locations?|job\s+location|office(?:\s+location)?|standort|arbeitsort|ort|"
    r"ubicaci[oó]n|localizaci[oó]n|lugar(?:\s+de\s+trabajo)?|localiza[cç][aã]o|local(?:\s+de\s+trabalho)?|"
    r"cidade|ciudad|city|sede|город|локация|местоположение|место\s+работы)\s*:\s*(?P<value>.+)$",
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
# "from our Boston office", "híbrido em Belo Horizonte", "on-site in Lyon"
_PROSE_MORE = re.compile(
    r"\bour\s+(?P<city>" + _CITY_ALT + r")\s+(?:office|offices|hq|headquarters|hub|studio|campus|team)\b|"
    r"\b(?:hybrid|h[ií]brid[oa]|on[- ]?site|presencial|in[- ]office|in[- ]person|hybride?)"
    r"(?:\s+(?:role|position|job))?\s*,?\s+"
    r"(?:in|em|en|at|na|no|à|a)\s+(?:our\s+|the\s+)?(?P<city2>" + _CITY_ALT + r")(?![^\W\d_])",
    re.IGNORECASE,
)
# A header line, or one part of it, that is a city: "Dublin, Ireland - on-site", "Hyderabad",
# "Санкт-Петербург, работа в офисе", "Toronto, ON - Hybrid (three days in office)".
_HEADER_SPLIT = re.compile(r"\s+[|•·–—-]\s+|\s*[|•·]\s*")
_HEADER_CITY = re.compile(r"^(?P<city>" + _CITY_ALT + r")(?![^\W\d_])(?P<rest>.*)$", re.IGNORECASE)
_HEADER_REGION = re.compile(r"^\s*,\s*(?P<region>[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,2}|[A-Z]{2})\s*$")
_HEADER_LINES = 6


def _header_location(doc: Document) -> Optional[str]:
    for line in doc.lines[:_HEADER_LINES]:
        if len(line) > 120:
            continue
        for segment in _HEADER_SPLIT.split(line.strip()):
            segment = segment.strip(" .")
            if len(segment) > 60:
                continue
            m = _HEADER_CITY.match(segment)
            if not m:
                continue
            city = m.group("city")
            rest = _PARENS.sub("", m.group("rest")).rstrip(" .")
            if not rest.strip():
                return city
            region = _HEADER_REGION.match(rest)
            if region and not _MODE_WORDS.search(rest):
                return city + ", " + region.group("region")
            words = re.sub(r"[,;:()]", " ", rest).split()
            if _MODE_WORDS.search(rest) and len(words) <= 5 and re.match(r"^\s*[,(]", rest):
                return city
    return None


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
    header = _header_location(doc)
    if header:
        return [header], 0.7
    m2 = _PROSE.search(doc.text)
    if m2:
        place = m2.group("city") + (m2.group("rest") or "")
        return [place.strip()], 0.6
    m3 = _PROSE_MORE.search(doc.text)
    if m3:
        return [m3.group("city") or m3.group("city2")], 0.6
    return None
