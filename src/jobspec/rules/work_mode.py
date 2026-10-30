"""Work mode (remote / hybrid / onsite) and the regions a remote hire may live in."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from ..text import Document
from ..vocab import COUNTRIES, COUNTRY_CODES_EXACT, REGIONS, REGIONS_EXACT

_LABEL = re.compile(
    r"^\W*(?:work(?:place)?\s*(?:mode|type|model|arrangement|setup|location)|location\s*type|"
    r"remote\s*(?:policy|status)?|arbeitsmodell|arbeitsort|modalidad(?:\s+de\s+trabajo)?|modalidade|"
    r"формат(?:\s+работы)?|режим\s+работы|location|locations|standort|ubicaci[oó]n|localiza[cç][aã]o|"
    r"local|локация|город)\s*:\s*(?P<value>.+)$",
    re.IGNORECASE | re.MULTILINE,
)

_HYBRID = re.compile(
    r"\bhybrid\w*|\bh[ií]brid[oa]s?\b|гибрид\w*|"
    r"\b\d\s*(?:-\s*\d\s*)?days?\s+(?:a|per)\s+week\s+(?:in|at|from)\s+(?:the\s+|our\s+)?office|"
    r"\b\d\s*(?:-\s*\d\s*)?days?\s+(?:in|at|from)\s+(?:the\s+|our\s+)?office|"
    r"\b\d\s+tage?\s+(?:pro\s+woche\s+)?im\s+büro|"
    r"\b\d\s+d[ií]as?\s+(?:a\s+la\s+semana\s+|por\s+semana\s+)?(?:en|de|no|na)\s+(?:la\s+)?(?:oficina|escrit[óo]rio)|"
    r"\bwfo\s+\d\s+days|\b\d\s+days\s+wfo",
    re.IGNORECASE,
)
_REMOTE = re.compile(
    r"\bremote\b|\bwork(?:ing)?\s+from\s+(?:home|anywhere)\b|\bwfh\b|\bremot[oa]s?\b|\bteletrabajo\b|"
    r"\bhome\s*office\s+(?:only|100\s*%)|\bremote[- ]first\b|удал[её]нн?\w*|\bудал[её]нно\b|\bна\s+удал[её]нке\b|"
    r"\btrabalho\s+remoto\b|\b100\s*%\s*remote\b",
    re.IGNORECASE,
)
# "remote" that does not describe where the job is done
_REMOTE_FALSE = re.compile(
    r"\bremote\s+(?:sensing|monitoring|control|access|desktop|firmware|software\s+updates?|updates?|"
    r"diagnostics|teams?|colleagues|employees|workers|support\s+tools?|devices?|servers?|sites?)\b",
    re.IGNORECASE,
)
_NOT_REMOTE = re.compile(
    r"\bnot\s+(?:a\s+|an?\s+)?(?:fully\s+)?remote\b|\bno\s+remote\b|\bnon[- ]remote\b|"
    r"\bremote\s+(?:work\s+)?(?:is\s+)?not\s+(?:possible|available|an\s+option|offered)|"
    r"\bnicht\s+remote\b|\bkein(?:e)?\s+remote\b|\bno\s+es\s+remoto\b|\bnão\s+é\s+remoto\b|"
    r"\bне\s+удал[её]нн?\w*|\bбез\s+удал[её]нк\w*",
    re.IGNORECASE,
)
_ONSITE = re.compile(
    r"\bon[- ]?site\b|\bin[- ]office\b|\boffice[- ]based\b|\bin[- ]person\b|\bwork\s+from\s+(?:the\s+|our\s+)?office\b|"
    r"\bwfo\b|\bpresencial\b|\bvor\s+ort\b|\bв\s+офисе\b|\bофис\w*|\bonsite\b",
    re.IGNORECASE,
)
_STRICT_ONSITE = re.compile(
    r"\b(?:fully|100\s*%|strictly)\s+on[- ]?site\b|\bon[- ]?site\s+only\b|\b5\s+days\s+(?:a|per)\s+week\s+in\s+"
    r"(?:the\s+)?office|\bwfo\s+5\s+days|\b5\s+days\s+wfo",
    re.IGNORECASE,
)

_REGION_LEAD = re.compile(
    r"\bremote\b|\bremot[oa]\b|удал[её]нн?\w*|\b(?:live|living|reside|residing|based|located|resident)\s+(?:in|within)\b|"
    r"\bresidents?\s+of\b|\bcandidates?\s+(?:from|in|based\s+in|located\s+in)\b|\banywhere\s+in\b|"
    r"\bwithin\s+(?:the\s+)?\b|\bdesde\b|\bopen\s+to\s+(?:candidates\s+)?(?:in|from)\b|\bfrom\s+anywhere\b|"
    r"\bworldwide\b|\banywhere\s+in\s+the\s+world\b|\bglobally\b|\bтолько\b|\bresidir\s+en\b|\bmorar\s+no\b",
    re.IGNORECASE,
)


# Office days written as weekdays: "in our Boston office Tuesday through Thursday".
_WEEKDAYS: List[Tuple[str, int]] = [
    (r"mon(?:day)?s?|montags?|lunes|segunda(?:s)?-feiras?|понедельник\w*", 0),
    (r"tue(?:s(?:day)?)?s?|dienstags?|martes|ter[çc]a(?:s)?-feiras?|вторник\w*", 1),
    (r"wed(?:nesday)?s?|mittwochs?|mi[ée]rcoles|quarta(?:s)?-feiras?|сред[аыу]", 2),
    (r"thu(?:r(?:s(?:day)?)?)?s?|donnerstags?|jueves|quinta(?:s)?-feiras?|четверг\w*", 3),
    (r"fri(?:day)?s?|freitags?|viernes|sexta(?:s)?-feiras?|пятниц\w*", 4),
]
_DAY = re.compile(
    "|".join(rf"(?P<d{i}>(?<![^\W\d_])(?:{p})(?![^\W\d_]))" for i, (p, _) in enumerate(_WEEKDAYS)),
    re.IGNORECASE,
)
_DAY_RANGE_WORD = re.compile(r"^\s*(?:-|–|—|through|thru|to|until|till|bis|a|hasta|à|até|по|до)\s*$", re.IGNORECASE)
_OFFICE_CLAUSE = re.compile(
    r"\b(?:in|at|from|to|into)\s+(?:the\s+|our\s+|an?\s+)?(?:[\w-]+\s+){0,2}?(?:office|hq|headquarters|studio|campus)\b|"
    r"\bon[- ]?site\b|\bin[- ]person\b|\bim\s+(?:\w+\s+)?b[üu]ro\b|\b(?:en\s+la|na|no)\s+(?:\w+\s+)?"
    r"(?:oficina|escrit[óo]rio|sede)\b|\bв\s+офис\w*",
    re.IGNORECASE,
)
_HOME_CLAUSE = re.compile(
    r"\b(?:from|at)\s+home\b|\bremotely\b|\bwork(?:ing)?\s+remote\b|\bwfh\b|\bhome\s*office\b|\bvon\s+zu\s*hause\b|"
    r"\bmobil(?:es)?\s+arbeiten\b|\bdesde\s+casa\b|\bem\s+casa\b|\bудал[её]нно\b|\bиз\s+дома\b",
    re.IGNORECASE,
)
# "the office is open on Fridays if you like": a remote job with an optional office
_OPTIONAL = re.compile(
    r"\boptional\b|\bif\s+you\s+(?:like|want|wish)\b|\bwelcome\s+to\b|\boccasional(?:ly)?\b|\bclosed\b|"
    r"\bfreiwillig\b|\bopcional\b",
    re.IGNORECASE,
)


def _day_index(m: re.Match[str]) -> int:
    for i in range(len(_WEEKDAYS)):
        if m.group(f"d{i}"):
            return i
    return -1  # pragma: no cover - one group always matches


def weekday_hybrid(text: str) -> bool:
    """True when a sentence puts some, not all, weekdays in the office (or the rest at home)."""
    for sentence in re.split(r"[.;!?\n]+", text):
        days = list(_DAY.finditer(sentence))
        if not days or not _OFFICE_CLAUSE.search(sentence) or _OPTIONAL.search(sentence):
            continue
        if _HOME_CLAUSE.search(sentence):
            return True
        office: Set[int] = set()
        for a, b in zip(days, days[1:]):
            if _DAY_RANGE_WORD.match(sentence[a.end() : b.start()]):
                office.update(range(_day_index(a), _day_index(b) + 1))
        office.update(_day_index(m) for m in days)
        if 1 <= len(office) <= 4:
            return True
    return False


def _classify(value: str) -> Optional[str]:
    if _NOT_REMOTE.search(value):
        return "hybrid" if _HYBRID.search(value) or weekday_hybrid(value) else "onsite"
    if _HYBRID.search(value) or weekday_hybrid(value):
        return "hybrid"
    if _REMOTE.search(_REMOTE_FALSE.sub(" ", value)):
        return "remote"
    if _ONSITE.search(value):
        return "onsite"
    return None


def extract_work_mode(doc: Document, title: Optional[str] = None) -> Optional[Tuple[str, float]]:
    for m in _LABEL.finditer(doc.text):
        mode = _classify(m.group("value"))
        if mode:
            return mode, 0.95
    head = title if title is not None else (doc.lines[0] if doc.lines else "")
    mode = _classify(head)
    if mode:
        return mode, 0.9
    text = doc.text
    if _NOT_REMOTE.search(text):
        return ("hybrid", 0.8) if _HYBRID.search(text) or weekday_hybrid(text) else ("onsite", 0.8)
    if _HYBRID.search(text):
        return "hybrid", 0.85
    if weekday_hybrid(text):
        return "hybrid", 0.75
    remote = _REMOTE.search(_REMOTE_FALSE.sub(" ", text))
    if remote and not _STRICT_ONSITE.search(text):
        return "remote", 0.8
    if _ONSITE.search(text) or _STRICT_ONSITE.search(text):
        return "onsite", 0.75
    return None


def _place_patterns() -> List[Tuple[re.Pattern[str], str]]:
    pats: List[Tuple[re.Pattern[str], str]] = []
    ci: Dict[str, str] = dict(COUNTRIES)
    ci.update(REGIONS)
    for name, code in sorted(ci.items(), key=lambda kv: -len(kv[0])):
        pats.append((re.compile(r"(?<![^\W\d_])" + re.escape(name) + r"(?![^\W\d_])", re.IGNORECASE), code))
    exact: Dict[str, str] = dict(COUNTRY_CODES_EXACT)
    exact.update(REGIONS_EXACT)
    for name, code in exact.items():
        pats.append((re.compile(r"(?<![^\W\d_])" + re.escape(name) + r"(?![^\W\d_])"), code))
    return pats


_PLACES = _place_patterns()


def places_in(text: str) -> List[Tuple[int, str]]:
    """Return (position, code) for every country or region named in ``text``."""
    found: List[Tuple[int, str]] = []
    taken: List[Tuple[int, int]] = []
    for pattern, code in _PLACES:
        for m in pattern.finditer(text):
            if any(s < m.end() and m.start() < e for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), code))
    return sorted(found)


def extract_remote_regions(doc: Document) -> Optional[Tuple[List[str], float]]:
    """Regions named right after a remote statement or a residency requirement."""
    text = _REMOTE_FALSE.sub(" ", doc.text)
    codes: List[str] = []
    for m in _REGION_LEAD.finditer(text):
        lead = m.group(0).lower()
        if lead in ("worldwide", "globally") or lead.startswith("anywhere in the world"):
            if "WORLDWIDE" not in codes:
                codes.append("WORLDWIDE")
            continue
        window = text[m.end() : m.end() + 70]
        cut = re.search(r"\n|\.\s|;|\bbut\b|\boffices?\b|\bheadquarter", window)
        if cut:
            window = window[: cut.start()]
        for _, code in places_in(window):
            if code not in codes:
                codes.append(code)
    if not codes:
        return None
    if "WORLDWIDE" in codes and len(codes) > 1:
        codes = [c for c in codes if c != "WORLDWIDE"]
    return codes, 0.75
