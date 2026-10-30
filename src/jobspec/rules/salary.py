"""Salary: amount range, currency, period and gross/net.

Handles "$120k-150k", "€60.000 - €75.000", "₹18-24 LPA", "12,00,000 - 18,00,000 INR",
"120 000 - 150 000 руб.", "R$ 9.000 a R$ 12.000 por mês", "up to ...", "from ..." and
"competitive salary" (no salary). Amounts are returned in full units for one period.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..schema import Salary
from ..text import Document
from ..vocab import CURRENCY_SYMBOLS, CURRENCY_WORDS, MONTHLY_PAY_CURRENCIES


def _alternation(items: List[str]) -> str:
    return "|".join(re.escape(item) for item in sorted(items, key=len, reverse=True))


_NOT_LETTER_BEFORE = r"(?<![^\W\d_])"
_NOT_LETTER_AFTER = r"(?![^\W\d_])"
_SYMBOL = _alternation([symbol for symbol, _ in CURRENCY_SYMBOLS])
_WORD = _alternation(list(CURRENCY_WORDS))
_CUR = f"(?:{_SYMBOL}|{_NOT_LETTER_BEFORE}(?:{_WORD}){_NOT_LETTER_AFTER})"
# Indian grouping (12,00,000), grouped thousands (120,000 / 60.000 / 120 000 / 120'000), plain numbers.
_NUM = r"(?:\d{1,2}(?:,\d{2})+,\d{3}|\d{1,3}(?:[ ,.'’]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d+)?)"
_UNIT = rf"(?:\s?(?:lakhs?|lacs?|lpa|crores?|cr|k|тыс\.?|т\.р\.|mil|l){_NOT_LETTER_AFTER})?"
_SEP = r"(?:-|to|till|bis|até|ate|a|до|hasta|au|à)"

_CANDIDATE = re.compile(
    r"(?<![\w.,])"
    r"(?:(?P<c1>" + _CUR + r")\s?)?"
    r"(?P<a1>" + _NUM + r")(?P<u1>" + _UNIT + r")"
    r"(?:\s?(?P<c2>" + _CUR + r"))?"
    r"(?:\s*" + _SEP + r"\s*"
    r"(?:(?P<c3>" + _CUR + r")\s?)?"
    r"(?P<a2>" + _NUM + r")(?P<u2>" + _UNIT + r")"
    r")?"
    r"(?:\s?(?P<c4>" + _CUR + r"))?",
    re.IGNORECASE,
)

_BETWEEN = re.compile(
    r"\b(between|entre|zwischen|между)\s+(\S+(?: \d{3})*)\s+(?:and|y|und|e|и)\s+(?=\S*\d)", re.IGNORECASE
)

_SALARY_WORDS = re.compile(
    r"salar|\bpay\b|\bpaid\b|compensation|\bcomp\b|\bbase\b|\brate\b|\bwage|\bctc\b|stipend|package|"
    r"\bearn|\bote\b|gehalt|vergütung|verdienst|sueldo|remunera|retribuci|bolsa|зарплат|з/п|\bзп\b|"
    r"оклад|доход|вознагражд|заработн",
    re.IGNORECASE,
)
_NEGATIVE_BEFORE = re.compile(
    r"raised|funding|revenue|valuation|\barr\b|turnover|invest|bonus|allowance|budget|insurance|"
    r"coverage|reimburs|equipment|learning|wellness|\bgym\b|\bmeal|401|referral|series [a-e]\b|"
    r"customers|\busers\b|grant|prize|donat",
    re.IGNORECASE,
)
_NEGATIVE_AFTER = re.compile(
    r"^\s*(?:m\b|mm\b|bn\b|b\b|million|billion|mill?ones|млн|млрд|"
    r"(?:in|of)\s+(?:funding|revenue|arr|sales|investment)|"
    r"(?:sign(?:ing|-on)\s+|relocation\s+|joining\s+|referral\s+)?bonus|"
    r"(?:home[- ]office\s+|learning\s+|training\s+|wellness\s+|equipment\s+|education\s+)"
    r"(?:budget|allowance|stipend)|budget|allowance|customers|users|employees)",
    re.IGNORECASE,
)
_MAX_PREFIX = re.compile(r"(?:up\s*to|bis\s+zu|hasta|até|ate|до|max(?:imum)?\.?|not\s+more\s+than)\s*$", re.IGNORECASE)
_MIN_PREFIX = re.compile(
    r"(?:from|starting\s+(?:at|from)|\bab|desde|a\s+partir\s+de|от|min(?:imum)?\.?|at\s+least)\s*$",
    re.IGNORECASE,
)

_PERIODS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "hour",
        re.compile(
            r"/\s?(?:hour|hr|h|std|stunde|hora|час|ч)\b|\bper\s+hour\b|\ban\s+hour\b|\bhourly\b|"
            r"\bpro\s+stunde\b|\bstundenlohn\b|\bpor\s+hora\b|\bв\s+час\b|\bp/h\b",
            re.IGNORECASE,
        ),
    ),
    (
        "day",
        re.compile(
            r"/\s?(?:day|tag|día|dia|день)\b|\bper\s+day\b|\ba\s+day\b|\bdaily\b|\bday\s+rate\b|"
            r"\bpro\s+tag\b|\btagessatz\b|\bpor\s+d[ií]a\b|\bв\s+день\b",
            re.IGNORECASE,
        ),
    ),
    (
        "week",
        re.compile(
            r"/\s?(?:week|wk|woche|semana|неделю)\b|\bper\s+week\b|\ba\s+week\b|\bweekly\b|"
            r"\bpro\s+woche\b|\bpor\s+semana\b|\bв\s+неделю\b",
            re.IGNORECASE,
        ),
    ),
    (
        "month",
        re.compile(
            r"/\s?(?:month|mo|mon|monat|mes|mês|мес|месяц)\b|\bper\s+month\b|\ba\s+month\b|\bmonthly\b|"
            r"\bp\.m\.|\bpro\s+monat\b|\bmonatlich|\bmonatsgehalt|\bal\s+mes\b|\bpor\s+m[eê]s\b|"
            r"\bmensual(?:es)?\b|\bmensais\b|\bmensal\b|\bв\s+месяц\b|\bежемесячно|\bper\s+mensem\b",
            re.IGNORECASE,
        ),
    ),
    (
        "year",
        re.compile(
            r"/\s?(?:year|yr|annum|jahr|año|ano|год)\b|\bper\s+(?:year|annum)\b|\ba\s+year\b|"
            r"\bannual(?:ly)?\b|\byearly\b|\bp\.?\s?a\.?(?![^\W\d_])|\bpro\s+jahr\b|\bj[äa]hrlich|"
            r"\bjahresgehalt|\bal\s+a[ñn]o\b|\banual(?:es)?\b|\bpor\s+a[ñn]o\b|\banuais\b|\bв\s+год\b|"
            r"\bгодов|\blpa\b|\bctc\b",
            re.IGNORECASE,
        ),
    ),
]

_NET = re.compile(
    r"(?<![.\w])net(?:to|o|os|a|t)?\b(?!\s*(?:work|core|developer|framework))|на\s+руки|после\s+вычета|"
    r"l[ií]quido|after\s+tax|take[- ]home",
    re.IGNORECASE,
)
_GROSS = re.compile(r"\bgross\b|\bbrutt?o?s?\b|\bbruto?s?\b|до\s+вычета|before\s+tax|pre[- ]tax", re.IGNORECASE)

_LAKH_UNITS = ("lakh", "lakhs", "lac", "lacs", "lpa", "l")
_CRORE_UNITS = ("cr", "crore", "crores")
_THOUSAND_UNITS = ("k", "тыс", "тыс.", "т.р.", "mil")


@dataclass
class _Candidate:
    start: int
    end: int
    salary: Salary
    score: float
    confidence: float


def parse_number(raw: str) -> float:
    """Parse "120,000", "60.000", "120 000", "12,00,000", "1.5" or "2,5" into a float."""
    s = raw.replace(" ", "").replace("'", "").replace("’", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", "") if len(parts) > 2 or len(parts[-1]) == 3 else s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 2 or len(parts[-1]) == 3:
            s = s.replace(".", "")
    return float(s)


def _multiplier(unit: str) -> float:
    unit = unit.strip().lower()
    if not unit:
        return 1.0
    if unit in _LAKH_UNITS:
        return 100_000.0
    if unit in _CRORE_UNITS:
        return 10_000_000.0
    if unit in _THOUSAND_UNITS or unit.startswith("тыс"):
        return 1_000.0
    return 1.0


def _currency(token: Optional[str]) -> Tuple[Optional[str], bool]:
    """Map a currency token to (ISO code, is_bare_dollar)."""
    if not token:
        return None, False
    if token == "$":
        return "USD", True
    for symbol, code in CURRENCY_SYMBOLS:
        if token.lower() == symbol.lower():
            return code, False
    return CURRENCY_WORDS.get(token.lower()), False


def _sentence_after(text: str, end: int, size: int = 45) -> str:
    window = text[end : end + size]
    cut = re.search(r"\n|\.\s|;", window)
    return window[: cut.start()] if cut else window


def _sentence_before(text: str, start: int, size: int = 45) -> str:
    window = text[max(0, start - size) : start]
    cuts = [m.end() for m in re.finditer(r"\n|\.\s|;", window)]
    return window[cuts[-1] :] if cuts else window


def _find_period(before: str, after: str) -> Optional[str]:
    best: Optional[Tuple[int, str]] = None
    for period, pattern in _PERIODS:
        m = pattern.search(after)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), period)
    if best:
        return best[1]
    best_before: Optional[Tuple[int, str]] = None
    for period, pattern in _PERIODS:
        for m in pattern.finditer(before):
            if best_before is None or m.end() > best_before[0]:
                best_before = (m.end(), period)
    return best_before[1] if best_before else None


def _infer_period(currency: str, top: float) -> Optional[str]:
    if currency in ("JPY", "KRW", "IDR", "VND"):
        return None
    if currency == "INR":
        return "year" if top >= 200_000 else "month"
    if currency in MONTHLY_PAY_CURRENCIES:
        return "month"
    if top < 300:
        return "hour"
    if top < 1_500:
        return "day"
    if top < 15_000:
        return "month"
    return "year"


def _gross_net(text: str, start: int, end: int) -> Optional[str]:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    window = text[max(line_start, start - 80) : min(line_end, end + 80)]
    net, gross = bool(_NET.search(window)), bool(_GROSS.search(window))
    if net and not gross:
        return "net"
    if gross and not net:
        return "gross"
    return None


def _candidates(text: str) -> List[_Candidate]:
    out: List[_Candidate] = []
    for m in _CANDIDATE.finditer(text):
        tokens = [m.group(name) for name in ("c1", "c2", "c3", "c4")]
        found = [_currency(t) for t in tokens if t]
        units = (m.group("u1") or "").strip().lower(), (m.group("u2") or "").strip().lower()
        currency: Optional[str] = None
        for code, bare in found:
            if code and not bare:
                currency = code
                break
        if currency is None and found:
            currency = found[0][0]
        lakh = any(u in _LAKH_UNITS or u in _CRORE_UNITS for u in units)
        if currency is None and lakh:
            currency = "INR"
        if currency is None:
            continue

        try:
            a1 = parse_number(m.group("a1"))
            a2 = parse_number(m.group("a2")) if m.group("a2") else None
        except ValueError:
            continue
        u1, u2 = units
        if a2 is not None:
            if u2 and not u1 and a1 < 1000:
                u1 = u2
            if u1 and not u2 and a2 < 1000:
                u2 = u1
        a1 *= _multiplier(u1)
        if a2 is not None:
            a2 *= _multiplier(u2)

        before = _sentence_before(text, m.start(), 80)
        after_raw = text[m.end() : m.end() + 40]
        if _NEGATIVE_AFTER.search(after_raw):
            continue
        offset = max(0, len(before) - 40)
        neg = list(_NEGATIVE_BEFORE.finditer(before, offset))
        kws = list(_SALARY_WORDS.finditer(before))
        kw_close = bool(kws)
        if neg and (not kws or neg[-1].start() > kws[-1].start()):
            continue

        low: Optional[float] = a1
        high: Optional[float] = a1
        if a2 is not None:
            low, high = min(a1, a2), max(a1, a2)
        else:
            head = text[max(0, m.start() - 20) : m.start()]
            if _MAX_PREFIX.search(head):
                low = None
            elif _MIN_PREFIX.search(head):
                high = None
        top = high if high is not None else low
        if top is None or top <= 0 or top > 1e10:
            continue

        period = "year" if lakh else _find_period(_sentence_before(text, m.start()), _sentence_after(text, m.end()))
        explicit = period is not None
        if period is None:
            period = _infer_period(currency, top)
        salary = Salary(
            min=low,
            max=high,
            currency=currency,
            period=period,
            gross_net=_gross_net(text, m.start(), m.end()),
        )
        score = 1.0 + (3.0 if kw_close else 0.0) + (2.0 if a2 is not None else 0.0) + (2.0 if explicit else 0.0)
        if a2 is None and top < 100 and not kw_close and not explicit:
            continue  # a lone small amount ("$5 coffee") is not pay
        confidence = 0.55 + (0.15 if kw_close else 0.0) + (0.2 if explicit else 0.0) + (0.05 if a2 else 0.0)
        if not explicit:
            confidence = min(confidence, 0.65)
        out.append(_Candidate(m.start(), m.end(), salary, score, min(confidence, 0.95)))
    return out


def extract_salary_text(text: str) -> Optional[Tuple[Salary, float]]:
    """Find the most salary-like amount in ``text``; return it with a confidence, or None."""
    text = _BETWEEN.sub(lambda m: f"{m.group(1)} {m.group(2)} - ", text)
    candidates = _candidates(text)
    if not candidates:
        return None
    best = max(candidates, key=lambda c: (c.score, -c.start))
    return best.salary, best.confidence


def extract_salary(doc: Document) -> Optional[Tuple[Salary, float]]:
    return extract_salary_text(doc.text)
