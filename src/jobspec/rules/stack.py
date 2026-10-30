"""Tech stack: technologies from the curated vocabulary, with care for short ambiguous names.

"Go", "R", "C", "Swift", "Spring", "Express", "Dart" and "Unity" are also ordinary words or
letters. They count only when written with the expected case and either sit next to a
programming context ("written in Go", "R programming", "C/C++") or stand in a list with another
technology ("Python, R and SQL").
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Tuple

from ..text import Document
from ..vocab import TECHNOLOGIES, Tech


class _Matcher(NamedTuple):
    tech: Tech
    pattern: re.Pattern[str]
    exact: Optional[re.Pattern[str]]


def _alias_regex(aliases: Tuple[str, ...], ambiguous: bool) -> str:
    body = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    after = r"(?![\w#+&$]|-\w|\.\w)" if ambiguous else r"(?![\w#+])"
    return r"(?<![\w.#+])(?:" + body + ")" + after


def _build() -> List[_Matcher]:
    out: List[_Matcher] = []
    for tech in TECHNOLOGIES:
        ci = re.compile(_alias_regex(tech.aliases, tech.ambiguous), re.IGNORECASE) if tech.aliases else None
        exact = re.compile(_alias_regex(tech.exact, tech.ambiguous)) if tech.exact else None
        out.append(_Matcher(tech, ci if ci is not None else re.compile(r"(?!x)x"), exact))
    return out


_MATCHERS = _build()

_CONTEXT_BEFORE = re.compile(
    r"(?:\bin|\bwith|\busing|\bembedded|\bmodern|\bidiomatic|\bproficien\w*\s+in|\bwritten\s+in|"
    r"\bexperience\s+(?:in|with)|\bknowledge\s+of|\bprogramming\s+in|\bcoding\s+in|\bskills\s+in)\s+$",
    re.IGNORECASE,
)
_CONTEXT_AFTER = re.compile(
    r"^\s*(?:developers?|engineers?|programmers?|programming|language|code(?:base)?|services?|microservices|"
    r"backend|back-end|stack|librar(?:y|ies)|toolchain|ecosystem|shiny|studio|scripts?|apps?|framework|"
    r"modules?|firmware|\d+(?:\.\d+)?\b)",
    re.IGNORECASE,
)
_LIST_GAP = re.compile(
    r"^(?:[\s,/&()|+·•:]|\band\b|\bor\b|\bund\b|\boder\b|\by\b|\be\b|\bou\b|\bи\b|\bили\b)*$", re.IGNORECASE
)


def _spans(text: str) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
    """Return (clear matches, ambiguous matches) as (start, end, canonical name)."""
    clear: List[Tuple[int, int, str]] = []
    ambiguous: List[Tuple[int, int, str]] = []
    for matcher in _MATCHERS:
        tech = matcher.tech
        for m in matcher.pattern.finditer(text):
            clear.append((m.start(), m.end(), tech.name))
        if matcher.exact is not None:
            for m in matcher.exact.finditer(text):
                (ambiguous if tech.ambiguous else clear).append((m.start(), m.end(), tech.name))
    return clear, ambiguous


def _has_context(text: str, start: int, end: int, neighbours: List[Tuple[int, int, str]]) -> bool:
    if _CONTEXT_BEFORE.search(text[max(0, start - 30) : start]):
        return True
    if _CONTEXT_AFTER.search(text[end : end + 30]):
        return True
    for s, e, _ in neighbours:
        if e <= start and start - e <= 30 and _LIST_GAP.match(text[e:start]):
            return True
        if s >= end and s - end <= 30 and _LIST_GAP.match(text[end:s]):
            return True
    return False


def extract_stack_text(text: str) -> List[str]:
    clear, ambiguous = _spans(text)
    # drop matches inside a longer match ("React" inside "React Native", "Spring" in "Spring Boot")
    every = sorted(clear, key=lambda t: (t[0], -(t[1] - t[0])))
    kept: List[Tuple[int, int, str]] = []
    for span in every:
        if any(s <= span[0] and span[1] <= e and (s, e) != (span[0], span[1]) for s, e, _ in kept):
            continue
        kept.append(span)
    accepted = list(kept)
    # ambiguous names: resolve repeatedly so "C, C++ and Go" chains through accepted neighbours
    pending = [a for a in ambiguous if not any(s <= a[0] and a[1] <= e for s, e, _ in kept)]
    changed = True
    while changed and pending:
        changed = False
        for span in list(pending):
            if _has_context(text, span[0], span[1], accepted):
                accepted.append(span)
                pending.remove(span)
                changed = True
    first: Dict[str, int] = {}
    for start, _, name in accepted:
        if name not in first or start < first[name]:
            first[name] = start
    return sorted(first, key=lambda n: first[n])


def extract_stack(doc: Document) -> Optional[Tuple[List[str], float]]:
    names = extract_stack_text(doc.text)
    if not names:
        return None
    return names, 0.8


def canonical_tech(name: str) -> str:
    """Map an alias or differently cased name to the vocabulary's canonical name, if known."""
    key = name.strip().lower()
    for tech in TECHNOLOGIES:
        if key == tech.name.lower() or key in tech.aliases or key in (e.lower() for e in tech.exact):
            return tech.name
    return name.strip()
