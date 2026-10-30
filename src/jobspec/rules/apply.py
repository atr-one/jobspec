"""Where to apply: an application URL and/or an email address."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..text import Document

_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_URL = re.compile(r"https?://[^\s<>\"'()\[\]]+", re.IGNORECASE)
_APPLY_WORDS = re.compile(
    r"appl(?:y|ication)|send\s+(?:us\s+)?(?:your\s+)?(?:cv|resume|résumé|application)|\bcv\b|resume|résumé|"
    r"bewerb|post[uú]la|envía|envia|candidat|currículo|curriculo|inscri|отклик|резюме|присылайте|отправ",
    re.IGNORECASE,
)
_APPLY_LINK_TEXT = re.compile(r"appl(?:y|ication)|bewerben|postular|candidatar|откликнуться|submit", re.IGNORECASE)
_RECRUITING_MAILBOX = re.compile(
    r"^(?:jobs?|careers?|hr|recruit\w*|talent|hiring|rh|vacanc\w*|karriere|bewerbung\w*|empleo|vagas|hh)@",
    re.IGNORECASE,
)
_ATS_URL = re.compile(
    r"apply|/jobs?/|/careers?/|/vacanc|/position|/empleos?/|/vagas?/|/karriere/|/stellen/|lever\.co|greenhouse\.io|workable\.com|ashbyhq\.com|"
    r"smartrecruiters|recruitee|personio|bamboohr|myworkdayjobs|breezy\.hr|teamtailor|jobs\.",
    re.IGNORECASE,
)


def _lead_in(text: str, start: int) -> str:
    """The words before ``start`` on its line, or the previous line when the line starts with it."""
    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[line_start:start]
    if prefix.strip(" -*:>\t"):
        return prefix
    previous_start = text.rfind("\n", 0, max(0, line_start - 1)) + 1
    return text[previous_start:line_start]


def _trim_url(url: str) -> str:
    return url.rstrip(".,;:!?)»”'\"")


def extract_apply_email(doc: Document) -> Optional[Tuple[str, float]]:
    text = doc.text
    emails: List[Tuple[int, str]] = [(m.start(), m.group(0).rstrip(".")) for m in _EMAIL.finditer(text)]
    if not emails:
        return None
    for start, email in emails:
        if _APPLY_WORDS.search(_lead_in(text, start)):
            return email, 0.9
    for _, email in emails:
        if _RECRUITING_MAILBOX.match(email):
            return email, 0.75
    return None


def extract_apply_url(doc: Document) -> Optional[Tuple[str, float]]:
    for href, label in doc.links:
        if href.lower().startswith("http") and _APPLY_LINK_TEXT.search(label):
            return _trim_url(href), 0.95
    text = doc.text
    urls = [(m.start(), _trim_url(m.group(0))) for m in _URL.finditer(text)]
    for start, url in urls:
        if _APPLY_WORDS.search(_lead_in(text, start)):
            return url, 0.85
    for _, url in urls:
        if _ATS_URL.search(url):
            return url, 0.7
    for href, _ in doc.links:
        if href.lower().startswith("http") and _ATS_URL.search(href):
            return _trim_url(href), 0.7
    return None
