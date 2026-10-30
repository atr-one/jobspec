"""Visa sponsorship: True when offered, False when ruled out, None when not mentioned."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from ..text import Document

_LABEL = re.compile(
    r"^\W*(?:visa(?:\s+sponsorship)?|sponsorship|visa\s+support)\s*:\s*(?P<value>.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_NO = re.compile(
    r"\b(?:no|not)\s+(?:(?:able|eligible)\s+(?:to\s+|for\s+))?(?:offer(?:ing)?\s+|provid(?:e|ing)\s+)?"
    r"(?:visa\s+)?sponsor(?:ship)?\b|"
    r"\b(?:unable|not\s+able)\s+to\s+(?:offer\s+|provide\s+)?(?:visa\s+)?sponsor\w*|"
    r"\b(?:cannot|can't|can\s+not|won't|will\s+not|do\s+not|don't|does\s+not)\s+"
    r"(?:offer\s+|provide\s+|support\s+)?(?:visa\s+)?sponsor\w*|"
    r"\bsponsorship\s+(?:is\s+)?(?:not\s+(?:available|offered|provided|possible)|unavailable)\b|"
    r"\bwithout\s+(?:the\s+need\s+for\s+|requiring\s+)?(?:current\s+or\s+future\s+)?(?:visa\s+)?sponsorship\b|"
    r"\bnot\s+(?:in\s+a\s+position\s+to\s+)?sponsor\b|"
    r"\bkeine\s+visa\w*|\bno\s+ofrecemos\s+patrocinio|\bsin\s+patrocinio\b|\bnão\s+oferecemos\s+(?:patroc[íi]nio|visto)|"
    r"\bбез\s+визов\w*\s+поддержк\w*",
    re.IGNORECASE,
)
_YES = re.compile(
    r"\bvisa\s+sponsorship\s+(?:is\s+)?(?:available|provided|offered|possible)\b|"
    r"\b(?:we|company)\s+(?:can\s+|will\s+|do\s+|happily\s+)?sponsor\w*\b[^.\n]{0,40}\bvisas?\b|"
    r"\bwe\s+(?:can\s+|will\s+|do\s+)?sponsor\s+(?:the\s+|your\s+)?(?:h-?1b|blue\s+card|work\s+permit|employment\s+pass|visa)\w*|"
    r"\b(?:offer|offers|offering|provide|provides|including|includes|with)\s+(?:full\s+)?(?:relocation\s+(?:and|&)\s+)?"
    r"visa\s+(?:sponsorship|support)\b|"
    r"\b(?:sponsorship|visa\s+support)\s+(?:is\s+)?(?:available|provided|offered)\b|"
    r"\b(?:h-?1b|employment\s+pass|blue\s+card|skilled\s+worker)\s+(?:visa\s+)?(?:sponsorship|support)\s+"
    r"(?:is\s+)?(?:available|provided|offered)\b|"
    r"\b(?:offer|offers|offering|provide|provides|including|includes)\b[^.\n]{0,40}\bvisa\s+(?:sponsorship|support)\b|"
    r"(?:^|\n)\W*(?:relocation\s+(?:support|assistance|package)\s+(?:and|&)\s+)?visa\s+(?:sponsorship|support)\b"
    r"(?!\s+(?:is\s+)?(?:not|unavailable))|"
    r"\bvisa\s+(?:sponsorship|support)\s+(?:and|&)\s+relocation\b|\brelocation\s+(?:and|&)\s+visa\s+(?:sponsorship|support)\b|"
    r"\bpatrocinio\s+de\s+visa\b|\bpatroc[íi]nio\s+de\s+visto\b|\bvisum\w*\s*(?:-|und)?\s*unterst[üu]tzung|"
    r"\bunterst[üu]tzung\s+(?:beim|bei\s+der)\s+visum|\bвизов\w*\s+поддержк\w*|\bпомо\w+\s+с\s+виз\w*",
    re.IGNORECASE,
)


def extract_visa(doc: Document) -> Optional[Tuple[bool, float]]:
    for m in _LABEL.finditer(doc.text):
        value = m.group("value").strip().lower()
        if re.match(r"(?:yes|available|provided|offered|sí|si|sim|да|ja)\b", value):
            return True, 0.95
        if re.match(r"(?:no|not|none|unavailable|nein|não|нет)\b", value):
            return False, 0.95
    text = doc.text
    if _NO.search(text):
        return False, 0.85
    if _YES.search(text):
        return True, 0.85
    return None
