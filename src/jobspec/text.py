"""Input preparation: HTML to text, whitespace and dash normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import List, Optional, Tuple

_BLOCK_TAGS = frozenset(
    [
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "table",
        "section",
        "article",
        "header",
        "footer",
        "dt",
        "dd",
        "pre",
        "blockquote",
        "hr",
        "main",
        "aside",
        "nav",
    ]
)
_SKIP_TAGS = frozenset(["script", "style", "noscript", "template", "svg"])
_HTML_HINT = re.compile(r"<(?:p|div|br|li|ul|ol|h[1-6]|html|body|span|a|strong|b|table)(?:\s[^>]*)?/?>", re.IGNORECASE)
_SPACES = re.compile(r"[ \t    ​　]+")
_DASHES = re.compile(r"[‐‑‒–—―−]")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.links: List[Tuple[str, str]] = []
        self.h1: Optional[str] = None
        self.title: Optional[str] = None
        self._skip = 0
        self._href: Optional[str] = None
        self._link_text: List[str] = []
        self._capture: Optional[str] = None
        self._captured: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._link_text = []
        if tag in ("h1", "title") and self._capture is None:
            self._capture = tag
            self._captured = []

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "a" and self._href:
            self.links.append((self._href.strip(), " ".join("".join(self._link_text).split())))
            self._href = None
        if tag == self._capture:
            text = " ".join("".join(self._captured).split())
            if tag == "h1" and self.h1 is None and text:
                self.h1 = text
            if tag == "title" and self.title is None and text:
                self.title = text
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._capture is not None:
            self._captured.append(data)
        if self._href is not None:
            self._link_text.append(data)
        if self._capture == "title":
            return  # the <title> text is kept aside, not in the body
        self.parts.append(data)


@dataclass
class Document:
    """A posting ready for the rules: normalized text plus what the HTML told us."""

    raw: str
    text: str
    is_html: bool = False
    html_title: Optional[str] = None
    links: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def lines(self) -> List[str]:
        return [line for line in self.text.split("\n") if line.strip()]


def looks_like_html(text: str) -> bool:
    return bool(_HTML_HINT.search(text[:5000]))


def normalize(text: str) -> str:
    """Unify spaces and dashes, drop zero-width characters and trailing blanks, keep line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _DASHES.sub("-", text)
    text = _SPACES.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    out: List[str] = []
    for line in lines:
        if not line and out and not out[-1]:
            continue
        out.append(line)
    return "\n".join(out).strip()


def html_to_text(html: str) -> Tuple[str, Optional[str], List[Tuple[str, str]]]:
    """Return (text, title, links) of an HTML fragment or page using only the standard library."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = unescape("".join(parser.parts))
    return normalize(text), parser.h1 or parser.title, parser.links


def prepare(raw: str, html: Optional[bool] = None) -> Document:
    """Turn raw posting text (plain or HTML) into a ``Document``.

    ``html=None`` detects HTML by its tags.
    """
    is_html = looks_like_html(raw) if html is None else html
    if is_html:
        text, title, links = html_to_text(raw)
        return Document(raw=raw, text=text, is_html=True, html_title=title, links=links)
    return Document(raw=raw, text=normalize(raw))
