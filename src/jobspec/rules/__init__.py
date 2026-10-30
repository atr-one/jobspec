"""The offline, deterministic rule extractor. One module per field group."""

from __future__ import annotations

import time
from typing import Optional

from ..base import ExtractionResult, Extractor
from ..schema import JobSpec
from ..text import Document, prepare
from .apply import extract_apply_email, extract_apply_url
from .employment import extract_employment_type
from .experience import extract_experience
from .location import extract_locations
from .salary import extract_salary, extract_salary_text, parse_number
from .seniority import extract_seniority, seniority_from_title
from .stack import canonical_tech, extract_stack, extract_stack_text
from .title import clean_title, extract_title
from .visa import extract_visa
from .work_mode import extract_remote_regions, extract_work_mode

__all__ = [
    "RuleExtractor",
    "canonical_tech",
    "clean_title",
    "extract_document",
    "extract_salary_text",
    "extract_stack_text",
    "parse_number",
    "seniority_from_title",
]


def extract_document(doc: Document) -> JobSpec:
    """Run every rule on a prepared document."""
    spec = JobSpec()
    raw_title_line = doc.lines[0] if doc.lines else None

    found_title = extract_title(doc)
    title = found_title[0] if found_title else None
    if found_title:
        spec.set("title", found_title[0], "rule", found_title[1])

    seniority = extract_seniority(doc, title)
    if seniority:
        spec.set("seniority", seniority[0], "rule", seniority[1])

    employment = extract_employment_type(doc, title)
    if employment:
        spec.set("employment_type", employment[0], "rule", employment[1])

    mode = extract_work_mode(doc, raw_title_line)
    if mode:
        spec.set("work_mode", mode[0], "rule", mode[1])
        if mode[0] == "remote":
            regions = extract_remote_regions(doc)
            if regions:
                spec.set("remote_regions", regions[0], "rule", regions[1])

    locations = extract_locations(doc)
    if locations and not (spec.work_mode == "remote" and locations[1] < 0.8):
        spec.set("locations", locations[0], "rule", locations[1])

    salary = extract_salary(doc)
    if salary:
        spec.set("salary", salary[0], "rule", salary[1])

    years = extract_experience(doc, raw_title_line)
    if years:
        spec.set("experience_years_min", years[0], "rule", years[1])

    stack = extract_stack(doc)
    if stack:
        spec.set("tech_stack", stack[0], "rule", stack[1])

    visa = extract_visa(doc)
    if visa:
        spec.set("visa_sponsorship", visa[0], "rule", visa[1])

    url = extract_apply_url(doc)
    if url:
        spec.set("apply_url", url[0], "rule", url[1])
    email = extract_apply_email(doc)
    if email:
        spec.set("apply_email", email[0], "rule", email[1])
    return spec


class RuleExtractor(Extractor):
    """Regex and vocabulary rules. No network, no randomness: the same input gives the same output."""

    name = "rules"

    def extract(self, text: str, html: Optional[bool] = None) -> ExtractionResult:
        started = time.perf_counter()
        spec = extract_document(prepare(text, html))
        warnings = spec.problems()
        return ExtractionResult(spec=spec, latency_s=time.perf_counter() - started, warnings=warnings)
