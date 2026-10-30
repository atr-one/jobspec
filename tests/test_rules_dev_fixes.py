"""Rule fixes for the misses found on data/dev.jsonl (the 0.1.0 held-out set).

The cases are written fresh here, in several wordings each, rather than copied from dev.jsonl.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from jobspec.rules import RuleExtractor
from jobspec.rules.employment import extract_employment_type
from jobspec.rules.experience import extract_experience
from jobspec.rules.location import extract_locations
from jobspec.rules.visa import extract_visa
from jobspec.rules.work_mode import extract_remote_regions, extract_work_mode, weekday_hybrid

from .conftest import doc

# --- hybrid described by weekdays -------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Engineer\nYou will be in the office Tuesday to Thursday.",
        "Engineer\nThe team works from our Denver office Monday through Wednesday and from home the rest of the week.",
        "Engineer\nWe meet in the office on Tuesdays and Thursdays.",
        "Engineer\nOn-site Mon-Wed, remotely on Thursday and Friday.",
        "Entwickler\nDienstag bis Donnerstag arbeiten wir im Büro.",
        "Desarrollador\nTrabajamos en la oficina de martes a jueves.",
    ],
)
def test_hybrid_by_weekdays(text: str) -> None:
    found = extract_work_mode(doc(text))
    assert found is not None and found[0] == "hybrid"


@pytest.mark.parametrize(
    "text",
    [
        "Engineer\nYou will work in our office Monday to Friday, 9 to 5.",
        "Engineer\nFully remote. The Berlin office is open on Fridays if you like to drop by.",
        "Engineer\nWe ship on Tuesdays and review on Thursdays.",
    ],
)
def test_weekdays_that_do_not_mean_hybrid(text: str) -> None:
    assert not weekday_hybrid(text)


def test_full_week_in_the_office_stays_onsite() -> None:
    found = extract_work_mode(doc("Engineer\nOn-site role, in the office Monday to Friday."))
    assert found is not None and found[0] == "onsite"


# --- "N or more years" ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,years",
    [
        ("4 or more years of professional software engineering experience", 4),
        ("You have six or more years building data pipelines.", 6),
        ("Experience: 3 years or more with Python", 3),
        ("Al menos 2 o más años de experiencia", 2),
        ("5 ou mais anos de experiência", 5),
        ("Our company has been around for 10 or more years.", None),
    ],
)
def test_or_more_years(text: str, years: Optional[int]) -> None:
    found = extract_experience(doc(text))
    assert (found[0] if found else None) == years


# --- PJ (Brazilian contractor regime) ---------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Engenheiro de Dados\nContratação: PJ",
        "Engenheiro de Dados\nForma de contratação: PJ",
        "Desenvolvedor\nVaga PJ, 100% remota.",
        "Desenvolvedor\nRegime de contratação: PJ (pessoa jurídica)",
        "Desenvolvedor\nVocê será contratado como PJ.",
    ],
)
def test_pj_is_contract(text: str) -> None:
    found = extract_employment_type(doc(text), None)
    assert found is not None and found[0] == "contract"


def test_clt_is_still_full_time() -> None:
    found = extract_employment_type(doc("Desenvolvedor\nContratação: CLT"), None)
    assert found is not None and found[0] == "full_time"


# --- "B2B" as a market, not a contract --------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Backend Developer\nWe are a B2B SaaS company for restaurants.",
        "Account Executive\nExperience in B2B sales is a plus.",
        "Engineer\nOur B2B platform serves 400 retailers.",
        "Engineer\nWe build tools for the B2B e-commerce market.",
    ],
)
def test_b2b_market_is_not_a_contract(text: str) -> None:
    found = extract_employment_type(doc(text), None)
    assert found is None or found[0] != "contract"


@pytest.mark.parametrize(
    "text",
    [
        "Java Developer\nB2B contract, 20 000 - 26 000 PLN net per month.",
        "Java Developer\nForm of employment: B2B",
        "Java Developer\nWe offer B2B or UoP.",
    ],
)
def test_b2b_contract_is_still_a_contract(text: str) -> None:
    found = extract_employment_type(doc(text), None)
    assert found is not None and found[0] == "contract"


# --- work-permit sponsorship without the word "visa" ------------------------------------------


@pytest.mark.parametrize(
    "text,visa",
    [
        ("We can sponsor Critical Skills Employment Permits for eligible graduates.", True),
        ("We will sponsor your work permit.", True),
        ("We are happy to sponsor work authorization for the right candidate.", True),
        ("Work permit sponsorship is available.", True),
        ("Sponsorship of a work permit is possible for this role.", True),
        ("We offer relocation and work permit support.", True),
        ("We cannot sponsor work permits for this role.", False),
        ("Applicants must be authorized to work in the U.S.; we do not provide sponsorship.", False),
        ("We sponsor local hackathons and meetups.", None),
    ],
)
def test_permit_sponsorship(text: str, visa: Optional[bool]) -> None:
    found = extract_visa(doc(text))
    assert (found[0] if found else None) == visa


# --- locations --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,locations",
    [
        ("Data Analyst\n\nCiudad: Medellín\nModalidad: presencial", ["Medellín"]),
        ("Backend Developer | 3-5 Yrs | Pune\n\nPackage: 10 LPA", ["Pune"]),
        ("Аналитик\n\nМосква, работа в офисе\nПолная занятость", ["Москва"]),
        ("Graduate Engineer\n\nCork, Ireland - on-site\nSalary: €40,000", ["Cork, Ireland"]),
        ("ML Engineer\n\nVancouver, BC - Hybrid (two days in office)", ["Vancouver, BC"]),
        ("Engineer\n\nModelo: híbrido em Curitiba (3x por semana)", ["Curitiba"]),
        ("Engineer\n\nOur product team works from our Chicago office three days a week.", ["Chicago"]),
        ("Engineer\n\nThis is an on-site role in Lyon.", ["Lyon"]),
        ("Engineer\n\nWe are a team of 40 people. Apply below.", None),
        ("Engineer\n\nLondon Market experience preferred.", None),
    ],
)
def test_locations_without_a_label(text: str, locations: Optional[List[str]]) -> None:
    found = extract_locations(doc(text))
    assert (found[0] if found else None) == locations


def test_header_city_is_not_kept_for_a_remote_job() -> None:
    spec = RuleExtractor().extract("Engineer\n\nBerlin - Remote (EU)\nWe build things.").spec
    assert spec.work_mode == "remote" and spec.locations == []


# --- remote within Germany ("innerhalb Deutschlands") -----------------------------------------


def test_remote_region_in_german_genitive() -> None:
    found = extract_remote_regions(doc("Arbeitsort: 100 % remote innerhalb Deutschlands"))
    assert found is not None and found[0] == ["DE"]
