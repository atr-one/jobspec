from __future__ import annotations

from typing import List, Optional

import pytest

from jobspec.rules import RuleExtractor, clean_title, seniority_from_title
from jobspec.rules.apply import extract_apply_email, extract_apply_url
from jobspec.rules.employment import extract_employment_type
from jobspec.rules.experience import extract_experience
from jobspec.rules.location import extract_locations
from jobspec.rules.visa import extract_visa
from jobspec.rules.work_mode import extract_remote_regions, extract_work_mode

from .conftest import doc

# --- work mode -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,mode",
    [
        ("Backend Engineer\nThis is a fully remote role.", "remote"),
        ("Engineer\nWork from home, anywhere in Spain.", "remote"),
        ("Engineer\nWe work 3 days a week in the office, the rest from home.", "hybrid"),
        ("Engineer\nLocation: London (Hybrid)", "hybrid"),
        ("Ingeniero\nModalidad: Híbrido", "hybrid"),
        ("Разработчик\nФормат работы: удалённо", "remote"),
        ("Разработчик\nРабота в офисе в Москве.", "onsite"),
        ("Engineer\nThis role is on-site in Austin.", "onsite"),
        ("Engineer\nThis is not a remote position; you will work from our office.", "onsite"),
        ("Engineer\nOur devices get remote firmware updates. The job is on-site.", "onsite"),
        ("Python Developer\nWork mode: WFO, 5 days a week", "onsite"),
        ("Senior Engineer (Remote, EU)\nWe build things.", "remote"),
    ],
)
def test_work_mode(text: str, mode: str) -> None:
    found = extract_work_mode(doc(text))
    assert found is not None and found[0] == mode


def test_work_mode_absent() -> None:
    assert extract_work_mode(doc("Engineer\nWe build accounting software.")) is None


@pytest.mark.parametrize(
    "text,regions",
    [
        ("Remote (US)", ["US"]),
        ("Fully remote, open to candidates located in the United States.", ["US"]),
        ("We are remote-first, but you must live in Poland.", ["PL"]),
        ("Remote within the EU", ["EU"]),
        ("Remote - anywhere in the world", ["WORLDWIDE"]),
        ("Location: Remote - US or Canada", ["US", "CA"]),
        ("100% remoto desde cualquier lugar de México", ["MX"]),
        ("Remote. Our offices are in Germany and France.", None),
    ],
)
def test_remote_regions(text: str, regions: Optional[List[str]]) -> None:
    found = extract_remote_regions(doc(text))
    assert (found[0] if found else None) == regions


def test_remote_location_label_is_not_an_office() -> None:
    assert extract_locations(doc("Location: Remote - US or Canada")) is None
    found = extract_locations(doc("Location: Berlin (Hybrid) / Munich"))
    assert found is not None and found[0] == ["Berlin", "Munich"]
    found = extract_locations(doc("We are based in our Denver office."))
    assert found is not None and found[0] == ["Denver"]


# --- seniority, employment, experience ------------------------------------------------------


@pytest.mark.parametrize(
    "title,band",
    [
        ("Senior Backend Engineer", "senior"),
        ("Sr. Data Engineer", "senior"),
        ("Junior QA Engineer", "junior"),
        ("Software Engineering Intern", "intern"),
        ("Werkstudent Data Science", "intern"),
        ("Staff Software Engineer, Platform", "staff"),
        ("Principal Data Scientist", "principal"),
        ("Tech Lead, Payments", "lead"),
        ("Engineering Manager", "lead"),
        ("Head of Engineering", "head"),
        ("VP of Engineering", "head"),
        ("Site Reliability Engineer II", "mid"),
        ("Ingeniero de Datos Semi Senior", "mid"),
        ("Desenvolvedor Pleno", "mid"),
        ("Старший Python-разработчик", "senior"),
        ("Ведущий инженер", "lead"),
        ("Backend Engineer", None),
        ("Lead Generation Specialist", None),
        ("Staffing Coordinator", None),
    ],
)
def test_seniority_from_title(title: str, band: Optional[str]) -> None:
    assert seniority_from_title(title) == band


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Engineer\nEmployment type: Full Time", "full_time"),
        ("Engineer\nContract: Permanent, full-time", "full_time"),
        ("Engineer\nThis is a 6-month contract role.", "contract"),
        ("Engineer\nType: Full-time contractor", "contract"),
        ("Engineer\nB2B contract, invoice monthly", "contract"),
        ("Bookkeeper\nWe need a part-time bookkeeper.", "part_time"),
        ("Arbeitszeit: Teilzeit, 20 Stunden", "part_time"),
        ("Software Engineering Intern\nA 6-month internship with a possible full-time offer.", "internship"),
        ("Desarrollador\nJornada: Completa", "full_time"),
        ("Desenvolvedor\nRegime: CLT", "full_time"),
    ],
)
def test_employment_type(text: str, kind: str) -> None:
    d = doc(text)
    found = extract_employment_type(d, d.lines[0])
    assert found is not None and found[0] == kind


@pytest.mark.parametrize(
    "text,years",
    [
        ("5+ years of backend experience", 5),
        ("At least 3 years of experience", 3),
        ("3-5 years of experience in Python", 3),
        ("three years of professional experience", 3),
        ("Mindestens 5 Jahre Berufserfahrung", 5),
        ("Al menos 4 años de experiencia", 4),
        ("1 año de experiencia en QA", 1),
        ("3 anos de experiência com web", 3),
        ("опыт коммерческой разработки от 4 лет", 4),
        ("8+ years of engineering experience, including 3+ years on infrastructure", 8),
        ("The company was founded 12 years ago.", None),
        ("A 6-month internship.", None),
    ],
)
def test_experience(text: str, years: Optional[int]) -> None:
    found = extract_experience(doc(text))
    assert (found[0] if found else None) == years


# --- visa, apply, title ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,visa",
    [
        ("We are unable to sponsor visas at this time.", False),
        ("We do not offer visa sponsorship.", False),
        ("Candidates must be authorized to work without sponsorship.", False),
        ("We sponsor H-1B visas and transfers.", True),
        ("Employment Pass sponsorship is available.", True),
        ("We offer relocation support and visa sponsorship.", True),
        ("- Relocation support and visa sponsorship for candidates moving to Germany", True),
        ("Visa sponsorship: No", False),
        ("We have a great team.", None),
    ],
)
def test_visa(text: str, visa: Optional[bool]) -> None:
    found = extract_visa(doc(text))
    assert (found[0] if found else None) == visa


def test_apply_email_and_url() -> None:
    d = doc("Questions: info@acme.example\nSend your CV to jobs@acme.example.\nAbout us: https://acme.example")
    assert extract_apply_email(d) == ("jobs@acme.example", 0.9)
    assert extract_apply_url(d) is None
    d = doc("Apply here: https://acme.example/careers/42).")
    found = extract_apply_url(d)
    assert found is not None and found[0] == "https://acme.example/careers/42"
    d = doc("More at https://boards.greenhouse.io/acme/jobs/1")
    found = extract_apply_url(d)
    assert found is not None and found[0].startswith("https://boards.greenhouse.io")


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("Senior iOS Engineer - $160k-$190k + equity", "Senior iOS Engineer"),
        ("Senior Java-Entwickler (m/w/d)", "Senior Java-Entwickler"),
        ("Python Developer (3-5 yrs) - Bengaluru", "Python Developer"),
        ("DevOps Engineer - 6-month contract (Remote, EU)", "DevOps Engineer"),
        ("Customer Support Specialist (Remote, LATAM)", "Customer Support Specialist"),
        ("Staff Software Engineer, Platform", "Staff Software Engineer, Platform"),
        ("Backend Engineer (Node.js)", "Backend Engineer (Node.js)"),
        ("## Data Analyst", "Data Analyst"),
    ],
)
def test_clean_title(raw: str, clean: str) -> None:
    assert clean_title(raw) == clean


def test_html_input() -> None:
    html = (
        "<html><head><title>Ignored | Acme</title><style>p{color:red}</style></head><body>"
        "<h1>Platform Engineer</h1><p>Location: Remote (EU)</p>"
        "<ul><li>Kubernetes</li><li>Terraform</li></ul>"
        "<script>var salary='$1';</script>"
        '<a href="https://acme.example/jobs/1/apply">Apply now</a></body></html>'
    )
    spec = RuleExtractor().extract(html).spec
    assert spec.title == "Platform Engineer"
    assert spec.work_mode == "remote" and spec.remote_regions == ["EU"]
    assert spec.tech_stack == ["Kubernetes", "Terraform"]
    assert spec.salary is None
    assert spec.apply_url == "https://acme.example/jobs/1/apply"


def test_rule_extractor_is_deterministic_and_carries_meta() -> None:
    text = "Senior Go Engineer\nRemote (US)\nSalary: $150k-$180k per year\n5+ years with Go and k8s."
    a = RuleExtractor().extract(text).spec
    b = RuleExtractor().extract(text).spec
    assert a.to_dict(include_meta=True) == b.to_dict(include_meta=True)
    assert all(m.source == "rule" for m in a.meta.values())
    assert set(a.meta) == {f for f in a.to_dict() if not a.is_empty(f)}
