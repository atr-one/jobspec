from __future__ import annotations

import json

import pytest

from jobspec.schema import FIELDS, FieldMeta, JobSpec, Salary, SchemaError, json_schema


def full_spec() -> JobSpec:
    spec = JobSpec(
        title="Senior Backend Engineer",
        seniority="senior",
        employment_type="full_time",
        work_mode="remote",
        locations=["Berlin, Germany"],
        remote_regions=["EU", "PL"],
        salary=Salary(min=60000, max=75000.5, currency="EUR", period="year", gross_net="gross"),
        experience_years_min=5,
        tech_stack=["Go", "PostgreSQL"],
        visa_sponsorship=False,
        apply_url="https://example.com/apply",
        apply_email="jobs@example.com",
    )
    spec.meta["salary"] = FieldMeta("rule", 0.9)
    spec.meta["title"] = FieldMeta("llm", 0.7)
    return spec


def test_round_trip_with_meta() -> None:
    spec = full_spec()
    data = spec.to_dict(include_meta=True)
    back = JobSpec.from_dict(json.loads(json.dumps(data)))
    assert back == spec
    assert back.meta["salary"].source == "rule"
    assert back.meta["title"].confidence == pytest.approx(0.7)


def test_to_dict_has_every_field_and_integral_numbers() -> None:
    data = full_spec().to_dict()
    assert list(data) == list(FIELDS)
    assert data["salary"]["min"] == 60000 and isinstance(data["salary"]["min"], int)
    assert data["salary"]["max"] == 75000.5


def test_empty_spec_round_trip() -> None:
    assert JobSpec.from_dict({}) == JobSpec()
    assert JobSpec.from_json(JobSpec().to_json()) == JobSpec()


@pytest.mark.parametrize(
    "data",
    [
        {"seniority": "guru"},
        {"work_mode": "sometimes"},
        {"salary": {"min": 1, "max": 2, "currency": "DOLLARS"}},
        {"salary": {"min": 5, "max": 2}},
        {"salary": {"min": "100"}},
        {"salary": {"min": 1, "period": "fortnight"}},
        {"remote_regions": ["Europe"]},
        {"experience_years_min": 3.5},
        {"visa_sponsorship": "yes"},
        {"apply_email": "not an email"},
        {"apply_url": "ftp://example.com"},
        {"tech_stack": "Python"},
        {"unknown_field": 1},
        {"meta": {"title": {"source": "human", "confidence": 1}}},
    ],
)
def test_from_dict_rejects_invalid(data: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(SchemaError):
        JobSpec.from_dict(data)


def test_set_records_provenance_and_clamps() -> None:
    spec = JobSpec()
    spec.set("work_mode", "hybrid", "rule", 1.7)
    assert spec.meta["work_mode"].confidence == 1.0
    with pytest.raises(ValueError):
        spec.set("work_mode", "hybrid", "oracle", 0.5)
    with pytest.raises(KeyError):
        spec.set("salary_text", "x", "rule", 0.5)


def test_is_empty() -> None:
    spec = JobSpec(salary=Salary(currency="USD"), tech_stack=[])
    assert spec.is_empty("salary") and spec.is_empty("tech_stack") and spec.is_empty("title")
    spec.visa_sponsorship = False
    assert not spec.is_empty("visa_sponsorship")


def test_json_schema_shape() -> None:
    schema = json_schema()
    json.dumps(schema)
    assert schema["required"] == list(FIELDS)
    assert set(schema["properties"]) == set(FIELDS) | {"meta"}
    assert None in schema["properties"]["seniority"]["enum"]
    assert "WORLDWIDE" in schema["properties"]["remote_regions"]["items"]["pattern"]
