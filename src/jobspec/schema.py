"""The ``JobSpec`` record, its validation and its JSON Schema."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .vocab import CURRENCY_CODES, REGION_TOKENS

SENIORITY_LEVELS: Tuple[str, ...] = (
    "intern",
    "junior",
    "mid",
    "senior",
    "staff",
    "principal",
    "lead",
    "head",
)
EMPLOYMENT_TYPES: Tuple[str, ...] = ("full_time", "part_time", "contract", "internship")
WORK_MODES: Tuple[str, ...] = ("remote", "hybrid", "onsite")
SALARY_PERIODS: Tuple[str, ...] = ("hour", "day", "week", "month", "year")
GROSS_NET: Tuple[str, ...] = ("gross", "net")
SOURCES: Tuple[str, ...] = ("rule", "llm")

FIELDS: Tuple[str, ...] = (
    "title",
    "seniority",
    "employment_type",
    "work_mode",
    "locations",
    "remote_regions",
    "salary",
    "experience_years_min",
    "tech_stack",
    "visa_sponsorship",
    "apply_url",
    "apply_email",
)
LIST_FIELDS: Tuple[str, ...] = ("locations", "remote_regions", "tech_stack")
ENUM_FIELDS: Dict[str, Tuple[str, ...]] = {
    "seniority": SENIORITY_LEVELS,
    "employment_type": EMPLOYMENT_TYPES,
    "work_mode": WORK_MODES,
}

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
URL_RE = re.compile(r"^https?://[^\s/$.?#][^\s]*$", re.IGNORECASE)
COUNTRY_CODE_RE = re.compile(r"^[A-Z]{2}$")


class SchemaError(ValueError):
    """Raised by ``JobSpec.from_dict`` when a value does not fit the schema."""


def _number(value: Optional[float]) -> Optional[float]:
    """Return integral floats as ints so JSON output reads ``120000`` and not ``120000.0``."""
    if value is None:
        return None
    if float(value).is_integer():
        return int(value)
    return float(value)


@dataclass
class Salary:
    """A pay range. ``min`` and ``max`` are in full currency units for one ``period``."""

    min: Optional[float] = None
    max: Optional[float] = None
    currency: Optional[str] = None
    period: Optional[str] = None
    gross_net: Optional[str] = None

    def is_empty(self) -> bool:
        return self.min is None and self.max is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min": _number(self.min),
            "max": _number(self.max),
            "currency": self.currency,
            "period": self.period,
            "gross_net": self.gross_net,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Salary:
        if not isinstance(data, dict):
            raise SchemaError("salary must be an object or null")
        unknown = set(data) - {"min", "max", "currency", "period", "gross_net"}
        if unknown:
            raise SchemaError("unknown salary keys: {}".format(", ".join(sorted(unknown))))
        salary = cls(
            min=_float_or_none(data.get("min"), "salary.min"),
            max=_float_or_none(data.get("max"), "salary.max"),
            currency=data.get("currency"),
            period=data.get("period"),
            gross_net=data.get("gross_net"),
        )
        problems = salary.problems()
        if problems:
            raise SchemaError("; ".join(problems))
        return salary

    def problems(self) -> List[str]:
        out: List[str] = []
        for name in ("min", "max"):
            value = getattr(self, name)
            if value is not None and value < 0:
                out.append(f"salary.{name} must not be negative")
        if self.min is not None and self.max is not None and self.min > self.max:
            out.append("salary.min is greater than salary.max")
        if self.currency is not None and self.currency not in CURRENCY_CODES:
            out.append(f"salary.currency {self.currency!r} is not an ISO 4217 code")
        if self.period is not None and self.period not in SALARY_PERIODS:
            out.append("salary.period {!r} is not one of {}".format(self.period, ", ".join(SALARY_PERIODS)))
        if self.gross_net is not None and self.gross_net not in GROSS_NET:
            out.append(f"salary.gross_net {self.gross_net!r} is not gross or net")
        return out


def _float_or_none(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{name} must be a number or null")
    return float(value)


@dataclass
class FieldMeta:
    """Where a field value came from and how sure the extractor is (0..1)."""

    source: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "confidence": round(self.confidence, 3)}


@dataclass
class JobSpec:
    """A structured job posting. Unknown values are ``None`` (scalars) or empty lists."""

    title: Optional[str] = None
    seniority: Optional[str] = None
    employment_type: Optional[str] = None
    work_mode: Optional[str] = None
    locations: List[str] = field(default_factory=list)
    remote_regions: List[str] = field(default_factory=list)
    salary: Optional[Salary] = None
    experience_years_min: Optional[int] = None
    tech_stack: List[str] = field(default_factory=list)
    visa_sponsorship: Optional[bool] = None
    apply_url: Optional[str] = None
    apply_email: Optional[str] = None
    meta: Dict[str, FieldMeta] = field(default_factory=dict, compare=False)

    # field access -------------------------------------------------------------------------

    def get(self, name: str) -> Any:
        if name not in FIELDS:
            raise KeyError(name)
        return getattr(self, name)

    def set(self, name: str, value: Any, source: str, confidence: float) -> None:
        """Set a field together with its provenance."""
        if name not in FIELDS:
            raise KeyError(name)
        if source not in SOURCES:
            raise ValueError("source must be one of {}".format(", ".join(SOURCES)))
        setattr(self, name, value)
        self.meta[name] = FieldMeta(source, max(0.0, min(1.0, confidence)))

    def is_empty(self, name: str) -> bool:
        value = self.get(name)
        if value is None:
            return True
        if isinstance(value, list):
            return not value
        if isinstance(value, Salary):
            return value.is_empty()
        return False

    def confidence(self, name: str) -> float:
        meta = self.meta.get(name)
        return meta.confidence if meta else 0.0

    # serialization ------------------------------------------------------------------------

    def to_dict(self, include_meta: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "title": self.title,
            "seniority": self.seniority,
            "employment_type": self.employment_type,
            "work_mode": self.work_mode,
            "locations": list(self.locations),
            "remote_regions": list(self.remote_regions),
            "salary": self.salary.to_dict() if self.salary is not None else None,
            "experience_years_min": self.experience_years_min,
            "tech_stack": list(self.tech_stack),
            "visa_sponsorship": self.visa_sponsorship,
            "apply_url": self.apply_url,
            "apply_email": self.apply_email,
        }
        if include_meta:
            data["meta"] = {name: self.meta[name].to_dict() for name in FIELDS if name in self.meta}
        return data

    def to_json(self, include_meta: bool = False, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(include_meta), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JobSpec:
        """Build a record from a dict, raising ``SchemaError`` on any invalid value.

        Missing keys mean unknown. A ``meta`` key, as written by ``to_dict(include_meta=True)``,
        is read back.
        """
        if not isinstance(data, dict):
            raise SchemaError("a job spec must be an object")
        unknown = set(data) - set(FIELDS) - {"meta"}
        if unknown:
            raise SchemaError("unknown keys: {}".format(", ".join(sorted(unknown))))
        spec = cls()
        for name in ("title", "apply_url", "apply_email", "seniority", "employment_type", "work_mode"):
            value = data.get(name)
            if value is not None and not isinstance(value, str):
                raise SchemaError(f"{name} must be a string or null")
            setattr(spec, name, value)
        for name in LIST_FIELDS:
            value = data.get(name) or []
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise SchemaError(f"{name} must be a list of strings")
            setattr(spec, name, list(value))
        salary = data.get("salary")
        spec.salary = Salary.from_dict(salary) if salary is not None else None
        years = data.get("experience_years_min")
        if years is not None and (isinstance(years, bool) or not isinstance(years, int)):
            raise SchemaError("experience_years_min must be an integer or null")
        spec.experience_years_min = years
        visa = data.get("visa_sponsorship")
        if visa is not None and not isinstance(visa, bool):
            raise SchemaError("visa_sponsorship must be true, false or null")
        spec.visa_sponsorship = visa
        meta = data.get("meta") or {}
        if not isinstance(meta, dict):
            raise SchemaError("meta must be an object")
        for name, item in meta.items():
            if name not in FIELDS or not isinstance(item, dict):
                raise SchemaError(f"invalid meta entry {name!r}")
            source, confidence = item.get("source"), item.get("confidence")
            if source not in SOURCES or not isinstance(confidence, (int, float)):
                raise SchemaError(f"invalid meta entry {name!r}")
            spec.meta[name] = FieldMeta(source, float(confidence))
        problems = spec.problems()
        if problems:
            raise SchemaError("; ".join(problems))
        return spec

    @classmethod
    def from_json(cls, text: str) -> JobSpec:
        return cls.from_dict(json.loads(text))

    # validation ---------------------------------------------------------------------------

    def problems(self) -> List[str]:
        """Return a list of schema violations (empty when the record is valid)."""
        out: List[str] = []
        for name, allowed in ENUM_FIELDS.items():
            value = getattr(self, name)
            if value is not None and value not in allowed:
                out.append("{} {!r} is not one of {}".format(name, value, ", ".join(allowed)))
        for region in self.remote_regions:
            if region not in REGION_TOKENS and not COUNTRY_CODE_RE.match(region):
                out.append(f"remote_regions entry {region!r} is not a country code or region token")
        if self.salary is not None:
            out.extend(self.salary.problems())
        years = self.experience_years_min
        if years is not None and not 0 <= years <= 60:
            out.append("experience_years_min must be between 0 and 60")
        if self.apply_email is not None and not EMAIL_RE.match(self.apply_email):
            out.append(f"apply_email {self.apply_email!r} is not an email address")
        if self.apply_url is not None and not URL_RE.match(self.apply_url):
            out.append(f"apply_url {self.apply_url!r} is not an http(s) URL")
        return out

    def is_valid(self) -> bool:
        return not self.problems()


def json_schema() -> Dict[str, Any]:
    """Return the JSON Schema (draft 2020-12) of ``JobSpec.to_dict()``."""

    def nullable(schema: Dict[str, Any]) -> Dict[str, Any]:
        types = schema.get("type")
        if isinstance(types, str):
            schema = dict(schema, type=[types, "null"])
        if "enum" in schema:
            schema = dict(schema, enum=[*list(schema["enum"]), None])
        return schema

    region_pattern = "^([A-Z]{{2}}|{})$".format("|".join(sorted(REGION_TOKENS)))
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/atr-one/jobspec/schema/jobspec.json",
        "title": "JobSpec",
        "description": "A structured job posting as produced by jobspec.",
        "type": "object",
        "additionalProperties": False,
        "required": list(FIELDS),
        "properties": {
            "title": nullable({"type": "string", "description": "Job title without salary or location."}),
            "seniority": nullable({"type": "string", "enum": list(SENIORITY_LEVELS)}),
            "employment_type": nullable({"type": "string", "enum": list(EMPLOYMENT_TYPES)}),
            "work_mode": nullable({"type": "string", "enum": list(WORK_MODES)}),
            "locations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Work locations as written (city, region or country).",
            },
            "remote_regions": {
                "type": "array",
                "items": {"type": "string", "pattern": region_pattern},
                "description": "Where a remote hire may live: ISO 3166-1 alpha-2 codes or region tokens.",
            },
            "salary": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["min", "max", "currency", "period", "gross_net"],
                "properties": {
                    "min": {"type": ["number", "null"], "minimum": 0},
                    "max": {"type": ["number", "null"], "minimum": 0},
                    "currency": {
                        "type": ["string", "null"],
                        "pattern": "^[A-Z]{3}$",
                        "description": "ISO 4217 code.",
                    },
                    "period": nullable({"type": "string", "enum": list(SALARY_PERIODS)}),
                    "gross_net": nullable({"type": "string", "enum": list(GROSS_NET)}),
                },
            },
            "experience_years_min": {"type": ["integer", "null"], "minimum": 0, "maximum": 60},
            "tech_stack": {"type": "array", "items": {"type": "string"}},
            "visa_sponsorship": {"type": ["boolean", "null"]},
            "apply_url": {"type": ["string", "null"], "format": "uri"},
            "apply_email": {"type": ["string", "null"], "format": "email"},
            "meta": {
                "type": "object",
                "description": "Optional provenance per field.",
                "additionalProperties": {
                    "type": "object",
                    "required": ["source", "confidence"],
                    "properties": {
                        "source": {"type": "string", "enum": list(SOURCES)},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
        },
    }
