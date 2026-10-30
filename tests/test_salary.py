from __future__ import annotations

from typing import Optional

import pytest

from jobspec.rules import extract_salary_text, parse_number


@pytest.mark.parametrize(
    "raw,value",
    [
        ("120,000", 120000),
        ("60.000", 60000),
        ("120 000", 120000),
        ("120'000", 120000),
        ("12,00,000", 1200000),
        ("1.5", 1.5),
        ("2,5", 2.5),
        ("1.234,56", 1234.56),
        ("1,234.56", 1234.56),
    ],
)
def test_parse_number(raw: str, value: float) -> None:
    assert parse_number(raw) == pytest.approx(value)


@pytest.mark.parametrize(
    "text,low,high,currency,period,gross_net",
    [
        ("Salary: $120k-150k", 120000, 150000, "USD", "year", None),
        ("Compensation: $150,000 - $180,000 per year plus equity", 150000, 180000, "USD", "year", None),
        ("Gehalt: €60.000 - €75.000 brutto/Jahr", 60000, 75000, "EUR", "year", "gross"),
        ("Gehalt: 70.000 € - 85.000 € brutto pro Jahr", 70000, 85000, "EUR", "year", "gross"),
        ("CTC: ₹18-24 LPA", 1800000, 2400000, "INR", "year", None),
        ("Salary: 12,00,000 - 18,00,000 INR per annum", 1200000, 1800000, "INR", "year", None),
        ("Stipend: Rs. 25,000 per month", 25000, 25000, "INR", "month", None),
        ("зарплата 120 000 - 150 000 руб.", 120000, 150000, "RUB", "month", None),
        ("зарплата от 250 000 до 320 000 руб. на руки", 250000, 320000, "RUB", "month", "net"),
        ("Salário: R$ 9.000 a R$ 12.000 por mês", 9000, 12000, "BRL", "month", None),
        ("Salario: 45.000 - 55.000 € brutos anuales", 45000, 55000, "EUR", "year", "gross"),
        ("Sueldo: $35,000 - $45,000 MXN mensuales netos", 35000, 45000, "MXN", "month", "net"),
        ("Salary: PLN 20 000 - 26 000 net/month + VAT", 20000, 26000, "PLN", "month", "net"),
        ("Pay: $28-$32/hour", 28, 32, "USD", "hour", None),
        ("Rate: €550-650 per day", 550, 650, "EUR", "day", None),
        ("Salary: S$220k - S$260k per annum", 220000, 260000, "SGD", "year", None),
        ("Salary: A$120,000 - A$140,000 + super", 120000, 140000, "AUD", "year", None),
        ("Salary range: CA$190,000 - CA$230,000", 190000, 230000, "CAD", "year", None),
        ("Pay: £40,000 - £48,000 per annum", 40000, 48000, "GBP", "year", None),
        ("Salary between $90,000 and $110,000", 90000, 110000, "USD", "year", None),
        ("Lohn: CHF 120'000 - 140'000", 120000, 140000, "CHF", "year", None),
    ],
)
def test_salary_formats(
    text: str, low: float, high: float, currency: str, period: str, gross_net: Optional[str]
) -> None:
    found = extract_salary_text(text)
    assert found is not None
    salary, confidence = found
    assert (salary.min, salary.max) == (low, high)
    assert salary.currency == currency
    assert salary.period == period
    assert salary.gross_net == gross_net
    assert 0 < confidence <= 1


def test_up_to_is_max_only_and_from_is_min_only() -> None:
    found = extract_salary_text("Зарплата: до 90 000 ₽ до вычета налогов.")
    assert found is not None
    salary = found[0]
    assert (salary.min, salary.max, salary.gross_net) == (None, 90000, "gross")
    found = extract_salary_text("Salary: from €50,000 per year")
    assert found is not None
    assert (found[0].min, found[0].max) == (50000, None)


def test_single_amount_is_min_and_max() -> None:
    found = extract_salary_text("Bolsa-auxílio: R$ 1.800/mês")
    assert found is not None
    assert (found[0].min, found[0].max, found[0].period) == (1800, 1800, "month")


@pytest.mark.parametrize(
    "text",
    [
        "Competitive salary and stock options.",
        "We raised $25M in our Series B.",
        "We offer a $2,000 learning budget and a home office stipend.",
        "Retirement plan with 401(k) match.",
        "Our customers process $3 billion a year.",
        "3-5 years of experience required.",
    ],
)
def test_no_salary(text: str) -> None:
    assert extract_salary_text(text) is None


def test_prefers_the_salary_over_other_amounts() -> None:
    text = "We raised $40M last year.\nSalary: $140,000 - $165,000 per year.\nSigning bonus $10,000."
    found = extract_salary_text(text)
    assert found is not None
    assert (found[0].min, found[0].max) == (140000, 165000)


def test_inferred_period_has_lower_confidence() -> None:
    stated = extract_salary_text("Salary: $150,000 - $180,000 per year")
    inferred = extract_salary_text("Salary: $150,000 - $180,000")
    assert stated is not None and inferred is not None
    assert inferred[0].period == "year"
    assert inferred[1] < stated[1]
