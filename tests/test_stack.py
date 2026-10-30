from __future__ import annotations

from typing import List

import pytest

from jobspec.rules import canonical_tech, extract_stack_text
from jobspec.vocab import TECHNOLOGIES


@pytest.mark.parametrize(
    "text,expected",
    [
        ("We run k8s on AWS", ["Kubernetes", "AWS"]),
        ("Postgres and Redis", ["PostgreSQL", "Redis"]),
        ("Strong JS and TS skills, Node.js a plus", ["JavaScript", "Node.js"]),
        ("Golang microservices", ["Go"]),
        ("Python 3, Django REST", ["Python", "Django"]),
        ("ASP.NET Core and .NET 8", ["ASP.NET", ".NET"]),
        ("React Native and React", ["React Native", "React"]),
        ("Spring Boot microservices in Java", ["Spring Boot", "Java"]),
        ("C# and C++", ["C#", "C++"]),
        ("Vue.js 3 with Nuxt", ["Vue.js", "Nuxt"]),
        ("PySpark jobs on Databricks", ["Spark", "Databricks"]),
        ("Ruby on Rails", ["Ruby on Rails"]),
    ],
)
def test_aliases(text: str, expected: List[str]) -> None:
    assert extract_stack_text(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        # Go: the language, or a verb
        ("Services written in Go and PostgreSQL", ["Go", "PostgreSQL"]),
        ("Python, Go, Rust", ["Python", "Go", "Rust"]),
        ("Go Developer", ["Go"]),
        ("Go beyond tickets and talk to users. Ready to go?", []),
        ("Go-to-market experience with Salesforce", ["Salesforce"]),
        # R: the language, or R&D / R$
        ("Build models in R or Python", ["R", "Python"]),
        ("Our R&D team uses Python", ["Python"]),
        ("Salário: R$ 9.000", []),
        ("R programming and SQL", ["R", "SQL"]),
        # C: the language, or Series C / C-level / grades
        ("Firmware in C and C++", ["C", "C++"]),
        ("C/C++ on Linux", ["C", "C++", "Linux"]),
        ("A Series C startup", []),
        ("Report to the C-level", []),
        ("English at C1 level", []),
        ("Objective-C and Swift", ["Objective-C", "Swift"]),
        # common words with a tech twin
        ("Swift and SwiftUI", ["Swift", "SwiftUI"]),
        ("Make a swift decision", []),
        ("Hiring starts this Spring.", []),
        ("Spring and Hibernate", ["Spring", "Hibernate"]),
        ("Please express interest by email", []),
        ("Node.js, Express and NestJS", ["Node.js", "Express", "NestJS"]),
        ("You react quickly to incidents", []),
        ("You excel at communication", []),
        ("Reports in Excel", ["Excel"]),
        ("Guard rails for the platform", []),
        ("Python-разработчик", ["Python"]),
        ("Java-Entwickler", ["Java"]),
    ],
)
def test_short_token_traps(text: str, expected: List[str]) -> None:
    assert extract_stack_text(text) == expected


def test_vocabulary_size_and_unique_names() -> None:
    names = [t.name for t in TECHNOLOGIES]
    assert len(names) >= 150
    assert len(names) == len(set(names))


def test_canonical_tech() -> None:
    assert canonical_tech("k8s") == "Kubernetes"
    assert canonical_tech("postgres") == "PostgreSQL"
    assert canonical_tech("golang") == "Go"
    assert canonical_tech("javascript") == "JavaScript"
    assert canonical_tech("Some New Thing") == "Some New Thing"
