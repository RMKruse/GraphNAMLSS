from __future__ import annotations

import re


_PROHIBITED_CLAIMS = (
    re.compile(
        r"\bsynthetic (?:concept )?demo(?: bundle)? metrics\b.{0,100}"
        r"\b(?:show|prove|demonstrate|establish|confirm)\b.{0,100}"
        r"\b(?:outperforms?|superior|improv(?:e|es|ed) calibration)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bgraphnamlss\b.{0,100}\b(?:outperforms?|is superior)\b.{0,100}"
        r"\b(?:based on|using) synthetic (?:concept )?demo(?: bundle)? metrics\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:concept demonstrator|concept demo bundle)\b\s+"
        r"(?:currently\s+)?(?:supports?|implements?|includes?|provides?|reports?|"
        r"demonstrates?)\b.{0,100}\b(?:bayesian graph streams?|"
        r"structural uncertainty|medical application results?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bconcept demo bundle\b\s+(?:currently\s+)?"
        r"(?:includes?|integrates?|implements?|provides?)\b.{0,100}"
        r"\b(?:gamlss|bamlss)\b",
        re.IGNORECASE,
    ),
)


def assert_concept_demo_scope(text: str) -> None:
    """Reject claims that exceed the current Concept Demonstrator evidence scope."""

    normalized = " ".join(text.split())
    for prohibited_claim in _PROHIBITED_CLAIMS:
        match = prohibited_claim.search(normalized)
        if match is not None:
            raise AssertionError(
                "Concept Demonstrator scope violation: " + match.group(0)
            )
