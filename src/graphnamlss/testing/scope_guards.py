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
    # The fitted graph `mu` path is weak and seed-dependent under the
    # Tracer-Bullet Fit, so no wording may claim it reliably shows the spillover
    # slowdown. A negation between the path and the adverb is honest disclosure.
    re.compile(
        r"\b(?:graph\s+`?mu`?|gated\s+`?mu`?(?:\s+graph)?|`?mu`?\s+graph)\s+"
        r"(?:path|contribution|contrast|head|signal)s?\b"
        r"(?:(?!\b(?:not|never|cannot)\b|n't\b).){0,80}?"
        r"\b(?:reliabl[ey]|consistently|always|robustly|"
        r"on (?:every|all|each) (?:declared )?(?:demo fixture )?seeds?)\b"
        r".{0,80}?\bslow(?:down|er)\b",
        re.IGNORECASE,
    ),
    # The Localized Congestion Spillover Demo Scenario describes the data; how a
    # fitted model splits the slowdown between `mu` and `sigma` is a Graph Gate
    # Diagnostic, so no wording may promise it through the location parameter.
    re.compile(
        r"(?<!not )(?<!n't )(?<!never )"
        r"\b(?:shows?|exposes?|captures?|reveals?|carry|carries)\b.{0,80}?"
        r"\bslowdown through the location parameter\b",
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
