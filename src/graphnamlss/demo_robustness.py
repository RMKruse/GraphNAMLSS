from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graphnamlss.demo import run_concept_demo_bundle
from graphnamlss.spillover_contrasts import (
    SPILLOVER_CHECKS,
    measure_spillover_contrasts,
)
from graphnamlss.testing import load_localized_congestion_spillover_demo_fixture


CONCEPT_DEMO_ROBUSTNESS_SEEDS: tuple[int, ...] = tuple(range(20250214, 20250224))
"""Declared Demo Fixture seeds: the default demo seed and the nine that follow."""

SUMMARY_FILENAME = "concept_demo_robustness_summary.json"

EVIDENCE_BOUNDARY = (
    "Multi-seed synthetic robustness is a teaching sanity check: it shows whether "
    "the Localized Congestion Spillover Demo Scenario reads the same way across "
    "Demo Fixture seeds. It stays inside the Concept Demo Evidence Boundary, is "
    "not METR-LA Merit Probe evidence, and cannot support the Further-Investment "
    "Gate."
)

COMPARISON = (
    "Per seed, downstream-sensor spillover prediction times are compared with "
    "routine prediction times from the Demo Fixture scenario, pooled over the "
    "train, validation and test splits of the Tracer-Bullet GraphNAMLSS run: "
    "target speed mean (expected lower during spillover) and standard deviation "
    "(expected higher), and mean gated mu (expected lower) and gated sigma "
    "(expected higher) graph contributions. A check holds when the "
    "spillover-minus-routine difference has the expected sign. "
    "gated_mu_share_of_slowdown divides the gated mu difference by the same "
    "difference of the standardized target mean, so it is the share of the "
    "standardized slowdown the gated mu graph path carries; it is reported, not "
    "enforced."
)

_GRAPH_RUN = Path("runs") / "tracer-bullet-graph-namlss"


def run_concept_demo_robustness_probe(
    output_directory: Path,
    *,
    seeds: Sequence[int] = CONCEPT_DEMO_ROBUSTNESS_SEEDS,
) -> dict[str, Any]:
    """Run the Concept Demo Bundle per seed and write one robustness summary."""

    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("The robustness probe needs at least one seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"The robustness probe seed set repeats a seed: {seeds}.")
    per_seed =[_run_seed(output_directory, seed) for seed in seeds]
    summary = {
        "probe": "concept_demo_robustness",
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "comparison": COMPARISON,
        "seed_set": (
            "declared" if seeds == CONCEPT_DEMO_ROBUSTNESS_SEEDS else "custom"
        ),
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _run_seed(output_directory: Path, seed: int) -> dict[str, Any]:
    bundle_directory = Path("seeds") / f"seed-{seed}"
    record: dict[str, Any] = {
        "seed": seed,
        "bundle_directory": bundle_directory.as_posix(),
    }
    # Any failure is recorded against its seed so the summary never drops a seed.
    try:
        run_concept_demo_bundle(output_directory / bundle_directory, seed=seed)
        scenario = load_localized_congestion_spillover_demo_fixture(
            seed=seed
        ).scenario
        contrasts = measure_spillover_contrasts(
            output_directory / bundle_directory / _GRAPH_RUN, scenario
        )
    except Exception as exc:  # noqa: BLE001
        return {**record, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return {**record, "status": "completed", **contrasts}


def _aggregate(per_seed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    checks = {}
    for name in SPILLOVER_CHECKS:
        evaluated = [record for record in per_seed if "checks" in record]
        checks[name] = {
            "held_on": [
                record["seed"] for record in evaluated if record["checks"][name]["held"]
            ],
            "not_held_on": [
                record["seed"]
                for record in evaluated
                if not record["checks"][name]["held"]
            ],
            "not_evaluated_on": [
                record["seed"] for record in per_seed if "checks" not in record
            ],
        }
    shares = [
        record["gated_mu_share_of_slowdown"]
        for record in per_seed
        if "gated_mu_share_of_slowdown" in record
    ]
    return {
        "checks": checks,
        "gated_mu_share_of_slowdown": {
            "min": min(shares, default=None),
            "max": max(shares, default=None),
        },
        "signal_holds_on_every_seed": all(
            not check["not_held_on"] and not check["not_evaluated_on"]
            for check in checks.values()
        ),
    }
