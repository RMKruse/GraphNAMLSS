from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from graphnamlss.demo import run_concept_demo_bundle
from graphnamlss.testing import (
    LocalizedCongestionSpilloverScenario,
    load_localized_congestion_spillover_demo_fixture,
)


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
    "spillover-minus-routine difference has the expected sign."
)

_GRAPH_RUN = Path("runs") / "tracer-bullet-graph-namlss"

# name -> (artifact, row value, statistic, expected spillover direction)
_CHECKS: dict[str, tuple[str, str, Callable[[Sequence[float]], float], str]] = {
    "target_location": ("predictions.json", "target", fmean, "lower"),
    "target_variability": ("predictions.json", "target", stdev, "higher"),
    "gated_mu_contribution": ("graph_contributions.json", "gated_mu", fmean, "lower"),
    "gated_sigma_contribution": (
        "graph_contributions.json",
        "gated_sigma",
        fmean,
        "higher",
    ),
}


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
        checks = _spillover_checks(
            output_directory / bundle_directory / _GRAPH_RUN, scenario
        )
    except Exception as exc:  # noqa: BLE001
        return {**record, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return {**record, "status": "completed", "checks": checks}


def _spillover_checks(
    graph_run: Path, scenario: LocalizedCongestionSpilloverScenario
) -> dict[str, dict[str, Any]]:
    artifacts = {
        artifact: json.loads((graph_run / artifact).read_text())
        for artifact in {artifact for artifact, *_ in _CHECKS.values()}
    }
    checks = {}
    for name, (artifact, value_key, statistic, direction) in _CHECKS.items():
        spillover = statistic(
            _downstream_values(
                artifacts[artifact],
                scenario.downstream_sensor_id,
                scenario.spillover_prediction_time_indices,
                value_key,
            )
        )
        routine = statistic(
            _downstream_values(
                artifacts[artifact],
                scenario.downstream_sensor_id,
                scenario.routine_prediction_time_indices,
                value_key,
            )
        )
        difference = spillover - routine
        checks[name] = {
            "spillover": round(spillover, 6),
            "routine": round(routine, 6),
            "difference": round(difference, 6),
            "held": difference < 0.0 if direction == "lower" else difference > 0.0,
        }
    return checks


def _downstream_values(
    artifact: dict[str, Any],
    sensor_id: str,
    time_indices: Sequence[int],
    value_key: str,
) -> list[float]:
    wanted = set(time_indices)
    return [
        row[value_key]
        for split_rows in artifact["splits"].values()
        for row in split_rows
        if row["sensor_id"] == sensor_id
        and row["time_index"] in wanted
        and row[value_key] is not None
    ]


def _aggregate(per_seed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    checks = {}
    for name in _CHECKS:
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
    return {
        "checks": checks,
        "signal_holds_on_every_seed": all(
            not check["not_held_on"] and not check["not_evaluated_on"]
            for check in checks.values()
        ),
    }
