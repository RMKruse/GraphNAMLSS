from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from graphnamlss.testing import LocalizedCongestionSpilloverScenario


# name -> (artifact, row value, statistic, expected spillover direction)
SPILLOVER_CHECKS: dict[
    str, tuple[str, str, Callable[[Sequence[float]], float], str]
] = {
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
"""Spillover contrasts of a Localized Congestion Spillover Demo Scenario run."""


def measure_spillover_contrasts(
    graph_run: Path, scenario: LocalizedCongestionSpilloverScenario
) -> dict[str, Any]:
    """Compare downstream spillover with routine prediction times in saved artifacts.

    Rows are pooled over the train, validation and test splits of a GraphNAMLSS
    run. Each check records spillover, routine, their difference and whether the
    difference has the expected sign. ``gated_mu_share_of_slowdown`` divides the
    gated ``mu`` difference by the standardized target-location difference.
    """

    artifacts = {
        artifact: json.loads((graph_run / artifact).read_text())
        for artifact in {artifact for artifact, *_ in SPILLOVER_CHECKS.values()}
    }
    checks = {}
    differences = {}
    for name, (artifact, value_key, statistic, direction) in SPILLOVER_CHECKS.items():
        spillover, routine = _spillover_and_routine(
            artifacts[artifact], scenario, value_key, statistic
        )
        difference = spillover - routine
        differences[name] = difference
        checks[name] = {
            "spillover": round(spillover, 6),
            "routine": round(routine, 6),
            "difference": round(difference, 6),
            "held": difference < 0.0 if direction == "lower" else difference > 0.0,
        }
    spillover, routine = _spillover_and_routine(
        artifacts["predictions.json"], scenario, "standardized.target", fmean
    )
    return {
        "checks": checks,
        "gated_mu_share_of_slowdown": round(
            differences["gated_mu_contribution"] / (spillover - routine), 6
        ),
    }


def _spillover_and_routine(
    artifact: dict[str, Any],
    scenario: LocalizedCongestionSpilloverScenario,
    value_key: str,
    statistic: Callable[[Sequence[float]], float],
) -> tuple[float, float]:
    spillover, routine = (
        statistic(
            _downstream_values(
                artifact, scenario.downstream_sensor_id, time_indices, value_key
            )
        )
        for time_indices in (
            scenario.spillover_prediction_time_indices,
            scenario.routine_prediction_time_indices,
        )
    )
    return spillover, routine


def _downstream_values(
    artifact: dict[str, Any],
    sensor_id: str,
    time_indices: Sequence[int],
    value_key: str,
) -> list[float]:
    """Collect one row value, where ``value_key`` may name a nested field by dots."""

    wanted = set(time_indices)
    values = []
    for split_rows in artifact["splits"].values():
        for row in split_rows:
            if row["sensor_id"] != sensor_id or row["time_index"] not in wanted:
                continue
            value = row
            for key in value_key.split("."):
                value = value[key]
            if value is not None:
                values.append(value)
    return values
