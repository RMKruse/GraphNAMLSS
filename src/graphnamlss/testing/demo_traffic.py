from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import random

from graphnamlss.testing.synthetic_traffic import (
    LOCAL_FEATURE_METADATA,
    NEIGHBOR_FEATURE_METADATA,
    SyntheticTrafficEdge,
    SyntheticTrafficFixture,
    SyntheticTrafficObservation,
    _local_covariates,
    _neighbor_covariates,
    _reading,
    _target_reading,
)


@dataclass(frozen=True)
class LocalizedCongestionSpilloverScenario:
    name: str
    upstream_sensor_id: str
    downstream_sensor_id: str
    congestion_time_indices: tuple[int, ...]
    spillover_prediction_time_indices: tuple[int, ...]
    routine_prediction_time_indices: tuple[int, ...]


@dataclass(frozen=True)
class LocalizedCongestionSpilloverDemoFixture(SyntheticTrafficFixture):
    scenario: LocalizedCongestionSpilloverScenario


def load_localized_congestion_spillover_demo_fixture(
    seed: int = 20250214,
) -> LocalizedCongestionSpilloverDemoFixture:
    """Build the synthetic, artifact-free Concept Demo traffic fixture."""

    rng = random.Random(seed)
    sensors = ("upstream", "downstream", "local_reference", "background")
    start = datetime(2026, 1, 5, 7, 0)
    time_points = tuple(start + timedelta(minutes=5 * step) for step in range(120))
    adjacency = (
        SyntheticTrafficEdge("upstream", "downstream", 1.0),
        SyntheticTrafficEdge("downstream", "local_reference", 1.0),
        SyntheticTrafficEdge("local_reference", "background", 1.0),
        SyntheticTrafficEdge("background", "upstream", 1.0),
    )
    congestion_time_indices = tuple(
        time_index
        for window_start in (15, 35, 55, 75, 95)
        for time_index in (window_start, window_start + 1)
    )
    routine_prediction_time_indices = tuple(
        time_index - 10 for time_index in congestion_time_indices
    )
    speeds = _demo_speed_table(
        sensors,
        time_points,
        congestion_time_indices=set(congestion_time_indices),
        rng=rng,
    )
    observations = _demo_observations(
        sensors=sensors,
        time_points=time_points,
        adjacency=adjacency,
        speeds=speeds,
    )

    return LocalizedCongestionSpilloverDemoFixture(
        sensors=sensors,
        time_points=time_points,
        adjacency=adjacency,
        local_feature_metadata=LOCAL_FEATURE_METADATA,
        neighbor_feature_metadata=NEIGHBOR_FEATURE_METADATA,
        observations=observations,
        scenario=LocalizedCongestionSpilloverScenario(
            name="localized_congestion_spillover",
            upstream_sensor_id="upstream",
            downstream_sensor_id="downstream",
            congestion_time_indices=congestion_time_indices,
            spillover_prediction_time_indices=congestion_time_indices,
            routine_prediction_time_indices=routine_prediction_time_indices,
        ),
    )


def _demo_speed_table(
    sensors: tuple[str, ...],
    time_points: tuple[datetime, ...],
    *,
    congestion_time_indices: set[int],
    rng: random.Random,
) -> dict[tuple[str, int], float]:
    base_speed = {
        "upstream": 58.0,
        "downstream": 54.0,
        "local_reference": 52.0,
        "background": 56.0,
    }
    spillover_time_indices = {
        time_index + 3 for time_index in congestion_time_indices
    }
    speeds: dict[tuple[str, int], float] = {}
    for sensor_id in sensors:
        for time_index, _timestamp in enumerate(time_points):
            routine_wave = 1.5 * math.sin(2.0 * math.pi * time_index / 10.0)
            speed = base_speed[sensor_id] + routine_wave + rng.gauss(0.0, 0.25)
            if sensor_id == "upstream" and time_index in congestion_time_indices:
                speed -= 16.0
            if sensor_id == "downstream" and time_index in spillover_time_indices:
                speed += rng.gauss(-8.0, 3.0)
            speeds[(sensor_id, time_index)] = round(speed, 3)
    return speeds


def _demo_observations(
    *,
    sensors: tuple[str, ...],
    time_points: tuple[datetime, ...],
    adjacency: tuple[SyntheticTrafficEdge, ...],
    speeds: dict[tuple[str, int], float],
) -> tuple[SyntheticTrafficObservation, ...]:
    current_missing: set[tuple[str, int]] = set()
    target_missing: set[tuple[str, int]] = set()
    observations: list[SyntheticTrafficObservation] = []
    for sensor_id in sensors:
        for time_index, timestamp in enumerate(time_points):
            current_speed = _reading(speeds, sensor_id, time_index, current_missing)
            target_speed = _target_reading(
                speeds, sensor_id, time_index, target_missing
            )
            observations.append(
                SyntheticTrafficObservation(
                    sensor_id=sensor_id,
                    time_index=time_index,
                    timestamp=timestamp,
                    current_speed=current_speed,
                    target_speed=target_speed,
                    current_speed_missing=False,
                    target_missing=target_speed is None,
                    local_covariates=_local_covariates(
                        speeds, sensor_id, time_index, timestamp, current_missing
                    ),
                    neighbor_covariates=_neighbor_covariates(
                        speeds, adjacency, sensor_id, time_index, current_missing
                    ),
                )
            )
    return tuple(observations)
