from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator
import math
import random


@dataclass(frozen=True)
class SyntheticTrafficEdge:
    source: str
    target: str
    weight: float


@dataclass(frozen=True)
class SyntheticTrafficObservation:
    sensor_id: str
    time_index: int
    timestamp: datetime
    current_speed: float | None
    target_speed: float | None
    current_speed_missing: bool
    target_missing: bool
    local_covariates: dict[str, float | bool | None]
    neighbor_covariates: dict[str, float | bool | None]


@dataclass(frozen=True)
class SyntheticTrafficFeatureMetadata:
    name: str
    source: str
    dtype: str
    unit: str
    offset_minutes: int | None


@dataclass(frozen=True)
class SyntheticTrafficSplit:
    sensors: tuple[str, ...]
    time_points: tuple[datetime, ...]
    adjacency: tuple[SyntheticTrafficEdge, ...]
    local_feature_metadata: tuple[SyntheticTrafficFeatureMetadata, ...]
    observations: tuple[SyntheticTrafficObservation, ...]

    def __iter__(self) -> Iterator[SyntheticTrafficObservation]:
        return iter(self.observations)


@dataclass(frozen=True)
class SyntheticTrafficFixture:
    sensors: tuple[str, ...]
    time_points: tuple[datetime, ...]
    adjacency: tuple[SyntheticTrafficEdge, ...]
    local_feature_metadata: tuple[SyntheticTrafficFeatureMetadata, ...]
    observations: tuple[SyntheticTrafficObservation, ...]

    def temporal_split_observations(
        self,
        train: float = 0.5,
        validation: float = 0.25,
        test: float = 0.25,
    ) -> dict[str, SyntheticTrafficSplit]:
        split_total = train + validation + test
        if abs(split_total - 1.0) > 1e-6:
            raise ValueError(
                "train, validation, and test fractions must sum to 1.0 "
                f"(got {split_total:g})."
            )

        time_count = len(self.time_points)
        train_end = round(time_count * train)
        validation_end = train_end + round(time_count * validation)
        split_for_time: dict[int, str] = {}
        for time_index in range(time_count):
            if time_index < train_end:
                split_for_time[time_index] = "train"
            elif time_index < validation_end:
                split_for_time[time_index] = "validation"
            else:
                split_for_time[time_index] = "test"

        return {
            split_name: SyntheticTrafficSplit(
                sensors=self.sensors,
                time_points=tuple(
                    timestamp
                    for time_index, timestamp in enumerate(self.time_points)
                    if split_for_time[time_index] == split_name
                ),
                adjacency=self.adjacency,
                local_feature_metadata=self.local_feature_metadata,
                observations=tuple(
                    sorted(
                        (
                            observation
                            for observation in self.observations
                            if split_for_time[observation.time_index] == split_name
                        ),
                        key=lambda observation: (
                            observation.time_index,
                            observation.sensor_id,
                        ),
                    )
                ),
            )
            for split_name in ("train", "validation", "test")
        }


LOCAL_FEATURE_METADATA = (
    SyntheticTrafficFeatureMetadata("current_speed", "traffic", "continuous", "speed", 0),
    SyntheticTrafficFeatureMetadata(
        "current_speed_missing", "traffic_missingness", "indicator", "flag", 0
    ),
    SyntheticTrafficFeatureMetadata("lag_5m", "traffic_lag", "continuous", "speed", -5),
    SyntheticTrafficFeatureMetadata("lag_10m", "traffic_lag", "continuous", "speed", -10),
    SyntheticTrafficFeatureMetadata("lag_15m", "traffic_lag", "continuous", "speed", -15),
    SyntheticTrafficFeatureMetadata("lag_30m", "traffic_lag", "continuous", "speed", -30),
    SyntheticTrafficFeatureMetadata("lag_60m", "traffic_lag", "continuous", "speed", -60),
    SyntheticTrafficFeatureMetadata(
        "time_of_day", "calendar", "continuous", "hour_of_day", 0
    ),
    SyntheticTrafficFeatureMetadata("day_of_week", "calendar", "ordinal", "weekday", 0),
    SyntheticTrafficFeatureMetadata("weekend", "calendar", "indicator", "flag", 0),
)


def load_synthetic_traffic_fixture(seed: int = 20250214) -> SyntheticTrafficFixture:
    """Build tiny deterministic traffic test data, not a scientific benchmark."""

    rng = random.Random(seed)
    sensors = ("sensor_a", "sensor_b", "sensor_c", "sensor_d")
    start = datetime(2026, 1, 5, 7, 0)
    time_points = tuple(start + timedelta(minutes=5 * step) for step in range(16))
    adjacency = (
        SyntheticTrafficEdge("sensor_a", "sensor_b", 0.8),
        SyntheticTrafficEdge("sensor_b", "sensor_c", 0.6),
        SyntheticTrafficEdge("sensor_c", "sensor_d", 0.7),
        SyntheticTrafficEdge("sensor_d", "sensor_a", 0.4),
    )

    speeds = _speed_table(sensors, time_points, rng)
    current_missing = {("sensor_b", 3)}
    target_missing = {("sensor_c", 5)}

    observations: list[SyntheticTrafficObservation] = []
    for sensor_id in sensors:
        for time_index, timestamp in enumerate(time_points):
            current_speed = _reading(speeds, sensor_id, time_index, current_missing)
            target_speed = _target_reading(speeds, sensor_id, time_index, target_missing)
            local_covariates = _local_covariates(
                speeds, sensor_id, time_index, timestamp, current_missing
            )
            neighbor_covariates = _neighbor_covariates(
                speeds, adjacency, sensor_id, time_index, current_missing
            )
            observations.append(
                SyntheticTrafficObservation(
                    sensor_id=sensor_id,
                    time_index=time_index,
                    timestamp=timestamp,
                    current_speed=current_speed,
                    target_speed=target_speed,
                    current_speed_missing=current_speed is None,
                    target_missing=target_speed is None,
                    local_covariates=local_covariates,
                    neighbor_covariates=neighbor_covariates,
                )
            )

    return SyntheticTrafficFixture(
        sensors=sensors,
        time_points=time_points,
        adjacency=adjacency,
        local_feature_metadata=LOCAL_FEATURE_METADATA,
        observations=tuple(observations),
    )


def _speed_table(
    sensors: tuple[str, ...],
    time_points: tuple[datetime, ...],
    rng: random.Random,
) -> dict[tuple[str, int], float]:
    speeds: dict[tuple[str, int], float] = {}
    for sensor_offset, sensor_id in enumerate(sensors):
        base_speed = 55.0 - sensor_offset * 3.0
        for time_index, _timestamp in enumerate(time_points):
            commute_wave = math.sin(time_index / 2.0) * 4.0
            local_variation = rng.uniform(-0.35, 0.35)
            speeds[(sensor_id, time_index)] = round(
                base_speed - time_index * 0.75 + commute_wave + local_variation,
                3,
            )
    return speeds


def _reading(
    speeds: dict[tuple[str, int], float],
    sensor_id: str,
    time_index: int,
    missing: set[tuple[str, int]],
) -> float | None:
    if (sensor_id, time_index) in missing:
        return None
    return speeds[(sensor_id, time_index)]


def _target_reading(
    speeds: dict[tuple[str, int], float],
    sensor_id: str,
    time_index: int,
    missing: set[tuple[str, int]],
) -> float | None:
    target_index = time_index + 3
    if (sensor_id, target_index) not in speeds:
        return None
    return _reading(speeds, sensor_id, target_index, missing)


def _local_covariates(
    speeds: dict[tuple[str, int], float],
    sensor_id: str,
    time_index: int,
    timestamp: datetime,
    current_missing: set[tuple[str, int]],
) -> dict[str, float | bool | None]:
    return {
        "current_speed": _reading(speeds, sensor_id, time_index, current_missing),
        "current_speed_missing": (sensor_id, time_index) in current_missing,
        "lag_5m": _lag_reading(speeds, sensor_id, time_index, 1, current_missing),
        "lag_10m": _lag_reading(speeds, sensor_id, time_index, 2, current_missing),
        "lag_15m": _lag_reading(speeds, sensor_id, time_index, 3, current_missing),
        "lag_30m": _lag_reading(speeds, sensor_id, time_index, 6, current_missing),
        "lag_60m": _lag_reading(speeds, sensor_id, time_index, 12, current_missing),
        "time_of_day": timestamp.hour + timestamp.minute / 60.0,
        "day_of_week": float(timestamp.weekday()),
        "weekend": timestamp.weekday() >= 5,
    }


def _lag_reading(
    speeds: dict[tuple[str, int], float],
    sensor_id: str,
    time_index: int,
    lag_steps: int,
    missing: set[tuple[str, int]],
) -> float | None:
    lag_index = time_index - lag_steps
    if lag_index < 0:
        return None
    return _reading(speeds, sensor_id, lag_index, missing)


def _neighbor_covariates(
    speeds: dict[tuple[str, int], float],
    adjacency: tuple[SyntheticTrafficEdge, ...],
    sensor_id: str,
    time_index: int,
    current_missing: set[tuple[str, int]],
) -> dict[str, float | bool | None]:
    incoming_edges = tuple(edge for edge in adjacency if edge.target == sensor_id)
    current = _weighted_neighbor_average(
        speeds, incoming_edges, time_index, 0, current_missing
    )
    lag_5m = _weighted_neighbor_average(speeds, incoming_edges, time_index, 1, current_missing)
    return {
        "neighbor_current_speed": current,
        "neighbor_current_speed_missing": current is None,
        "neighbor_lag_5m": lag_5m,
        "neighbor_lag_5m_missing": lag_5m is None,
    }


def _weighted_neighbor_average(
    speeds: dict[tuple[str, int], float],
    incoming_edges: tuple[SyntheticTrafficEdge, ...],
    time_index: int,
    lag_steps: int,
    missing: set[tuple[str, int]],
) -> float | None:
    lag_index = time_index - lag_steps
    if lag_index < 0:
        return None

    weighted_sum = 0.0
    total_weight = 0.0
    for edge in incoming_edges:
        reading = _reading(speeds, edge.source, lag_index, missing)
        if reading is None:
            continue
        weighted_sum += reading * edge.weight
        total_weight += edge.weight

    if total_weight == 0.0:
        return None
    return round(weighted_sum / total_weight, 3)
