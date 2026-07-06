from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
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
class SyntheticTrafficMissingnessReport:
    observation_count: int
    target_missing_count: int
    target_missing_rate: float
    current_missing_count: int
    current_missing_rate: float
    imputation_strategy: str
    imputed_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticTrafficSplit:
    sensors: tuple[str, ...]
    time_points: tuple[datetime, ...]
    adjacency: tuple[SyntheticTrafficEdge, ...]
    local_feature_metadata: tuple[SyntheticTrafficFeatureMetadata, ...]
    neighbor_feature_metadata: tuple[SyntheticTrafficFeatureMetadata, ...]
    observations: tuple[SyntheticTrafficObservation, ...]

    def __iter__(self) -> Iterator[SyntheticTrafficObservation]:
        return iter(self.observations)

    def observed_target_observations(self) -> tuple[SyntheticTrafficObservation, ...]:
        return tuple(
            observation
            for observation in self.observations
            if not observation.target_missing
        )


@dataclass(frozen=True)
class SyntheticTrafficFixture:
    sensors: tuple[str, ...]
    time_points: tuple[datetime, ...]
    adjacency: tuple[SyntheticTrafficEdge, ...]
    local_feature_metadata: tuple[SyntheticTrafficFeatureMetadata, ...]
    neighbor_feature_metadata: tuple[SyntheticTrafficFeatureMetadata, ...]
    observations: tuple[SyntheticTrafficObservation, ...]

    def missingness_report(self) -> SyntheticTrafficMissingnessReport:
        observation_count = len(self.observations)
        target_missing_count = sum(
            1 for observation in self.observations if observation.target_missing
        )
        current_missing_count = sum(
            1 for observation in self.observations if observation.current_speed_missing
        )
        imputed_feature_names = _imputed_feature_names(
            self.local_feature_metadata + self.neighbor_feature_metadata
        )

        return SyntheticTrafficMissingnessReport(
            observation_count=observation_count,
            target_missing_count=target_missing_count,
            target_missing_rate=target_missing_count / observation_count,
            current_missing_count=current_missing_count,
            current_missing_rate=current_missing_count / observation_count,
            imputation_strategy=(
                "past_observed_lag" if imputed_feature_names else "none"
            ),
            imputed_feature_names=imputed_feature_names,
        )

    def with_past_lag_imputation(self) -> SyntheticTrafficFixture:
        readings = {
            (observation.sensor_id, observation.time_index): observation.current_speed
            for observation in self.observations
        }
        incoming_edges_by_target = {
            sensor_id: tuple(
                edge for edge in self.adjacency if edge.target == sensor_id
            )
            for sensor_id in self.sensors
        }

        return replace(
            self,
            local_feature_metadata=_with_local_lag_imputation_metadata(
                self.local_feature_metadata
            ),
            neighbor_feature_metadata=_with_neighbor_lag_imputation_metadata(
                self.neighbor_feature_metadata
            ),
            observations=tuple(
                replace(
                    observation,
                    local_covariates=_impute_local_lags_from_past(
                        observation, readings
                    ),
                    neighbor_covariates=_impute_neighbor_lags_from_past(
                        observation,
                        readings,
                        incoming_edges_by_target[observation.sensor_id],
                    ),
                )
                for observation in self.observations
            ),
        )

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
                neighbor_feature_metadata=self.neighbor_feature_metadata,
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


NEIGHBOR_LAG_STEPS = (
    ("neighbor_lag_5m", 1),
    ("neighbor_lag_10m", 2),
    ("neighbor_lag_15m", 3),
    ("neighbor_lag_30m", 6),
    ("neighbor_lag_60m", 12),
)


LOCAL_LAG_STEPS = (
    ("lag_5m", 1),
    ("lag_10m", 2),
    ("lag_15m", 3),
    ("lag_30m", 6),
    ("lag_60m", 12),
)


LOCAL_LAG_IMPUTATION_METADATA = tuple(
    SyntheticTrafficFeatureMetadata(
        f"{feature_name}_imputed",
        "traffic_imputation",
        "indicator",
        "flag",
        lag_steps * -5,
    )
    for feature_name, lag_steps in LOCAL_LAG_STEPS
)


NEIGHBOR_LAG_IMPUTATION_METADATA = tuple(
    SyntheticTrafficFeatureMetadata(
        f"{feature_name}_imputed",
        "neighbor_traffic_imputation",
        "indicator",
        "flag",
        lag_steps * -5,
    )
    for feature_name, lag_steps in NEIGHBOR_LAG_STEPS
)


NEIGHBOR_FEATURE_METADATA = (
    SyntheticTrafficFeatureMetadata(
        "neighbor_current_speed", "neighbor_traffic", "continuous", "speed", 0
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_current_speed_missing",
        "neighbor_traffic_missingness",
        "indicator",
        "flag",
        0,
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_lag_5m", "neighbor_traffic_lag", "continuous", "speed", -5
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_lag_5m_missing",
        "neighbor_traffic_missingness",
        "indicator",
        "flag",
        -5,
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_lag_10m", "neighbor_traffic_lag", "continuous", "speed", -10
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_lag_10m_missing",
        "neighbor_traffic_missingness",
        "indicator",
        "flag",
        -10,
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_lag_15m", "neighbor_traffic_lag", "continuous", "speed", -15
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_lag_15m_missing",
        "neighbor_traffic_missingness",
        "indicator",
        "flag",
        -15,
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_lag_30m", "neighbor_traffic_lag", "continuous", "speed", -30
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_lag_30m_missing",
        "neighbor_traffic_missingness",
        "indicator",
        "flag",
        -30,
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_lag_60m", "neighbor_traffic_lag", "continuous", "speed", -60
    ),
    SyntheticTrafficFeatureMetadata(
        "neighbor_lag_60m_missing",
        "neighbor_traffic_missingness",
        "indicator",
        "flag",
        -60,
    ),
)


def load_synthetic_traffic_fixture(
    seed: int = 20250214, adjacency_path: str | Path | None = None
) -> SyntheticTrafficFixture:
    """Build tiny deterministic traffic test data, not a scientific benchmark."""

    rng = random.Random(seed)
    sensors = ("sensor_a", "sensor_b", "sensor_c", "sensor_d")
    start = datetime(2026, 1, 5, 7, 0)
    time_points = tuple(start + timedelta(minutes=5 * step) for step in range(16))
    adjacency = (
        _load_dataset_provided_adjacency(adjacency_path, sensors)
        if adjacency_path is not None
        else _default_adjacency()
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
        neighbor_feature_metadata=NEIGHBOR_FEATURE_METADATA,
        observations=tuple(observations),
    )


def load_synthetic_traffic_fixture_from_config(
    config: Any, seed: int = 20250214
) -> SyntheticTrafficFixture:
    adjacency_path = config.dataset.artifacts.get("adjacency")
    fixture = load_synthetic_traffic_fixture(seed=seed, adjacency_path=adjacency_path)

    if (
        config.missingness is not None
        and config.missingness.covariates == "past_observed_lag_imputation"
    ):
        return fixture.with_past_lag_imputation()

    return fixture


def _default_adjacency() -> tuple[SyntheticTrafficEdge, ...]:
    return (
        SyntheticTrafficEdge("sensor_a", "sensor_b", 0.8),
        SyntheticTrafficEdge("sensor_b", "sensor_c", 0.6),
        SyntheticTrafficEdge("sensor_c", "sensor_d", 0.7),
        SyntheticTrafficEdge("sensor_d", "sensor_a", 0.4),
    )


def _load_dataset_provided_adjacency(
    adjacency_path: str | Path,
    sensors: tuple[str, ...],
) -> tuple[SyntheticTrafficEdge, ...]:
    known_sensors = set(sensors)
    with Path(adjacency_path).open(newline="") as adjacency_file:
        rows = csv.DictReader(adjacency_file)
        adjacency = tuple(_traffic_edge_from_row(row, known_sensors) for row in rows)
    return adjacency


def _traffic_edge_from_row(
    row: dict[str, str],
    known_sensors: set[str],
) -> SyntheticTrafficEdge:
    edge = SyntheticTrafficEdge(
        source=row["source"],
        target=row["target"],
        weight=float(row["weight"]),
    )
    unknown_sensors = sorted(
        sensor_id
        for sensor_id in (edge.source, edge.target)
        if sensor_id not in known_sensors
    )
    if unknown_sensors:
        raise ValueError(
            "Dataset-provided adjacency references unknown sensors: "
            + ", ".join(unknown_sensors)
        )
    return edge


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


def _impute_local_lags_from_past(
    observation: SyntheticTrafficObservation,
    readings: dict[tuple[str, int], float | None],
) -> dict[str, float | bool | None]:
    covariates = dict(observation.local_covariates)
    for feature_name, lag_steps in LOCAL_LAG_STEPS:
        imputed_name = f"{feature_name}_imputed"
        original_value = covariates[feature_name]
        if original_value is not None:
            covariates[imputed_name] = False
            continue
        lag_index = observation.time_index - lag_steps
        if lag_index < 0:
            covariates[imputed_name] = False
            continue
        imputed_value = _previous_observed_reading(
            readings, observation.sensor_id, lag_index
        )
        covariates[feature_name] = imputed_value
        covariates[imputed_name] = imputed_value is not None
    return covariates


def _impute_neighbor_lags_from_past(
    observation: SyntheticTrafficObservation,
    readings: dict[tuple[str, int], float | None],
    incoming_edges: tuple[SyntheticTrafficEdge, ...],
) -> dict[str, float | bool | None]:
    covariates = dict(observation.neighbor_covariates)
    for feature_name, lag_steps in NEIGHBOR_LAG_STEPS:
        imputed_name = f"{feature_name}_imputed"
        original_value = covariates[feature_name]
        if original_value is not None:
            covariates[imputed_name] = False
            continue
        lag_index = observation.time_index - lag_steps
        if lag_index < 0:
            covariates[imputed_name] = False
            continue
        imputed_value = _weighted_previous_neighbor_average(
            readings, incoming_edges, lag_index
        )
        covariates[feature_name] = imputed_value
        covariates[imputed_name] = imputed_value is not None
    return covariates


def _weighted_previous_neighbor_average(
    readings: dict[tuple[str, int], float | None],
    incoming_edges: tuple[SyntheticTrafficEdge, ...],
    latest_time_index: int,
) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for edge in incoming_edges:
        reading = _previous_observed_reading(
            readings, edge.source, latest_time_index
        )
        if reading is None:
            continue
        weighted_sum += reading * edge.weight
        total_weight += edge.weight

    if total_weight == 0.0:
        return None
    return round(weighted_sum / total_weight, 3)


def _with_local_lag_imputation_metadata(
    metadata: tuple[SyntheticTrafficFeatureMetadata, ...],
) -> tuple[SyntheticTrafficFeatureMetadata, ...]:
    return _with_imputation_metadata(metadata, LOCAL_LAG_IMPUTATION_METADATA)


def _with_neighbor_lag_imputation_metadata(
    metadata: tuple[SyntheticTrafficFeatureMetadata, ...],
) -> tuple[SyntheticTrafficFeatureMetadata, ...]:
    return _with_imputation_metadata(metadata, NEIGHBOR_LAG_IMPUTATION_METADATA)


def _with_imputation_metadata(
    metadata: tuple[SyntheticTrafficFeatureMetadata, ...],
    imputation_metadata: tuple[SyntheticTrafficFeatureMetadata, ...],
) -> tuple[SyntheticTrafficFeatureMetadata, ...]:
    names = {feature.name for feature in metadata}
    return metadata + tuple(
        feature
        for feature in imputation_metadata
        if feature.name not in names
    )


def _imputed_feature_names(
    metadata: tuple[SyntheticTrafficFeatureMetadata, ...],
) -> tuple[str, ...]:
    return tuple(
        feature.name.removesuffix("_imputed")
        for feature in metadata
        if feature.source.endswith("_imputation")
    )


def _previous_observed_reading(
    readings: dict[tuple[str, int], float | None],
    sensor_id: str,
    latest_time_index: int,
) -> float | None:
    for time_index in range(latest_time_index, -1, -1):
        reading = readings[(sensor_id, time_index)]
        if reading is not None:
            return reading
    return None


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
    covariates: dict[str, float | bool | None] = {
        "neighbor_current_speed": current,
        "neighbor_current_speed_missing": current is None,
    }
    for feature_name, lag_steps in NEIGHBOR_LAG_STEPS:
        value = _weighted_neighbor_average(
            speeds, incoming_edges, time_index, lag_steps, current_missing
        )
        covariates[feature_name] = value
        covariates[f"{feature_name}_missing"] = value is None
    return covariates


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
