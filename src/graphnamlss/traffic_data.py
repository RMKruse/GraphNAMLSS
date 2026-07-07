from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import pickle
from typing import Any

from graphnamlss.config import ConfigError
from graphnamlss.testing.synthetic_traffic import (
    LOCAL_FEATURE_METADATA,
    NEIGHBOR_FEATURE_METADATA,
    SyntheticTrafficEdge,
    SyntheticTrafficFixture,
    SyntheticTrafficObservation,
)


__all__ = ["load_metr_la_traffic_fixture_from_config"]


def load_metr_la_traffic_fixture_from_config(config: Any) -> SyntheticTrafficFixture:
    readings_path = _required_artifact(config, "readings")
    adjacency_path = _required_artifact(config, "adjacency")

    sensors, time_points, readings = _load_wide_readings(readings_path)
    adjacency = _load_adjacency(adjacency_path, sensors, config)
    horizon_steps = config.temporal.horizon_steps if config.temporal is not None else 3

    observations: list[SyntheticTrafficObservation] = []
    for sensor_id in sensors:
        for time_index, timestamp in enumerate(time_points):
            current_speed = readings[(sensor_id, time_index)]
            target_speed = _target_reading(
                readings, sensor_id, time_index, horizon_steps
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
                    local_covariates=_local_covariates(
                        readings, sensor_id, time_index, timestamp
                    ),
                    neighbor_covariates=_neighbor_covariates(
                        readings, adjacency, sensor_id, time_index
                    ),
                )
            )

    fixture = SyntheticTrafficFixture(
        sensors=sensors,
        time_points=time_points,
        adjacency=adjacency,
        local_feature_metadata=LOCAL_FEATURE_METADATA,
        neighbor_feature_metadata=NEIGHBOR_FEATURE_METADATA,
        observations=tuple(observations),
    )
    if (
        config.missingness is not None
        and config.missingness.covariates == "past_observed_lag_imputation"
    ):
        return fixture.with_past_lag_imputation()
    return fixture


def _required_artifact(config: Any, name: str) -> Path:
    try:
        return config.dataset.artifacts[name]
    except KeyError as exc:
        raise ConfigError(
            f"METR-LA training requires dataset.artifacts.{name}."
        ) from exc


def _load_wide_readings(
    path: Path,
) -> tuple[tuple[str, ...], tuple[datetime, ...], dict[tuple[str, int], float | None]]:
    if path.suffix.lower() == ".csv":
        return _load_wide_readings_csv(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return _load_wide_readings_hdf(path)
    raise ConfigError(
        "METR-LA readings must be a wide CSV or HDF5 table with timestamps "
        "and one column per sensor."
    )


def _load_wide_readings_csv(
    path: Path,
) -> tuple[tuple[str, ...], tuple[datetime, ...], dict[tuple[str, int], float | None]]:
    with path.open(newline="") as readings_file:
        rows = tuple(csv.DictReader(readings_file))
    if not rows:
        raise ConfigError(f"METR-LA readings artifact is empty: {path}")

    fieldnames = tuple(rows[0])
    if not fieldnames or fieldnames[0] != "timestamp":
        raise ConfigError("METR-LA CSV readings must start with a timestamp column.")
    sensors = tuple(fieldnames[1:])
    if not sensors:
        raise ConfigError("METR-LA readings must include at least one sensor column.")

    time_points = tuple(_parse_timestamp(row["timestamp"]) for row in rows)
    readings: dict[tuple[str, int], float | None] = {}
    for time_index, row in enumerate(rows):
        for sensor_id in sensors:
            readings[(sensor_id, time_index)] = _parse_optional_float(row[sensor_id])
    return sensors, time_points, readings


def _load_wide_readings_hdf(
    path: Path,
) -> tuple[tuple[str, ...], tuple[datetime, ...], dict[tuple[str, int], float | None]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ConfigError(
            "Reading canonical METR-LA HDF5 artifacts requires pandas; "
            "install the traffic data extra before training from .h5 files."
        ) from exc

    frame = pd.read_hdf(path)
    if frame.empty:
        raise ConfigError(f"METR-LA readings artifact is empty: {path}")
    sensors = tuple(str(column) for column in frame.columns)
    time_points = tuple(_parse_timestamp(value) for value in frame.index)
    readings: dict[tuple[str, int], float | None] = {}
    for time_index, (_index, row) in enumerate(frame.iterrows()):
        for sensor_id in sensors:
            readings[(sensor_id, time_index)] = _parse_optional_float(row[sensor_id])
    return sensors, time_points, readings


def _load_adjacency(
    path: Path, sensors: tuple[str, ...], config: Any
) -> tuple[SyntheticTrafficEdge, ...]:
    graph_format = config.graph.format if config.graph is not None else ""
    if path.suffix.lower() == ".csv" or graph_format == "edge_list_csv":
        return _load_edge_list_csv(path, sensors)
    if path.suffix.lower() == ".pkl" or graph_format == "dcrnn_adjacency_pickle":
        return _load_dcrnn_adjacency_pickle(path, sensors)
    raise ConfigError(
        "METR-LA adjacency must be an edge-list CSV or DCRNN adjacency pickle."
    )


def _load_edge_list_csv(
    path: Path, sensors: tuple[str, ...]
) -> tuple[SyntheticTrafficEdge, ...]:
    known_sensors = set(sensors)
    with path.open(newline="") as adjacency_file:
        rows = csv.DictReader(adjacency_file)
        return tuple(_edge_from_row(row, known_sensors) for row in rows)


def _load_dcrnn_adjacency_pickle(
    path: Path, sensors: tuple[str, ...]
) -> tuple[SyntheticTrafficEdge, ...]:
    try:
        with path.open("rb") as adjacency_file:
            sensor_ids, _sensor_id_to_index, adjacency_matrix = pickle.load(
                adjacency_file, encoding="latin1"
            )
    except ModuleNotFoundError as exc:
        raise ConfigError(
            "Reading canonical METR-LA adjacency pickles requires the packages "
            "used to serialize the artifact, typically numpy."
        ) from exc

    sensor_ids = tuple(str(sensor_id) for sensor_id in sensor_ids)
    configured_sensors = set(sensors)
    if set(sensor_ids) != configured_sensors:
        raise ConfigError(
            "METR-LA readings sensors and adjacency sensors do not match."
        )

    edges: list[SyntheticTrafficEdge] = []
    for source_index, source in enumerate(sensor_ids):
        for target_index, target in enumerate(sensor_ids):
            weight = float(adjacency_matrix[source_index][target_index])
            if weight != 0.0:
                edges.append(SyntheticTrafficEdge(source, target, weight))
    return tuple(edges)


def _edge_from_row(
    row: dict[str, str], known_sensors: set[str]
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
        raise ConfigError(
            "Dataset-provided adjacency references unknown sensors: "
            + ", ".join(unknown_sensors)
        )
    return edge


def _target_reading(
    readings: dict[tuple[str, int], float | None],
    sensor_id: str,
    time_index: int,
    horizon_steps: int,
) -> float | None:
    return readings.get((sensor_id, time_index + horizon_steps))


def _local_covariates(
    readings: dict[tuple[str, int], float | None],
    sensor_id: str,
    time_index: int,
    timestamp: datetime,
) -> dict[str, float | bool | None]:
    current_speed = readings[(sensor_id, time_index)]
    return {
        "current_speed": current_speed,
        "current_speed_missing": current_speed is None,
        "lag_5m": _lag_reading(readings, sensor_id, time_index, 1),
        "lag_10m": _lag_reading(readings, sensor_id, time_index, 2),
        "lag_15m": _lag_reading(readings, sensor_id, time_index, 3),
        "lag_30m": _lag_reading(readings, sensor_id, time_index, 6),
        "lag_60m": _lag_reading(readings, sensor_id, time_index, 12),
        "time_of_day": timestamp.hour + timestamp.minute / 60.0,
        "day_of_week": float(timestamp.weekday()),
        "weekend": timestamp.weekday() >= 5,
    }


def _neighbor_covariates(
    readings: dict[tuple[str, int], float | None],
    adjacency: tuple[SyntheticTrafficEdge, ...],
    sensor_id: str,
    time_index: int,
) -> dict[str, float | bool | None]:
    incoming_edges = tuple(edge for edge in adjacency if edge.target == sensor_id)
    current = _weighted_neighbor_average(readings, incoming_edges, time_index, 0)
    covariates: dict[str, float | bool | None] = {
        "neighbor_current_speed": current,
        "neighbor_current_speed_missing": current is None,
    }
    for feature_name, lag_steps in (
        ("neighbor_lag_5m", 1),
        ("neighbor_lag_10m", 2),
        ("neighbor_lag_15m", 3),
        ("neighbor_lag_30m", 6),
        ("neighbor_lag_60m", 12),
    ):
        value = _weighted_neighbor_average(readings, incoming_edges, time_index, lag_steps)
        covariates[feature_name] = value
        covariates[f"{feature_name}_missing"] = value is None
    return covariates


def _weighted_neighbor_average(
    readings: dict[tuple[str, int], float | None],
    incoming_edges: tuple[SyntheticTrafficEdge, ...],
    time_index: int,
    lag_steps: int,
) -> float | None:
    lag_index = time_index - lag_steps
    if lag_index < 0:
        return None

    weighted_sum = 0.0
    total_weight = 0.0
    for edge in incoming_edges:
        reading = readings[(edge.source, lag_index)]
        if reading is None:
            continue
        weighted_sum += reading * edge.weight
        total_weight += edge.weight

    if total_weight == 0.0:
        return None
    return round(weighted_sum / total_weight, 3)


def _lag_reading(
    readings: dict[tuple[str, int], float | None],
    sensor_id: str,
    time_index: int,
    lag_steps: int,
) -> float | None:
    lag_index = time_index - lag_steps
    if lag_index < 0:
        return None
    return readings[(sensor_id, lag_index)]


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return datetime.fromisoformat(str(value))


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    numeric = float(value)
    if numeric != numeric:
        return None
    return numeric
