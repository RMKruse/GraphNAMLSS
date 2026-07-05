from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any


__all__ = [
    "TrafficGaussianPrediction",
    "TrafficScalingPolicy",
    "TrafficScalingStatistic",
    "fit_traffic_scaling_policy",
]


@dataclass(frozen=True)
class TrafficScalingStatistic:
    mean: float
    scale: float


@dataclass(frozen=True)
class TrafficGaussianPrediction:
    mu: float
    sigma: float
    unit: str


@dataclass(frozen=True)
class TrafficScalingPolicy:
    target: TrafficScalingStatistic
    features: dict[str, TrafficScalingStatistic]

    def inverse_transform_gaussian(
        self, mu: float, sigma: float
    ) -> TrafficGaussianPrediction:
        if sigma < 0.0:
            raise ValueError("Gaussian sigma must be non-negative.")

        return TrafficGaussianPrediction(
            mu=mu * self.target.scale + self.target.mean,
            sigma=sigma * self.target.scale,
            unit="original_traffic",
        )

    def transform_split(self, split: Any) -> Any:
        return replace(
            split,
            observations=tuple(
                self._transform_observation(observation)
                for observation in split.observations
            ),
        )

    def _transform_observation(self, observation: Any) -> Any:
        return replace(
            observation,
            current_speed=_scale_optional(
                observation.current_speed, self.features.get("current_speed")
            ),
            target_speed=_scale_optional(observation.target_speed, self.target),
            local_covariates=self._transform_covariates(observation.local_covariates),
            neighbor_covariates=self._transform_covariates(
                observation.neighbor_covariates
            ),
        )

    def _transform_covariates(
        self, covariates: dict[str, float | bool | None]
    ) -> dict[str, float | bool | None]:
        return {
            name: _scale_optional(value, self.features.get(name))
            for name, value in covariates.items()
        }


def fit_traffic_scaling_policy(train_split: Any) -> TrafficScalingPolicy:
    target_values = tuple(
        observation.target_speed
        for observation in train_split.observed_target_observations()
        if observation.target_speed is not None
    )
    feature_values = _feature_values_by_name(train_split)

    return TrafficScalingPolicy(
        target=_fit_statistic(target_values, "target_speed"),
        features={
            name: _fit_statistic(values, name)
            for name, values in feature_values.items()
            if values
        },
    )


def _feature_values_by_name(train_split: Any) -> dict[str, tuple[float, ...]]:
    feature_values: dict[str, list[float]] = {}
    for metadata in train_split.local_feature_metadata:
        if metadata.dtype != "indicator":
            feature_values[metadata.name] = []
    for metadata in train_split.neighbor_feature_metadata:
        if metadata.dtype != "indicator":
            feature_values[metadata.name] = []

    for observation in train_split.observations:
        for name, value in observation.local_covariates.items():
            _append_numeric_feature_value(feature_values, name, value)
        for name, value in observation.neighbor_covariates.items():
            _append_numeric_feature_value(feature_values, name, value)

    return {name: tuple(values) for name, values in feature_values.items()}


def _append_numeric_feature_value(
    feature_values: dict[str, list[float]], name: str, value: float | bool | None
) -> None:
    if name not in feature_values or value is None or isinstance(value, bool):
        return
    feature_values[name].append(float(value))


def _scale_optional(
    value: float | bool | None, statistic: TrafficScalingStatistic | None
) -> float | bool | None:
    if value is None or isinstance(value, bool) or statistic is None:
        return value
    return (value - statistic.mean) / statistic.scale


def _fit_statistic(values: tuple[float, ...], label: str) -> TrafficScalingStatistic:
    if not values:
        raise ValueError(f"Cannot fit traffic scaling statistic without {label} values.")

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)

    return TrafficScalingStatistic(mean=mean, scale=scale if scale > 0.0 else 1.0)
