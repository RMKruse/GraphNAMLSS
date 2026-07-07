"""Centered local additive shape functions for NAMLSS predictors."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal


CenteringMode = Literal["none", "batch", "reference"]
ShapeFunction = Callable[[float], Real]

__all__ = [
    "CenteredLocalShapeFunctions",
    "CenteringMode",
    "LocalShapeResult",
    "ShapeFunction",
]


@dataclass(frozen=True)
class LocalShapeResult:
    """Predictions and local contributions from a local additive predictor."""

    parameter_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_values: tuple[tuple[float, ...], ...]
    eta: tuple[tuple[float, ...], ...]
    contributions: tuple[tuple[tuple[float, ...], ...], ...]

    @property
    def eta_shape(self) -> tuple[int, int]:
        return (len(self.eta), len(self.parameter_names))

    @property
    def contribution_shape(self) -> tuple[int, int, int]:
        return (
            len(self.contributions),
            len(self.parameter_names),
            len(self.feature_names),
        )

    def contribution_rows(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for observation_index, observation_contributions in enumerate(self.contributions):
            for parameter_index, parameter_contributions in enumerate(
                observation_contributions
            ):
                for feature_index, contribution in enumerate(parameter_contributions):
                    rows.append(
                        {
                            "observation": observation_index,
                            "parameter": self.parameter_names[parameter_index],
                            "feature": self.feature_names[feature_index],
                            "feature_value": self.feature_values[observation_index][
                                feature_index
                            ],
                            "contribution": contribution,
                        }
                    )
        return rows


class CenteredLocalShapeFunctions:
    """Local additive predictor with centered per-parameter feature effects."""

    def __init__(
        self,
        *,
        parameter_names: Sequence[str],
        feature_names: Sequence[str],
        shape_functions: Mapping[str, Mapping[str, ShapeFunction]],
        intercepts: Mapping[str, Real] | None = None,
        centering: CenteringMode = "batch",
    ) -> None:
        self.parameter_names = _names_tuple(parameter_names, "parameter_names")
        self.feature_names = _names_tuple(feature_names, "feature_names")
        self.centering = _validate_centering(centering)
        self._shape_functions = _shape_function_grid(
            self.parameter_names, self.feature_names, shape_functions
        )
        self._intercepts = _intercept_tuple(self.parameter_names, intercepts or {})
        self._reference_centers: tuple[tuple[float, ...], ...] | None = None

    def fit_reference(
        self, observations: Sequence[Mapping[str, Real] | Sequence[Real]]
    ) -> CenteredLocalShapeFunctions:
        """Store training-reference contribution centers for later predictions."""
        feature_values = self._coerce_observations(observations)
        self._reference_centers = _mean_contributions(
            self._raw_contributions(feature_values)
        )
        return self

    def predict(
        self,
        observations: Sequence[Mapping[str, Real] | Sequence[Real]],
        *,
        centering: CenteringMode | None = None,
        reference: Sequence[Mapping[str, Real] | Sequence[Real]] | None = None,
    ) -> LocalShapeResult:
        """Return additive predictors and local contributions for observations."""
        centering_mode = (
            self.centering if centering is None else _validate_centering(centering)
        )
        feature_values = self._coerce_observations(observations)
        raw_contributions = self._raw_contributions(feature_values)
        centers = self._centers_for(centering_mode, raw_contributions, reference)
        contributions = _subtract_centers(raw_contributions, centers)
        eta = _eta_from_contributions(contributions, self._intercepts)

        return LocalShapeResult(
            parameter_names=self.parameter_names,
            feature_names=self.feature_names,
            feature_values=feature_values,
            eta=eta,
            contributions=contributions,
        )

    __call__ = predict

    def _coerce_observations(
        self, observations: Sequence[Mapping[str, Real] | Sequence[Real]]
    ) -> tuple[tuple[float, ...], ...]:
        if not observations:
            raise ValueError("Local shape prediction requires at least one observation.")

        rows = []
        for observation in observations:
            if isinstance(observation, Mapping):
                rows.append(
                    tuple(
                        _finite_float(observation[name], name)
                        for name in self.feature_names
                    )
                )
                continue

            if len(observation) != len(self.feature_names):
                raise ValueError(
                    "Observation feature count must match feature_names length."
                )
            rows.append(
                tuple(
                    _finite_float(value, self.feature_names[index])
                    for index, value in enumerate(observation)
                )
            )
        return tuple(rows)

    def _raw_contributions(
        self, feature_values: tuple[tuple[float, ...], ...]
    ) -> tuple[tuple[tuple[float, ...], ...], ...]:
        rows = []
        for observation in feature_values:
            parameter_rows = []
            for parameter_functions in self._shape_functions:
                parameter_rows.append(
                    tuple(
                        _finite_float(shape_function(value), "shape contribution")
                        for shape_function, value in zip(
                            parameter_functions, observation, strict=True
                        )
                    )
                )
            rows.append(tuple(parameter_rows))
        return tuple(rows)

    def _centers_for(
        self,
        centering: CenteringMode,
        raw_contributions: tuple[tuple[tuple[float, ...], ...], ...],
        reference: Sequence[Mapping[str, Real] | Sequence[Real]] | None,
    ) -> tuple[tuple[float, ...], ...]:
        if centering == "none":
            return tuple((0.0,) * len(self.feature_names) for _ in self.parameter_names)

        if centering == "batch":
            return _mean_contributions(raw_contributions)

        if reference is not None:
            return _mean_contributions(
                self._raw_contributions(self._coerce_observations(reference))
            )

        if self._reference_centers is None:
            raise ValueError(
                "Reference centering requires fit_reference(...) or a reference batch."
            )
        return self._reference_centers


def _names_tuple(names: Sequence[str], field_name: str) -> tuple[str, ...]:
    if not names:
        raise ValueError(f"{field_name} must contain at least one name.")
    names_tuple = tuple(names)
    if any(not isinstance(name, str) or not name for name in names_tuple):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    if len(set(names_tuple)) != len(names_tuple):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return names_tuple


def _validate_centering(centering: str) -> CenteringMode:
    if centering not in {"none", "batch", "reference"}:
        raise ValueError("centering must be one of: none, batch, reference.")
    return centering  # type: ignore[return-value]


def _shape_function_grid(
    parameter_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    shape_functions: Mapping[str, Mapping[str, ShapeFunction]],
) -> tuple[tuple[ShapeFunction, ...], ...]:
    grid = []
    for parameter_name in parameter_names:
        if parameter_name not in shape_functions:
            raise ValueError(f"Missing shape functions for parameter {parameter_name!r}.")
        parameter_functions = shape_functions[parameter_name]
        grid.append(
            tuple(
                _shape_function(parameter_functions, parameter_name, feature_name)
                for feature_name in feature_names
            )
        )
    return tuple(grid)


def _shape_function(
    parameter_functions: Mapping[str, ShapeFunction],
    parameter_name: str,
    feature_name: str,
) -> ShapeFunction:
    if feature_name not in parameter_functions:
        raise ValueError(
            f"Missing shape function for parameter {parameter_name!r} "
            f"and feature {feature_name!r}."
        )
    shape_function = parameter_functions[feature_name]
    if not callable(shape_function):
        raise TypeError("Shape functions must be callable.")
    return shape_function


def _intercept_tuple(
    parameter_names: tuple[str, ...], intercepts: Mapping[str, Real]
) -> tuple[float, ...]:
    return tuple(
        _finite_float(intercepts.get(name, 0.0), name) for name in parameter_names
    )


def _finite_float(value: Real, field_name: str) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number.")
    float_value = float(value)
    if not math.isfinite(float_value):
        raise ValueError(f"{field_name} must be finite.")
    return float_value


def _mean_contributions(
    contributions: tuple[tuple[tuple[float, ...], ...], ...]
) -> tuple[tuple[float, ...], ...]:
    observation_count = len(contributions)
    parameter_count = len(contributions[0])
    feature_count = len(contributions[0][0])

    centers = []
    for parameter_index in range(parameter_count):
        centers.append(
            tuple(
                sum(
                    observation[parameter_index][feature_index]
                    for observation in contributions
                )
                / observation_count
                for feature_index in range(feature_count)
            )
        )
    return tuple(centers)


def _subtract_centers(
    contributions: tuple[tuple[tuple[float, ...], ...], ...],
    centers: tuple[tuple[float, ...], ...],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(
                contribution - centers[parameter_index][feature_index]
                for feature_index, contribution in enumerate(parameter_contributions)
            )
            for parameter_index, parameter_contributions in enumerate(
                observation_contributions
            )
        )
        for observation_contributions in contributions
    )


def _eta_from_contributions(
    contributions: tuple[tuple[tuple[float, ...], ...], ...],
    intercepts: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            intercepts[parameter_index] + sum(parameter_contributions)
            for parameter_index, parameter_contributions in enumerate(
                observation_contributions
            )
        )
        for observation_contributions in contributions
    )
