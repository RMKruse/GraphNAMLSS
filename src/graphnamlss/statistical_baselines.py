from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from graphnamlss.metrics import GaussianMetricReport, gaussian_metric_report


__all__ = [
    "StatisticalBaselineAdapterContract",
    "StatisticalBaselineAdapterInput",
    "StatisticalBaselinePrediction",
]


@dataclass(frozen=True)
class StatisticalBaselineAdapterContract:
    """Named statistical baseline contract for future external adapters."""

    baseline_name: str
    backend: str
    uses_graph_features: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_name",
            _non_empty_string(self.baseline_name, "baseline_name"),
        )
        object.__setattr__(self, "backend", _non_empty_string(self.backend, "backend"))
        object.__setattr__(self, "uses_graph_features", bool(self.uses_graph_features))

    @classmethod
    def gamlss(cls) -> StatisticalBaselineAdapterContract:
        return cls(baseline_name="gamlss", backend="gamlss")

    @classmethod
    def bamlss(cls) -> StatisticalBaselineAdapterContract:
        return cls(baseline_name="bamlss", backend="bamlss")

    @classmethod
    def graph_feature_gamlss(cls) -> StatisticalBaselineAdapterContract:
        return cls(
            baseline_name="gamlss_graph_features",
            backend="gamlss",
            uses_graph_features=True,
        )

    @classmethod
    def graph_feature_bamlss(cls) -> StatisticalBaselineAdapterContract:
        return cls(
            baseline_name="bamlss_graph_features",
            backend="bamlss",
            uses_graph_features=True,
        )

    def model_covariate_columns(
        self, adapter_input: StatisticalBaselineAdapterInput
    ) -> tuple[str, ...]:
        if self.uses_graph_features:
            if not adapter_input.graph_feature_covariate_columns:
                raise ValueError(
                    "Graph-feature statistical baselines require graph-feature "
                    "covariate columns."
                )
            return adapter_input.covariate_columns
        return tuple(adapter_input.local_covariate_columns)


@dataclass(frozen=True)
class StatisticalBaselineAdapterInput:
    """Validated table-shaped input for statistical baseline adapters."""

    rows: Sequence[Mapping[str, Any]]
    target_column: str
    local_covariate_columns: Iterable[str]
    graph_feature_covariate_columns: Iterable[str] = ()
    exposure_columns: Iterable[str] = ()
    split_column: str = "split"

    def __post_init__(self) -> None:
        rows = tuple(dict(row) for row in self.rows)
        if not rows:
            raise ValueError("Statistical baseline adapter input requires rows.")

        target_column = _column_name(self.target_column, "target_column")
        local_covariate_columns = _column_names(
            self.local_covariate_columns, "local_covariate_columns"
        )
        graph_feature_covariate_columns = _column_names(
            self.graph_feature_covariate_columns,
            "graph_feature_covariate_columns",
        )
        exposure_columns = _column_names(self.exposure_columns, "exposure_columns")
        split_column = _column_name(self.split_column, "split_column")

        all_columns = (
            (target_column,)
            + local_covariate_columns
            + graph_feature_covariate_columns
            + exposure_columns
            + (split_column,)
        )
        _require_unique_columns(all_columns)
        _require_row_columns(rows, all_columns)

        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "target_column", target_column)
        object.__setattr__(self, "local_covariate_columns", local_covariate_columns)
        object.__setattr__(
            self,
            "graph_feature_covariate_columns",
            graph_feature_covariate_columns,
        )
        object.__setattr__(self, "exposure_columns", exposure_columns)
        object.__setattr__(self, "split_column", split_column)

    @property
    def covariate_columns(self) -> tuple[str, ...]:
        return (
            tuple(self.local_covariate_columns)
            + tuple(self.graph_feature_covariate_columns)
        )

    @property
    def target_values(self) -> tuple[float, ...]:
        return tuple(
            _finite_number(row[self.target_column], self.target_column)
            for row in self.rows
        )

    @property
    def split_values(self) -> tuple[str, ...]:
        return tuple(str(row[self.split_column]) for row in self.rows)

    def exposure_values(self, column: str) -> tuple[float, ...]:
        exposure_column = _column_name(column, "exposure column")
        if exposure_column not in self.exposure_columns:
            raise ValueError(
                f"Column {exposure_column!r} is not declared as an exposure column."
            )
        return tuple(
            _finite_number(row[exposure_column], exposure_column) for row in self.rows
        )


@dataclass(frozen=True)
class StatisticalBaselinePrediction:
    """Distribution parameter predictions returned by a statistical adapter."""

    distribution: str
    parameters: Mapping[str, Sequence[Real]]
    unit: str

    def __post_init__(self) -> None:
        distribution = _distribution_name(self.distribution)
        unit = _non_empty_string(self.unit, "unit")
        parameters = {
            _non_empty_string(name, "parameter name"): tuple(values)
            for name, values in self.parameters.items()
        }

        if distribution != "gaussian":
            raise ValueError(
                "Only gaussian statistical baseline predictions are supported."
            )
        if set(parameters) != {"mu", "sigma"}:
            raise ValueError(
                "Gaussian statistical baseline predictions require mu and sigma."
            )
        _require_matching_parameter_lengths(parameters)

        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "unit", unit)

    @classmethod
    def gaussian(
        cls, *, mu: Sequence[Real], sigma: Sequence[Real], unit: str
    ) -> StatisticalBaselinePrediction:
        return cls(
            distribution="gaussian",
            parameters={"mu": tuple(mu), "sigma": tuple(sigma)},
            unit=unit,
        )

    def gaussian_metric_report(
        self, *, target: Any, mask: Any | None = None
    ) -> GaussianMetricReport:
        if self.distribution != "gaussian":
            raise ValueError("Gaussian metrics require gaussian predictions.")
        return gaussian_metric_report(
            mu=self.parameters["mu"],
            sigma=self.parameters["sigma"],
            target=target,
            mask=mask,
            unit=self.unit,
        )


def _column_names(columns: Iterable[str], label: str) -> tuple[str, ...]:
    return tuple(_column_name(column, label) for column in columns)


def _column_name(column: str, label: str) -> str:
    return _non_empty_string(column, label)


def _non_empty_string(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} entries must be non-empty strings.")
    return value


def _require_unique_columns(columns: tuple[str, ...]) -> None:
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(
            "Statistical baseline adapter columns must be unique: "
            + ", ".join(duplicates)
        )


def _require_row_columns(
    rows: tuple[Mapping[str, Any], ...], columns: tuple[str, ...]
) -> None:
    for row_index, row in enumerate(rows):
        missing = [column for column in columns if column not in row]
        if missing:
            raise ValueError(
                f"Statistical baseline row {row_index} is missing columns: "
                + ", ".join(missing)
            )


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, Real):
        raise ValueError(f"{label} values must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} values must be finite.")
    return number


def _distribution_name(distribution: str) -> str:
    return _non_empty_string(distribution, "distribution").lower()


def _require_matching_parameter_lengths(
    parameters: Mapping[str, tuple[Real, ...]]
) -> None:
    lengths = {name: len(values) for name, values in parameters.items()}
    if any(length == 0 for length in lengths.values()):
        raise ValueError("Statistical baseline prediction parameters cannot be empty.")
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "Statistical baseline prediction parameters must have matching lengths."
        )
