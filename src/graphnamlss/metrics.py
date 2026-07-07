from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from numbers import Real
from statistics import NormalDist
from typing import Any

from graphnamlss.likelihood import gaussian_nll


__all__ = [
    "GaussianMetricReport",
    "gaussian_metric_report",
]


_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class GaussianMetricReport:
    unit: str
    observed_count: int
    nll: float
    crps: float
    interval_level: float
    interval_coverage: float
    pit_values: tuple[float, ...]
    mae: float
    rmse: float


def gaussian_metric_report(
    *,
    mu: Any,
    sigma: Any,
    target: Any,
    mask: Any | None = None,
    unit: str,
    interval_level: float = 0.9,
) -> GaussianMetricReport:
    """Return Gaussian distributional metrics over observed targets."""
    checked_interval_level = _interval_level(interval_level)
    observations = _observed_gaussian_observations(
        mu=mu,
        sigma=sigma,
        target=target,
        mask=mask,
    )
    standardized_residuals = tuple(
        (observed_target - mean) / scale
        for mean, scale, observed_target in observations
    )
    errors = tuple(
        mean - observed_target for mean, _, observed_target in observations
    )

    return GaussianMetricReport(
        unit=_metric_unit(unit),
        observed_count=len(observations),
        nll=float(gaussian_nll(mu=mu, sigma=sigma, target=target, mask=mask)),
        crps=_mean(
            _gaussian_crps(mean, scale, observed_target)
            for mean, scale, observed_target in observations
        ),
        interval_level=checked_interval_level,
        interval_coverage=_interval_coverage(
            standardized_residuals, checked_interval_level
        ),
        pit_values=tuple(
            _STANDARD_NORMAL.cdf(residual) for residual in standardized_residuals
        ),
        mae=sum(abs(error) for error in errors) / len(errors),
        rmse=math.sqrt(sum(error**2 for error in errors) / len(errors)),
    )


def _gaussian_crps(mean: float, scale: float, observed_target: float) -> float:
    standardized_residual = (observed_target - mean) / scale
    return scale * (
        standardized_residual
        * (2.0 * _STANDARD_NORMAL.cdf(standardized_residual) - 1.0)
        + 2.0 * _normal_pdf(standardized_residual)
        - 1.0 / math.sqrt(math.pi)
    )


def _observed_gaussian_observations(
    *,
    mu: Any,
    sigma: Any,
    target: Any,
    mask: Any | None,
) -> tuple[tuple[float, float, float], ...]:
    targets = _as_float_tuple(target, "target")
    mus = _broadcast_or_match(_as_float_tuple(mu, "mu"), len(targets), "mu")
    sigmas = _broadcast_or_match(
        _as_float_tuple(sigma, "sigma"), len(targets), "sigma"
    )
    masks = (
        (True,) * len(targets)
        if mask is None
        else _broadcast_or_match(_as_bool_tuple(mask), len(targets), "mask")
    )

    observations = []
    for mean, scale, observed_target, observed in zip(
        mus, sigmas, targets, masks, strict=True
    ):
        if not observed:
            continue
        if not math.isfinite(mean) or not math.isfinite(observed_target):
            raise ValueError("Gaussian metric values must be finite.")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Gaussian sigma values must be positive and finite.")
        observations.append((mean, scale, observed_target))

    if not observations:
        raise ValueError("Gaussian metrics require at least one observed target.")
    return tuple(observations)


def _metric_unit(unit: str) -> str:
    if not unit:
        raise ValueError("Metric unit label must be non-empty.")
    return unit


def _interval_level(interval_level: float) -> float:
    if not math.isfinite(interval_level) or not 0.0 < interval_level < 1.0:
        raise ValueError("Prediction interval level must be between 0 and 1.")
    return interval_level


def _interval_coverage(
    standardized_residuals: tuple[float, ...], interval_level: float
) -> float:
    tail_probability = (1.0 - interval_level) / 2.0
    lower = _STANDARD_NORMAL.inv_cdf(tail_probability)
    upper = _STANDARD_NORMAL.inv_cdf(1.0 - tail_probability)
    covered = sum(lower <= residual <= upper for residual in standardized_residuals)
    return covered / len(standardized_residuals)


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value**2) / math.sqrt(2.0 * math.pi)


def _mean(values: Iterable[float]) -> float:
    values_tuple = tuple(values)
    return sum(values_tuple) / len(values_tuple)


def _as_float_tuple(value: Real | Iterable[Real], label: str) -> tuple[float, ...]:
    if isinstance(value, Real):
        return (float(value),)

    values = tuple(float(item) for item in value)
    if not values:
        raise ValueError(f"Gaussian metrics require at least one {label} value.")
    return values


def _as_bool_tuple(value: bool | Iterable[bool]) -> tuple[bool, ...]:
    if isinstance(value, bool):
        return (value,)
    return tuple(bool(item) for item in value)


def _broadcast_or_match(
    values: tuple[float, ...] | tuple[bool, ...], size: int, label: str
) -> tuple[float, ...] | tuple[bool, ...]:
    if len(values) == size:
        return values
    if len(values) == 1:
        return values * size
    raise ValueError(
        f"Gaussian metrics {label} length must be 1 or match target length."
    )
