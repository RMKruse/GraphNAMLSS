from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from graphnamlss.config import ConfigError, ExperimentConfig
from graphnamlss.likelihood import positive_scale
from graphnamlss.local_shape import CenteredLocalShapeFunctions
from graphnamlss.metrics import GaussianMetricReport, gaussian_metric_report
from graphnamlss.testing import load_synthetic_traffic_fixture_from_config
from graphnamlss.traffic_scaling import TrafficScalingPolicy, fit_traffic_scaling_policy


__all__ = ["train_deterministic_namlss_on_synthetic_traffic"]


def train_deterministic_namlss_on_synthetic_traffic(
    config: ExperimentConfig, *, seed: int = 20250214
) -> dict[str, Any]:
    _validate_synthetic_training_config(config)

    fixture = load_synthetic_traffic_fixture_from_config(config, seed=seed)
    splits = fixture.temporal_split_observations(
        train=config.split.train,
        validation=config.split.validation,
        test=config.split.test,
    )

    scaling_policy = fit_traffic_scaling_policy(splits["train"])
    scaled_splits = {
        split_name: scaling_policy.transform_split(split)
        for split_name, split in splits.items()
    }

    feature_names = config.features.local
    train_rows = _feature_rows(scaled_splits["train"].observations, feature_names)
    observed_train = _observed_training_examples(
        scaled_splits["train"].observations, feature_names
    )
    feature_centers = _feature_centers(train_rows, feature_names)
    intercept, weights = _fit_linear_location(observed_train, feature_centers)
    sigma = _training_sigma(observed_train, feature_centers, intercept, weights)

    model = _local_shape_model(
        feature_names=feature_names,
        intercept=intercept,
        weights=weights,
        sigma=sigma,
    ).fit_reference(train_rows)

    predictions = {
        split_name: _predict_split(model, split.observations, feature_names)
        for split_name, split in scaled_splits.items()
    }
    metrics = {
        split_name: _metrics_for_split(predictions[split_name], split.observations)
        for split_name, split in scaled_splits.items()
    }

    resolved = config.to_resolved_dict()
    resolved["train"] = True
    resolved["training_seed"] = seed

    loss = {split_name: report.nll for split_name, report in metrics.items()}
    history = [
        {
            "epoch": 1,
            "loss": loss["train"],
            "optimizer": config.optimizer.name,
        }
    ]
    split_summary = {
        split_name: _split_summary(split.observations)
        for split_name, split in scaled_splits.items()
    }

    output = {
        "run_name": config.output.run_name,
        "model": {
            "family": config.model.family,
            "variant": config.model.variant,
            "distribution": config.model.distribution,
        },
        "loss": loss,
        "metrics": {
            split_name: _metric_report_dict(report)
            for split_name, report in metrics.items()
        },
        "resolved_config": resolved,
        "scaling": _scaling_summary(config, scaling_policy),
        "missingness": {
            "target_policy": config.missingness.target,
            "covariate_policy": config.missingness.covariates,
            "fixture": fixture.missingness_report().__dict__,
        },
        "splits": split_summary,
        "training_history": history,
        "parameters": {
            "intercept_mu": intercept,
            "weights_mu": dict(zip(feature_names, weights, strict=True)),
            "sigma": sigma,
        },
    }

    _write_training_artifacts(config.output.directory, resolved, output)
    return output


def _validate_synthetic_training_config(config: ExperimentConfig) -> None:
    if "synthetic" not in config.dataset.name.lower():
        raise ConfigError(
            "The train command currently supports the synthetic traffic fixture only."
        )
    if config.model.distribution != "gaussian_location_scale":
        raise ConfigError("Deterministic NAMLSS training requires gaussian_location_scale.")
    if config.missingness is None:
        raise ConfigError("Training requires an explicit traffic missingness policy.")
    if config.missingness.target != "mask_missing_targets":
        raise ConfigError("Training requires missingness.target=mask_missing_targets.")
    if config.missingness.future_looking_imputation:
        raise ConfigError("Training does not allow future-looking imputation.")
    if config.scaling is None:
        raise ConfigError("Training requires an explicit traffic scaling policy.")
    if (
        config.scaling.mode != "train_period_standardization"
        or config.scaling.fit_on != "train"
    ):
        raise ConfigError("Training requires train-period standardization.")


def _feature_rows(
    observations: Sequence[Any], feature_names: Sequence[str]
) -> tuple[dict[str, float], ...]:
    return tuple(_feature_row(observation, feature_names) for observation in observations)


def _observed_training_examples(
    observations: Sequence[Any], feature_names: Sequence[str]
) -> tuple[tuple[dict[str, float], float], ...]:
    examples = tuple(
        (_feature_row(observation, feature_names), float(observation.target_speed))
        for observation in observations
        if not observation.target_missing and observation.target_speed is not None
    )
    if not examples:
        raise ConfigError("Training requires at least one observed target.")
    return examples


def _feature_row(observation: Any, feature_names: Sequence[str]) -> dict[str, float]:
    return {
        name: _feature_value(observation.local_covariates.get(name), name)
        for name in feature_names
    }


def _feature_value(value: float | bool | None, name: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, Real):
        raise ConfigError(f"Feature {name} must be numeric, boolean, or missing.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ConfigError(f"Feature {name} must be finite.")
    return numeric


def _feature_centers(
    rows: Sequence[Mapping[str, float]], feature_names: Sequence[str]
) -> dict[str, float]:
    if not rows:
        raise ConfigError("Training requires at least one traffic observation.")
    return {
        name: sum(row[name] for row in rows) / len(rows)
        for name in feature_names
    }


def _fit_linear_location(
    examples: Sequence[tuple[Mapping[str, float], float]],
    feature_centers: Mapping[str, float],
) -> tuple[float, tuple[float, ...]]:
    feature_names = tuple(feature_centers)
    x_rows = [
        (1.0,)
        + tuple(row[name] - feature_centers[name] for name in feature_names)
        for row, _target in examples
    ]
    targets = [target for _row, target in examples]
    coefficients = _solve_ridge_normal_equations(x_rows, targets)
    return coefficients[0], tuple(coefficients[1:])


def _solve_ridge_normal_equations(
    x_rows: Sequence[Sequence[float]], targets: Sequence[float], ridge: float = 1e-6
) -> tuple[float, ...]:
    width = len(x_rows[0])
    matrix = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]

    for x_row, target in zip(x_rows, targets, strict=True):
        for row_index, row_value in enumerate(x_row):
            rhs[row_index] += row_value * target
            for column_index, column_value in enumerate(x_row):
                matrix[row_index][column_index] += row_value * column_value

    for index in range(1, width):
        matrix[index][index] += ridge

    return _solve_linear_system(matrix, rhs)


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> tuple[float, ...]:
    size = len(rhs)
    augmented = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs, strict=True)]

    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ConfigError("Could not fit deterministic NAMLSS location model.")
        augmented[pivot_index], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot_index],
        )

        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            augmented[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row_index], augmented[pivot_index], strict=True
                )
            ]

    return tuple(row[-1] for row in augmented)


def _training_sigma(
    examples: Sequence[tuple[Mapping[str, float], float]],
    feature_centers: Mapping[str, float],
    intercept: float,
    weights: Sequence[float],
) -> float:
    feature_names = tuple(feature_centers)
    residuals = []
    for row, target in examples:
        mu = intercept + sum(
            weight * (row[name] - feature_centers[name])
            for weight, name in zip(weights, feature_names, strict=True)
        )
        residuals.append(target - mu)
    variance = sum(residual**2 for residual in residuals) / len(residuals)
    return max(math.sqrt(variance), 1e-6)


def _local_shape_model(
    *,
    feature_names: Sequence[str],
    intercept: float,
    weights: Sequence[float],
    sigma: float,
) -> CenteredLocalShapeFunctions:
    sigma_eta = _inverse_softplus(sigma)
    zero_shapes = {name: _constant_zero for name in feature_names}
    return CenteredLocalShapeFunctions(
        parameter_names=("mu", "sigma"),
        feature_names=feature_names,
        intercepts={"mu": intercept, "sigma": sigma_eta},
        centering="reference",
        shape_functions={
            "mu": {
                name: _linear_shape(weight)
                for name, weight in zip(feature_names, weights, strict=True)
            },
            "sigma": zero_shapes,
        },
    )


def _linear_shape(weight: float):
    def shape(value: float) -> float:
        return weight * value

    return shape


def _constant_zero(_value: float) -> float:
    return 0.0


def _inverse_softplus(value: float) -> float:
    adjusted = value - 1e-6
    if adjusted <= 0.0:
        return -50.0
    if adjusted > 20.0:
        return adjusted
    return math.log(math.expm1(adjusted))


def _predict_split(
    model: CenteredLocalShapeFunctions,
    observations: Sequence[Any],
    feature_names: Sequence[str],
) -> dict[str, tuple[float, ...]]:
    rows = _feature_rows(observations, feature_names)
    local = model.predict(rows)
    mu = tuple(eta_row[0] for eta_row in local.eta)
    sigma = tuple(float(positive_scale(eta_row[1])) for eta_row in local.eta)
    return {"mu": mu, "sigma": sigma}


def _metrics_for_split(
    prediction: Mapping[str, tuple[float, ...]], observations: Sequence[Any]
) -> GaussianMetricReport:
    targets = tuple(
        0.0 if observation.target_speed is None else float(observation.target_speed)
        for observation in observations
    )
    mask = tuple(
        (not observation.target_missing) and observation.target_speed is not None
        for observation in observations
    )
    return gaussian_metric_report(
        mu=prediction["mu"],
        sigma=prediction["sigma"],
        target=targets,
        mask=mask,
        unit="standardized_traffic",
    )


def _metric_report_dict(report: GaussianMetricReport) -> dict[str, Any]:
    return {
        "unit": report.unit,
        "observed_count": report.observed_count,
        "nll": report.nll,
        "crps": report.crps,
        "interval_level": report.interval_level,
        "interval_coverage": report.interval_coverage,
        "pit_values": list(report.pit_values),
        "mae": report.mae,
        "rmse": report.rmse,
    }


def _split_summary(observations: Sequence[Any]) -> dict[str, int]:
    prediction_count = len(observations)
    observed_target_count = sum(
        (not observation.target_missing) and observation.target_speed is not None
        for observation in observations
    )
    return {
        "prediction_count": prediction_count,
        "observed_target_count": observed_target_count,
        "masked_target_count": prediction_count - observed_target_count,
    }


def _scaling_summary(
    config: ExperimentConfig, scaling_policy: TrafficScalingPolicy
) -> dict[str, Any]:
    return {
        "mode": config.scaling.mode,
        "target": {
            "fit_on": config.scaling.fit_on,
            "mean": scaling_policy.target.mean,
            "scale": scaling_policy.target.scale,
        },
        "features": {
            name: {"mean": statistic.mean, "scale": statistic.scale}
            for name, statistic in scaling_policy.features.items()
        },
    }


def _write_training_artifacts(
    output_directory: Path, resolved: Mapping[str, Any], summary: Mapping[str, Any]
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "resolved_config.json", resolved)
    _write_json(output_directory / "metrics.json", summary["metrics"])
    _write_json(output_directory / "training_history.json", summary["training_history"])
    _write_json(output_directory / "train_summary.json", summary)


def _write_json(path: Path, payload: Mapping[str, Any] | Iterable[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
