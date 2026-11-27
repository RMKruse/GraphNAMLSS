from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from graphnamlss.config import ConfigError, ExperimentConfig
from graphnamlss.graph_encoder import GraphMetadata
from graphnamlss.likelihood import positive_scale
from graphnamlss.local_shape import CenteredLocalShapeFunctions
from graphnamlss.metrics import (
    GaussianMetricReport,
    PointMetricReport,
    gaussian_metric_report,
    point_metric_report,
)
from graphnamlss.testing import (
    load_localized_congestion_spillover_demo_fixture,
    load_synthetic_traffic_fixture_from_config,
)
from graphnamlss.traffic_data import load_metr_la_traffic_fixture_from_config
from graphnamlss.traffic_scaling import TrafficScalingPolicy, fit_traffic_scaling_policy


TRACER_BULLET_SCALE_FIT_STEPS = 3000
TRACER_BULLET_SCALE_LEARNING_RATE = 0.05
TRACER_BULLET_SCALE_RIDGE = 1e-3


__all__ = [
    "train_namlss_on_traffic",
    "train_deterministic_namlss_on_synthetic_traffic",
    "train_deterministic_namlss_on_traffic",
    "train_tracer_bullet_graph_namlss_on_traffic",
    "train_distributional_gnn_baseline_on_traffic",
    "train_point_gnn_reference_on_traffic",
]


def train_namlss_on_traffic(
    config: ExperimentConfig, *, seed: int = 20250214
) -> dict[str, Any]:
    if _is_point_gnn_reference_config(config):
        return train_point_gnn_reference_on_traffic(config, seed=seed)
    if _is_distributional_gnn_baseline_config(config):
        return train_distributional_gnn_baseline_on_traffic(config, seed=seed)
    if _is_tracer_bullet_graph_config(config):
        return train_tracer_bullet_graph_namlss_on_traffic(config, seed=seed)
    return train_deterministic_namlss_on_traffic(config, seed=seed)


def train_deterministic_namlss_on_traffic(
    config: ExperimentConfig, *, seed: int = 20250214
) -> dict[str, Any]:
    _validate_deterministic_traffic_training_config(config)

    fixture = _load_traffic_fixture(config, seed=seed)
    return _train_deterministic_namlss_on_fixture(config, fixture, seed=seed)


def train_deterministic_namlss_on_synthetic_traffic(
    config: ExperimentConfig, *, seed: int = 20250214
) -> dict[str, Any]:
    _validate_synthetic_training_config(config)
    fixture = load_synthetic_traffic_fixture_from_config(config, seed=seed)
    return _train_deterministic_namlss_on_fixture(config, fixture, seed=seed)


def train_tracer_bullet_graph_namlss_on_traffic(
    config: ExperimentConfig, *, seed: int = 20250214
) -> dict[str, Any]:
    _validate_tracer_bullet_graph_training_config(config)
    fixture = _load_traffic_fixture(config, seed=seed)
    return _train_tracer_bullet_graph_namlss_on_fixture(config, fixture, seed=seed)


def train_distributional_gnn_baseline_on_traffic(
    config: ExperimentConfig, *, seed: int = 20250214
) -> dict[str, Any]:
    _validate_distributional_gnn_training_config(config)
    fixture = _load_traffic_fixture(config, seed=seed)
    return _train_distributional_gnn_baseline_on_fixture(config, fixture, seed=seed)


def train_point_gnn_reference_on_traffic(
    config: ExperimentConfig, *, seed: int = 20250214
) -> dict[str, Any]:
    _validate_point_gnn_reference_training_config(config)
    fixture = _load_traffic_fixture(config, seed=seed)
    return _train_point_gnn_reference_on_fixture(config, fixture, seed=seed)


def _train_deterministic_namlss_on_fixture(
    config: ExperimentConfig, fixture: Any, *, seed: int
) -> dict[str, Any]:
    prepared = _prepare_deterministic_local_training(config, fixture)
    scaled_splits = prepared["scaled_splits"]
    feature_names = prepared["feature_names"]
    scaling_policy = prepared["scaling_policy"]
    model = prepared["model"]
    intercept = prepared["intercept"]
    weights = prepared["weights"]
    sigma = prepared["sigma"]

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
        "features": {
            "local": list(config.features.local),
            "neighbor": list(config.features.neighbor),
            "local_metadata": [
                feature.__dict__ for feature in fixture.local_feature_metadata
            ],
            "neighbor_metadata": [
                feature.__dict__ for feature in fixture.neighbor_feature_metadata
            ],
        },
        "splits": split_summary,
        "training_history": history,
        "prediction_artifact": "predictions.json",
        "parameters": {
            "intercept_mu": intercept,
            "weights_mu": dict(zip(feature_names, weights, strict=True)),
            "sigma": sigma,
        },
    }

    _write_training_artifacts(config.output.directory, resolved, output)
    _write_json(
        config.output.directory / "predictions.json",
        _prediction_artifact(predictions, scaled_splits, scaling_policy),
    )
    return output


def _train_distributional_gnn_baseline_on_fixture(
    config: ExperimentConfig, fixture: Any, *, seed: int
) -> dict[str, Any]:
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
    graph_state = _fit_distributional_gnn_state(
        config,
        scaled_splits["train"].observations,
        fixture,
    )
    predictions = {
        split_name: _distributional_gnn_prediction_for_split(
            graph_state,
            split.observations,
        )
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
            "epoch": epoch,
            "loss": loss["train"],
            "optimizer": config.optimizer.name,
        }
        for epoch in range(1, config.optimizer.max_epochs + 1)
    ]
    split_summary = {
        split_name: _split_summary(split.observations)
        for split_name, split in scaled_splits.items()
    }
    train_sigma = predictions["train"]["sigma"]

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
        "features": {
            "local": list(config.features.local),
            "neighbor": list(config.features.neighbor),
            "local_metadata": [
                feature.__dict__ for feature in fixture.local_feature_metadata
            ],
            "neighbor_metadata": [
                feature.__dict__ for feature in fixture.neighbor_feature_metadata
            ],
        },
        "graph": {
            "representation": config.model.graph_representation,
            "encoder": config.model.graph_encoder,
            "injection": config.model.graph_injection,
            "self_feature_access": config.model.graph_stream_self_feature_access,
            "node_count": len(fixture.sensors),
            "edge_count": len(fixture.adjacency),
            "feature_names": list(graph_state["feature_names"]),
            "embedding_dim": graph_state["embedding_dim"],
        },
        "splits": split_summary,
        "training_history": history,
        "prediction_artifact": "predictions.json",
        "parameters": {
            "intercept_mu": graph_state["intercept_mu"],
            "graph_weights_mu": dict(
                zip(
                    graph_state["embedding_names"],
                    graph_state["graph_weights_mu"],
                    strict=True,
                )
            ),
            "intercept_sigma": graph_state["intercept_sigma"],
            "graph_weights_sigma": dict(
                zip(
                    graph_state["embedding_names"],
                    graph_state["graph_weights_sigma"],
                    strict=True,
                )
            ),
            "sigma": sum(train_sigma) / len(train_sigma),
        },
    }

    _write_training_artifacts(config.output.directory, resolved, output)
    _write_json(
        config.output.directory / "predictions.json",
        _prediction_artifact(predictions, scaled_splits, scaling_policy),
    )
    return output


def _train_point_gnn_reference_on_fixture(
    config: ExperimentConfig, fixture: Any, *, seed: int
) -> dict[str, Any]:
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
    graph_state = _fit_point_gnn_state(
        config,
        scaled_splits["train"].observations,
        fixture,
    )
    predictions = {
        split_name: _point_gnn_prediction_for_split(
            graph_state,
            split.observations,
        )
        for split_name, split in scaled_splits.items()
    }
    metrics = {
        split_name: _point_metrics_for_split(
            predictions[split_name], split.observations
        )
        for split_name, split in scaled_splits.items()
    }

    resolved = config.to_resolved_dict()
    resolved["train"] = True
    resolved["training_seed"] = seed

    loss = {split_name: report.rmse for split_name, report in metrics.items()}
    history = [
        {
            "epoch": epoch,
            "loss": loss["train"],
            "optimizer": config.optimizer.name,
        }
        for epoch in range(1, config.optimizer.max_epochs + 1)
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
            "prediction_type": "point",
            "role": "point_prediction_reference",
        },
        "loss": loss,
        "metrics": {
            split_name: _point_metric_report_dict(report)
            for split_name, report in metrics.items()
        },
        "resolved_config": resolved,
        "reference": {
            "prediction_type": "point",
            "role": "point_prediction_reference",
            "evaluated_with": ["mae", "rmse"],
        },
        "scaling": _scaling_summary(config, scaling_policy),
        "missingness": {
            "target_policy": config.missingness.target,
            "covariate_policy": config.missingness.covariates,
            "fixture": fixture.missingness_report().__dict__,
        },
        "features": {
            "local": list(config.features.local),
            "neighbor": list(config.features.neighbor),
            "local_metadata": [
                feature.__dict__ for feature in fixture.local_feature_metadata
            ],
            "neighbor_metadata": [
                feature.__dict__ for feature in fixture.neighbor_feature_metadata
            ],
        },
        "graph": {
            "representation": config.model.graph_representation,
            "encoder": config.model.graph_encoder,
            "injection": config.model.graph_injection,
            "self_feature_access": config.model.graph_stream_self_feature_access,
            "node_count": len(fixture.sensors),
            "edge_count": len(fixture.adjacency),
            "feature_names": list(graph_state["feature_names"]),
            "embedding_dim": graph_state["embedding_dim"],
        },
        "splits": split_summary,
        "training_history": history,
        "prediction_artifact": "predictions.json",
        "parameters": {
            "intercept_mu": graph_state["intercept_mu"],
            "graph_weights_mu": dict(
                zip(
                    graph_state["embedding_names"],
                    graph_state["graph_weights_mu"],
                    strict=True,
                )
            ),
            "sigma": None,
        },
    }

    _write_training_artifacts(config.output.directory, resolved, output)
    _write_json(
        config.output.directory / "predictions.json",
        _prediction_artifact(predictions, scaled_splits, scaling_policy),
    )
    return output


def _train_tracer_bullet_graph_namlss_on_fixture(
    config: ExperimentConfig, fixture: Any, *, seed: int
) -> dict[str, Any]:
    prepared = _prepare_deterministic_local_training(config, fixture)
    scaled_splits = prepared["scaled_splits"]
    feature_names = prepared["feature_names"]
    scaling_policy = prepared["scaling_policy"]
    model = prepared["model"]
    intercept = prepared["intercept"]
    weights = prepared["weights"]
    sigma = prepared["sigma"]

    local_results = {
        split_name: model.predict(_feature_rows(split.observations, feature_names))
        for split_name, split in scaled_splits.items()
    }
    graph_state = _fit_tracer_bullet_graph_state(
        config,
        scaled_splits["train"].observations,
        fixture,
        local_results["train"].eta,
    )
    graph_results = {
        split_name: _graph_contribution_for_split(
            graph_state,
            split_name=split_name,
            observations=split.observations,
            local_eta=local_results[split_name].eta,
        )
        for split_name, split in scaled_splits.items()
    }
    predictions = {
        split_name: _prediction_from_eta(graph_results[split_name]["eta"])
        for split_name in scaled_splits
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
            "epoch": epoch,
            "loss": loss["train"],
            "optimizer": config.optimizer.name,
        }
        for epoch in range(1, config.optimizer.max_epochs + 1)
    ]
    split_summary = {
        split_name: _split_summary(split.observations)
        for split_name, split in scaled_splits.items()
    }
    graph_gates = _graph_gate_rows(graph_state)
    graph_contributions = {
        "parameter_names": list(graph_state["parameter_names"]),
        "splits": {
            split_name: graph_results[split_name]["rows"]
            for split_name in scaled_splits
        },
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
        "features": {
            "local": list(config.features.local),
            "neighbor": list(config.features.neighbor),
            "local_metadata": [
                feature.__dict__ for feature in fixture.local_feature_metadata
            ],
            "neighbor_metadata": [
                feature.__dict__ for feature in fixture.neighbor_feature_metadata
            ],
        },
        "graph": {
            "representation": config.model.graph_representation,
            "encoder": config.model.graph_encoder,
            "injection": config.model.graph_injection,
            "self_feature_access": config.model.graph_stream_self_feature_access,
            "node_count": len(fixture.sensors),
            "edge_count": len(fixture.adjacency),
            "feature_names": list(graph_state["feature_names"]),
            "embedding_dim": graph_state["embedding_dim"],
            "gate_estimation": graph_state["gate_estimation"],
            "graph_head_normalization": graph_state["graph_head_normalization"],
            "gate_artifact": "graph_gates.json",
            "contribution_artifact": "graph_contributions.json",
        },
        "splits": split_summary,
        "training_history": history,
        "prediction_artifact": "predictions.json",
        "parameters": {
            "intercept_mu": intercept,
            "weights_mu": dict(zip(feature_names, weights, strict=True)),
            "sigma": sigma,
            "graph_weights_mu": dict(
                zip(
                    graph_state["embedding_names"],
                    graph_state["graph_weights_mu"],
                    strict=True,
                )
            ),
            "graph_weights_sigma": dict(
                zip(
                    graph_state["embedding_names"],
                    graph_state["graph_weights_sigma"],
                    strict=True,
                )
            ),
        },
    }

    _write_training_artifacts(config.output.directory, resolved, output)
    _write_json(
        config.output.directory / "predictions.json",
        _prediction_artifact(predictions, scaled_splits, scaling_policy),
    )
    _write_json(config.output.directory / "graph_gates.json", graph_gates)
    _write_json(
        config.output.directory / "graph_contributions.json", graph_contributions
    )
    return output


def _prepare_deterministic_local_training(
    config: ExperimentConfig, fixture: Any
) -> dict[str, Any]:
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

    return {
        "scaled_splits": scaled_splits,
        "scaling_policy": scaling_policy,
        "feature_names": feature_names,
        "model": model,
        "intercept": intercept,
        "weights": weights,
        "sigma": sigma,
    }


SUPPORTED_REAL_TRAFFIC_DATASETS = frozenset({"METR-LA", "PEMS-BAY"})


def _validate_deterministic_traffic_training_config(config: ExperimentConfig) -> None:
    if _is_synthetic_config(config):
        _validate_synthetic_training_config(config)
        return
    if not _is_supported_real_traffic_config(config):
        raise ConfigError(
            "Deterministic NAMLSS training supports synthetic traffic, "
            "METR-LA, and PEMS-BAY."
        )
    _validate_shared_training_config(config)


def _validate_synthetic_training_config(config: ExperimentConfig) -> None:
    if "synthetic" not in config.dataset.name.lower():
        raise ConfigError(
            "The train command currently supports the synthetic traffic fixture only."
        )
    _validate_shared_training_config(config)


def _validate_shared_training_config(config: ExperimentConfig) -> None:
    if config.model.distribution != "gaussian_location_scale":
        raise ConfigError("Deterministic NAMLSS training requires gaussian_location_scale.")
    _validate_deterministic_model_config(config)
    _validate_traffic_policy_config(config)


def _validate_tracer_bullet_graph_training_config(config: ExperimentConfig) -> None:
    if not _is_synthetic_config(config) and not _is_supported_real_traffic_config(config):
        raise ConfigError(
            "Tracer-bullet GraphNAMLSS training supports synthetic traffic, "
            "METR-LA, and PEMS-BAY."
        )
    if config.model.distribution != "gaussian_location_scale":
        raise ConfigError("Tracer-bullet GraphNAMLSS training requires gaussian_location_scale.")
    if not _is_tracer_bullet_graph_config(config):
        raise ConfigError(
            "Tracer-bullet GraphNAMLSS training requires model.family=GraphNAMLSS, "
            "model.variant=tracer_bullet, graph_representation=shared, "
            "graph_injection=additive_graph_contribution, "
            "graph_stream_self_feature_access=neighbor_focused, and "
            "graph_encoder=graphsage."
        )
    if config.graph is None:
        raise ConfigError("Tracer-bullet GraphNAMLSS training requires graph metadata.")
    if not config.features.neighbor:
        raise ConfigError(
            "Tracer-bullet GraphNAMLSS training requires neighbor graph features."
        )
    _validate_traffic_policy_config(config)


def _validate_distributional_gnn_training_config(config: ExperimentConfig) -> None:
    if not _is_synthetic_config(config) and not _is_supported_real_traffic_config(config):
        raise ConfigError(
            "Distributional GNN baseline training supports synthetic traffic, "
            "METR-LA, and PEMS-BAY."
        )
    if config.model.distribution != "gaussian_location_scale":
        raise ConfigError(
            "Distributional GNN baseline training requires gaussian_location_scale."
        )
    if not _is_distributional_gnn_baseline_config(config):
        raise ConfigError(
            "Distributional GNN baseline training requires "
            "model.family=DistributionalGNN, model.variant=baseline, "
            "graph_representation=shared, "
            "graph_injection=direct_distributional_parameters, "
            "graph_stream_self_feature_access=neighbor_focused, and "
            "graph_encoder=graphsage."
        )
    if config.graph is None:
        raise ConfigError("Distributional GNN baseline training requires graph metadata.")
    if not config.features.neighbor:
        raise ConfigError(
            "Distributional GNN baseline training requires neighbor graph features."
        )
    _validate_traffic_policy_config(config)


def _validate_point_gnn_reference_training_config(config: ExperimentConfig) -> None:
    if not _is_synthetic_config(config) and not _is_supported_real_traffic_config(config):
        raise ConfigError(
            "Point GNN reference training supports synthetic traffic, "
            "METR-LA, and PEMS-BAY."
        )
    if config.model.distribution != "conditional_mean":
        raise ConfigError("Point GNN reference training requires conditional_mean.")
    if not _is_point_gnn_reference_config(config):
        raise ConfigError(
            "Point GNN reference training requires "
            "model.family=PointGNN, model.variant=reference, "
            "graph_representation=shared, graph_injection=direct_mean, "
            "graph_stream_self_feature_access=neighbor_focused, and "
            "graph_encoder=graphsage."
        )
    if config.graph is None:
        raise ConfigError("Point GNN reference training requires graph metadata.")
    if not config.features.neighbor:
        raise ConfigError(
            "Point GNN reference training requires neighbor graph features."
        )
    _validate_traffic_policy_config(config)


def _validate_deterministic_model_config(config: ExperimentConfig) -> None:
    if config.model.family != "NAMLSS" or config.model.variant != "deterministic_local":
        raise ConfigError(
            "Deterministic NAMLSS training requires model.family=NAMLSS "
            "and model.variant=deterministic_local."
        )


def _validate_traffic_policy_config(config: ExperimentConfig) -> None:
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


def _is_tracer_bullet_graph_config(config: ExperimentConfig) -> bool:
    return (
        config.model.family == "GraphNAMLSS"
        and config.model.variant == "tracer_bullet"
        and config.model.graph_representation == "shared"
        and config.model.graph_injection == "additive_graph_contribution"
        and config.model.graph_stream_self_feature_access == "neighbor_focused"
        and config.model.graph_encoder == "graphsage"
    )


def _is_distributional_gnn_baseline_config(config: ExperimentConfig) -> bool:
    return (
        config.model.family == "DistributionalGNN"
        and config.model.variant == "baseline"
        and config.model.graph_representation == "shared"
        and config.model.graph_injection == "direct_distributional_parameters"
        and config.model.graph_stream_self_feature_access == "neighbor_focused"
        and config.model.graph_encoder == "graphsage"
    )


def _is_point_gnn_reference_config(config: ExperimentConfig) -> bool:
    return (
        config.model.family == "PointGNN"
        and config.model.variant == "reference"
        and config.model.graph_representation == "shared"
        and config.model.graph_injection == "direct_mean"
        and config.model.graph_stream_self_feature_access == "neighbor_focused"
        and config.model.graph_encoder == "graphsage"
    )


def _load_traffic_fixture(config: ExperimentConfig, *, seed: int) -> Any:
    if config.dataset.name == "synthetic-concept-demo":
        return load_localized_congestion_spillover_demo_fixture(seed=seed)
    if _is_synthetic_config(config):
        return load_synthetic_traffic_fixture_from_config(config, seed=seed)
    return load_metr_la_traffic_fixture_from_config(config)


def _is_synthetic_config(config: ExperimentConfig) -> bool:
    return "synthetic" in config.dataset.name.lower()


def _is_supported_real_traffic_config(config: ExperimentConfig) -> bool:
    return config.dataset.name in SUPPORTED_REAL_TRAFFIC_DATASETS


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


def _fit_tracer_bullet_graph_state(
    config: ExperimentConfig,
    train_observations: Sequence[Any],
    fixture: Any,
    local_eta: Sequence[Sequence[float]],
) -> dict[str, Any]:
    feature_names = tuple(config.features.neighbor)
    metadata = GraphMetadata.from_edges(
        node_ids=fixture.sensors,
        edges=fixture.adjacency,
        use_edge_weights=True,
    )
    train_embeddings = _graph_embeddings_by_observation(
        train_observations,
        feature_names=feature_names,
        metadata=metadata,
    )
    embedding_centers = _embedding_centers(train_embeddings.values(), feature_names)
    training_rows = tuple(
        (
            _center_embedding(
                train_embeddings[(observation.sensor_id, observation.time_index)],
                embedding_centers,
            ),
            float(observation.target_speed),
            float(eta_row[0]),
            float(eta_row[1]),
        )
        for observation, eta_row in zip(train_observations, local_eta, strict=True)
        if not observation.target_missing and observation.target_speed is not None
    )
    if not training_rows:
        raise ConfigError("GraphNAMLSS training requires observed training targets.")

    fitted_mu_coefficients = _fit_graph_residual_weights(
        tuple(
            (embedding, target - local_mu)
            for embedding, target, local_mu, _local_sigma_eta in training_rows
        ),
        feature_names,
    )
    fitted_sigma_coefficients = _fit_graph_scale_weights(
        tuple(
            (
                embedding,
                target
                - (
                    local_mu
                    + _dot(fitted_mu_coefficients, embedding)
                ),
                local_sigma_eta,
            )
            for embedding, target, local_mu, local_sigma_eta in training_rows
        ),
        feature_names,
    )
    graph_weights_mu, mu_gate = _factor_graph_head(fitted_mu_coefficients)
    graph_weights_sigma, sigma_gate = _factor_graph_head(
        fitted_sigma_coefficients
    )
    graph_state = {
        "metadata": metadata,
        "feature_names": feature_names,
        "embedding_names": feature_names,
        "embedding_dim": len(feature_names),
        "embedding_centers": embedding_centers,
        "graph_weights_mu": graph_weights_mu,
        "graph_weights_sigma": graph_weights_sigma,
        "gates": {"mu": mu_gate, "sigma": sigma_gate},
        "gate_estimation": "deterministic_unit_l2_head_factorization",
        "graph_head_normalization": "unit_l2",
        "parameter_names": ("mu", "sigma"),
    }
    # Centered Graph Contributions are centered on training statistics only, so a
    # held-out prediction depends on its own row and this fitted state alone.
    train_raw_rows = _raw_graph_contributions(graph_state, train_observations)
    graph_state["contribution_centers"] = tuple(
        sum(row[index] for row in train_raw_rows) / len(train_raw_rows)
        for index in range(len(graph_state["parameter_names"]))
    )
    return graph_state


def _fit_distributional_gnn_state(
    config: ExperimentConfig,
    train_observations: Sequence[Any],
    fixture: Any,
) -> dict[str, Any]:
    feature_names = tuple(config.features.neighbor)
    metadata = GraphMetadata.from_edges(
        node_ids=fixture.sensors,
        edges=fixture.adjacency,
        use_edge_weights=True,
    )
    train_embeddings = _graph_embeddings_by_observation(
        train_observations,
        feature_names=feature_names,
        metadata=metadata,
    )
    embedding_centers = _embedding_centers(train_embeddings.values(), feature_names)
    examples = tuple(
        (
            _center_embedding(
                train_embeddings[(observation.sensor_id, observation.time_index)],
                embedding_centers,
            ),
            float(observation.target_speed),
        )
        for observation in train_observations
        if not observation.target_missing and observation.target_speed is not None
    )
    if not examples:
        raise ConfigError(
            "Distributional GNN baseline training requires observed training targets."
        )

    intercept_mu, graph_weights_mu = _fit_graph_parameter(examples, feature_names)
    scale_examples = tuple(
        (
            embedding,
            _inverse_softplus(
                max(
                    abs(
                        target
                        - _linear_predict(
                            intercept_mu,
                            graph_weights_mu,
                            embedding,
                        )
                    ),
                    1e-6,
                )
            ),
        )
        for embedding, target in examples
    )
    intercept_sigma, graph_weights_sigma = _fit_graph_parameter(
        scale_examples,
        feature_names,
    )
    return {
        "metadata": metadata,
        "feature_names": feature_names,
        "embedding_names": feature_names,
        "embedding_dim": len(feature_names),
        "embedding_centers": embedding_centers,
        "intercept_mu": intercept_mu,
        "graph_weights_mu": graph_weights_mu,
        "intercept_sigma": intercept_sigma,
        "graph_weights_sigma": graph_weights_sigma,
        "parameter_names": ("mu", "sigma"),
    }


def _fit_point_gnn_state(
    config: ExperimentConfig,
    train_observations: Sequence[Any],
    fixture: Any,
) -> dict[str, Any]:
    feature_names = tuple(config.features.neighbor)
    metadata = GraphMetadata.from_edges(
        node_ids=fixture.sensors,
        edges=fixture.adjacency,
        use_edge_weights=True,
    )
    train_embeddings = _graph_embeddings_by_observation(
        train_observations,
        feature_names=feature_names,
        metadata=metadata,
    )
    embedding_centers = _embedding_centers(train_embeddings.values(), feature_names)
    examples = tuple(
        (
            _center_embedding(
                train_embeddings[(observation.sensor_id, observation.time_index)],
                embedding_centers,
            ),
            float(observation.target_speed),
        )
        for observation in train_observations
        if not observation.target_missing and observation.target_speed is not None
    )
    if not examples:
        raise ConfigError(
            "Point GNN reference training requires observed training targets."
        )

    intercept_mu, graph_weights_mu = _fit_graph_parameter(examples, feature_names)
    return {
        "metadata": metadata,
        "feature_names": feature_names,
        "embedding_names": feature_names,
        "embedding_dim": len(feature_names),
        "embedding_centers": embedding_centers,
        "intercept_mu": intercept_mu,
        "graph_weights_mu": graph_weights_mu,
        "parameter_names": ("mu",),
    }


def _distributional_gnn_prediction_for_split(
    graph_state: Mapping[str, Any],
    observations: Sequence[Any],
) -> dict[str, tuple[float, ...]]:
    embeddings = _graph_embeddings_by_observation(
        observations,
        feature_names=graph_state["feature_names"],
        metadata=graph_state["metadata"],
    )
    eta = tuple(
        (
            _linear_predict(
                graph_state["intercept_mu"],
                graph_state["graph_weights_mu"],
                _center_embedding(
                    embeddings[(observation.sensor_id, observation.time_index)],
                    graph_state["embedding_centers"],
                ),
            ),
            _linear_predict(
                graph_state["intercept_sigma"],
                graph_state["graph_weights_sigma"],
                _center_embedding(
                    embeddings[(observation.sensor_id, observation.time_index)],
                    graph_state["embedding_centers"],
                ),
            ),
        )
        for observation in observations
    )
    return _prediction_from_eta(eta)


def _point_gnn_prediction_for_split(
    graph_state: Mapping[str, Any],
    observations: Sequence[Any],
) -> dict[str, tuple[float, ...]]:
    embeddings = _graph_embeddings_by_observation(
        observations,
        feature_names=graph_state["feature_names"],
        metadata=graph_state["metadata"],
    )
    mu = tuple(
        _linear_predict(
            graph_state["intercept_mu"],
            graph_state["graph_weights_mu"],
            _center_embedding(
                embeddings[(observation.sensor_id, observation.time_index)],
                graph_state["embedding_centers"],
            ),
        )
        for observation in observations
    )
    return {"mu": mu}


def _graph_contribution_for_split(
    graph_state: Mapping[str, Any],
    *,
    split_name: str,
    observations: Sequence[Any],
    local_eta: Sequence[Sequence[float]],
) -> dict[str, Any]:
    raw_rows = _raw_graph_contributions(graph_state, observations)
    centers = graph_state["contribution_centers"]
    rows = []
    eta = []
    for observation, eta_row, raw_row in zip(
        observations, local_eta, raw_rows, strict=True
    ):
        centered_mu = raw_row[0] - centers[0]
        centered_sigma = raw_row[1] - centers[1]
        gated_mu = centered_mu * graph_state["gates"]["mu"]
        gated_sigma = centered_sigma * graph_state["gates"]["sigma"]
        eta.append((float(eta_row[0]) + gated_mu, float(eta_row[1]) + gated_sigma))
        rows.append(
            {
                "split": split_name,
                "sensor_id": observation.sensor_id,
                "time_index": observation.time_index,
                "mu": centered_mu,
                "sigma": centered_sigma,
                "gated_mu": gated_mu,
                "gated_sigma": gated_sigma,
            }
        )

    return {"eta": tuple(eta), "rows": rows}


def _raw_graph_contributions(
    graph_state: Mapping[str, Any], observations: Sequence[Any]
) -> list[tuple[float, float]]:
    embeddings = _graph_embeddings_by_observation(
        observations,
        feature_names=graph_state["feature_names"],
        metadata=graph_state["metadata"],
    )
    raw_rows = []
    for observation in observations:
        centered_embedding = _center_embedding(
            embeddings[(observation.sensor_id, observation.time_index)],
            graph_state["embedding_centers"],
        )
        raw_rows.append(
            (
                _dot(graph_state["graph_weights_mu"], centered_embedding),
                _dot(graph_state["graph_weights_sigma"], centered_embedding),
            )
        )
    return raw_rows


def _graph_embeddings_by_observation(
    observations: Sequence[Any],
    *,
    feature_names: Sequence[str],
    metadata: GraphMetadata,
) -> dict[tuple[str, int], tuple[float, ...]]:
    observations_by_key = {
        (observation.sensor_id, observation.time_index): observation
        for observation in observations
    }
    embeddings: dict[tuple[str, int], tuple[float, ...]] = {}
    for time_index in sorted({observation.time_index for observation in observations}):
        node_features = tuple(
            tuple(
                _feature_value(
                    observations_by_key[(sensor_id, time_index)].neighbor_covariates.get(
                        feature_name
                    ),
                    feature_name,
                )
                for feature_name in feature_names
            )
            for sensor_id in metadata.node_ids
        )
        neighbor_summary = _aggregate_graph_feature_rows(node_features, metadata)
        for sensor_id, self_features, neighbor_features in zip(
            metadata.node_ids, node_features, neighbor_summary, strict=True
        ):
            embeddings[(sensor_id, time_index)] = tuple(
                max(0.0, 0.5 * self_value + 0.5 * neighbor_value)
                for self_value, neighbor_value in zip(
                    self_features, neighbor_features, strict=True
                )
            )
    return embeddings


def _aggregate_graph_feature_rows(
    node_features: Sequence[Sequence[float]], metadata: GraphMetadata
) -> tuple[tuple[float, ...], ...]:
    feature_count = len(node_features[0])
    aggregated = [[0.0] * feature_count for _node_id in metadata.node_ids]
    denominator = [0.0] * len(metadata.node_ids)
    weights = metadata.edge_weight or (1.0,) * len(metadata.edge_index)
    for (source_index, target_index), weight in zip(
        metadata.edge_index, weights, strict=True
    ):
        for feature_index, value in enumerate(node_features[source_index]):
            aggregated[target_index][feature_index] += value * weight
        denominator[target_index] += weight

    return tuple(
        tuple(
            0.0 if denominator[node_index] == 0.0 else value / denominator[node_index]
            for value in feature_values
        )
        for node_index, feature_values in enumerate(aggregated)
    )


def _embedding_centers(
    embeddings: Iterable[Sequence[float]], feature_names: Sequence[str]
) -> tuple[float, ...]:
    rows = tuple(embeddings)
    if not rows:
        raise ConfigError("GraphNAMLSS training requires graph embeddings.")
    return tuple(
        sum(row[index] for row in rows) / len(rows)
        for index, _name in enumerate(feature_names)
    )


def _center_embedding(
    embedding: Sequence[float], centers: Sequence[float]
) -> tuple[float, ...]:
    return tuple(
        value - center for value, center in zip(embedding, centers, strict=True)
    )


def _fit_graph_residual_weights(
    examples: Sequence[tuple[Sequence[float], float]], feature_names: Sequence[str]
) -> tuple[float, ...]:
    x_rows = [row for row, _target in examples]
    targets = [target for _row, target in examples]
    matrix = [[0.0 for _ in feature_names] for _ in feature_names]
    rhs = [0.0 for _ in feature_names]

    for x_row, target in zip(x_rows, targets, strict=True):
        for row_index, row_value in enumerate(x_row):
            rhs[row_index] += row_value * target
            for column_index, column_value in enumerate(x_row):
                matrix[row_index][column_index] += row_value * column_value

    for index in range(len(feature_names)):
        matrix[index][index] += 1e-6

    return _solve_linear_system(matrix, rhs)


def _fit_graph_scale_weights(
    examples: Sequence[tuple[Sequence[float], float, float]],
    feature_names: Sequence[str],
) -> tuple[float, ...]:
    """Fit a centered graph scale head against Gaussian training NLL."""

    weights = [0.0 for _ in feature_names]
    for _step in range(TRACER_BULLET_SCALE_FIT_STEPS):
        gradients = [0.0 for _ in feature_names]
        for embedding, residual, local_sigma_eta in examples:
            sigma_eta = local_sigma_eta + _dot(weights, embedding)
            sigma = float(positive_scale(sigma_eta))
            derivative = (
                (1.0 / sigma) - ((residual * residual) / (sigma * sigma * sigma))
            ) * _sigmoid(sigma_eta)
            for index, value in enumerate(embedding):
                gradients[index] += derivative * value

        mean_gradients = tuple(
            (gradient / len(examples))
            + (TRACER_BULLET_SCALE_RIDGE * weights[index])
            for index, gradient in enumerate(gradients)
        )
        if max(abs(gradient) for gradient in mean_gradients) < 1e-8:
            break
        for index, gradient in enumerate(mean_gradients):
            weights[index] -= TRACER_BULLET_SCALE_LEARNING_RATE * gradient

    if any(not math.isfinite(weight) for weight in weights):
        raise ConfigError("Could not fit deterministic GraphNAMLSS scale head.")
    return tuple(weights)


def _factor_graph_head(
    coefficients: Sequence[float],
) -> tuple[tuple[float, ...], float]:
    """Identify graph head and gate via a unit-L2 head convention."""

    gate = math.sqrt(sum(coefficient * coefficient for coefficient in coefficients))
    if gate < 1e-12:
        return (0.0,) * len(coefficients), 0.0
    return tuple(coefficient / gate for coefficient in coefficients), gate


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _fit_graph_parameter(
    examples: Sequence[tuple[Sequence[float], float]], feature_names: Sequence[str]
) -> tuple[float, tuple[float, ...]]:
    x_rows = [(1.0, *row) for row, _target in examples]
    targets = [target for _row, target in examples]
    coefficients = _solve_ridge_normal_equations(x_rows, targets)
    return coefficients[0], tuple(coefficients[1:])


def _linear_predict(
    intercept: float, weights: Sequence[float], values: Sequence[float]
) -> float:
    return intercept + _dot(weights, values)


def _dot(weights: Sequence[float], values: Sequence[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, values, strict=True))


def _prediction_from_eta(eta: Sequence[Sequence[float]]) -> dict[str, tuple[float, ...]]:
    return {
        "mu": tuple(float(row[0]) for row in eta),
        "sigma": tuple(float(positive_scale(row[1])) for row in eta),
    }


def _graph_gate_rows(graph_state: Mapping[str, Any]) -> list[dict[str, float | str]]:
    return [
        {"parameter": parameter_name, "gate": graph_state["gates"][parameter_name]}
        for parameter_name in graph_state["parameter_names"]
    ]


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


def _point_metrics_for_split(
    prediction: Mapping[str, tuple[float, ...]], observations: Sequence[Any]
) -> PointMetricReport:
    targets = tuple(
        0.0 if observation.target_speed is None else float(observation.target_speed)
        for observation in observations
    )
    mask = tuple(
        (not observation.target_missing) and observation.target_speed is not None
        for observation in observations
    )
    return point_metric_report(
        mu=prediction["mu"],
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


def _point_metric_report_dict(report: PointMetricReport) -> dict[str, Any]:
    return {
        "unit": report.unit,
        "observed_count": report.observed_count,
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


def _prediction_artifact(
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    scaled_splits: Mapping[str, Any],
    scaling_policy: TrafficScalingPolicy,
) -> dict[str, Any]:
    """Serialize bounded, reader-facing predictions in original traffic units."""

    has_sigma = all(
        "sigma" in split_predictions for split_predictions in predictions.values()
    )
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_predictions in predictions.items():
        rows = []
        for index, observation in enumerate(scaled_splits[split_name].observations):
            standardized_mu = float(split_predictions["mu"][index])
            mu = (
                standardized_mu * scaling_policy.target.scale
                + scaling_policy.target.mean
            )
            standardized_target = (
                None
                if observation.target_speed is None
                else float(observation.target_speed)
            )
            target = (
                None
                if standardized_target is None
                else standardized_target * scaling_policy.target.scale
                + scaling_policy.target.mean
            )
            standardized_sigma = (
                float(split_predictions["sigma"][index]) if has_sigma else None
            )
            sigma = (
                standardized_sigma * scaling_policy.target.scale
                if standardized_sigma is not None
                else None
            )
            timestamp = getattr(observation, "timestamp", None)
            rows.append(
                {
                    "split": split_name,
                    "sensor_id": observation.sensor_id,
                    "time_index": observation.time_index,
                    "timestamp": (
                        timestamp.isoformat() if timestamp is not None else None
                    ),
                    "target": target,
                    "target_missing": bool(observation.target_missing),
                    "mu": mu,
                    "sigma": sigma,
                    "standardized": {
                        "target": standardized_target,
                        "mu": standardized_mu,
                        "sigma": standardized_sigma,
                    },
                }
            )
        split_rows[split_name] = rows
    return {
        "prediction_type": "distributional" if has_sigma else "point",
        "unit": "original_traffic",
        "splits": split_rows,
    }


def _write_json(path: Path, payload: Mapping[str, Any] | Iterable[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
