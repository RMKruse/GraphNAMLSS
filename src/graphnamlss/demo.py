from __future__ import annotations

from pathlib import Path
from typing import Any

from graphnamlss.config import (
    DatasetConfig,
    ExperimentConfig,
    FeatureConfig,
    GraphConfig,
    MissingnessConfig,
    ModelConfig,
    OptimizerConfig,
    OutputConfig,
    ScalingConfig,
    SplitConfig,
    TemporalConfig,
    validate_experiment_artifacts,
    validate_experiment_config,
)
from graphnamlss.report import generate_tracer_bullet_evaluation_report
from graphnamlss.testing import load_localized_congestion_spillover_demo_fixture
from graphnamlss.training import train_namlss_on_traffic


MODEL_ORDER = (
    "deterministic_namlss",
    "tracer_bullet_graph_namlss",
    "distributional_gnn_baseline",
    "point_gnn_reference",
)


def run_concept_demo_bundle(
    output_directory: Path, *, seed: int = 20250214
) -> dict[str, Any]:
    """Run the synthetic Concept Demo Bundle and report from saved artifacts."""

    run_directories = {
        "deterministic_namlss": output_directory / "runs" / "deterministic-namlss",
        "tracer_bullet_graph_namlss": (
            output_directory / "runs" / "tracer-bullet-graph-namlss"
        ),
        "distributional_gnn_baseline": (
            output_directory / "runs" / "distributional-gnn-baseline"
        ),
        "point_gnn_reference": output_directory / "runs" / "point-gnn-reference",
    }
    configs = _concept_demo_configs(run_directories)
    for model_key in MODEL_ORDER:
        config = configs[model_key]
        validate_experiment_config(config)
        validate_experiment_artifacts(config)
        train_namlss_on_traffic(config, seed=seed)

    report_summary = generate_tracer_bullet_evaluation_report(
        deterministic_run=run_directories["deterministic_namlss"],
        graph_run=run_directories["tracer_bullet_graph_namlss"],
        distributional_gnn_run=run_directories["distributional_gnn_baseline"],
        point_gnn_run=run_directories["point_gnn_reference"],
        output_directory=output_directory / "report",
        spillover_scenario=load_localized_congestion_spillover_demo_fixture(
            seed=seed
        ).scenario,
    )
    return {
        "bundle_directory": str(output_directory),
        "models": report_summary["models"],
        "run_directories": {
            model_key: str(run_directories[model_key]) for model_key in MODEL_ORDER
        },
        "report_path": report_summary["report_path"],
        "summary_path": report_summary["summary_path"],
    }


def _concept_demo_configs(
    run_directories: dict[str, Path],
) -> dict[str, ExperimentConfig]:
    dataset = DatasetConfig(
        name="synthetic-concept-demo",
        task="fifteen_minute_ahead_speed",
        adjacency="synthetic_physical",
    )
    split = SplitConfig(strategy="temporal", train=0.5, validation=0.25, test=0.25)
    temporal = TemporalConfig(
        frequency_minutes=5,
        horizon_minutes=15,
        horizon_steps=3,
        lag_minutes=(0, 5, 10, 15, 30, 60),
    )
    features = FeatureConfig(
        local=(
            "current_speed",
            "current_speed_missing",
            "lag_5m",
            "lag_10m",
            "lag_15m",
            "lag_30m",
            "lag_60m",
            "time_of_day",
            "day_of_week",
            "weekend",
        ),
        neighbor=(
            "neighbor_current_speed",
            "neighbor_lag_5m",
            "neighbor_lag_10m",
            "neighbor_lag_15m",
            "neighbor_lag_30m",
            "neighbor_lag_60m",
        ),
    )
    scaling = ScalingConfig(
        mode="train_period_standardization",
        fit_on="train",
        target="standardize",
        features="standardize_continuous",
    )
    missingness = MissingnessConfig(
        target="mask_missing_targets",
        covariates="past_observed_lag_imputation",
        future_looking_imputation=False,
    )
    graph = GraphConfig(
        source="synthetic_physical",
        format="fixture_edges",
        edge_weight="fixture_weight",
        directed=True,
    )

    model_settings = {
        "deterministic_namlss": (
            ModelConfig(
                family="NAMLSS",
                variant="deterministic_local",
                distribution="gaussian_location_scale",
                graph_representation="none",
                graph_injection="none",
                graph_stream_self_feature_access="none",
                graph_encoder="none",
            ),
            OptimizerConfig(
                name="deterministic_closed_form",
                learning_rate=0.0,
                max_epochs=1,
            ),
            None,
        ),
        "tracer_bullet_graph_namlss": (
            ModelConfig(
                family="GraphNAMLSS",
                variant="tracer_bullet",
                distribution="gaussian_location_scale",
                graph_representation="shared",
                graph_injection="additive_graph_contribution",
                graph_stream_self_feature_access="neighbor_focused",
                graph_encoder="graphsage",
            ),
            OptimizerConfig(name="adam", learning_rate=0.001, max_epochs=1),
            graph,
        ),
        "distributional_gnn_baseline": (
            ModelConfig(
                family="DistributionalGNN",
                variant="baseline",
                distribution="gaussian_location_scale",
                graph_representation="shared",
                graph_injection="direct_distributional_parameters",
                graph_stream_self_feature_access="neighbor_focused",
                graph_encoder="graphsage",
            ),
            OptimizerConfig(name="adam", learning_rate=0.001, max_epochs=1),
            graph,
        ),
        "point_gnn_reference": (
            ModelConfig(
                family="PointGNN",
                variant="reference",
                distribution="conditional_mean",
                graph_representation="shared",
                graph_injection="direct_mean",
                graph_stream_self_feature_access="neighbor_focused",
                graph_encoder="graphsage",
            ),
            OptimizerConfig(name="adam", learning_rate=0.001, max_epochs=1),
            graph,
        ),
    }

    return {
        model_key: ExperimentConfig(
            dataset=dataset,
            split=split,
            features=features,
            model=model,
            optimizer=optimizer,
            output=OutputConfig(
                directory=run_directories[model_key],
                run_name=model_key.replace("_", "-"),
            ),
            temporal=temporal,
            scaling=scaling,
            missingness=missingness,
            graph=model_graph,
        )
        for model_key, (model, optimizer, model_graph) in model_settings.items()
    }
