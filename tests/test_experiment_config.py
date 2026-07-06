from graphnamlss.config import (
    ConfigError,
    ExperimentConfig,
    load_experiment_config,
    validate_experiment_artifacts,
    validate_experiment_config,
)
from graphnamlss.testing import load_synthetic_traffic_fixture_from_config
from graphnamlss.traffic_scaling import fit_traffic_scaling_policy


def test_sample_tracer_bullet_config_loads_as_typed_experiment_config():
    config = load_experiment_config("configs/tracer_bullet.yaml")
    validate_experiment_config(config)

    assert isinstance(config, ExperimentConfig)
    assert config.dataset.name == "METR-LA"
    assert config.dataset.task == "fifteen_minute_ahead_speed"
    assert config.split.strategy == "temporal"
    assert config.features.local[0] == "current_speed"
    assert config.model.variant == "tracer_bullet"
    assert config.optimizer.name == "adam"
    assert config.output.run_name == "tracer-bullet-dry-run"


def test_metr_la_config_resolves_tracer_dataset_contract():
    config = load_experiment_config("configs/metr_la.yaml")
    validate_experiment_config(config)

    resolved = config.to_resolved_dict()

    assert resolved["dataset"]["name"] == "METR-LA"
    assert resolved["dataset"]["artifacts"]["readings"].endswith(
        "data/raw/metr-la/metr-la.h5"
    )
    assert resolved["dataset"]["artifacts"]["adjacency"].endswith(
        "data/raw/metr-la/adj_mx.pkl"
    )
    assert resolved["split"]["boundaries"] == {
        "train": [0.0, 0.7],
        "validation": [0.7, 0.8],
        "test": [0.8, 1.0],
    }
    assert resolved["temporal"]["horizon_minutes"] == 15
    assert resolved["temporal"]["lag_minutes"] == [0, 5, 10, 15, 30, 60]
    assert resolved["scaling"]["mode"] == "train_period_standardization"
    assert resolved["missingness"]["target"] == "mask_missing_targets"
    assert resolved["missingness"]["covariates"] == "past_observed_lag_imputation"
    assert resolved["graph"] == {
        "source": "dataset_provided",
        "format": "dcrnn_adjacency_pickle",
        "edge_weight": "dataset_provided_distance_kernel",
        "directed": True,
    }


def test_metr_la_config_reports_missing_required_artifacts():
    config = load_experiment_config("configs/metr_la.yaml")

    try:
        validate_experiment_artifacts(config)
    except ConfigError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected absent METR-LA artifacts to fail validation.")

    assert "Missing required dataset artifacts for METR-LA" in message
    assert "readings:" in message
    assert "adjacency:" in message


def test_extended_dataset_config_runs_against_synthetic_traffic_smoke_path(tmp_path):
    adjacency_path = tmp_path / "adjacency.csv"
    adjacency_path.write_text(
        "source,target,weight\n"
        "sensor_b,sensor_a,0.25\n"
        "sensor_c,sensor_a,0.75\n"
        "sensor_a,sensor_d,1.0\n"
    )
    config_path = tmp_path / "synthetic_metr_la.yaml"
    config_path.write_text(
        f"""
dataset:
  name: synthetic-METR-LA-smoke
  task: fifteen_minute_ahead_speed
  adjacency: dataset_provided
  artifacts:
    adjacency: {adjacency_path}
split:
  strategy: temporal
  train: 0.5
  validation: 0.25
  test: 0.25
temporal:
  frequency_minutes: 5
  horizon_minutes: 15
  horizon_steps: 3
  lag_minutes:
    - 0
    - 5
    - 10
    - 15
    - 30
    - 60
features:
  local:
    - current_speed
    - current_speed_missing
    - lag_5m
    - lag_10m
    - lag_15m
    - lag_30m
    - lag_60m
    - time_of_day
    - day_of_week
    - weekend
  neighbor:
    - neighbor_current_speed
    - neighbor_current_speed_missing
    - neighbor_lag_5m
    - neighbor_lag_5m_missing
    - neighbor_lag_10m
    - neighbor_lag_10m_missing
    - neighbor_lag_15m
    - neighbor_lag_15m_missing
    - neighbor_lag_30m
    - neighbor_lag_30m_missing
    - neighbor_lag_60m
    - neighbor_lag_60m_missing
missingness:
  target: mask_missing_targets
  covariates: past_observed_lag_imputation
  future_looking_imputation: false
scaling:
  mode: train_period_standardization
  fit_on: train
  target: standardize
  features: standardize_continuous
graph:
  source: dataset_provided
  format: edge_list_csv
  edge_weight: dataset_provided_weight
  directed: true
model:
  family: GraphNAMLSS
  variant: tracer_bullet
  distribution: gaussian_location_scale
  graph_representation: shared
  graph_injection: additive_graph_contribution
  graph_stream_self_feature_access: neighbor_focused
  graph_encoder: graphsage
optimizer:
  name: adam
  learning_rate: 0.001
  max_epochs: 1
output:
  directory: {tmp_path / "runs"}
  run_name: synthetic-smoke
""".strip()
    )

    config = load_experiment_config(config_path)
    validate_experiment_config(config)
    validate_experiment_artifacts(config)

    fixture = load_synthetic_traffic_fixture_from_config(config, seed=7)
    splits = fixture.temporal_split_observations(
        train=config.split.train,
        validation=config.split.validation,
        test=config.split.test,
    )
    policy = fit_traffic_scaling_policy(splits["train"])
    scaled_validation = policy.transform_split(splits["validation"])

    assert [edge.weight for edge in fixture.adjacency] == [0.25, 0.75, 1.0]
    assert fixture.missingness_report().imputation_strategy == "past_observed_lag"
    assert set(splits) == {"train", "validation", "test"}
    assert scaled_validation.observations
