import json
import math
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml


def test_train_command_runs_deterministic_namlss_on_synthetic_traffic(tmp_path):
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "synthetic_train.yaml"
    config_path.write_text(
        f"""
dataset:
  name: synthetic-METR-LA-smoke
  task: fifteen_minute_ahead_speed
  adjacency: synthetic_physical
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
missingness:
  target: mask_missing_targets
  covariates: past_observed_lag_imputation
  future_looking_imputation: false
scaling:
  mode: train_period_standardization
  fit_on: train
  target: standardize
  features: standardize_continuous
model:
  family: NAMLSS
  variant: deterministic_local
  distribution: gaussian_location_scale
  graph_representation: none
  graph_injection: none
  graph_stream_self_feature_access: none
  graph_encoder: none
optimizer:
  name: deterministic_closed_form
  learning_rate: 0.0
  max_epochs: 1
output:
  directory: {output_dir}
  run_name: deterministic-synthetic
""".strip()
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "train",
            "--config",
            str(config_path),
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    assert emitted["run_name"] == "deterministic-synthetic"
    assert emitted["model"]["family"] == "NAMLSS"
    assert math.isfinite(emitted["loss"]["train"])

    resolved = json.loads((output_dir / "resolved_config.json").read_text())
    summary = json.loads((output_dir / "train_summary.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert resolved["dataset"]["name"] == "synthetic-METR-LA-smoke"
    assert summary["scaling"]["target"]["fit_on"] == "train"
    assert summary["missingness"]["target_policy"] == "mask_missing_targets"
    assert summary["splits"]["train"]["masked_target_count"] > 0
    assert summary["splits"]["train"]["observed_target_count"] < summary["splits"]["train"]["prediction_count"]

    for split_name in ("train", "validation", "test"):
        split_metrics = metrics[split_name]
        assert split_metrics["unit"] == "standardized_traffic"
        assert split_metrics["observed_count"] == summary["splits"][split_name]["observed_target_count"]
        for field in ("nll", "crps", "mae", "rmse"):
            assert math.isfinite(split_metrics[field])


def test_train_command_runs_tracer_bullet_graph_namlss_on_synthetic_traffic(tmp_path):
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "synthetic_graph_train.yaml"
    config_path.write_text(
        f"""
dataset:
  name: synthetic-METR-LA-smoke
  task: fifteen_minute_ahead_speed
  adjacency: synthetic_physical
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
    - neighbor_lag_5m
    - neighbor_lag_10m
    - neighbor_lag_15m
    - neighbor_lag_30m
    - neighbor_lag_60m
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
  source: synthetic_physical
  format: fixture_edges
  edge_weight: fixture_weight
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
  directory: {output_dir}
  run_name: tracer-bullet-synthetic
""".strip()
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "train",
            "--config",
            str(config_path),
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    assert emitted["run_name"] == "tracer-bullet-synthetic"
    assert emitted["model"] == {
        "family": "GraphNAMLSS",
        "variant": "tracer_bullet",
        "distribution": "gaussian_location_scale",
    }
    assert math.isfinite(emitted["loss"]["train"])

    summary = json.loads((output_dir / "train_summary.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())
    graph_gates = json.loads((output_dir / "graph_gates.json").read_text())
    graph_contributions = json.loads(
        (output_dir / "graph_contributions.json").read_text()
    )

    assert summary["graph"]["encoder"] == "graphsage"
    assert summary["graph"]["injection"] == "additive_graph_contribution"
    assert summary["graph"]["contribution_artifact"] == "graph_contributions.json"
    assert summary["graph"]["gate_artifact"] == "graph_gates.json"
    assert [row["parameter"] for row in graph_gates] == ["mu", "sigma"]
    assert all(row["gate"] >= 0.0 for row in graph_gates)
    assert summary["graph"]["gate_estimation"] == (
        "deterministic_unit_l2_head_factorization"
    )
    assert summary["graph"]["graph_head_normalization"] == "unit_l2"
    assert any(
        abs(weight) > 0.0
        for weight in summary["parameters"]["graph_weights_sigma"].values()
    )
    assert graph_contributions["parameter_names"] == ["mu", "sigma"]
    assert set(graph_contributions["splits"]) == {"train", "validation", "test"}
    assert len(graph_contributions["splits"]["train"]) == summary["splits"]["train"]["prediction_count"]
    assert all(
        set(row) == {
            "split",
            "sensor_id",
            "time_index",
            "mu",
            "sigma",
            "gated_mu",
            "gated_sigma",
        }
        for row in graph_contributions["splits"]["train"]
    )

    for split_name in ("train", "validation", "test"):
        split_metrics = metrics[split_name]
        assert split_metrics["unit"] == "standardized_traffic"
        assert split_metrics["observed_count"] == summary["splits"][split_name]["observed_target_count"]
        for field in ("nll", "crps", "mae", "rmse"):
            assert math.isfinite(split_metrics[field])


def test_train_command_runs_distributional_gnn_baseline_on_synthetic_traffic(tmp_path):
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "synthetic_distributional_gnn_train.yaml"
    config_path.write_text(
        f"""
dataset:
  name: synthetic-METR-LA-smoke
  task: fifteen_minute_ahead_speed
  adjacency: synthetic_physical
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
    - neighbor_lag_5m
    - neighbor_lag_10m
    - neighbor_lag_15m
    - neighbor_lag_30m
    - neighbor_lag_60m
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
  source: synthetic_physical
  format: fixture_edges
  edge_weight: fixture_weight
  directed: true
model:
  family: DistributionalGNN
  variant: baseline
  distribution: gaussian_location_scale
  graph_representation: shared
  graph_injection: direct_distributional_parameters
  graph_stream_self_feature_access: neighbor_focused
  graph_encoder: graphsage
optimizer:
  name: adam
  learning_rate: 0.001
  max_epochs: 1
output:
  directory: {output_dir}
  run_name: distributional-gnn-synthetic
""".strip()
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "train",
            "--config",
            str(config_path),
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    assert emitted["run_name"] == "distributional-gnn-synthetic"
    assert emitted["model"] == {
        "family": "DistributionalGNN",
        "variant": "baseline",
        "distribution": "gaussian_location_scale",
    }
    assert math.isfinite(emitted["loss"]["train"])

    summary = json.loads((output_dir / "train_summary.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert summary["graph"]["injection"] == "direct_distributional_parameters"
    assert summary["graph"]["feature_names"] == [
        "neighbor_current_speed",
        "neighbor_lag_5m",
        "neighbor_lag_10m",
        "neighbor_lag_15m",
        "neighbor_lag_30m",
        "neighbor_lag_60m",
    ]
    assert summary["parameters"]["sigma"] > 0.0

    metric_fields = set(metrics["train"])
    for split_name in ("train", "validation", "test"):
        split_metrics = metrics[split_name]
        assert set(split_metrics) == metric_fields
        assert split_metrics["unit"] == "standardized_traffic"
        assert (
            split_metrics["observed_count"]
            == summary["splits"][split_name]["observed_target_count"]
        )
        for field in ("nll", "crps", "mae", "rmse"):
            assert math.isfinite(split_metrics[field])


def test_train_command_runs_point_gnn_reference_on_synthetic_traffic(tmp_path):
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "synthetic_point_gnn_train.yaml"
    config_path.write_text(
        f"""
dataset:
  name: synthetic-METR-LA-smoke
  task: fifteen_minute_ahead_speed
  adjacency: synthetic_physical
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
    - neighbor_lag_5m
    - neighbor_lag_10m
    - neighbor_lag_15m
    - neighbor_lag_30m
    - neighbor_lag_60m
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
  source: synthetic_physical
  format: fixture_edges
  edge_weight: fixture_weight
  directed: true
model:
  family: PointGNN
  variant: reference
  distribution: conditional_mean
  graph_representation: shared
  graph_injection: direct_mean
  graph_stream_self_feature_access: neighbor_focused
  graph_encoder: graphsage
optimizer:
  name: adam
  learning_rate: 0.001
  max_epochs: 1
output:
  directory: {output_dir}
  run_name: point-gnn-synthetic
""".strip()
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "train",
            "--config",
            str(config_path),
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    assert emitted["run_name"] == "point-gnn-synthetic"
    assert emitted["model"] == {
        "family": "PointGNN",
        "variant": "reference",
        "distribution": "conditional_mean",
        "prediction_type": "point",
        "role": "point_prediction_reference",
    }
    assert math.isfinite(emitted["loss"]["train"])

    summary = json.loads((output_dir / "train_summary.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert summary["graph"]["injection"] == "direct_mean"
    assert summary["reference"]["prediction_type"] == "point"
    assert summary["reference"]["evaluated_with"] == ["mae", "rmse"]
    assert summary["parameters"]["sigma"] is None

    metric_fields = {"unit", "observed_count", "mae", "rmse"}
    for split_name in ("train", "validation", "test"):
        split_metrics = metrics[split_name]
        assert set(split_metrics) == metric_fields
        assert split_metrics["unit"] == "standardized_traffic"
        assert (
            split_metrics["observed_count"]
            == summary["splits"][split_name]["observed_target_count"]
        )
        assert math.isfinite(split_metrics["mae"])
        assert math.isfinite(split_metrics["rmse"])


def test_train_command_reports_missing_metr_la_artifacts_before_training():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "train",
            "--config",
            "configs/metr_la.yaml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Missing required dataset artifacts for METR-LA" in result.stderr
    assert "readings:" in result.stderr
    assert "adjacency:" in result.stderr


def test_train_command_reports_missing_metr_la_graph_artifacts_before_training():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "train",
            "--config",
            "configs/metr_la_graph_tracer_bullet.yaml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Missing required dataset artifacts for METR-LA" in result.stderr
    assert "readings:" in result.stderr
    assert "adjacency:" in result.stderr


def test_train_command_runs_deterministic_namlss_on_metr_la_artifacts(tmp_path):
    readings_path = tmp_path / "metr_la_readings.csv"
    adjacency_path = tmp_path / "adjacency.csv"
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "metr_la_train.yaml"
    _write_wide_metr_la_readings(readings_path)
    adjacency_path.write_text(
        "source,target,weight\n"
        "sensor_b,sensor_a,0.25\n"
        "sensor_c,sensor_a,0.75\n"
        "sensor_a,sensor_b,0.50\n"
        "sensor_b,sensor_c,1.00\n"
    )
    config_path.write_text(
        f"""
dataset:
  name: METR-LA
  task: fifteen_minute_ahead_speed
  adjacency: dataset_provided
  artifacts:
    readings: {readings_path}
    adjacency: {adjacency_path}
split:
  strategy: temporal
  train: 0.6
  validation: 0.2
  test: 0.2
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
  family: NAMLSS
  variant: deterministic_local
  distribution: gaussian_location_scale
  graph_representation: none
  graph_injection: none
  graph_stream_self_feature_access: none
  graph_encoder: none
optimizer:
  name: deterministic_closed_form
  learning_rate: 0.0
  max_epochs: 1
output:
  directory: {output_dir}
  run_name: deterministic-metr-la
""".strip()
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "train",
            "--config",
            str(config_path),
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    assert emitted["run_name"] == "deterministic-metr-la"
    assert emitted["model"] == {
        "family": "NAMLSS",
        "variant": "deterministic_local",
        "distribution": "gaussian_location_scale",
    }
    assert math.isfinite(emitted["loss"]["train"])

    resolved = json.loads((output_dir / "resolved_config.json").read_text())
    summary = json.loads((output_dir / "train_summary.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert resolved["dataset"]["name"] == "METR-LA"
    assert summary["features"]["local"] == resolved["features"]["local"]
    assert summary["missingness"]["target_policy"] == "mask_missing_targets"
    assert summary["missingness"]["covariate_policy"] == "past_observed_lag_imputation"
    assert summary["splits"]["train"]["observed_target_count"] > 0
    assert set(metrics) == {"train", "validation", "test"}

    metric_fields = set(metrics["train"])
    for split_name in ("train", "validation", "test"):
        split_metrics = metrics[split_name]
        assert set(split_metrics) == metric_fields
        assert split_metrics["unit"] == "standardized_traffic"
        assert split_metrics["observed_count"] == summary["splits"][split_name]["observed_target_count"]
        for field in ("nll", "crps", "mae", "rmse"):
            assert math.isfinite(split_metrics[field])


def test_train_command_runs_deterministic_namlss_on_pems_bay_artifacts(tmp_path):
    readings_path = tmp_path / "pems_bay_readings.csv"
    adjacency_path = tmp_path / "adjacency.csv"
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "pems_bay_train.yaml"
    _write_wide_metr_la_readings(readings_path)
    adjacency_path.write_text(
        "source,target,weight\n"
        "sensor_b,sensor_a,0.25\n"
        "sensor_c,sensor_a,0.75\n"
        "sensor_a,sensor_b,0.50\n"
        "sensor_b,sensor_c,1.00\n"
    )

    raw_config = yaml.safe_load(Path("configs/pems_bay.yaml").read_text())
    raw_config["dataset"]["artifacts"] = {
        "readings": str(readings_path),
        "adjacency": str(adjacency_path),
    }
    raw_config["graph"]["format"] = "edge_list_csv"
    raw_config["graph"]["edge_weight"] = "dataset_provided_weight"
    raw_config["output"]["directory"] = str(output_dir)
    config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "train",
            "--config",
            str(config_path),
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    assert emitted["run_name"] == "deterministic-pems-bay"
    assert math.isfinite(emitted["loss"]["train"])

    resolved = json.loads((output_dir / "resolved_config.json").read_text())
    summary = json.loads((output_dir / "train_summary.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert resolved["dataset"]["name"] == "PEMS-BAY"
    assert summary["features"]["local"] == resolved["features"]["local"]
    assert summary["missingness"]["target_policy"] == "mask_missing_targets"
    assert summary["missingness"]["covariate_policy"] == "past_observed_lag_imputation"
    assert summary["splits"]["train"]["observed_target_count"] > 0
    assert set(metrics) == {"train", "validation", "test"}


def test_train_command_runs_tracer_bullet_graph_namlss_on_metr_la_artifacts(tmp_path):
    readings_path = tmp_path / "metr_la_readings.csv"
    adjacency_path = tmp_path / "adjacency.csv"
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "metr_la_graph_train.yaml"
    _write_wide_metr_la_readings(readings_path)
    adjacency_path.write_text(
        "source,target,weight\n"
        "sensor_b,sensor_a,0.25\n"
        "sensor_c,sensor_a,0.75\n"
        "sensor_a,sensor_b,0.50\n"
        "sensor_b,sensor_c,1.00\n"
    )
    config_path.write_text(
        f"""
dataset:
  name: METR-LA
  task: fifteen_minute_ahead_speed
  adjacency: dataset_provided
  artifacts:
    readings: {readings_path}
    adjacency: {adjacency_path}
split:
  strategy: temporal
  train: 0.6
  validation: 0.2
  test: 0.2
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
    - neighbor_lag_5m
    - neighbor_lag_10m
    - neighbor_lag_15m
    - neighbor_lag_30m
    - neighbor_lag_60m
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
  directory: {output_dir}
  run_name: tracer-bullet-metr-la
""".strip()
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "train",
            "--config",
            str(config_path),
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    assert emitted["run_name"] == "tracer-bullet-metr-la"
    assert emitted["model"] == {
        "family": "GraphNAMLSS",
        "variant": "tracer_bullet",
        "distribution": "gaussian_location_scale",
    }
    assert math.isfinite(emitted["loss"]["train"])

    resolved = json.loads((output_dir / "resolved_config.json").read_text())
    summary = json.loads((output_dir / "train_summary.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())
    graph_gates = json.loads((output_dir / "graph_gates.json").read_text())
    graph_contributions = json.loads(
        (output_dir / "graph_contributions.json").read_text()
    )

    assert resolved["dataset"]["name"] == "METR-LA"
    assert summary["graph"]["representation"] == "shared"
    assert summary["graph"]["encoder"] == "graphsage"
    assert summary["graph"]["injection"] == "additive_graph_contribution"
    assert summary["graph"]["self_feature_access"] == "neighbor_focused"
    assert summary["graph"]["node_count"] == 3
    assert summary["graph"]["edge_count"] == 4
    assert summary["graph"]["feature_names"] == [
        "neighbor_current_speed",
        "neighbor_lag_5m",
        "neighbor_lag_10m",
        "neighbor_lag_15m",
        "neighbor_lag_30m",
        "neighbor_lag_60m",
    ]
    assert summary["graph"]["gate_artifact"] == "graph_gates.json"
    assert summary["graph"]["contribution_artifact"] == "graph_contributions.json"
    assert summary["scaling"]["target"]["fit_on"] == "train"
    assert summary["missingness"]["target_policy"] == "mask_missing_targets"
    assert summary["missingness"]["covariate_policy"] == "past_observed_lag_imputation"
    assert [row["parameter"] for row in graph_gates] == ["mu", "sigma"]
    assert all(row["gate"] >= 0.0 for row in graph_gates)
    assert summary["graph"]["gate_estimation"] == (
        "deterministic_unit_l2_head_factorization"
    )
    assert summary["graph"]["graph_head_normalization"] == "unit_l2"
    assert graph_contributions["parameter_names"] == ["mu", "sigma"]
    assert set(graph_contributions["splits"]) == {"train", "validation", "test"}

    metric_fields = set(metrics["train"])
    for split_name in ("train", "validation", "test"):
        split_metrics = metrics[split_name]
        assert set(split_metrics) == metric_fields
        assert split_metrics["unit"] == "standardized_traffic"
        assert split_metrics["observed_count"] == summary["splits"][split_name]["observed_target_count"]
        assert len(graph_contributions["splits"][split_name]) == summary["splits"][split_name]["prediction_count"]
        for field in ("nll", "crps", "mae", "rmse"):
            assert math.isfinite(split_metrics[field])


def test_train_command_runs_distributional_gnn_baseline_on_metr_la_artifacts(tmp_path):
    readings_path = tmp_path / "metr_la_readings.csv"
    adjacency_path = tmp_path / "adjacency.csv"
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "metr_la_distributional_gnn_train.yaml"
    _write_wide_metr_la_readings(readings_path)
    adjacency_path.write_text(
        "source,target,weight\n"
        "sensor_b,sensor_a,0.25\n"
        "sensor_c,sensor_a,0.75\n"
        "sensor_a,sensor_b,0.50\n"
        "sensor_b,sensor_c,1.00\n"
    )
    config_path.write_text(
        f"""
dataset:
  name: METR-LA
  task: fifteen_minute_ahead_speed
  adjacency: dataset_provided
  artifacts:
    readings: {readings_path}
    adjacency: {adjacency_path}
split:
  strategy: temporal
  train: 0.6
  validation: 0.2
  test: 0.2
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
    - neighbor_lag_5m
    - neighbor_lag_10m
    - neighbor_lag_15m
    - neighbor_lag_30m
    - neighbor_lag_60m
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
  family: DistributionalGNN
  variant: baseline
  distribution: gaussian_location_scale
  graph_representation: shared
  graph_injection: direct_distributional_parameters
  graph_stream_self_feature_access: neighbor_focused
  graph_encoder: graphsage
optimizer:
  name: adam
  learning_rate: 0.001
  max_epochs: 1
output:
  directory: {output_dir}
  run_name: distributional-gnn-metr-la
""".strip()
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "train",
            "--config",
            str(config_path),
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    assert emitted["run_name"] == "distributional-gnn-metr-la"
    assert emitted["model"] == {
        "family": "DistributionalGNN",
        "variant": "baseline",
        "distribution": "gaussian_location_scale",
    }

    resolved = json.loads((output_dir / "resolved_config.json").read_text())
    summary = json.loads((output_dir / "train_summary.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert resolved["dataset"]["name"] == "METR-LA"
    assert summary["graph"]["representation"] == "shared"
    assert summary["graph"]["encoder"] == "graphsage"
    assert summary["graph"]["injection"] == "direct_distributional_parameters"
    assert summary["graph"]["self_feature_access"] == "neighbor_focused"
    assert summary["graph"]["node_count"] == 3
    assert summary["graph"]["edge_count"] == 4
    assert summary["parameters"]["sigma"] > 0.0

    metric_fields = set(metrics["train"])
    for split_name in ("train", "validation", "test"):
        split_metrics = metrics[split_name]
        assert set(split_metrics) == metric_fields
        assert split_metrics["unit"] == "standardized_traffic"
        assert (
            split_metrics["observed_count"]
            == summary["splits"][split_name]["observed_target_count"]
        )
        for field in ("nll", "crps", "mae", "rmse"):
            assert math.isfinite(split_metrics[field])


def _write_wide_metr_la_readings(path):
    start = datetime(2026, 1, 5, 7, 0)
    sensors = ("sensor_a", "sensor_b", "sensor_c")
    lines = ["timestamp," + ",".join(sensors)]
    for time_index in range(20):
        timestamp = start + timedelta(minutes=5 * time_index)
        values = []
        for sensor_offset, _sensor in enumerate(sensors):
            if sensor_offset == 1 and time_index == 4:
                values.append("")
            else:
                speed = 60.0 - sensor_offset * 2.5 - time_index * 0.4
                values.append(f"{speed:.2f}")
        lines.append(timestamp.isoformat() + "," + ",".join(values))
    path.write_text("\n".join(lines) + "\n")
