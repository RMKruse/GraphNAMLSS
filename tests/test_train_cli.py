import json
import math
import subprocess
import sys
from datetime import datetime, timedelta


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
