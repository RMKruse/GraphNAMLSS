import json
import math
import subprocess
import sys


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
