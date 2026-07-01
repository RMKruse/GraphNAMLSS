import json
import subprocess
import sys
from pathlib import Path


def test_dry_run_records_resolved_experiment_metadata(tmp_path):
    output_dir = tmp_path / "runs" / "tracer-bullet"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        f"""
dataset:
  name: METR-LA
  task: fifteen_minute_ahead_speed
  adjacency: dataset_provided
split:
  strategy: temporal
  train: 0.7
  validation: 0.1
  test: 0.2
features:
  local:
    - current_speed
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
  run_name: dry-run-smoke
""".strip()
    )

    result = subprocess.run(
        [sys.executable, "-m", "graphnamlss", "dry-run", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "METR-LA" in result.stdout
    assert "temporal" in result.stdout
    assert "current_speed" in result.stdout
    assert "GraphNAMLSS" in result.stdout
    assert "adam" in result.stdout
    assert str(output_dir) in result.stdout

    resolved_path = output_dir / "resolved_config.json"
    assert resolved_path.exists()

    resolved = json.loads(resolved_path.read_text())
    assert resolved["dataset"]["name"] == "METR-LA"
    assert resolved["split"]["strategy"] == "temporal"
    assert resolved["features"]["neighbor"] == [
        "neighbor_current_speed",
        "neighbor_lag_5m",
        "neighbor_lag_10m",
        "neighbor_lag_15m",
        "neighbor_lag_30m",
        "neighbor_lag_60m",
    ]
    assert resolved["output"]["run_name"] == "dry-run-smoke"


def test_dry_run_reports_missing_required_config_field(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
dataset:
  name: METR-LA
  task: fifteen_minute_ahead_speed
  adjacency: dataset_provided
split:
  strategy: temporal
  train: 0.7
  validation: 0.1
  test: 0.2
features:
  local:
    - current_speed
  neighbor:
    - neighbor_current_speed
model:
  family: GraphNAMLSS
  variant: tracer_bullet
  graph_representation: shared
  graph_injection: additive_graph_contribution
  graph_stream_self_feature_access: neighbor_focused
  graph_encoder: graphsage
optimizer:
  name: adam
  learning_rate: 0.001
  max_epochs: 1
output:
  directory: runs/tracer-bullet
  run_name: dry-run-smoke
""".strip()
    )

    result = subprocess.run(
        [sys.executable, "-m", "graphnamlss", "dry-run", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "model.distribution" in result.stderr
    assert "Missing required string" in result.stderr
