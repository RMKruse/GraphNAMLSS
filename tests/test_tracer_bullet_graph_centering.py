"""Centered Graph Contributions of the Tracer-Bullet Fit use training statistics.

A held-out prediction must depend only on its own row and the fitted state, never
on which other rows happen to share its split.
"""

import json
from statistics import fmean

import pytest

from graphnamlss.config import load_experiment_config
from graphnamlss.training import train_namlss_on_traffic


DEMO_SEED = 20250214


def _tracer_bullet_demo_config(tmp_path, *, name, validation, test):
    output_dir = tmp_path / name
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(
        f"""
dataset:
  name: synthetic-concept-demo
  task: fifteen_minute_ahead_speed
  adjacency: synthetic_physical
split:
  strategy: temporal
  train: 0.5
  validation: {validation}
  test: {test}
temporal:
  frequency_minutes: 5
  horizon_minutes: 15
  horizon_steps: 3
  lag_minutes: [0, 5, 10, 15, 30, 60]
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
  run_name: {name}
""".strip()
    )
    return load_experiment_config(config_path)


def _train(tmp_path, *, name, validation, test):
    config = _tracer_bullet_demo_config(
        tmp_path, name=name, validation=validation, test=test
    )
    train_namlss_on_traffic(config, seed=DEMO_SEED)
    output_dir = config.output.directory
    return (
        json.loads((output_dir / "predictions.json").read_text()),
        json.loads((output_dir / "graph_contributions.json").read_text()),
    )


def _rows_by_key(artifact):
    return {
        (row["sensor_id"], row["time_index"]): row
        for split_rows in artifact["splits"].values()
        for row in split_rows
    }


def _split_keys(artifact, split_name):
    return {
        (row["sensor_id"], row["time_index"]) for row in artifact["splits"][split_name]
    }


@pytest.fixture(scope="module")
def demo_split_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("demo-split")
    return _train(tmp_path, name="demo-split", validation=0.25, test=0.25)


@pytest.fixture(scope="module")
def shorter_validation_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("shorter-validation")
    return _train(tmp_path, name="shorter-validation", validation=0.125, test=0.375)


def test_held_out_predictions_do_not_depend_on_the_other_rows_of_their_split(
    demo_split_run, shorter_validation_run
):
    demo_predictions, demo_contributions = demo_split_run
    shorter_predictions, shorter_contributions = shorter_validation_run

    # The training split is identical; validation loses its later half to test.
    assert _split_keys(shorter_predictions, "train") == _split_keys(
        demo_predictions, "train"
    )
    moved_to_test = _split_keys(demo_predictions, "validation") & _split_keys(
        shorter_predictions, "test"
    )
    kept_in_validation = _split_keys(demo_predictions, "validation") & _split_keys(
        shorter_predictions, "validation"
    )
    assert moved_to_test
    assert kept_in_validation

    demo_rows = _rows_by_key(demo_predictions)
    shorter_rows = _rows_by_key(shorter_predictions)
    assert demo_rows.keys() == shorter_rows.keys()
    for key, demo_row in demo_rows.items():
        for field in ("mu", "sigma"):
            assert shorter_rows[key]["standardized"][field] == pytest.approx(
                demo_row["standardized"][field], abs=1e-12
            ), (key, field)

    demo_graph_rows = _rows_by_key(demo_contributions)
    shorter_graph_rows = _rows_by_key(shorter_contributions)
    for key, demo_row in demo_graph_rows.items():
        for field in ("mu", "sigma", "gated_mu", "gated_sigma"):
            assert shorter_graph_rows[key][field] == pytest.approx(
                demo_row[field], abs=1e-12
            ), (key, field)


def test_training_graph_contributions_are_centred_per_parameter(demo_split_run):
    _predictions, contributions = demo_split_run
    train_rows = contributions["splits"]["train"]

    for parameter in contributions["parameter_names"]:
        assert fmean(row[parameter] for row in train_rows) == pytest.approx(
            0.0, abs=1e-9
        )
