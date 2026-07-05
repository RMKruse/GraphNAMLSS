import math

import pytest

from graphnamlss.testing import load_synthetic_traffic_fixture
from graphnamlss.traffic_scaling import fit_traffic_scaling_policy


def test_traffic_scaling_policy_fits_targets_from_training_period_only():
    fixture = load_synthetic_traffic_fixture(seed=7).with_past_lag_imputation()
    splits = fixture.temporal_split_observations(train=0.5, validation=0.25, test=0.25)

    policy = fit_traffic_scaling_policy(splits["train"])

    train_targets = [
        row.target_speed
        for row in splits["train"].observed_target_observations()
        if row.target_speed is not None
    ]
    all_targets = [
        row.target_speed
        for split in splits.values()
        for row in split.observed_target_observations()
        if row.target_speed is not None
    ]

    assert policy.target.mean == pytest.approx(sum(train_targets) / len(train_targets))
    assert policy.target.scale == pytest.approx(_population_scale(train_targets))
    assert policy.target.mean != pytest.approx(sum(all_targets) / len(all_targets))

    train_current_speeds = _feature_values(splits["train"], "current_speed")
    all_current_speeds = [
        value
        for split in splits.values()
        for value in _feature_values(split, "current_speed")
    ]
    assert policy.features["current_speed"].mean == pytest.approx(
        sum(train_current_speeds) / len(train_current_speeds)
    )
    assert policy.features["current_speed"].mean != pytest.approx(
        sum(all_current_speeds) / len(all_current_speeds)
    )


def test_traffic_scaling_policy_transforms_each_partition_with_training_statistics():
    fixture = load_synthetic_traffic_fixture(seed=7).with_past_lag_imputation()
    splits = fixture.temporal_split_observations(train=0.5, validation=0.25, test=0.25)
    policy = fit_traffic_scaling_policy(splits["train"])

    scaled_splits = {
        split_name: policy.transform_split(split)
        for split_name, split in splits.items()
    }

    assert set(scaled_splits) == {"train", "validation", "test"}
    assert all(
        len(scaled_splits[split_name].observations) == len(split.observations)
        for split_name, split in splits.items()
    )

    original_row = _observed(splits["validation"])
    scaled_row = _row_like(scaled_splits["validation"], original_row)

    assert scaled_row.target_speed == pytest.approx(
        (original_row.target_speed - policy.target.mean) / policy.target.scale
    )
    assert scaled_row.local_covariates["current_speed"] == pytest.approx(
        (
            original_row.local_covariates["current_speed"]
            - policy.features["current_speed"].mean
        )
        / policy.features["current_speed"].scale
    )
    assert scaled_row.local_covariates["weekend"] is original_row.local_covariates[
        "weekend"
    ]
    assert scaled_row.neighbor_covariates["neighbor_lag_5m"] == pytest.approx(
        (
            original_row.neighbor_covariates["neighbor_lag_5m"]
            - policy.features["neighbor_lag_5m"].mean
        )
        / policy.features["neighbor_lag_5m"].scale
    )

    test_row = _observed(splits["test"])
    scaled_test_row = _row_like(scaled_splits["test"], test_row)
    assert scaled_test_row.target_speed == pytest.approx(
        (test_row.target_speed - policy.target.mean) / policy.target.scale
    )

    first_scaled_train_row = _row_like(
        scaled_splits["train"],
        next(row for row in splits["train"] if row.local_covariates["lag_5m"] is None),
    )
    assert first_scaled_train_row.local_covariates["lag_5m"] is None


def test_traffic_scaling_policy_restores_gaussian_predictions_to_original_units():
    fixture = load_synthetic_traffic_fixture(seed=7).with_past_lag_imputation()
    splits = fixture.temporal_split_observations(train=0.5, validation=0.25, test=0.25)
    policy = fit_traffic_scaling_policy(splits["train"])

    prediction = policy.inverse_transform_gaussian(mu=-0.25, sigma=1.5)

    assert prediction.unit == "original_traffic"
    assert prediction.mu == pytest.approx(
        -0.25 * policy.target.scale + policy.target.mean
    )
    assert prediction.sigma == pytest.approx(1.5 * policy.target.scale)


def _population_scale(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _feature_values(split, name):
    return [
        row.local_covariates[name]
        for row in split.observations
        if row.local_covariates[name] is not None
    ]


def _observed(split):
    return next(
        row
        for row in split.observations
        if row.target_speed is not None
        and row.local_covariates["current_speed"] is not None
        and row.neighbor_covariates["neighbor_lag_5m"] is not None
    )


def _row_like(split, original_row):
    return next(
        row
        for row in split.observations
        if row.sensor_id == original_row.sensor_id
        and row.time_index == original_row.time_index
    )
