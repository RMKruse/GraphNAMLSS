import pytest

from graphnamlss.testing import load_synthetic_traffic_fixture


def test_synthetic_traffic_fixture_behaves_like_sensor_time_observations():
    fixture = load_synthetic_traffic_fixture(seed=7)

    assert len(fixture.sensors) >= 3
    assert len(fixture.time_points) >= 6
    assert len(fixture.observations) == len(fixture.sensors) * len(fixture.time_points)

    observation = fixture.observations[0]
    assert observation.sensor_id in fixture.sensors
    assert observation.time_index == 0
    assert observation.target_speed is not None
    assert "lag_5m" in observation.local_covariates
    assert "neighbor_current_speed" in observation.neighbor_covariates

    assert fixture.adjacency
    assert all(edge.source in fixture.sensors for edge in fixture.adjacency)
    assert all(edge.target in fixture.sensors for edge in fixture.adjacency)
    assert any(row.target_missing for row in fixture.observations)
    assert any(row.current_speed_missing for row in fixture.observations)


def test_synthetic_traffic_fixture_aligns_fifteen_minute_target_per_sensor():
    fixture = load_synthetic_traffic_fixture(seed=7)

    prediction_time = _observation(fixture, "sensor_a", 2)
    horizon_time = _observation(fixture, "sensor_a", 5)

    assert prediction_time.target_speed == horizon_time.current_speed
    assert not prediction_time.target_missing

    for time_index in (13, 14, 15):
        tail_observation = _observation(fixture, "sensor_a", time_index)
        assert tail_observation.target_speed is None
        assert tail_observation.target_missing


def test_synthetic_traffic_fixture_aligns_full_local_covariate_set_from_history():
    fixture = load_synthetic_traffic_fixture(seed=7)

    observation = _observation(fixture, "sensor_a", 12)
    covariates = observation.local_covariates

    assert covariates["current_speed"] == observation.current_speed
    assert covariates["lag_5m"] == _observation(fixture, "sensor_a", 11).current_speed
    assert covariates["lag_10m"] == _observation(fixture, "sensor_a", 10).current_speed
    assert covariates["lag_15m"] == _observation(fixture, "sensor_a", 9).current_speed
    assert covariates["lag_30m"] == _observation(fixture, "sensor_a", 6).current_speed
    assert covariates["lag_60m"] == _observation(fixture, "sensor_a", 0).current_speed

    first_observation = _observation(fixture, "sensor_a", 0)
    assert first_observation.local_covariates["lag_5m"] is None
    assert first_observation.local_covariates["lag_60m"] is None


def test_synthetic_traffic_fixture_aligns_neighbor_covariates_from_graph_history():
    fixture = load_synthetic_traffic_fixture(seed=7)

    observation = _observation(fixture, "sensor_a", 12)
    covariates = observation.neighbor_covariates

    assert covariates["neighbor_current_speed"] == _observation(
        fixture, "sensor_d", 12
    ).current_speed
    assert covariates["neighbor_lag_5m"] == _observation(
        fixture, "sensor_d", 11
    ).current_speed
    assert covariates["neighbor_lag_10m"] == _observation(
        fixture, "sensor_d", 10
    ).current_speed
    assert covariates["neighbor_lag_15m"] == _observation(
        fixture, "sensor_d", 9
    ).current_speed
    assert covariates["neighbor_lag_30m"] == _observation(
        fixture, "sensor_d", 6
    ).current_speed
    assert covariates["neighbor_lag_60m"] == _observation(
        fixture, "sensor_d", 0
    ).current_speed

    assert not {"time_of_day", "day_of_week", "weekend"}.intersection(covariates)

    first_observation = _observation(fixture, "sensor_a", 0)
    assert first_observation.neighbor_covariates["neighbor_lag_5m"] is None
    assert first_observation.neighbor_covariates["neighbor_lag_60m"] is None


def test_synthetic_traffic_fixture_loads_dataset_provided_adjacency(tmp_path):
    adjacency_path = tmp_path / "adjacency.csv"
    adjacency_path.write_text(
        "source,target,weight\n"
        "sensor_b,sensor_a,0.25\n"
        "sensor_c,sensor_a,0.75\n"
        "sensor_a,sensor_d,1.0\n"
    )

    fixture = load_synthetic_traffic_fixture(seed=7, adjacency_path=adjacency_path)

    assert [(edge.source, edge.target, edge.weight) for edge in fixture.adjacency] == [
        ("sensor_b", "sensor_a", 0.25),
        ("sensor_c", "sensor_a", 0.75),
        ("sensor_a", "sensor_d", 1.0),
    ]

    observation = _observation(fixture, "sensor_a", 12)
    sensor_b_speed = _observation(fixture, "sensor_b", 12).current_speed
    sensor_c_speed = _observation(fixture, "sensor_c", 12).current_speed
    expected_neighbor_speed = round((sensor_b_speed * 0.25 + sensor_c_speed * 0.75), 3)

    assert observation.neighbor_covariates["neighbor_current_speed"] == (
        expected_neighbor_speed
    )


def test_synthetic_traffic_fixture_rejects_adjacency_for_unknown_sensors(tmp_path):
    adjacency_path = tmp_path / "adjacency.csv"
    adjacency_path.write_text(
        "source,target,weight\n"
        "sensor_z,sensor_a,1.0\n"
    )

    with pytest.raises(ValueError, match="unknown sensors"):
        load_synthetic_traffic_fixture(seed=7, adjacency_path=adjacency_path)


def test_synthetic_traffic_fixture_exposes_local_feature_metadata():
    fixture = load_synthetic_traffic_fixture(seed=7)

    observation = _observation(fixture, "sensor_a", 12)
    metadata_by_name = {
        feature.name: feature for feature in fixture.local_feature_metadata
    }

    assert tuple(metadata_by_name) == tuple(observation.local_covariates)
    assert observation.local_covariates["time_of_day"] == 8.0
    assert observation.local_covariates["day_of_week"] == 0.0
    assert observation.local_covariates["weekend"] is False

    assert metadata_by_name["lag_60m"].source == "traffic_lag"
    assert metadata_by_name["lag_60m"].offset_minutes == -60
    assert metadata_by_name["time_of_day"].source == "calendar"
    assert metadata_by_name["time_of_day"].unit == "hour_of_day"
    assert metadata_by_name["weekend"].dtype == "indicator"


def test_synthetic_traffic_fixture_exposes_neighbor_feature_metadata_without_calendar():
    fixture = load_synthetic_traffic_fixture(seed=7)

    observation = _observation(fixture, "sensor_a", 12)
    metadata_by_name = {
        feature.name: feature for feature in fixture.neighbor_feature_metadata
    }

    assert tuple(metadata_by_name) == tuple(observation.neighbor_covariates)
    assert {feature.source for feature in metadata_by_name.values()} == {
        "neighbor_traffic",
        "neighbor_traffic_lag",
        "neighbor_traffic_missingness",
    }
    assert metadata_by_name["neighbor_lag_60m"].source == "neighbor_traffic_lag"
    assert metadata_by_name["neighbor_lag_60m"].offset_minutes == -60
    assert "calendar" not in {
        feature.source for feature in fixture.neighbor_feature_metadata
    }


def test_synthetic_traffic_fixture_is_deterministic_under_fixed_seed():
    first = load_synthetic_traffic_fixture(seed=11)
    second = load_synthetic_traffic_fixture(seed=11)
    different_seed = load_synthetic_traffic_fixture(seed=12)

    assert first == second
    assert first.sensors == different_seed.sensors
    assert first.time_points == different_seed.time_points
    assert first.adjacency == different_seed.adjacency
    assert first.observations[0].target_speed != different_seed.observations[0].target_speed


def test_synthetic_traffic_fixture_provides_temporal_split_observations():
    fixture = load_synthetic_traffic_fixture(seed=3)

    splits = fixture.temporal_split_observations(train=0.5, validation=0.25, test=0.25)

    assert set(splits) == {"train", "validation", "test"}
    assert {row.time_index for row in splits["train"]} == set(range(8))
    assert {row.time_index for row in splits["validation"]} == set(range(8, 12))
    assert {row.time_index for row in splits["test"]} == set(range(12, 16))
    assert {row.sensor_id for row in splits["train"]} == set(fixture.sensors)


def test_synthetic_traffic_split_partitions_share_sensor_graph_metadata():
    fixture = load_synthetic_traffic_fixture(seed=3)

    splits = fixture.temporal_split_observations(train=0.5, validation=0.25, test=0.25)

    for partition in splits.values():
        assert partition.sensors == fixture.sensors
        assert partition.adjacency == fixture.adjacency
        assert partition.local_feature_metadata == fixture.local_feature_metadata
        assert partition.neighbor_feature_metadata == fixture.neighbor_feature_metadata

    assert splits["train"].time_points == fixture.time_points[:8]
    assert splits["validation"].time_points == fixture.time_points[8:12]
    assert splits["test"].time_points == fixture.time_points[12:]
    assert {row.time_index for row in splits["validation"].observations} == set(range(8, 12))


def test_synthetic_traffic_split_observations_are_chronological():
    fixture = load_synthetic_traffic_fixture(seed=3)

    splits = fixture.temporal_split_observations(train=0.5, validation=0.25, test=0.25)

    for partition in splits.values():
        sensor_times = [(row.time_index, row.sensor_id) for row in partition.observations]
        assert sensor_times == sorted(sensor_times)


def test_synthetic_traffic_split_does_not_leak_sensor_times_between_partitions():
    fixture = load_synthetic_traffic_fixture(seed=3)

    splits = fixture.temporal_split_observations(train=0.5, validation=0.25, test=0.25)
    train_times = {row.time_index for row in splits["train"].observations}
    validation_times = {row.time_index for row in splits["validation"].observations}
    test_times = {row.time_index for row in splits["test"].observations}

    assert max(train_times) < min(validation_times)
    assert max(validation_times) < min(test_times)

    split_sensor_times = [
        {(row.sensor_id, row.time_index) for row in partition.observations}
        for partition in splits.values()
    ]
    assert split_sensor_times[0].isdisjoint(split_sensor_times[1])
    assert split_sensor_times[0].isdisjoint(split_sensor_times[2])
    assert split_sensor_times[1].isdisjoint(split_sensor_times[2])


def test_synthetic_traffic_split_masks_missing_targets_for_loss_and_metrics():
    fixture = load_synthetic_traffic_fixture(seed=7)

    splits = fixture.temporal_split_observations(train=0.5, validation=0.25, test=0.25)
    train_rows = splits["train"].observed_target_observations()

    assert train_rows
    assert all(not row.target_missing for row in train_rows)
    assert len(train_rows) == sum(
        1 for row in splits["train"].observations if not row.target_missing
    )


def test_synthetic_traffic_lag_imputation_uses_only_past_observations():
    fixture = load_synthetic_traffic_fixture(seed=7)

    imputed = fixture.with_past_lag_imputation()
    row = _observation(imputed, "sensor_b", 4)

    assert _observation(fixture, "sensor_b", 4).local_covariates["lag_5m"] is None
    assert row.local_covariates["lag_5m"] == _observation(
        fixture, "sensor_b", 2
    ).current_speed
    assert row.local_covariates["lag_5m"] != _observation(
        fixture, "sensor_b", 4
    ).current_speed


def test_synthetic_traffic_neighbor_lag_imputation_uses_only_past_observations():
    fixture = load_synthetic_traffic_fixture(seed=7)

    imputed = fixture.with_past_lag_imputation()
    row = _observation(imputed, "sensor_c", 4)

    assert _observation(fixture, "sensor_c", 4).neighbor_covariates[
        "neighbor_lag_5m"
    ] is None
    assert row.neighbor_covariates["neighbor_lag_5m"] == _observation(
        fixture, "sensor_b", 2
    ).current_speed
    assert row.neighbor_covariates["neighbor_lag_5m_missing"] is True
    assert row.neighbor_covariates["neighbor_lag_5m_imputed"] is True


def test_synthetic_traffic_lag_imputation_exposes_indicator_metadata():
    fixture = load_synthetic_traffic_fixture(seed=7)

    imputed = fixture.with_past_lag_imputation()
    imputed_row = _observation(imputed, "sensor_b", 4)
    observed_row = _observation(imputed, "sensor_b", 5)
    metadata_by_name = {
        feature.name: feature for feature in imputed.local_feature_metadata
    }
    neighbor_metadata_by_name = {
        feature.name: feature for feature in imputed.neighbor_feature_metadata
    }

    assert imputed_row.local_covariates["lag_5m_imputed"] is True
    assert observed_row.local_covariates["lag_5m_imputed"] is False
    assert metadata_by_name["lag_5m_imputed"].source == "traffic_imputation"
    assert metadata_by_name["lag_5m_imputed"].dtype == "indicator"
    assert (
        neighbor_metadata_by_name["neighbor_lag_5m_imputed"].source
        == "neighbor_traffic_imputation"
    )


def test_synthetic_traffic_fixture_reports_missingness_policy():
    fixture = load_synthetic_traffic_fixture(seed=7)

    report = fixture.with_past_lag_imputation().missingness_report()

    assert report.target_missing_count == sum(
        1 for row in fixture.observations if row.target_missing
    )
    assert report.target_missing_rate == report.target_missing_count / len(
        fixture.observations
    )
    assert report.imputation_strategy == "past_observed_lag"
    assert "lag_5m" in report.imputed_feature_names
    assert "neighbor_lag_5m" in report.imputed_feature_names


def _observation(fixture, sensor_id, time_index):
    return next(
        row
        for row in fixture.observations
        if row.sensor_id == sensor_id and row.time_index == time_index
    )
