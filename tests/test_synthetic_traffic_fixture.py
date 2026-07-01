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
    assert {row.time_index for row in splits["train"]} == {0, 1, 2, 3}
    assert {row.time_index for row in splits["validation"]} == {4, 5}
    assert {row.time_index for row in splits["test"]} == {6, 7}
    assert {row.sensor_id for row in splits["train"]} == set(fixture.sensors)
