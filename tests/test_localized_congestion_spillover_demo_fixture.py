from statistics import fmean, stdev

from graphnamlss.testing import (
    load_localized_congestion_spillover_demo_fixture,
    load_synthetic_traffic_fixture,
)


def test_demo_fixture_is_separate_and_loads_without_benchmark_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    demo_fixture = load_localized_congestion_spillover_demo_fixture(seed=7)
    contract_fixture = load_synthetic_traffic_fixture(seed=7)

    assert demo_fixture.scenario.name == "localized_congestion_spillover"
    assert demo_fixture != contract_fixture
    assert demo_fixture.observations


def test_neighbor_congestion_lowers_downstream_expected_speed():
    fixture = load_localized_congestion_spillover_demo_fixture(seed=7)
    scenario = fixture.scenario

    spillover_rows = _rows_at(
        fixture,
        scenario.downstream_sensor_id,
        scenario.spillover_prediction_time_indices,
    )
    routine_rows = _rows_at(
        fixture,
        scenario.downstream_sensor_id,
        scenario.routine_prediction_time_indices,
    )

    spillover_neighbor_speed = fmean(
        row.neighbor_covariates["neighbor_current_speed"] for row in spillover_rows
    )
    routine_neighbor_speed = fmean(
        row.neighbor_covariates["neighbor_current_speed"] for row in routine_rows
    )
    spillover_local_speed = fmean(row.current_speed for row in spillover_rows)
    routine_local_speed = fmean(row.current_speed for row in routine_rows)
    spillover_target_speed = fmean(row.target_speed for row in spillover_rows)
    routine_target_speed = fmean(row.target_speed for row in routine_rows)

    assert spillover_neighbor_speed < routine_neighbor_speed - 10.0
    assert abs(spillover_local_speed - routine_local_speed) < 1.0
    assert spillover_target_speed < routine_target_speed - 5.0


def test_downstream_target_variability_increases_during_spillover_windows():
    fixture = load_localized_congestion_spillover_demo_fixture(seed=7)
    scenario = fixture.scenario

    spillover_targets = tuple(
        row.target_speed
        for row in _rows_at(
            fixture,
            scenario.downstream_sensor_id,
            scenario.spillover_prediction_time_indices,
        )
    )
    routine_targets = tuple(
        row.target_speed
        for row in _rows_at(
            fixture,
            scenario.downstream_sensor_id,
            scenario.routine_prediction_time_indices,
        )
    )

    assert stdev(spillover_targets) > stdev(routine_targets) * 3.0


def _rows_at(fixture, sensor_id, time_indices):
    return tuple(
        row
        for row in fixture.observations
        if row.sensor_id == sensor_id and row.time_index in time_indices
    )
