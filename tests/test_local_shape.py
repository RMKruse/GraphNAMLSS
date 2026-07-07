import pytest

from graphnamlss.local_shape import CenteredLocalShapeFunctions


def test_centered_local_shape_functions_return_parameter_feature_contributions():
    model = CenteredLocalShapeFunctions(
        parameter_names=("mu", "sigma"),
        feature_names=("speed", "lag_5"),
        shape_functions={
            "mu": {
                "speed": lambda value: value,
                "lag_5": lambda value: 2.0 * value,
            },
            "sigma": {
                "speed": lambda value: -value,
                "lag_5": lambda value: value + 1.0,
            },
        },
        intercepts={"mu": 10.0, "sigma": -3.0},
        centering="none",
    )

    result = model.predict(
        [
            {"speed": 1.0, "lag_5": 2.0},
            {"speed": 3.0, "lag_5": 4.0},
            {"speed": 5.0, "lag_5": 6.0},
        ]
    )

    assert result.eta_shape == (3, 2)
    assert result.contribution_shape == (3, 2, 2)
    assert result.eta == (
        (15.0, -1.0),
        (21.0, -1.0),
        (27.0, -1.0),
    )
    assert result.contributions == (
        ((1.0, 4.0), (-1.0, 3.0)),
        ((3.0, 8.0), (-3.0, 5.0)),
        ((5.0, 12.0), (-5.0, 7.0)),
    )
    contribution_rows = result.contribution_rows()
    assert len(contribution_rows) == 12
    assert contribution_rows[0] == {
        "observation": 0,
        "parameter": "mu",
        "feature": "speed",
        "feature_value": 1.0,
        "contribution": 1.0,
    }


def test_batch_centering_keeps_each_parameter_feature_contribution_zero_mean():
    model = CenteredLocalShapeFunctions(
        parameter_names=("mu", "sigma"),
        feature_names=("speed", "lag_5"),
        shape_functions={
            "mu": {
                "speed": lambda value: value,
                "lag_5": lambda value: 10.0 * value,
            },
            "sigma": {
                "speed": lambda value: 2.0 * value,
                "lag_5": lambda value: -value,
            },
        },
        intercepts={"mu": 100.0, "sigma": 1.0},
        centering="batch",
    )

    result = model.predict(
        [
            {"speed": 1.0, "lag_5": 10.0},
            {"speed": 3.0, "lag_5": 20.0},
            {"speed": 5.0, "lag_5": 30.0},
        ]
    )

    for parameter_index in range(2):
        for feature_index in range(2):
            centered_sum = sum(
                observation[parameter_index][feature_index]
                for observation in result.contributions
            )
            assert centered_sum == pytest.approx(0.0)

    assert result.eta == (
        (-2.0, 7.0),
        (100.0, 1.0),
        (202.0, -5.0),
    )


def test_reference_centering_anchors_contributions_to_training_reference():
    model = CenteredLocalShapeFunctions(
        parameter_names=("mu", "sigma"),
        feature_names=("speed", "lag_5"),
        shape_functions={
            "mu": {
                "speed": lambda value: value,
                "lag_5": lambda value: 2.0 * value,
            },
            "sigma": {
                "speed": lambda value: -value,
                "lag_5": lambda value: value + 1.0,
            },
        },
        intercepts={"mu": 10.0, "sigma": -3.0},
        centering="reference",
    )

    model.fit_reference(
        [
            {"speed": 10.0, "lag_5": 20.0},
            {"speed": 14.0, "lag_5": 24.0},
        ]
    )
    result = model.predict(
        [
            {"speed": 12.0, "lag_5": 22.0},
            {"speed": 20.0, "lag_5": 30.0},
        ]
    )

    assert result.contributions == (
        ((0.0, 0.0), (0.0, 0.0)),
        ((8.0, 16.0), (-8.0, 8.0)),
    )
    assert result.eta == (
        (10.0, -3.0),
        (34.0, -3.0),
    )

    mu_speed_sum = sum(
        observation[0][0] for observation in result.contributions
    )
    assert mu_speed_sum == pytest.approx(8.0)
