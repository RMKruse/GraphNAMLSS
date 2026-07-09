import pytest

from graphnamlss.statistical_baselines import (
    StatisticalBaselineAdapterContract,
    StatisticalBaselineAdapterInput,
    StatisticalBaselinePrediction,
)


def test_statistical_baseline_input_schema_exposes_adapter_columns():
    adapter_input = StatisticalBaselineAdapterInput(
        rows=[
            {
                "sensor_id": "A",
                "split": "train",
                "speed": 42.0,
                "lag_5": 40.5,
                "neighbor_lag_5_mean": 38.0,
                "exposure": 1.0,
            },
            {
                "sensor_id": "A",
                "split": "test",
                "speed": 43.0,
                "lag_5": 42.0,
                "neighbor_lag_5_mean": 39.5,
                "exposure": 0.5,
            },
        ],
        target_column="speed",
        local_covariate_columns=["lag_5"],
        graph_feature_covariate_columns=["neighbor_lag_5_mean"],
        exposure_columns=["exposure"],
        split_column="split",
    )

    assert adapter_input.covariate_columns == ("lag_5", "neighbor_lag_5_mean")
    assert adapter_input.target_values == (42.0, 43.0)
    assert adapter_input.exposure_values("exposure") == (1.0, 0.5)
    assert adapter_input.split_values == ("train", "test")


def test_statistical_baseline_input_schema_rejects_missing_graph_features():
    with pytest.raises(ValueError, match="neighbor_lag_5_mean"):
        StatisticalBaselineAdapterInput(
            rows=[
                {
                    "split": "train",
                    "speed": 42.0,
                    "lag_5": 40.5,
                },
            ],
            target_column="speed",
            local_covariate_columns=["lag_5"],
            graph_feature_covariate_columns=["neighbor_lag_5_mean"],
            split_column="split",
        )


def test_statistical_baseline_predictions_are_compatible_with_gaussian_metrics():
    predictions = StatisticalBaselinePrediction.gaussian(
        mu=[0.0, 100.0],
        sigma=[1.0, 1.0],
        unit="standardized",
    )

    report = predictions.gaussian_metric_report(
        target=[0.0, -999.0],
        mask=[True, False],
    )

    assert report.unit == "standardized"
    assert report.observed_count == 1
    assert report.mae == pytest.approx(0.0)


def test_graph_feature_statistical_baseline_contract_uses_neighbor_summaries():
    adapter_input = StatisticalBaselineAdapterInput(
        rows=[
            {
                "split": "train",
                "speed": 42.0,
                "lag_5": 40.5,
                "neighbor_lag_5_mean": 38.0,
            },
        ],
        target_column="speed",
        local_covariate_columns=["lag_5"],
        graph_feature_covariate_columns=["neighbor_lag_5_mean"],
        split_column="split",
    )
    contract = StatisticalBaselineAdapterContract.graph_feature_gamlss()

    assert contract.baseline_name == "gamlss_graph_features"
    assert contract.model_covariate_columns(adapter_input) == (
        "lag_5",
        "neighbor_lag_5_mean",
    )
