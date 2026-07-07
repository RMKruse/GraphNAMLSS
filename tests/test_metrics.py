import math

import pytest

from graphnamlss.metrics import gaussian_metric_report


def test_gaussian_metric_report_includes_nll_and_crps_for_standard_normal_examples():
    report = gaussian_metric_report(
        mu=[0.0, 0.0],
        sigma=[1.0, 1.0],
        target=[0.0, 1.0],
        unit="standardized",
    )

    expected_nll = (
        0.5 * math.log(2.0 * math.pi)
        + 0.5 * math.log(2.0 * math.pi)
        + 0.5
    ) / 2.0
    standard_normal_crps_at_mean = 2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(
        math.pi
    )
    standard_normal_crps_one_sd_high = (
        2.0 * _normal_cdf(1.0) - 1.0
        + 2.0 * _normal_pdf(1.0)
        - 1.0 / math.sqrt(math.pi)
    )

    assert report.unit == "standardized"
    assert report.observed_count == 2
    assert report.nll == pytest.approx(expected_nll)
    assert report.crps == pytest.approx(
        (standard_normal_crps_at_mean + standard_normal_crps_one_sd_high) / 2.0
    )


def test_gaussian_metric_report_masks_distributional_and_point_metrics():
    report = gaussian_metric_report(
        mu=[0.0, 100.0, 2.0],
        sigma=[1.0, 1.0, 2.0],
        target=[0.0, -999.0, 4.0],
        mask=[True, False, True],
        unit="original_traffic",
        interval_level=0.5,
    )

    assert report.unit == "original_traffic"
    assert report.observed_count == 2
    assert report.pit_values == pytest.approx((0.5, _normal_cdf(1.0)))
    assert report.interval_level == pytest.approx(0.5)
    assert report.interval_coverage == pytest.approx(0.5)
    assert report.mae == pytest.approx(1.0)
    assert report.rmse == pytest.approx(math.sqrt(2.0))


def _normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_pdf(value):
    return math.exp(-0.5 * value**2) / math.sqrt(2.0 * math.pi)
