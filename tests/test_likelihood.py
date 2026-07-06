import math

import pytest

from graphnamlss.likelihood import gaussian_nll, positive_scale


def test_positive_scale_link_maps_finite_predictors_to_positive_scales():
    scale_values = positive_scale([-1000.0, -2.0, 0.0, 2.0, 1000.0])

    assert all(math.isfinite(value) for value in scale_values)
    assert all(value > 0.0 for value in scale_values)


def test_gaussian_nll_averages_observed_targets_only():
    loss = gaussian_nll(
        mu=[0.0, 100.0],
        sigma=[1.0, 1.0],
        target=[0.0, 0.0],
        mask=[True, False],
    )

    assert loss == pytest.approx(0.5 * math.log(2.0 * math.pi))


def test_gaussian_nll_matches_closed_form_small_example():
    loss = gaussian_nll(
        mu=[0.0, 0.0],
        sigma=[1.0, 2.0],
        target=[0.0, 2.0],
    )

    standard_normal_at_mean = 0.5 * math.log(2.0 * math.pi)
    sigma_two_one_sd_high = 0.5 * math.log(2.0 * math.pi) + math.log(2.0) + 0.5
    assert loss == pytest.approx(
        (standard_normal_at_mean + sigma_two_one_sd_high) / 2.0
    )


def test_torch_likelihood_path_is_differentiable_when_torch_is_installed():
    torch = pytest.importorskip("torch")
    scale_predictor = torch.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)

    sigma = positive_scale(scale_predictor)
    loss = gaussian_nll(
        mu=torch.zeros(3),
        sigma=sigma,
        target=torch.tensor([0.0, 1.0, float("nan")]),
        mask=torch.tensor([True, True, False]),
    )
    loss.backward()

    assert torch.isfinite(sigma).all()
    assert (sigma > 0.0).all()
    assert scale_predictor.grad is not None
    assert torch.isfinite(scale_predictor.grad).all()
