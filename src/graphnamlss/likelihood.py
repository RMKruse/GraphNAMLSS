from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real
from typing import Any


__all__ = [
    "DEFAULT_SCALE_FLOOR",
    "gaussian_nll",
    "positive_scale",
]


DEFAULT_SCALE_FLOOR = 1e-6


def positive_scale(
    unconstrained: Any, *, floor: float = DEFAULT_SCALE_FLOOR
) -> Any:
    """Map unconstrained scale predictors to strictly positive scales."""
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("Scale floor must be a positive finite value.")

    if _is_torch_tensor(unconstrained):
        torch = _torch_module()
        return torch.nn.functional.softplus(unconstrained) + floor

    if isinstance(unconstrained, Real):
        return _positive_scale_scalar(float(unconstrained), floor)

    return [_positive_scale_scalar(float(value), floor) for value in unconstrained]


def gaussian_nll(
    *,
    mu: Any,
    sigma: Any,
    target: Any,
    mask: Any | None = None,
) -> Any:
    """Return mean Gaussian negative log-likelihood over observed targets."""
    if _contains_torch_tensor(mu, sigma, target, mask):
        return _gaussian_nll_torch(mu=mu, sigma=sigma, target=target, mask=mask)

    targets = _as_float_tuple(target, "target")
    mus = _as_float_tuple(mu, "mu")
    sigmas = _as_float_tuple(sigma, "sigma")
    masks = (True,) * len(targets) if mask is None else _as_bool_tuple(mask)

    mus = _broadcast_or_match(mus, len(targets), "mu")
    sigmas = _broadcast_or_match(sigmas, len(targets), "sigma")
    masks = _broadcast_or_match(masks, len(targets), "mask")

    terms = []
    for mean, scale, observed_target, observed in zip(
        mus, sigmas, targets, masks, strict=True
    ):
        if not observed:
            continue
        if not math.isfinite(mean) or not math.isfinite(observed_target):
            raise ValueError("Gaussian NLL values must be finite.")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Gaussian sigma values must be positive and finite.")

        standardized_residual = (observed_target - mean) / scale
        terms.append(
            0.5 * math.log(2.0 * math.pi)
            + math.log(scale)
            + 0.5 * standardized_residual**2
        )

    if not terms:
        raise ValueError("Gaussian NLL requires at least one observed target.")
    return sum(terms) / len(terms)


def _positive_scale_scalar(unconstrained: float, floor: float) -> float:
    if not math.isfinite(unconstrained):
        raise ValueError("Scale predictor values must be finite.")

    if unconstrained > 0.0:
        return unconstrained + math.log1p(math.exp(-unconstrained)) + floor
    return math.log1p(math.exp(unconstrained)) + floor


def _gaussian_nll_torch(*, mu: Any, sigma: Any, target: Any, mask: Any | None) -> Any:
    torch = _torch_module()
    reference = _first_torch_tensor(mu, sigma, target)
    dtype = reference.dtype if reference.is_floating_point() else torch.float32
    device = reference.device

    mu_tensor = torch.as_tensor(mu, dtype=dtype, device=device)
    sigma_tensor = torch.as_tensor(sigma, dtype=dtype, device=device)
    target_tensor = torch.as_tensor(target, dtype=dtype, device=device)

    loss = (
        0.5 * math.log(2.0 * math.pi)
        + torch.log(sigma_tensor)
        + 0.5 * ((target_tensor - mu_tensor) / sigma_tensor) ** 2
    )

    if mask is not None:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        mask_tensor = torch.broadcast_to(mask_tensor, loss.shape)
        loss = loss[mask_tensor]

    if loss.numel() == 0:
        raise ValueError("Gaussian NLL requires at least one observed target.")
    if not bool(torch.isfinite(loss).all().item()):
        raise ValueError("Observed Gaussian NLL values must be finite.")

    return loss.mean()


def _as_float_tuple(value: Real | Iterable[Real], label: str) -> tuple[float, ...]:
    if isinstance(value, Real):
        return (float(value),)

    values = tuple(float(item) for item in value)
    if not values:
        raise ValueError(f"Gaussian NLL requires at least one {label} value.")
    return values


def _as_bool_tuple(value: bool | Iterable[bool]) -> tuple[bool, ...]:
    if isinstance(value, bool):
        return (value,)
    return tuple(bool(item) for item in value)


def _broadcast_or_match(
    values: tuple[float, ...] | tuple[bool, ...], size: int, label: str
) -> tuple[float, ...] | tuple[bool, ...]:
    if len(values) == size:
        return values
    if len(values) == 1:
        return values * size
    raise ValueError(f"Gaussian NLL {label} length must be 1 or match target length.")


def _contains_torch_tensor(*values: Any) -> bool:
    return any(_is_torch_tensor(value) for value in values if value is not None)


def _first_torch_tensor(*values: Any) -> Any:
    for value in values:
        if _is_torch_tensor(value):
            return value
    raise TypeError("Expected at least one torch tensor.")


def _is_torch_tensor(value: Any) -> bool:
    try:
        torch = _torch_module()
    except ModuleNotFoundError:
        return False
    return isinstance(value, torch.Tensor)


def _torch_module() -> Any:
    import torch

    return torch
