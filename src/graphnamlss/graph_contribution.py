"""Centered additive graph contributions for GraphNAMLSS predictors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

try:
    import torch
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None


CenteringMode = Literal["none", "batch"]
_Module = torch.nn.Module if torch is not None else object

__all__ = [
    "CenteredGraphContribution",
    "CenteringMode",
    "GraphContributionResult",
]


@dataclass(frozen=True)
class GraphContributionResult:
    """Predictor-scale graph contributions and gates."""

    parameter_names: tuple[str, ...]
    raw_contributions: Any
    centered_contributions: Any
    gated_contributions: Any
    eta: Any
    gate_values: dict[str, float]

    @property
    def eta_shape(self) -> tuple[int, ...]:
        return tuple(self.eta.shape)

    @property
    def contribution_shape(self) -> tuple[int, ...]:
        return tuple(self.centered_contributions.shape)

    def gate_rows(self) -> list[dict[str, float | str]]:
        return [
            {"parameter": parameter_name, "gate": self.gate_values[parameter_name]}
            for parameter_name in self.parameter_names
        ]


class CenteredGraphContribution(_Module):
    """Add centered, gated graph contributions to distributional predictors."""

    def __init__(
        self,
        *,
        parameter_names: Sequence[str],
        embedding_dim: int,
        gate_init: Mapping[str, Real] | None = None,
        centering: CenteringMode = "batch",
    ) -> None:
        torch_module = _require_torch()
        super().__init__()
        self.parameter_names = _names_tuple(parameter_names, "parameter_names")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.embedding_dim = embedding_dim
        self.centering = _validate_centering(centering)
        gate_values = gate_init or {}
        self.heads = torch_module.nn.ModuleDict(
            {
                parameter_name: torch_module.nn.Linear(embedding_dim, 1)
                for parameter_name in self.parameter_names
            }
        )
        self.gates = torch_module.nn.ParameterDict(
            {
                parameter_name: torch_module.nn.Parameter(
                    torch_module.tensor(
                        _finite_float(
                            gate_values.get(parameter_name, 1.0), parameter_name
                        ),
                        dtype=torch_module.float32,
                    )
                )
                for parameter_name in self.parameter_names
            }
        )

    def forward(
        self,
        embeddings: Any,
        *,
        local_eta: Any | None = None,
        centering: CenteringMode | None = None,
    ) -> GraphContributionResult:
        _validate_embedding_shape(embeddings, self.embedding_dim)
        centering_mode = (
            self.centering if centering is None else _validate_centering(centering)
        )
        raw_contributions = self._raw_contributions(embeddings)
        centered_contributions = _center_contributions(
            raw_contributions, centering=centering_mode
        )
        gate_tensor = self._gate_tensor(centered_contributions)
        gated_contributions = centered_contributions * gate_tensor
        if local_eta is None:
            eta = gated_contributions
        else:
            _validate_eta_shape(local_eta, gated_contributions)
            eta = local_eta + gated_contributions

        return GraphContributionResult(
            parameter_names=self.parameter_names,
            raw_contributions=raw_contributions,
            centered_contributions=centered_contributions,
            gated_contributions=gated_contributions,
            eta=eta,
            gate_values=self.gate_values(),
        )

    def gate_values(self) -> dict[str, float]:
        return {
            parameter_name: float(self.gates[parameter_name].detach().cpu().item())
            for parameter_name in self.parameter_names
        }

    def _raw_contributions(self, embeddings: Any) -> Any:
        torch_module = _require_torch()
        return torch_module.cat(
            [
                self.heads[parameter_name](embeddings)
                for parameter_name in self.parameter_names
            ],
            dim=-1,
        )

    def _gate_tensor(self, contributions: Any) -> Any:
        torch_module = _require_torch()
        gates = torch_module.stack(
            [self.gates[parameter_name] for parameter_name in self.parameter_names]
        ).to(dtype=contributions.dtype, device=contributions.device)
        return gates.reshape(
            (1,) * (contributions.ndim - 1) + (len(self.parameter_names),)
        )


def _names_tuple(names: Sequence[str], field_name: str) -> tuple[str, ...]:
    if not names:
        raise ValueError(f"{field_name} must contain at least one name.")
    names_tuple = tuple(names)
    if any(not isinstance(name, str) or not name for name in names_tuple):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    if len(set(names_tuple)) != len(names_tuple):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return names_tuple


def _validate_centering(centering: str) -> CenteringMode:
    if centering not in {"none", "batch"}:
        raise ValueError("centering must be one of: none, batch.")
    return centering  # type: ignore[return-value]


def _finite_float(value: Real, field_name: str) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number.")
    float_value = float(value)
    if not math.isfinite(float_value):
        raise ValueError(f"{field_name} must be finite.")
    return float_value


def _validate_embedding_shape(embeddings: Any, embedding_dim: int) -> None:
    if embeddings.ndim < 2:
        raise ValueError("embeddings must have shape (..., node, embedding).")
    if embeddings.shape[-1] != embedding_dim:
        raise ValueError(
            "embeddings last dimension must match embedding_dim "
            f"({embeddings.shape[-1]} != {embedding_dim})."
        )


def _validate_eta_shape(local_eta: Any, contributions: Any) -> None:
    if tuple(local_eta.shape) != tuple(contributions.shape):
        raise ValueError(
            "local_eta shape must match graph contribution shape "
            f"({tuple(local_eta.shape)} != {tuple(contributions.shape)})."
        )


def _center_contributions(contributions: Any, *, centering: CenteringMode) -> Any:
    if centering == "none":
        return contributions
    center_dims = tuple(range(contributions.ndim - 1))
    return contributions - contributions.mean(dim=center_dims, keepdim=True)


def _require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError(
            "CenteredGraphContribution requires the optional 'torch' dependency."
        )
    return torch
