from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from graphnamlss.graph_encoder import (
    GraphEncoderInput,
    GraphMetadata,
    TracerBulletGraphEncoder,
)
from graphnamlss.likelihood import positive_scale

try:
    import torch
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None


_Module = torch.nn.Module if torch is not None else object


class DistributionalGNNBaseline(_Module):
    """Non-additive graph baseline that predicts distributional parameters directly."""

    def __init__(
        self,
        *,
        input_dim: int,
        embedding_dim: int,
        parameter_names: Sequence[str] = ("mu", "sigma"),
        num_layers: int = 1,
    ) -> None:
        torch_module = _require_torch()
        super().__init__()
        parameter_names = tuple(parameter_names)
        if "mu" not in parameter_names or "sigma" not in parameter_names:
            raise ValueError("DistributionalGNNBaseline requires mu and sigma heads.")

        self.parameter_names = parameter_names
        self.encoder = TracerBulletGraphEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )
        self.heads = torch_module.nn.ModuleDict(
            {
                parameter_name: torch_module.nn.Linear(embedding_dim, 1)
                for parameter_name in parameter_names
            }
        )

    def forward(
        self,
        graph_input: GraphEncoderInput | Any,
        metadata: GraphMetadata | None = None,
        *,
        use_edge_weights: bool = True,
    ) -> dict[str, Any]:
        embeddings = self.encoder(
            graph_input,
            metadata,
            use_edge_weights=use_edge_weights,
        )
        eta = {
            parameter_name: self.heads[parameter_name](embeddings).squeeze(-1)
            for parameter_name in self.parameter_names
        }
        return {
            "mu": eta["mu"],
            "sigma": positive_scale(eta["sigma"]),
        }


def _require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError(
            "DistributionalGNNBaseline requires the optional 'torch' dependency."
        )
    return torch


__all__ = ["DistributionalGNNBaseline"]
