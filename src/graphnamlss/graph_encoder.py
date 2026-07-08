from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

try:
    import torch
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None


_Module = torch.nn.Module if torch is not None else object


@dataclass(frozen=True)
class GraphMetadata:
    node_ids: tuple[str, ...]
    edge_index: tuple[tuple[int, int], ...]
    edge_weight: tuple[float, ...] | None = None

    @classmethod
    def from_edges(
        cls,
        *,
        node_ids: Iterable[str],
        edges: Iterable[Any],
        use_edge_weights: bool = True,
    ) -> GraphMetadata:
        nodes = tuple(node_ids)
        if len(set(nodes)) != len(nodes):
            raise ValueError("node_ids must be unique.")

        node_index = {node_id: index for index, node_id in enumerate(nodes)}
        edge_index: list[tuple[int, int]] = []
        edge_weight: list[float] = []
        for edge in edges:
            source = _edge_value(edge, "source")
            target = _edge_value(edge, "target")
            if source not in node_index or target not in node_index:
                raise ValueError(
                    "edges must reference only node_ids "
                    f"(got edge {source!r} -> {target!r})."
                )
            edge_index.append((node_index[source], node_index[target]))
            if use_edge_weights:
                edge_weight.append(float(_edge_value(edge, "weight", default=1.0)))

        return cls(
            node_ids=nodes,
            edge_index=tuple(edge_index),
            edge_weight=tuple(edge_weight) if use_edge_weights else None,
        )

    def without_edge_weights(self) -> GraphMetadata:
        return GraphMetadata(
            node_ids=self.node_ids,
            edge_index=self.edge_index,
            edge_weight=None,
        )


@dataclass(frozen=True)
class GraphEncoderInput:
    features: Any
    metadata: GraphMetadata
    use_edge_weights: bool = True


class TracerBulletGraphEncoder(_Module):
    def __init__(
        self,
        *,
        input_dim: int,
        embedding_dim: int,
        num_layers: int = 1,
    ) -> None:
        torch_module = _require_torch()
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.layers = torch_module.nn.ModuleList(
            torch_module.nn.Linear(
                (input_dim if layer_index == 0 else embedding_dim) * 2,
                embedding_dim,
            )
            for layer_index in range(num_layers)
        )
        self.activation = torch_module.nn.ReLU()

    def forward(
        self,
        graph_input: GraphEncoderInput | Any,
        metadata: GraphMetadata | None = None,
        *,
        use_edge_weights: bool = True,
    ) -> Any:
        if isinstance(graph_input, GraphEncoderInput):
            features = graph_input.features
            metadata = graph_input.metadata
            use_edge_weights = graph_input.use_edge_weights
        elif metadata is not None:
            features = graph_input
        else:
            raise TypeError(
                "forward expects GraphEncoderInput or features with metadata."
            )

        _validate_feature_shape(features, metadata, self.input_dim)
        embeddings = features
        for layer in self.layers:
            neighbor_summary = aggregate_neighbor_features(
                embeddings,
                metadata,
                use_edge_weights=use_edge_weights,
            )
            embeddings = self.activation(
                layer(torch.cat((embeddings, neighbor_summary), dim=-1))
            )
        return embeddings


def aggregate_neighbor_features(
    features: Any,
    metadata: GraphMetadata,
    *,
    use_edge_weights: bool = True,
) -> Any:
    torch_module = _require_torch()
    if not metadata.edge_index:
        return torch_module.zeros_like(features)

    device = features.device
    source_index = torch_module.tensor(
        [source for source, _target in metadata.edge_index],
        dtype=torch_module.long,
        device=device,
    )
    target_index = torch_module.tensor(
        [target for _source, target in metadata.edge_index],
        dtype=torch_module.long,
        device=device,
    )
    messages = features.index_select(-2, source_index)
    weights = _edge_weights(
        features,
        metadata,
        use_edge_weights=use_edge_weights,
    )
    weighted_messages = messages * _edge_weight_view(weights, features)

    node_dim = features.ndim - 2
    aggregated = torch_module.zeros_like(features)
    aggregated.index_add_(node_dim, target_index, weighted_messages)

    denominator = features.new_zeros(*features.shape[:-1], 1)
    denominator_increment = _edge_weight_view(weights, features).expand(
        *messages.shape[:-1], 1
    )
    denominator.index_add_(node_dim, target_index, denominator_increment)
    return aggregated / denominator.clamp_min(torch_module.finfo(features.dtype).eps)


def _edge_weights(
    features: Any,
    metadata: GraphMetadata,
    *,
    use_edge_weights: bool,
) -> Any:
    torch_module = _require_torch()
    if use_edge_weights and metadata.edge_weight is not None:
        values = metadata.edge_weight
    else:
        values = (1.0,) * len(metadata.edge_index)
    return torch_module.tensor(values, dtype=features.dtype, device=features.device)


def _edge_weight_view(weights: Any, features: Any) -> Any:
    return weights.reshape((1,) * (features.ndim - 2) + (weights.numel(), 1))


def _validate_feature_shape(features: Any, metadata: GraphMetadata, input_dim: int) -> None:
    if features.ndim < 2:
        raise ValueError("features must have shape (..., node, feature).")
    if features.shape[-2] != len(metadata.node_ids):
        raise ValueError(
            "features node dimension must match graph metadata "
            f"({features.shape[-2]} != {len(metadata.node_ids)})."
        )
    if features.shape[-1] != input_dim:
        raise ValueError(
            f"features last dimension must match input_dim ({features.shape[-1]} != "
            f"{input_dim})."
        )


def _edge_value(edge: Any, name: str, *, default: Any = None) -> Any:
    if isinstance(edge, Mapping):
        return edge.get(name, default)
    return getattr(edge, name, default)


def _require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError(
            "TracerBulletGraphEncoder requires the optional 'torch' dependency."
        )
    return torch


__all__ = [
    "GraphEncoderInput",
    "GraphMetadata",
    "TracerBulletGraphEncoder",
    "aggregate_neighbor_features",
]
