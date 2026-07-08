import pytest

from graphnamlss.graph_encoder import (
    GraphEncoderInput,
    GraphMetadata,
    TracerBulletGraphEncoder,
    aggregate_neighbor_features,
)
from graphnamlss.testing.synthetic_traffic import (
    SyntheticTrafficEdge,
    load_synthetic_traffic_fixture,
)


def test_graph_metadata_exposes_weighted_and_unweighted_physical_graph_forms():
    fixture = load_synthetic_traffic_fixture(seed=11)

    metadata = GraphMetadata.from_edges(
        node_ids=fixture.sensors,
        edges=fixture.adjacency,
    )
    fallback = metadata.without_edge_weights()

    assert metadata.node_ids == fixture.sensors
    assert metadata.edge_index == ((0, 1), (1, 2), (2, 3), (3, 0))
    assert metadata.edge_weight == pytest.approx((0.8, 0.6, 0.7, 0.4))
    assert fallback.edge_index == metadata.edge_index
    assert fallback.edge_weight is None


def test_tracer_bullet_graph_encoder_embeds_synthetic_sensor_time_neighbor_features():
    torch = pytest.importorskip("torch")
    fixture = load_synthetic_traffic_fixture(seed=11)
    feature_names = tuple(
        metadata.name
        for metadata in fixture.neighbor_feature_metadata
        if metadata.dtype == "continuous"
    )
    features = _sensor_time_neighbor_features(
        torch, fixture, time_indices=(12, 13), feature_names=feature_names
    )
    metadata = GraphMetadata.from_edges(
        node_ids=fixture.sensors,
        edges=fixture.adjacency,
    )
    encoder = TracerBulletGraphEncoder(
        input_dim=len(feature_names),
        embedding_dim=5,
    )

    embeddings = encoder(GraphEncoderInput(features=features, metadata=metadata))

    assert embeddings.shape == (2, len(fixture.sensors), 5)
    assert torch.isfinite(embeddings).all()


def test_neighbor_aggregation_uses_edge_weights_with_unweighted_fallback():
    torch = pytest.importorskip("torch")
    fixture = load_synthetic_traffic_fixture(seed=11)
    metadata = GraphMetadata.from_edges(
        node_ids=fixture.sensors,
        edges=(
            SyntheticTrafficEdge("sensor_b", "sensor_a", 0.25),
            SyntheticTrafficEdge("sensor_c", "sensor_a", 0.75),
        ),
    )
    features = torch.tensor(
        [[10.0], [20.0], [40.0], [80.0]],
        dtype=torch.float32,
    )

    weighted = aggregate_neighbor_features(features, metadata)
    unweighted = aggregate_neighbor_features(
        features,
        metadata,
        use_edge_weights=False,
    )

    assert weighted[0, 0].item() == pytest.approx(35.0)
    assert unweighted[0, 0].item() == pytest.approx(30.0)
    assert torch.equal(weighted[1:], torch.zeros_like(weighted[1:]))


def _sensor_time_neighbor_features(torch, fixture, *, time_indices, feature_names):
    observations = {
        (observation.sensor_id, observation.time_index): observation
        for observation in fixture.observations
    }
    return torch.tensor(
        [
            [
                [
                    observations[(sensor_id, time_index)].neighbor_covariates[
                        feature_name
                    ]
                    for feature_name in feature_names
                ]
                for sensor_id in fixture.sensors
            ]
            for time_index in time_indices
        ],
        dtype=torch.float32,
    )
