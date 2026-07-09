import pytest

from graphnamlss.distributional_gnn import DistributionalGNNBaseline, PointGNNReference
from graphnamlss.graph_encoder import GraphEncoderInput, GraphMetadata
from graphnamlss.testing.synthetic_traffic import load_synthetic_traffic_fixture


def test_distributional_gnn_baseline_predicts_positive_location_scale_shapes():
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
    model = DistributionalGNNBaseline(
        input_dim=len(feature_names),
        embedding_dim=5,
        parameter_names=("mu", "sigma"),
    )

    prediction = model(GraphEncoderInput(features=features, metadata=metadata))

    assert prediction["mu"].shape == (2, len(fixture.sensors))
    assert prediction["sigma"].shape == (2, len(fixture.sensors))
    assert torch.isfinite(prediction["mu"]).all()
    assert torch.isfinite(prediction["sigma"]).all()
    assert torch.all(prediction["sigma"] > 0.0)


def test_point_gnn_reference_predicts_mean_only_shapes():
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
    model = PointGNNReference(
        input_dim=len(feature_names),
        embedding_dim=5,
    )

    prediction = model(GraphEncoderInput(features=features, metadata=metadata))

    assert set(prediction) == {"mu"}
    assert prediction["mu"].shape == (2, len(fixture.sensors))
    assert torch.isfinite(prediction["mu"]).all()


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
