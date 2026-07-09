import pytest

from graphnamlss.graph_encoder import (
    GraphEncoderInput,
    GraphMetadata,
    TracerBulletGraphEncoder,
)
from graphnamlss.graph_contribution import CenteredGraphContribution
from graphnamlss.testing.synthetic_traffic import load_synthetic_traffic_fixture


def test_centered_graph_contribution_keeps_each_parameter_zero_mean():
    torch = pytest.importorskip("torch")
    embeddings = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
            [[13.0, 17.0], [19.0, 23.0], [29.0, 31.0]],
        ],
        dtype=torch.float32,
    )
    contribution = CenteredGraphContribution(
        parameter_names=("mu", "sigma"),
        embedding_dim=2,
        centering="batch",
    )
    with torch.no_grad():
        contribution.heads["mu"].weight.copy_(torch.tensor([[1.0, 0.0]]))
        contribution.heads["mu"].bias.zero_()
        contribution.heads["sigma"].weight.copy_(torch.tensor([[0.0, 1.0]]))
        contribution.heads["sigma"].bias.zero_()

    result = contribution(embeddings)

    assert result.eta_shape == (2, 3, 2)
    assert result.contribution_shape == (2, 3, 2)
    assert result.centered_contributions[..., 0].mean().item() == pytest.approx(0.0)
    assert result.centered_contributions[..., 1].mean().item() == pytest.approx(0.0)
    assert torch.equal(result.eta, result.centered_contributions)


def test_graph_gates_modulate_contributions_before_predictor_addition():
    torch = pytest.importorskip("torch")
    embeddings = torch.tensor([[[1.0], [3.0], [5.0]]], dtype=torch.float32)
    local_eta = torch.tensor(
        [[[10.0, -1.0], [10.0, -1.0], [10.0, -1.0]]],
        dtype=torch.float32,
    )
    contribution = CenteredGraphContribution(
        parameter_names=("mu", "sigma"),
        embedding_dim=1,
        gate_init={"mu": 0.0, "sigma": 2.0},
        centering="none",
    )
    with torch.no_grad():
        contribution.heads["mu"].weight.fill_(1.0)
        contribution.heads["mu"].bias.zero_()
        contribution.heads["sigma"].weight.fill_(1.0)
        contribution.heads["sigma"].bias.zero_()

    result = contribution(embeddings, local_eta=local_eta)

    assert torch.equal(result.eta[..., 0], local_eta[..., 0])
    assert result.eta[..., 1].tolist() == pytest.approx([[1.0, 5.0, 9.0]])
    assert result.gate_values == pytest.approx({"mu": 0.0, "sigma": 2.0})
    assert result.gate_rows() == [
        {"parameter": "mu", "gate": 0.0},
        {"parameter": "sigma", "gate": 2.0},
    ]


def test_graph_contribution_shapes_on_synthetic_graph_encoder_inputs():
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
    contribution = CenteredGraphContribution(
        parameter_names=("mu", "sigma"),
        embedding_dim=5,
        gate_init={"mu": 0.5, "sigma": 1.5},
    )

    embeddings = encoder(GraphEncoderInput(features=features, metadata=metadata))
    result = contribution(embeddings)

    assert result.eta_shape == (2, len(fixture.sensors), 2)
    assert result.contribution_shape == (2, len(fixture.sensors), 2)
    assert result.gate_rows() == [
        {"parameter": "mu", "gate": pytest.approx(0.5)},
        {"parameter": "sigma", "gate": pytest.approx(1.5)},
    ]


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
