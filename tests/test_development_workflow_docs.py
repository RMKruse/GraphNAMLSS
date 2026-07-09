from pathlib import Path


def test_readme_documents_plain_pyproject_and_pyg_workflow():
    readme = Path("README.md").read_text()

    assert "Plain Pyproject Workflow" in readme
    assert "python -m pip install -e \".[dev]\"" in readme
    assert "python -m pytest" in readme
    assert "PyTorch Geometric" in readme
    assert "python -m pip install -e \".[dev,pyg]\"" in readme


def test_readme_documents_synthetic_traffic_fixture_as_test_data():
    readme = Path("README.md").read_text()

    assert "Synthetic Traffic Fixture" in readme
    assert "test data" in readme
    assert "not a scientific benchmark" in readme


def test_readme_documents_pems_bay_replication_artifacts():
    readme = Path("README.md").read_text()

    assert "PEMS-BAY Replication Dataset" in readme
    assert "configs/pems_bay.yaml" in readme
    assert "data/raw/pems-bay/pems-bay.h5" in readme
    assert "data/raw/pems-bay/adj_mx_bay.pkl" in readme


def test_merit_probe_training_standard_is_recorded_as_an_adr():
    decision = Path(
        "docs/adr/0004-require-optimizer-trained-graphnamlss-for-merit-probe.md"
    ).read_text()

    assert "optimizer-trained GraphNAMLSS" in decision
    assert "deterministic Tracer-Bullet Fit" in decision
    assert "optimizer_trained_graph_namlss_training" in decision
    assert "Further-Investment Gate" in decision
