from pathlib import Path


def test_readme_documents_plain_pyproject_and_pyg_workflow():
    readme = Path("README.md").read_text()

    assert "Plain Pyproject Workflow" in readme
    assert "python -m pip install -e \".[dev]\"" in readme
    assert "python -m pytest" in readme
    assert "PyTorch Geometric" in readme
    assert "python -m pip install -e \".[pyg]\"" in readme


def test_readme_documents_synthetic_traffic_fixture_as_test_data():
    readme = Path("README.md").read_text()

    assert "Synthetic Traffic Fixture" in readme
    assert "test data" in readme
    assert "not a scientific benchmark" in readme
