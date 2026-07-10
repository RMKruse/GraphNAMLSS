import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/01_graphnamlss_concept_walkthrough.ipynb")


def _source(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def test_concept_notebook_is_reader_facing_and_package_backed():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    markdown = "\n".join(
        _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for section in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps"):
        assert section in markdown
    assert "run_concept_demo_bundle" in code
    assert "predictions.json" in code
    assert "graphnamlss.training" not in code
    assert "Concept Demonstrator" in markdown
    assert "METR-LA Merit Probe" in markdown


def test_concept_notebook_is_executed_without_errors():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]

    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert all(
        output.get("output_type") != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )
    assert sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell["outputs"]
    ) >= 3
