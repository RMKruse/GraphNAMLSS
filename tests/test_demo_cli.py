import json
import subprocess
import sys


def test_demo_command_produces_complete_concept_demo_bundle(tmp_path):
    output_directory = tmp_path / "concept-demo"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "demo",
            "--output",
            str(output_directory),
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    assert emitted["bundle_directory"] == str(output_directory)
    assert emitted["models"] == [
        "deterministic_namlss",
        "tracer_bullet_graph_namlss",
        "distributional_gnn_baseline",
        "point_gnn_reference",
    ]

    run_directories = {
        "deterministic_namlss": output_directory / "runs" / "deterministic-namlss",
        "tracer_bullet_graph_namlss": (
            output_directory / "runs" / "tracer-bullet-graph-namlss"
        ),
        "distributional_gnn_baseline": (
            output_directory / "runs" / "distributional-gnn-baseline"
        ),
        "point_gnn_reference": output_directory / "runs" / "point-gnn-reference",
    }
    assert emitted["run_directories"] == {
        model: str(path) for model, path in run_directories.items()
    }

    for run_directory in run_directories.values():
        assert (run_directory / "resolved_config.json").exists()
        assert (run_directory / "metrics.json").exists()
        assert (run_directory / "training_history.json").exists()
        assert (run_directory / "train_summary.json").exists()
        resolved = json.loads((run_directory / "resolved_config.json").read_text())
        assert resolved["dataset"]["name"] == "synthetic-concept-demo"
        assert resolved["dataset"]["artifacts"] == {}

    graph_run = run_directories["tracer_bullet_graph_namlss"]
    assert (graph_run / "graph_contributions.json").exists()
    assert (graph_run / "graph_gates.json").exists()

    report_directory = output_directory / "report"
    report_path = report_directory / "tracer_bullet_evaluation_report.md"
    assert emitted["report_path"] == str(report_path)
    assert report_path.exists()
    assert (report_directory / "report_summary.json").exists()
    for asset_name in (
        "local_shape_mu.svg",
        "local_shape_sigma.svg",
        "graph_contribution_mu.svg",
        "graph_contribution_sigma.svg",
    ):
        assert (report_directory / "assets" / asset_name).exists()

    report = report_path.read_text()
    assert "## Demo Data-Generating Story" in report
    assert "## Concept Demo Evidence Boundary" in report
    assert "## Tracer-Bullet Fit Disclosure" in report
