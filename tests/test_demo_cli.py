import json
import subprocess
import sys
from statistics import fmean


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
        assert (run_directory / "predictions.json").exists()
        resolved = json.loads((run_directory / "resolved_config.json").read_text())
        assert resolved["dataset"]["name"] == "synthetic-concept-demo"
        assert resolved["dataset"]["artifacts"] == {}

        predictions = json.loads((run_directory / "predictions.json").read_text())
        assert predictions["unit"] == "original_traffic"
        assert set(predictions["splits"]) == {"train", "validation", "test"}
        assert all(predictions["splits"].values())

    graph_run = run_directories["tracer_bullet_graph_namlss"]
    assert (graph_run / "graph_contributions.json").exists()
    assert (graph_run / "graph_gates.json").exists()
    graph_summary = json.loads((graph_run / "train_summary.json").read_text())
    graph_gates = json.loads((graph_run / "graph_gates.json").read_text())
    graph_contributions = json.loads(
        (graph_run / "graph_contributions.json").read_text()
    )
    assert graph_summary["graph"]["gate_estimation"] == (
        "deterministic_unit_l2_head_factorization"
    )
    assert [row["parameter"] for row in graph_gates] == ["mu", "sigma"]
    assert all(row["gate"] > 0.0 for row in graph_gates)

    contribution_rows = [
        row
        for split_rows in graph_contributions["splits"].values()
        for row in split_rows
        if row["sensor_id"] == "downstream"
    ]
    spillover_indices = {
        time_index
        for window_start in (15, 35, 55, 75, 95)
        for time_index in (window_start, window_start + 1)
    }
    routine_indices = {time_index - 10 for time_index in spillover_indices}
    spillover_scale_contribution = fmean(
        row["gated_sigma"]
        for row in contribution_rows
        if row["time_index"] in spillover_indices
    )
    routine_scale_contribution = fmean(
        row["gated_sigma"]
        for row in contribution_rows
        if row["time_index"] in routine_indices
    )
    assert spillover_scale_contribution > routine_scale_contribution + 0.1

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
    assert "normalized to unit L2 norm" in report
    assert "scale graph head is fitted against Gaussian training NLL" in report
