import json
import re
import subprocess
import sys
from statistics import fmean, stdev

import pytest

from graphnamlss.testing.scope_guards import assert_concept_demo_scope


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


def test_demo_report_shows_this_runs_spillover_contrasts_and_discloses_the_weak_mu_path(
    tmp_path,
):
    output_directory = tmp_path / "concept-demo"
    result = subprocess.run(
        [sys.executable, "-m", "graphnamlss", "demo", "--output", str(output_directory)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    graph_run = output_directory / "runs" / "tracer-bullet-graph-namlss"
    predictions = _downstream_rows(graph_run / "predictions.json")
    contributions = _downstream_rows(graph_run / "graph_contributions.json")
    spillover = {
        time_index
        for window_start in (15, 35, 55, 75, 95)
        for time_index in (window_start, window_start + 1)
    }
    routine = {time_index - 10 for time_index in spillover}

    def contrast(rows, value, statistic=fmean):
        def observed(time_indices):
            values = [value(row) for row in rows if row["time_index"] in time_indices]
            return [observed for observed in values if observed is not None]

        return statistic(observed(spillover)) - statistic(observed(routine))

    expected = {
        "target_location": contrast(predictions, lambda row: row["target"]),
        "target_variability": contrast(predictions, lambda row: row["target"], stdev),
        "gated_mu_contribution": contrast(contributions, lambda row: row["gated_mu"]),
        "gated_sigma_contribution": contrast(
            contributions, lambda row: row["gated_sigma"]
        ),
    }
    expected_share = expected["gated_mu_contribution"] / contrast(
        predictions, lambda row: row["standardized"]["target"]
    )

    summary = json.loads(
        (output_directory / "report" / "report_summary.json").read_text()
    )
    contrasts = summary["spillover_contrasts"]
    for name, difference in expected.items():
        assert contrasts["checks"][name]["difference"] == pytest.approx(
            difference, abs=1e-6
        )
    assert contrasts["gated_mu_share_of_slowdown"] == pytest.approx(
        expected_share, abs=1e-6
    )

    report = (
        output_directory / "report" / "tracer_bullet_evaluation_report.md"
    ).read_text()
    assert_concept_demo_scope(report)
    positions = [
        report.find(heading)
        for heading in (
            "## Graph Contributions",
            "### Spillover Contrasts",
            "## Graph Gate Diagnostics",
        )
    ]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    section = report[positions[1] : positions[2]]
    for value in (*expected.values(), expected_share):
        assert f"{value:.4g}" in section

    disclosure = section.strip().split("\n\n")[-1]
    for wording in (
        "`mu` graph path is weak and seed-dependent",
        "Tracer-Bullet Fit",
        "Localized Congestion Spillover Demo Scenario",
        "Scale Absorption",
        "not pure Heteroskedastic Spillover Uncertainty",
        "Graph Gate Diagnostic",
        "`graphnamlss demo-robustness`",
    ):
        assert wording in " ".join(disclosure.split())
    assert re.search(r"\d", disclosure) is None


def _downstream_rows(artifact_path):
    artifact = json.loads(artifact_path.read_text())
    return [
        row
        for split_rows in artifact["splits"].values()
        for row in split_rows
        if row["sensor_id"] == "downstream"
    ]
