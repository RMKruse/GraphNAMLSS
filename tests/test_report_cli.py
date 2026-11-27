import json
import subprocess
import sys

from graphnamlss.testing.scope_guards import assert_concept_demo_scope


def test_report_command_generates_tracer_bullet_evaluation_bundle(tmp_path):
    deterministic_run = _write_run(
        tmp_path / "deterministic",
        run_name="deterministic-metr-la",
        family="NAMLSS",
        variant="deterministic_local",
        distribution="gaussian_location_scale",
        parameters={
            "intercept_mu": 0.1,
            "weights_mu": {"current_speed": 0.7, "lag_5m": -0.2},
            "sigma": 0.4,
        },
    )
    graph_run = _write_run(
        tmp_path / "graph",
        run_name="tracer-bullet-metr-la",
        family="GraphNAMLSS",
        variant="tracer_bullet",
        distribution="gaussian_location_scale",
        parameters={
            "intercept_mu": 0.05,
            "weights_mu": {"current_speed": 0.6, "lag_5m": -0.1},
            "sigma": 0.35,
            "graph_weights_mu": {"neighbor_current_speed": 0.3},
            "graph_weights_sigma": {"neighbor_current_speed": 0.0},
        },
        graph={
            "representation": "shared",
            "encoder": "graphsage",
            "injection": "additive_graph_contribution",
            "self_feature_access": "neighbor_focused",
            "node_count": 3,
            "edge_count": 4,
            "gate_artifact": "graph_gates.json",
            "contribution_artifact": "graph_contributions.json",
        },
    )
    _write_graph_artifacts(graph_run)
    distributional_run = _write_run(
        tmp_path / "distributional_gnn",
        run_name="distributional-gnn-metr-la",
        family="DistributionalGNN",
        variant="baseline",
        distribution="gaussian_location_scale",
        parameters={
            "intercept_mu": 0.2,
            "graph_weights_mu": {"neighbor_current_speed": 0.4},
            "intercept_sigma": -0.9,
            "graph_weights_sigma": {"neighbor_current_speed": 0.05},
            "sigma": 0.45,
        },
        graph={
            "representation": "shared",
            "encoder": "graphsage",
            "injection": "direct_distributional_parameters",
            "self_feature_access": "neighbor_focused",
            "node_count": 3,
            "edge_count": 4,
        },
    )
    point_run = _write_run(
        tmp_path / "point_gnn",
        run_name="point-gnn-metr-la",
        family="PointGNN",
        variant="reference",
        distribution="conditional_mean",
        parameters={
            "intercept_mu": 0.3,
            "graph_weights_mu": {"neighbor_current_speed": 0.5},
            "sigma": None,
        },
        graph={
            "representation": "shared",
            "encoder": "graphsage",
            "injection": "direct_mean",
            "self_feature_access": "neighbor_focused",
            "node_count": 3,
            "edge_count": 4,
        },
        point=True,
    )
    output_dir = tmp_path / "report"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "report",
            "--deterministic-run",
            str(deterministic_run),
            "--graph-run",
            str(graph_run),
            "--distributional-gnn-run",
            str(distributional_run),
            "--point-gnn-run",
            str(point_run),
            "--output",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    assert emitted["report_path"] == str(
        output_dir / "tracer_bullet_evaluation_report.md"
    )
    assert emitted["models"] == [
        "deterministic_namlss",
        "tracer_bullet_graph_namlss",
        "distributional_gnn_baseline",
        "point_gnn_reference",
    ]

    report = (output_dir / "tracer_bullet_evaluation_report.md").read_text()
    assert_concept_demo_scope(report)
    assert "# Tracer-Bullet Evaluation Report" in report
    ordered_sections = [
        "## GraphNAMLSS Model Decomposition",
        "## Demo Data-Generating Story",
        "## Local Shape Contributions",
        "## Graph Contributions",
        "## Graph Gate Diagnostics",
        "## Metrics",
        "## Concept Demo Evidence Boundary",
        "## Current Concept Demonstrator Scope",
        "## Tracer-Bullet Fit Disclosure",
    ]
    section_positions = [report.find(section) for section in ordered_sections]
    assert all(position >= 0 for position in section_positions)
    assert section_positions == sorted(section_positions)
    assert "local lags explain routine traffic behavior" in report
    assert "upstream congestion propagates through the graph" in report
    assert "makes the downstream target both slower and more variable" in report
    assert "Neighbor lag features expose the spillover" in report
    assert (
        "How a fitted model splits it between its `mu` and `sigma` graph paths is "
        "a Graph Gate Diagnostic, not part of the scenario." in report
    )
    assert "### Spillover Contrasts" not in report
    assert "parameter-specific graph reliance diagnostic" in report
    assert "not causal proof" in report
    assert "not standalone evidence" in report
    assert "metrics are sanity checks" in report
    assert "not evidence that GraphNAMLSS outperforms baselines" in report
    assert (
        "does not implement Bayesian graph streams, structural uncertainty "
        "experiments, or medical application results" in report
    )
    assert (
        "GAMLSS and BAMLSS remain future Statistical Baseline integrations through "
        "External Code Adapters" in report
    )
    assert "not completed Concept Demo Bundle requirements" in report
    assert "lightweight deterministic Tracer-Bullet Fit" in report
    assert "does not satisfy the Merit Probe Training Standard" in report
    assert "deterministic-metr-la" in report
    assert "NLL" in report
    assert "CRPS" in report
    assert "PIT Mean" in report
    assert "local_shape_mu.svg" in report
    assert "local_shape_sigma.svg" in report
    assert "graph_contribution_mu.svg" in report
    assert "Graph Contribution Gates" in report

    summary = json.loads((output_dir / "report_summary.json").read_text())
    assert summary["metric_tables"]["distributional"]["test"][0]["nll"] == 1.3
    assert summary["metric_tables"]["point"]["test"][0]["rmse"] == 0.7
    assert summary["pit_summary"]["tracer_bullet_graph_namlss"]["test"] == {
        "count": 3,
        "mean": 0.5,
        "below_0_1": 1,
        "above_0_9": 1,
    }
    assert summary["graph_contribution_maps"]["mu"]["sensor_a"] == 0.2
    assert summary["graph_contribution_gates"] == [
        {"parameter": "mu", "gate": 0.8},
        {"parameter": "sigma", "gate": 0.5},
    ]

    assert (output_dir / "assets" / "local_shape_mu.svg").exists()
    assert (output_dir / "assets" / "local_shape_sigma.svg").exists()
    assert (output_dir / "assets" / "graph_contribution_mu.svg").exists()
    assert (output_dir / "assets" / "graph_contribution_sigma.svg").exists()


def test_report_command_requires_saved_graph_interpretability_artifacts(tmp_path):
    deterministic_run = _write_run(
        tmp_path / "deterministic",
        run_name="deterministic-metr-la",
        family="NAMLSS",
        variant="deterministic_local",
        distribution="gaussian_location_scale",
        parameters={
            "intercept_mu": 0.1,
            "weights_mu": {"current_speed": 0.7},
            "sigma": 0.4,
        },
    )
    graph_run = _write_run(
        tmp_path / "graph",
        run_name="tracer-bullet-metr-la",
        family="GraphNAMLSS",
        variant="tracer_bullet",
        distribution="gaussian_location_scale",
        parameters={
            "intercept_mu": 0.05,
            "weights_mu": {"current_speed": 0.6},
            "sigma": 0.35,
        },
        graph={
            "gate_artifact": "graph_gates.json",
            "contribution_artifact": "graph_contributions.json",
        },
    )
    distributional_run = _write_run(
        tmp_path / "distributional_gnn",
        run_name="distributional-gnn-metr-la",
        family="DistributionalGNN",
        variant="baseline",
        distribution="gaussian_location_scale",
        parameters={"intercept_mu": 0.2, "graph_weights_mu": {}, "sigma": 0.45},
    )
    point_run = _write_run(
        tmp_path / "point_gnn",
        run_name="point-gnn-metr-la",
        family="PointGNN",
        variant="reference",
        distribution="conditional_mean",
        parameters={"intercept_mu": 0.3, "graph_weights_mu": {}, "sigma": None},
        point=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "report",
            "--deterministic-run",
            str(deterministic_run),
            "--graph-run",
            str(graph_run),
            "--distributional-gnn-run",
            str(distributional_run),
            "--point-gnn-run",
            str(point_run),
            "--output",
            str(tmp_path / "report"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Missing GraphNAMLSS gate artifact" in result.stderr
    assert not (tmp_path / "report" / "tracer_bullet_evaluation_report.md").exists()


def _write_run(
    path,
    *,
    run_name,
    family,
    variant,
    distribution,
    parameters,
    graph=None,
    point=False,
):
    path.mkdir()
    metrics = _point_metrics() if point else _distributional_metrics()
    summary = {
        "run_name": run_name,
        "model": {
            "family": family,
            "variant": variant,
            "distribution": distribution,
        },
        "metrics": metrics,
        "features": {
            "local": ["current_speed", "lag_5m"],
            "neighbor": ["neighbor_current_speed"],
        },
        "parameters": parameters,
        "splits": {
            "train": {"prediction_count": 4, "observed_target_count": 3},
            "validation": {"prediction_count": 4, "observed_target_count": 3},
            "test": {"prediction_count": 4, "observed_target_count": 3},
        },
        "scaling": {"target": {"fit_on": "train"}},
        "missingness": {"target_policy": "mask_missing_targets"},
        "training_history": [
            {"epoch": 1, "loss": metrics["train"].get("nll", metrics["train"]["rmse"])}
        ],
    }
    if graph is not None:
        summary["graph"] = graph
    (path / "train_summary.json").write_text(json.dumps(summary) + "\n")
    (path / "metrics.json").write_text(json.dumps(metrics) + "\n")
    return path


def _distributional_metrics():
    return {
        "train": _gaussian_split(1.1, 0.2, 0.4, 0.6, (0.2, 0.5, 0.8)),
        "validation": _gaussian_split(1.2, 0.3, 0.5, 0.65, (0.3, 0.5, 0.7)),
        "test": _gaussian_split(1.3, 0.4, 0.6, 0.8, (0.05, 0.5, 0.95)),
    }


def _gaussian_split(nll, crps, mae, rmse, pit_values):
    return {
        "unit": "standardized_traffic",
        "observed_count": 3,
        "nll": nll,
        "crps": crps,
        "interval_level": 0.9,
        "interval_coverage": 2 / 3,
        "pit_values": list(pit_values),
        "mae": mae,
        "rmse": rmse,
    }


def _point_metrics():
    return {
        "train": {
            "unit": "standardized_traffic",
            "observed_count": 3,
            "mae": 0.4,
            "rmse": 0.5,
        },
        "validation": {
            "unit": "standardized_traffic",
            "observed_count": 3,
            "mae": 0.5,
            "rmse": 0.6,
        },
        "test": {
            "unit": "standardized_traffic",
            "observed_count": 3,
            "mae": 0.6,
            "rmse": 0.7,
        },
    }


def _write_graph_artifacts(path):
    (path / "graph_gates.json").write_text(
        json.dumps(
            [
                {"parameter": "mu", "gate": 0.8},
                {"parameter": "sigma", "gate": 0.5},
            ]
        )
        + "\n"
    )
    (path / "graph_contributions.json").write_text(
        json.dumps(
            {
                "parameter_names": ["mu", "sigma"],
                "splits": {
                    "test": [
                        {
                            "split": "test",
                            "sensor_id": "sensor_a",
                            "time_index": 1,
                            "mu": 0.3,
                            "sigma": 0.1,
                            "gated_mu": 0.2,
                            "gated_sigma": 0.05,
                        },
                        {
                            "split": "test",
                            "sensor_id": "sensor_b",
                            "time_index": 1,
                            "mu": -0.1,
                            "sigma": -0.2,
                            "gated_mu": -0.08,
                            "gated_sigma": -0.1,
                        },
                    ]
                },
            }
        )
        + "\n"
    )
