from __future__ import annotations

from collections.abc import Mapping, Sequence
import html
import json
from pathlib import Path
from typing import Any


MODEL_ORDER = (
    "deterministic_namlss",
    "tracer_bullet_graph_namlss",
    "distributional_gnn_baseline",
    "point_gnn_reference",
)
SPLIT_ORDER = ("train", "validation", "test")


def generate_tracer_bullet_evaluation_report(
    *,
    deterministic_run: Path,
    graph_run: Path,
    distributional_gnn_run: Path,
    point_gnn_run: Path,
    output_directory: Path,
) -> dict[str, Any]:
    runs = {
        "deterministic_namlss": _load_run(deterministic_run),
        "tracer_bullet_graph_namlss": _load_run(graph_run),
        "distributional_gnn_baseline": _load_run(distributional_gnn_run),
        "point_gnn_reference": _load_run(point_gnn_run),
    }
    graph_artifacts = _load_graph_artifacts(
        graph_run, runs["tracer_bullet_graph_namlss"]
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    assets_directory = output_directory / "assets"
    assets_directory.mkdir(parents=True, exist_ok=True)

    metric_tables = _metric_tables(runs)
    pit_summary = _pit_summary(runs)
    local_shape_plots = _local_shape_plots(runs)
    graph_contribution_maps = _graph_contribution_maps(graph_artifacts["contributions"])
    graph_contribution_gates = graph_artifacts["gates"]

    _write_local_shape_assets(assets_directory, local_shape_plots)
    _write_graph_contribution_assets(assets_directory, graph_contribution_maps)

    summary = {
        "models": list(MODEL_ORDER),
        "metric_tables": metric_tables,
        "pit_summary": pit_summary,
        "local_shape_plots": local_shape_plots,
        "graph_contribution_maps": graph_contribution_maps,
        "graph_contribution_gates": graph_contribution_gates,
        "source_runs": {
            model_key: str(runs[model_key]["path"]) for model_key in MODEL_ORDER
        },
    }
    summary_path = output_directory / "report_summary.json"
    _write_json(summary_path, summary)

    report_path = output_directory / "tracer_bullet_evaluation_report.md"
    report_path.write_text(_render_markdown(summary, runs), encoding="utf-8")

    return {
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "models": list(MODEL_ORDER),
    }


def _load_run(path: Path) -> dict[str, Any]:
    summary_path = path / "train_summary.json"
    metrics_path = path / "metrics.json"
    if not summary_path.exists():
        raise ValueError(f"Missing run artifact: {summary_path}")
    if not metrics_path.exists():
        raise ValueError(f"Missing run artifact: {metrics_path}")
    summary = _read_json(summary_path)
    metrics = _read_json(metrics_path)
    return {"path": path, "summary": summary, "metrics": metrics}


def _load_graph_artifacts(
    graph_run: Path, graph_run_payload: Mapping[str, Any]
) -> dict[str, Any]:
    graph_summary = graph_run_payload["summary"].get("graph", {})
    gate_artifact = graph_summary.get("gate_artifact", "graph_gates.json")
    contribution_artifact = graph_summary.get(
        "contribution_artifact", "graph_contributions.json"
    )
    gates_path = graph_run / gate_artifact
    contributions_path = graph_run / contribution_artifact
    if not gates_path.exists():
        raise ValueError(f"Missing GraphNAMLSS gate artifact: {gates_path}")
    if not contributions_path.exists():
        raise ValueError(
            f"Missing GraphNAMLSS contribution artifact: {contributions_path}"
        )
    return {
        "gates": _read_json(gates_path),
        "contributions": _read_json(contributions_path),
    }


def _metric_tables(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    distributional: dict[str, list[dict[str, Any]]] = {
        split_name: [] for split_name in SPLIT_ORDER
    }
    point: dict[str, list[dict[str, Any]]] = {
        split_name: [] for split_name in SPLIT_ORDER
    }

    for model_key in MODEL_ORDER:
        run = runs[model_key]
        for split_name in SPLIT_ORDER:
            split_metrics = run["metrics"][split_name]
            row = {
                "model": model_key,
                "run_name": run["summary"]["run_name"],
                "observed_count": split_metrics["observed_count"],
                "mae": split_metrics["mae"],
                "rmse": split_metrics["rmse"],
            }
            if "nll" in split_metrics:
                row.update(
                    {
                        "nll": split_metrics["nll"],
                        "crps": split_metrics["crps"],
                        "coverage": split_metrics["interval_coverage"],
                    }
                )
                distributional[split_name].append(row)
            else:
                point[split_name].append(row)

    return {"distributional": distributional, "point": point}


def _pit_summary(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for model_key, run in runs.items():
        split_summaries: dict[str, dict[str, float | int]] = {}
        for split_name in SPLIT_ORDER:
            pit_values = run["metrics"][split_name].get("pit_values")
            if pit_values is None:
                continue
            values = [float(value) for value in pit_values]
            split_summaries[split_name] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "below_0_1": sum(value < 0.1 for value in values),
                "above_0_9": sum(value > 0.9 for value in values),
            }
        if split_summaries:
            summaries[model_key] = split_summaries
    return summaries


def _local_shape_plots(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    plots = {"mu": [], "sigma": []}
    for model_key in ("deterministic_namlss", "tracer_bullet_graph_namlss"):
        run = runs[model_key]
        parameters = run["summary"].get("parameters", {})
        for feature_name, weight in parameters.get("weights_mu", {}).items():
            plots["mu"].append(
                {
                    "model": model_key,
                    "run_name": run["summary"]["run_name"],
                    "feature": feature_name,
                    "weight": float(weight),
                    "values": [
                        {"x": x_value, "contribution": float(weight) * x_value}
                        for x_value in (-2.0, -1.0, 0.0, 1.0, 2.0)
                    ],
                }
            )
        plots["sigma"].append(
            {
                "model": model_key,
                "run_name": run["summary"]["run_name"],
                "feature": "constant_scale",
                "weight": 0.0,
                "sigma": parameters.get("sigma"),
                "values": [
                    {"x": x_value, "contribution": 0.0}
                    for x_value in (-2.0, -1.0, 0.0, 1.0, 2.0)
                ],
            }
        )
    return plots


def _graph_contribution_maps(contributions: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    split_rows = contributions.get("splits", {})
    if "test" in split_rows:
        rows = split_rows["test"]
    else:
        for split_name in SPLIT_ORDER:
            rows.extend(split_rows.get(split_name, []))

    maps: dict[str, dict[str, float]] = {}
    for parameter_name in contributions.get("parameter_names", ("mu", "sigma")):
        field = f"gated_{parameter_name}"
        by_sensor: dict[str, list[float]] = {}
        for row in rows:
            sensor_id = str(row["sensor_id"])
            by_sensor.setdefault(sensor_id, []).append(float(row[field]))
        maps[parameter_name] = {
            sensor_id: sum(values) / len(values)
            for sensor_id, values in sorted(by_sensor.items())
        }
    return maps


def _write_local_shape_assets(
    assets_directory: Path, local_shape_plots: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    for parameter_name, rows in local_shape_plots.items():
        _write_line_svg(
            assets_directory / f"local_shape_{parameter_name}.svg",
            title=f"Local Shape Plot: {parameter_name}",
            series={
                f"{row['run_name']} {row['feature']}": row["values"] for row in rows
            },
        )


def _write_graph_contribution_assets(
    assets_directory: Path, graph_contribution_maps: Mapping[str, Mapping[str, float]]
) -> None:
    for parameter_name, values in graph_contribution_maps.items():
        _write_bar_svg(
            assets_directory / f"graph_contribution_{parameter_name}.svg",
            title=f"Average Graph Contribution: {parameter_name}",
            values=values,
        )


def _write_bar_svg(path: Path, *, title: str, values: Mapping[str, float]) -> None:
    width = 760
    row_height = 34
    margin_left = 250
    margin_right = 40
    margin_top = 54
    height = margin_top + row_height * max(1, len(values)) + 28
    plot_width = width - margin_left - margin_right
    max_abs = max((abs(value) for value in values.values()), default=1.0) or 1.0
    zero_x = margin_left + plot_width / 2
    scale = (plot_width / 2) / max_abs

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="30" font-family="Arial" font-size="18" '
        f'font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{zero_x:.1f}" y1="42" x2="{zero_x:.1f}" '
        f'y2="{height - 18}" stroke="#777" stroke-width="1"/>',
    ]
    for index, (label, value) in enumerate(values.items()):
        y = margin_top + index * row_height
        bar_width = abs(value) * scale
        x = zero_x if value >= 0 else zero_x - bar_width
        color = "#2f6f8f" if value >= 0 else "#b85c38"
        elements.extend(
            [
                f'<text x="24" y="{y + 18}" font-family="Arial" '
                f'font-size="12">{html.escape(label)}</text>',
                f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" '
                f'height="20" fill="{color}"/>',
                f'<text x="{margin_left + plot_width + 8}" y="{y + 15}" '
                f'font-family="Arial" font-size="12">{value:.4g}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_line_svg(
    path: Path, *, title: str, series: Mapping[str, Sequence[Mapping[str, float]]]
) -> None:
    width = 760
    height = 360
    margin_left = 70
    margin_right = 210
    margin_top = 58
    margin_bottom = 48
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_points = [
        (float(point["x"]), float(point["contribution"]))
        for points in series.values()
        for point in points
    ]
    x_values = [point[0] for point in all_points] or [-1.0, 1.0]
    y_values = [point[1] for point in all_points] or [-1.0, 1.0]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def x_coord(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def y_coord(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_height

    colors = ("#2f6f8f", "#b85c38", "#5f7f3a", "#7b5aa6", "#8a6f2a", "#3c7480")
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="30" font-family="Arial" font-size="18" '
        f'font-weight="700">{html.escape(title)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" '
        f'height="{plot_height}" fill="none" stroke="#555" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{y_coord(0.0):.1f}" '
        f'x2="{margin_left + plot_width}" y2="{y_coord(0.0):.1f}" '
        'stroke="#999" stroke-width="1"/>',
        f'<text x="{margin_left}" y="{height - 16}" font-family="Arial" '
        'font-size="12">standardized feature value</text>',
        '<text x="18" y="52" font-family="Arial" font-size="12">contribution</text>',
    ]
    for index, (label, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        coordinates = " ".join(
            f'{x_coord(float(point["x"])):.1f},{y_coord(float(point["contribution"])):.1f}'
            for point in points
        )
        legend_y = margin_top + index * 22
        elements.extend(
            [
                f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                f'points="{coordinates}"/>',
                f'<line x1="{width - margin_right + 18}" y1="{legend_y}" '
                f'x2="{width - margin_right + 42}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="2"/>',
                f'<text x="{width - margin_right + 48}" y="{legend_y + 4}" '
                f'font-family="Arial" font-size="11">{html.escape(label)}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _render_markdown(
    summary: Mapping[str, Any], runs: Mapping[str, Mapping[str, Any]]
) -> str:
    lines = [
        "# Tracer-Bullet Evaluation Report",
        "",
        "This report consumes saved run artifacts only; it does not retrain models.",
        "",
        "## GraphNAMLSS Model Decomposition",
        "",
        "GraphNAMLSS combines local additive predictors with gated, centered graph "
        "contributions for each distributional parameter.",
        "",
        "`parameter predictor = intercept + local shape contributions "
        "+ gate × centered graph contribution`",
        "",
        "The local terms preserve NAMLSS-style feature interpretation; the "
        "neighbor-focused graph stream adds parameter-specific graph context for "
        "location (`mu`) and scale (`sigma`).",
        "",
        "### Source Runs",
        "",
        "| Role | Run | Family | Variant |",
        "|---|---|---|---|",
    ]
    for model_key in MODEL_ORDER:
        run_summary = runs[model_key]["summary"]
        model = run_summary["model"]
        lines.append(
            f"| {_label(model_key)} | {run_summary['run_name']} | "
            f"{model['family']} | {model['variant']} |"
        )

    lines.extend(
        [
            "",
            "## Demo Data-Generating Story",
            "",
            "The Localized Congestion Spillover Demo Scenario separates ordinary "
            "traffic behavior from graph-neighbor spillover; local lags explain routine "
            "traffic behavior, while upstream congestion propagates through the graph "
            "to a downstream sensor. Neighbor lag features expose the spillover as an "
            "expected slowdown through the location parameter and increased uncertainty "
            "through the scale parameter.",
            "",
            "## Local Shape Contributions",
            "",
            "These plots expose feature-wise local contributions before graph context "
            "is added, keeping routine local behavior separately inspectable for `mu` "
            "and `sigma`.",
            "",
            "![Local shape mu](assets/local_shape_mu.svg)",
            "",
            "![Local shape sigma](assets/local_shape_sigma.svg)",
            "",
            "## Graph Contributions",
            "",
            "The maps summarize saved, gated graph contributions by sensor and "
            "distributional parameter. They complement rather than replace the local "
            "shape contributions.",
            "",
            "### Graph Contribution Maps",
            "",
            "![Graph contribution mu](assets/graph_contribution_mu.svg)",
            "",
            "![Graph contribution sigma](assets/graph_contribution_sigma.svg)",
            "",
            "## Graph Gate Diagnostics",
            "",
            "Each gate is a parameter-specific graph reliance diagnostic for this fitted "
            "demo model. Gate magnitude is not causal proof and is not standalone "
            "evidence that graph information improves a distributional parameter.",
            "",
            "### Graph Contribution Gates",
            "",
            "| Parameter | Gate |",
            "|---|---:|",
        ]
    )
    for gate in summary["graph_contribution_gates"]:
        lines.append(f"| {gate['parameter']} | {_fmt(gate['gate'])} |")

    lines.extend(["", "## Metrics", "", "### Distributional Metrics", ""])
    for split_name in SPLIT_ORDER:
        lines.extend(
            [
                f"#### {split_name.title()}",
                "",
                "| Model | Observed | NLL | CRPS | Coverage | MAE | RMSE |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["metric_tables"]["distributional"][split_name]:
            lines.append(
                f"| {_label(row['model'])} | {row['observed_count']} | "
                f"{_fmt(row['nll'])} | {_fmt(row['crps'])} | "
                f"{_fmt(row['coverage'])} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "### Point Accuracy Reference",
            "",
            "| Split | Model | Observed | MAE | RMSE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for split_name in SPLIT_ORDER:
        for row in summary["metric_tables"]["point"][split_name]:
            lines.append(
                f"| {split_name} | {_label(row['model'])} | {row['observed_count']} | "
                f"{_fmt(row['mae'])} | {_fmt(row['rmse'])} |"
            )

    lines.extend(
        [
            "",
            "### PIT Summaries",
            "",
            "| Model | Split | Count | PIT Mean | Below 0.1 | Above 0.9 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model_key, split_summaries in summary["pit_summary"].items():
        for split_name in SPLIT_ORDER:
            if split_name not in split_summaries:
                continue
            row = split_summaries[split_name]
            lines.append(
                f"| {_label(model_key)} | {split_name} | {row['count']} | "
                f"{_fmt(row['mean'])} | {row['below_0_1']} | {row['above_0_9']} |"
            )

    lines.extend(
        [
            "",
            "## Concept Demo Evidence Boundary",
            "",
            "The synthetic Concept Demo Bundle metrics are sanity checks for a runnable "
            "and inspectable experiment contract. They are not evidence that GraphNAMLSS "
            "outperforms baselines or improves calibration, and they cannot support the "
            "Distributional Calibration Claim.",
            "",
            "## Current Concept Demonstrator Scope",
            "",
            "The Concept Demonstrator does not implement Bayesian graph streams, "
            "structural uncertainty experiments, or medical application results. GAMLSS "
            "and BAMLSS remain future Statistical Baseline integrations through External "
            "Code Adapters; they are not completed Concept Demo Bundle requirements.",
            "",
            "## Tracer-Bullet Fit Disclosure",
            "",
            "The Concept Demonstrator uses a lightweight deterministic Tracer-Bullet Fit "
            "to make the model decomposition and saved artifacts inspectable. This fit "
            "does not satisfy the Merit Probe Training Standard; empirical claims require "
            "an optimizer-trained GraphNAMLSS candidate evaluated on the METR-LA Merit "
            "Probe.",
            "",
        ]
    )
    return "\n".join(lines)


def _label(model_key: str) -> str:
    return {
        "deterministic_namlss": "Deterministic NAMLSS",
        "tracer_bullet_graph_namlss": "Tracer-Bullet GraphNAMLSS",
        "distributional_gnn_baseline": "Distributional GNN Baseline",
        "point_gnn_reference": "Point GNN Reference",
    }[model_key]


def _fmt(value: float) -> str:
    return f"{float(value):.4g}"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
