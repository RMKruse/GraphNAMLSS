from pathlib import Path

from graphnamlss.testing.scope_guards import assert_concept_demo_scope


def test_readme_documents_guided_demo_command_and_report_location():
    readme = Path("README.md").read_text()

    assert "Guided Demo Path" in readme
    assert "graphnamlss demo --output outputs/concept-demo" in readme
    assert (
        "outputs/concept-demo/report/tracer_bullet_evaluation_report.md" in readme
    )


def test_readme_guides_colleagues_through_demo_report_and_artifacts():
    readme = Path("README.md").read_text()

    reading_order = [
        "Model Decomposition",
        "Local Shape Contributions",
        "Graph Contributions",
        "Graph Gate Diagnostics",
        "Metrics",
        "Concept Demo Evidence Boundary",
    ]
    positions = [readme.find(section) for section in reading_order]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)

    for artifact in (
        "local_shape_mu.svg",
        "local_shape_sigma.svg",
        "graph_contribution_mu.svg",
        "graph_contribution_sigma.svg",
        "graph_contributions.json",
        "graph_gates.json",
        "metrics.json",
    ):
        assert artifact in readme


def test_readme_keeps_synthetic_demo_evidence_within_its_boundary():
    readme = " ".join(Path("README.md").read_text().split())

    assert "Synthetic demo metrics are sanity checks" in readme
    assert "not evidence that GraphNAMLSS outperforms baselines" in readme
    assert "lightweight deterministic Tracer-Bullet Fit" in readme
    assert "not the METR-LA Merit Probe" in readme
    assert "does not satisfy the Merit Probe Training Standard" in readme


def test_readme_keeps_unimplemented_tracks_outside_the_concept_demonstrator():
    readme = " ".join(Path("README.md").read_text().split())

    assert_concept_demo_scope(readme)

    assert (
        "does not implement Bayesian graph streams, structural uncertainty "
        "experiments, or medical application results" in readme
    )
    assert (
        "GAMLSS and BAMLSS remain future Statistical Baseline integrations through "
        "External Code Adapters" in readme
    )
    assert "not completed Concept Demo Bundle requirements" in readme


def test_readme_guided_demo_path_discloses_the_weak_mu_path_and_scale_absorption():
    guided_demo_path = _section(Path("README.md").read_text(), "## Guided Demo Path")

    assert "`mu` graph path is weak and seed-dependent" in guided_demo_path
    assert "Tracer-Bullet Fit" in guided_demo_path
    assert "Scale Absorption" in guided_demo_path
    assert "`graphnamlss demo-robustness`" in guided_demo_path


def test_readme_robustness_probe_reports_the_gated_mu_share_without_enforcing_it():
    probe = _section(Path("README.md").read_text(), "## Concept Demo Robustness Probe")

    assert "`gated_mu_share_of_slowdown`" in probe
    assert "reported, not enforced" in probe


def test_review_guide_asks_why_the_sigma_spillover_contrast_is_not_pure_heteroskedasticity():
    guide = " ".join(Path("docs/concept-demonstrator-review.md").read_text().split())

    assert "Why is the `sigma` spillover contrast not pure heteroskedasticity?" in guide
    assert "can explain all five points" in guide
    assert "Scale Absorption teach-back: pending" in guide
    assert_concept_demo_scope(guide)


def _section(markdown, heading):
    start = markdown.index(heading)
    end = markdown.find("\n## ", start + len(heading))
    return " ".join(markdown[start : end if end >= 0 else None].split())


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
