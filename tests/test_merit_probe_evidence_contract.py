from pathlib import Path

import pytest

from graphnamlss.evidence import (
    EvidenceContractError,
    load_merit_probe_evidence_contract,
)


CONTRACT_PATH = Path("configs/metr_la_merit_probe.yaml")


def _load_contract():
    return load_merit_probe_evidence_contract(CONTRACT_PATH)


def _write_mutated_contract(tmp_path, original: str, replacement: str) -> Path:
    canonical = CONTRACT_PATH.read_text()
    assert original in canonical
    invalid_path = tmp_path / "invalid_merit_probe.yaml"
    invalid_path.write_text(canonical.replace(original, replacement))
    return invalid_path


def test_metr_la_merit_probe_exposes_the_first_real_data_primary_comparison():
    contract = _load_contract()

    assert contract.stage == "first_real_data_evidence_gate"
    assert contract.dataset == "metr_la"
    assert (
        contract.primary_comparison.reference_model == "deterministic_namlss"
    )
    assert (
        contract.primary_comparison.candidate_model
        == "tracer_bullet_graph_namlss"
    )


def test_metr_la_merit_probe_labels_secondary_context_by_evidence_role():
    contract = _load_contract()

    roles = {entry.model: entry.role for entry in contract.secondary_context}

    assert roles == {
        "distributional_gnn_baseline": "distributional_secondary_context",
        "point_gnn_reference": "point_accuracy_guardrail",
    }


def test_further_investment_gate_prefers_distributional_signals_with_guardrails():
    contract = _load_contract()

    gate = contract.further_investment_gate

    assert gate.preferred_distributional_signals == ("nll", "crps")
    assert gate.guardrail_metrics == ("coverage", "pit", "mae", "rmse")


def test_failed_merit_probe_routes_to_diagnosis_before_expansion():
    contract = _load_contract()

    failure_rule = contract.failure_rule

    assert failure_rule.next_step == "diagnose"
    assert failure_rule.diagnostic_areas == (
        "data",
        "baseline",
        "fitting_method",
        "graph_stream",
        "scale_predictor",
        "horizon_choice",
    )
    assert failure_rule.blocked_expansions == (
        "pems_bay_replication",
        "primary_experiment_matrix",
        "bayesian_graph_streams",
        "medical_applications",
    )


def test_pems_bay_is_a_replication_gate_after_a_promising_merit_probe():
    contract = _load_contract()

    replication_gate = contract.replication_gate

    assert replication_gate.dataset == "pems_bay"
    assert replication_gate.after == "promising_metr_la_merit_probe"
    assert replication_gate.purpose == "test_generalization"


def test_loader_rejects_a_point_gnn_without_its_guardrail_role(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path,
        "role: point_accuracy_guardrail",
        "role: distributional_secondary_context",
    )

    with pytest.raises(EvidenceContractError, match="point_accuracy_guardrail"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_gate_without_the_required_metric_sets(tmp_path):
    invalid_path = _write_mutated_contract(tmp_path, "    - crps\n", "")

    with pytest.raises(EvidenceContractError, match="NLL and CRPS"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_pems_bay_before_a_promising_metr_la_result(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path,
        "after: promising_metr_la_merit_probe",
        "after: concept_demonstrator",
    )

    with pytest.raises(EvidenceContractError, match="PEMS-BAY"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_drift_from_the_metr_la_primary_comparison(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path,
        "candidate_model: tracer_bullet_graph_namlss",
        "candidate_model: distributional_gnn_baseline",
    )

    with pytest.raises(EvidenceContractError, match="primary comparison"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_expansion_on_the_failure_path(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path, "    - pems_bay_replication\n", ""
    )

    with pytest.raises(EvidenceContractError, match="Failure Rule"):
        load_merit_probe_evidence_contract(invalid_path)
