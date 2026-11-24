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


def test_merit_probe_requires_optimizer_trained_graph_namlss_evidence():
    contract = _load_contract()

    assert (
        contract.training_standard.merit_probe_fit
        == "optimizer_trained_graph_namlss"
    )


def test_training_standard_identifies_the_concept_demonstrator_fit():
    contract = _load_contract()

    assert (
        contract.training_standard.concept_demonstrator_fit
        == "deterministic_tracer_bullet_fit"
    )


def test_training_standard_names_the_next_implementation_path():
    contract = _load_contract()

    assert (
        contract.training_standard.next_implementation_path
        == "optimizer_trained_graph_namlss_training"
    )


def test_metr_la_merit_probe_labels_secondary_context_by_evidence_role():
    contract = _load_contract()

    roles = {entry.model: entry.role for entry in contract.secondary_context}

    assert roles == {
        "distributional_gnn_baseline": "distributional_secondary_context",
        "point_gnn_reference": "point_accuracy_guardrail",
    }


def test_further_investment_gate_names_primary_signals_and_guardrails():
    contract = _load_contract()

    gate = contract.further_investment_gate

    assert gate.primary_distributional_signals == ("nll", "crps")
    assert gate.guardrail_metrics == ("coverage", "pit", "mae", "rmse")


def test_gate_rules_are_stated_over_seed_paired_test_differences():
    rules = _load_contract().further_investment_gate.rules

    assert rules.evaluation_partition == "test"
    assert rules.paired_difference == "candidate_minus_reference"


def test_gate_rules_declare_the_gate_seed_set_and_its_completion_floor():
    rules = _load_contract().further_investment_gate.rules

    assert rules.gate_seeds == (1, 2, 3, 4, 5)
    assert rules.minimum_completed_seeds == 4
    assert rules.maximum_dissenting_seeds == 1


def test_qualification_seed_sits_outside_the_gate_under_a_sealed_test_partition():
    rules = _load_contract().further_investment_gate.rules

    assert rules.qualification_run.seed == 0
    assert rules.qualification_run.seed not in rules.gate_seeds
    assert rules.qualification_run.evaluated_partitions == ("train", "validation")


def test_gate_rules_declare_clear_and_meaningful_primary_improvements():
    rules = _load_contract().further_investment_gate.rules

    signals = {signal.metric: signal for signal in rules.primary_signals}

    assert tuple(signals) == ("nll", "crps")
    assert signals["nll"].difference == "absolute"
    assert signals["nll"].clear_improvement == pytest.approx(0.02)
    assert signals["nll"].meaningful_improvement == pytest.approx(0.01)
    assert signals["crps"].difference == "relative"
    assert signals["crps"].clear_improvement == pytest.approx(0.02)
    assert signals["crps"].meaningful_improvement == pytest.approx(0.01)


def test_gate_rules_declare_serious_guardrail_degradation_tolerances():
    rules = _load_contract().further_investment_gate.rules

    guardrails = {guardrail.metric: guardrail for guardrail in rules.guardrails}

    assert tuple(guardrails) == ("coverage", "pit", "mae", "rmse")
    assert guardrails["coverage"].statistic == "absolute_coverage_error"
    assert guardrails["coverage"].interval_level == pytest.approx(0.9)
    assert guardrails["coverage"].difference == "absolute"
    assert guardrails["pit"].statistic == "ks_distance_from_uniform"
    assert guardrails["pit"].difference == "absolute"
    assert guardrails["mae"].difference == "relative"
    assert guardrails["rmse"].difference == "relative"
    for guardrail in guardrails.values():
        assert guardrail.serious_degradation == pytest.approx(0.02)


def test_gate_rules_declare_outcome_precedence():
    rules = _load_contract().further_investment_gate.rules

    assert rules.outcome_precedence == (
        "inconclusive_when_too_few_completed_seeds",
        "fail_when_no_meaningful_primary_improvement",
        "pass_when_gate_met",
        "inconclusive_otherwise",
    )


def test_gate_rules_never_replace_or_retry_a_diverged_seed():
    rules = _load_contract().further_investment_gate.rules

    assert rules.failed_seeds.infrastructure_failure == (
        "rerun_identical_seed_and_recipe"
    )
    assert rules.failed_seeds.training_failure == "record_without_rerun"
    assert rules.failed_seeds.replacement == "never"


def test_gate_rules_point_to_their_decision_record():
    rules = _load_contract().further_investment_gate.rules

    assert Path(rules.decision_record).is_file()


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


def test_loader_rejects_the_demo_fit_as_merit_probe_evidence(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path,
        "merit_probe_fit: optimizer_trained_graph_namlss",
        "merit_probe_fit: deterministic_tracer_bullet_fit",
    )

    with pytest.raises(EvidenceContractError, match="Training Standard"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_gate_without_the_required_metric_sets(tmp_path):
    invalid_path = _write_mutated_contract(tmp_path, "    - crps\n", "")

    with pytest.raises(EvidenceContractError, match="NLL and CRPS"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_gate_without_declared_rules(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path, "  declared_rules:\n", "  undeclared_rules:\n"
    )

    with pytest.raises(EvidenceContractError, match="declared_rules"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_qualification_seed_inside_the_gate_seed_set(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path, "gate_seeds: [1, 2, 3, 4, 5]", "gate_seeds: [0, 1, 2, 3, 4]"
    )

    with pytest.raises(EvidenceContractError, match="Qualification Seed"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_qualification_run_that_reads_the_test_partition(
    tmp_path,
):
    invalid_path = _write_mutated_contract(
        tmp_path,
        "evaluated_partitions: [train, validation]",
        "evaluated_partitions: [train, validation, test]",
    )

    with pytest.raises(EvidenceContractError, match="Sealed Test Partition"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_duplicate_gate_seeds(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path, "gate_seeds: [1, 2, 3, 4, 5]", "gate_seeds: [1, 2, 3, 4, 4]"
    )

    with pytest.raises(EvidenceContractError, match="distinct"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_completion_floor_above_the_gate_seed_count(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path, "minimum_completed_seeds: 4", "minimum_completed_seeds: 6"
    )

    with pytest.raises(EvidenceContractError, match="minimum_completed_seeds"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_dissent_allowance_that_swallows_the_completion_floor(
    tmp_path,
):
    invalid_path = _write_mutated_contract(
        tmp_path, "maximum_dissenting_seeds: 1", "maximum_dissenting_seeds: 4"
    )

    with pytest.raises(EvidenceContractError, match="maximum_dissenting_seeds"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_meaningful_improvement_at_or_above_clear(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path,
        "meaningful_improvement: 0.01",
        "meaningful_improvement: 0.02",
    )

    with pytest.raises(EvidenceContractError, match="meaningful_improvement"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_relative_nll_difference(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path,
        "nll:\n        difference: absolute",
        "nll:\n        difference: relative",
    )

    with pytest.raises(EvidenceContractError, match="NLL"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_guardrail_without_a_declared_tolerance(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path,
        "statistic: ks_distance_from_uniform\n"
        "        difference: absolute\n"
        "        serious_degradation: 0.02\n",
        "statistic: ks_distance_from_uniform\n"
        "        difference: absolute\n",
    )

    with pytest.raises(EvidenceContractError, match="serious_degradation"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_a_reordered_outcome_precedence(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path,
        "      - fail_when_no_meaningful_primary_improvement\n"
        "      - pass_when_gate_met\n",
        "      - pass_when_gate_met\n"
        "      - fail_when_no_meaningful_primary_improvement\n",
    )

    with pytest.raises(EvidenceContractError, match="outcome precedence"):
        load_merit_probe_evidence_contract(invalid_path)


def test_loader_rejects_replacing_a_failed_seed(tmp_path):
    invalid_path = _write_mutated_contract(
        tmp_path, "replacement: never", "replacement: next_unused_seed"
    )

    with pytest.raises(EvidenceContractError, match="failed gate seed"):
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
