from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class EvidenceContractError(ValueError):
    """Raised when an evidence contract cannot be loaded."""


@dataclass(frozen=True)
class MeritProbePrimaryComparison:
    reference_model: str
    candidate_model: str


@dataclass(frozen=True)
class MeritProbeSecondaryContext:
    model: str
    role: str


@dataclass(frozen=True)
class MeritProbeDecisionGate:
    preferred_distributional_signals: tuple[str, ...]
    guardrail_metrics: tuple[str, ...]


@dataclass(frozen=True)
class MeritProbeTrainingStandard:
    concept_demonstrator_fit: str
    merit_probe_fit: str
    next_implementation_path: str


@dataclass(frozen=True)
class MeritProbeFailureRule:
    next_step: str
    diagnostic_areas: tuple[str, ...]
    blocked_expansions: tuple[str, ...]


@dataclass(frozen=True)
class MeritProbeReplicationGate:
    dataset: str
    after: str
    purpose: str


@dataclass(frozen=True)
class MeritProbeEvidenceContract:
    stage: str
    dataset: str
    primary_comparison: MeritProbePrimaryComparison
    secondary_context: tuple[MeritProbeSecondaryContext, ...]
    training_standard: MeritProbeTrainingStandard
    further_investment_gate: MeritProbeDecisionGate
    failure_rule: MeritProbeFailureRule
    replication_gate: MeritProbeReplicationGate


def load_merit_probe_evidence_contract(
    path: str | Path,
) -> MeritProbeEvidenceContract:
    contract_path = Path(path)
    try:
        raw = yaml.safe_load(contract_path.read_text())
    except OSError as exc:
        raise EvidenceContractError(
            f"Could not read evidence contract {contract_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise EvidenceContractError(
            f"Could not parse YAML evidence contract {contract_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise EvidenceContractError(
            "Merit Probe evidence contract must be a YAML mapping."
        )

    comparison = _required_mapping(raw, "primary_comparison")
    training_standard = _required_mapping(raw, "training_standard")
    decision_gate = _required_mapping(raw, "further_investment_gate")
    failure_rule = _required_mapping(raw, "failure_rule")
    replication_gate = _required_mapping(raw, "replication_gate")
    contract = MeritProbeEvidenceContract(
        stage=_required_str(raw, "stage"),
        dataset=_required_str(raw, "dataset"),
        primary_comparison=MeritProbePrimaryComparison(
            reference_model=_required_str(comparison, "reference_model"),
            candidate_model=_required_str(comparison, "candidate_model"),
        ),
        secondary_context=tuple(
            MeritProbeSecondaryContext(
                model=_required_str(entry, "model"),
                role=_required_str(entry, "role"),
            )
            for entry in _required_mapping_sequence(raw, "secondary_context")
        ),
        training_standard=MeritProbeTrainingStandard(
            concept_demonstrator_fit=_required_str(
                training_standard, "concept_demonstrator_fit"
            ),
            merit_probe_fit=_required_str(training_standard, "merit_probe_fit"),
            next_implementation_path=_required_str(
                training_standard, "next_implementation_path"
            ),
        ),
        further_investment_gate=MeritProbeDecisionGate(
            preferred_distributional_signals=_required_str_sequence(
                decision_gate, "preferred_distributional_signals"
            ),
            guardrail_metrics=_required_str_sequence(
                decision_gate, "guardrail_metrics"
            ),
        ),
        failure_rule=MeritProbeFailureRule(
            next_step=_required_str(failure_rule, "next_step"),
            diagnostic_areas=_required_str_sequence(
                failure_rule, "diagnostic_areas"
            ),
            blocked_expansions=_required_str_sequence(
                failure_rule, "blocked_expansions"
            ),
        ),
        replication_gate=MeritProbeReplicationGate(
            dataset=_required_str(replication_gate, "dataset"),
            after=_required_str(replication_gate, "after"),
            purpose=_required_str(replication_gate, "purpose"),
        ),
    )
    _validate_merit_probe_evidence_contract(contract)
    return contract


def _validate_merit_probe_evidence_contract(
    contract: MeritProbeEvidenceContract,
) -> None:
    comparison = contract.primary_comparison
    if (
        contract.stage != "first_real_data_evidence_gate"
        or contract.dataset != "metr_la"
        or comparison.reference_model != "deterministic_namlss"
        or comparison.candidate_model != "tracer_bullet_graph_namlss"
    ):
        raise EvidenceContractError(
            "METR-LA Merit Probe primary comparison must be deterministic NAMLSS "
            "versus Tracer-Bullet GraphNAMLSS at the first real-data evidence gate."
        )

    expected_roles = {
        "distributional_gnn_baseline": "distributional_secondary_context",
        "point_gnn_reference": "point_accuracy_guardrail",
    }
    observed_roles = {
        entry.model: entry.role for entry in contract.secondary_context
    }
    if observed_roles != expected_roles:
        raise EvidenceContractError(
            "Secondary context must label distributional_gnn_baseline as "
            "distributional_secondary_context and point_gnn_reference as "
            "point_accuracy_guardrail."
        )

    if (
        contract.training_standard.merit_probe_fit
        != "optimizer_trained_graph_namlss"
    ):
        raise EvidenceContractError(
            "Merit Probe Training Standard requires optimizer-trained "
            "GraphNAMLSS evidence; the deterministic Tracer-Bullet Fit is limited "
            "to the Concept Demonstrator."
        )

    gate = contract.further_investment_gate
    if gate.preferred_distributional_signals != ("nll", "crps"):
        raise EvidenceContractError(
            "Further-Investment Gate must prefer NLL and CRPS."
        )
    if gate.guardrail_metrics != ("coverage", "pit", "mae", "rmse"):
        raise EvidenceContractError(
            "Further-Investment Gate must guard coverage, PIT, MAE, and RMSE."
        )

    failure_rule = contract.failure_rule
    if (
        failure_rule.next_step != "diagnose"
        or failure_rule.diagnostic_areas
        != (
            "data",
            "baseline",
            "fitting_method",
            "graph_stream",
            "scale_predictor",
            "horizon_choice",
        )
        or failure_rule.blocked_expansions
        != (
            "pems_bay_replication",
            "primary_experiment_matrix",
            "bayesian_graph_streams",
            "medical_applications",
        )
    ):
        raise EvidenceContractError(
            "Merit Probe Failure Rule must diagnose unclear evidence before expansion."
        )

    replication_gate = contract.replication_gate
    if (
        replication_gate.dataset != "pems_bay"
        or replication_gate.after != "promising_metr_la_merit_probe"
        or replication_gate.purpose != "test_generalization"
    ):
        raise EvidenceContractError(
            "PEMS-BAY must be the generalization Replication Gate after a promising "
            "METR-LA Merit Probe."
        )


def _required_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise EvidenceContractError(f"Expected '{key}' to be a mapping.")
    return value


def _required_mapping_sequence(
    mapping: dict[str, Any], key: str
) -> tuple[dict[str, Any], ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(entry, dict) for entry in value
    ):
        raise EvidenceContractError(
            f"Expected '{key}' to be a non-empty list of mappings."
        )
    return tuple(value)


def _required_str_sequence(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(entry, str) and entry.strip() for entry in value
    ):
        raise EvidenceContractError(
            f"Expected '{key}' to be a non-empty list of strings."
        )
    return tuple(value)


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvidenceContractError(f"Expected '{key}' to be a non-empty string.")
    return value
