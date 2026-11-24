from __future__ import annotations

from dataclasses import dataclass
import math
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
class GatePrimarySignalRule:
    """Thresholds are positive magnitudes on the mean paired difference."""

    metric: str
    difference: str
    clear_improvement: float
    meaningful_improvement: float


@dataclass(frozen=True)
class GateGuardrailRule:
    """A degradation beyond serious_degradation on the mean paired difference."""

    metric: str
    statistic: str
    difference: str
    serious_degradation: float
    interval_level: float | None


@dataclass(frozen=True)
class GateQualificationRun:
    seed: int
    evaluated_partitions: tuple[str, ...]


@dataclass(frozen=True)
class GateFailedSeedPolicy:
    infrastructure_failure: str
    training_failure: str
    replacement: str


@dataclass(frozen=True)
class FurtherInvestmentGateRules:
    """The pass, fail and inconclusive rules declared before the Merit Probe run."""

    decision_record: str
    evaluation_partition: str
    paired_difference: str
    gate_seeds: tuple[int, ...]
    qualification_run: GateQualificationRun
    minimum_completed_seeds: int
    maximum_dissenting_seeds: int
    primary_signals: tuple[GatePrimarySignalRule, ...]
    guardrails: tuple[GateGuardrailRule, ...]
    failed_seeds: GateFailedSeedPolicy
    outcome_precedence: tuple[str, ...]


@dataclass(frozen=True)
class MeritProbeDecisionGate:
    primary_distributional_signals: tuple[str, ...]
    guardrail_metrics: tuple[str, ...]
    rules: FurtherInvestmentGateRules


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
            primary_distributional_signals=_required_str_sequence(
                decision_gate, "primary_distributional_signals"
            ),
            guardrail_metrics=_required_str_sequence(
                decision_gate, "guardrail_metrics"
            ),
            rules=_further_investment_gate_rules(
                _required_mapping(decision_gate, "declared_rules")
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

    _validate_further_investment_gate(contract.further_investment_gate)

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


_PRIMARY_SIGNALS = ("nll", "crps")
_GUARDRAIL_STATISTICS = {
    "coverage": "absolute_coverage_error",
    "pit": "ks_distance_from_uniform",
    "mae": "mae",
    "rmse": "rmse",
}
_DIFFERENCE_FORMS = ("absolute", "relative")
_QUALIFICATION_PARTITIONS = ("train", "validation")
_FAILED_SEED_POLICY = GateFailedSeedPolicy(
    infrastructure_failure="rerun_identical_seed_and_recipe",
    training_failure="record_without_rerun",
    replacement="never",
)
_OUTCOME_PRECEDENCE = (
    "inconclusive_when_too_few_completed_seeds",
    "fail_when_no_meaningful_primary_improvement",
    "pass_when_gate_met",
    "inconclusive_otherwise",
)


def _further_investment_gate_rules(
    rules: dict[str, Any],
) -> FurtherInvestmentGateRules:
    qualification_run = _required_mapping(rules, "qualification_run")
    failed_seeds = _required_mapping(rules, "failed_seeds")
    return FurtherInvestmentGateRules(
        decision_record=_required_str(rules, "decision_record"),
        evaluation_partition=_required_str(rules, "evaluation_partition"),
        paired_difference=_required_str(rules, "paired_difference"),
        gate_seeds=_required_int_sequence(rules, "gate_seeds"),
        qualification_run=GateQualificationRun(
            seed=_required_int(qualification_run, "seed"),
            evaluated_partitions=_required_str_sequence(
                qualification_run, "evaluated_partitions"
            ),
        ),
        minimum_completed_seeds=_required_int(rules, "minimum_completed_seeds"),
        maximum_dissenting_seeds=_required_int(rules, "maximum_dissenting_seeds"),
        primary_signals=tuple(
            GatePrimarySignalRule(
                metric=metric,
                difference=_required_str(entry, "difference"),
                clear_improvement=_required_positive_float(
                    entry, "clear_improvement"
                ),
                meaningful_improvement=_required_positive_float(
                    entry, "meaningful_improvement"
                ),
            )
            for metric, entry in _named_mappings(rules, "primary_signals")
        ),
        guardrails=tuple(
            GateGuardrailRule(
                metric=metric,
                statistic=_required_str(entry, "statistic"),
                difference=_required_str(entry, "difference"),
                serious_degradation=_required_positive_float(
                    entry, "serious_degradation"
                ),
                interval_level=_optional_probability(entry, "interval_level"),
            )
            for metric, entry in _named_mappings(rules, "guardrails")
        ),
        failed_seeds=GateFailedSeedPolicy(
            infrastructure_failure=_required_str(
                failed_seeds, "infrastructure_failure"
            ),
            training_failure=_required_str(failed_seeds, "training_failure"),
            replacement=_required_str(failed_seeds, "replacement"),
        ),
        outcome_precedence=_required_str_sequence(rules, "outcome_precedence"),
    )


def _validate_further_investment_gate(gate: MeritProbeDecisionGate) -> None:
    if gate.primary_distributional_signals != _PRIMARY_SIGNALS:
        raise EvidenceContractError(
            "Further-Investment Gate must use NLL and CRPS as its primary "
            "distributional signals."
        )
    if gate.guardrail_metrics != tuple(_GUARDRAIL_STATISTICS):
        raise EvidenceContractError(
            "Further-Investment Gate must guard coverage, PIT, MAE, and RMSE."
        )

    rules = gate.rules
    if (
        rules.evaluation_partition != "test"
        or rules.paired_difference != "candidate_minus_reference"
    ):
        raise EvidenceContractError(
            "Further-Investment Gate rules must be stated over test-partition "
            "paired differences, candidate minus reference."
        )
    _validate_gate_seeds(rules)
    _validate_primary_signal_rules(rules.primary_signals)
    _validate_guardrail_rules(rules.guardrails)

    if rules.failed_seeds != _FAILED_SEED_POLICY:
        raise EvidenceContractError(
            "A failed gate seed may be rerun only with the identical seed and "
            "recipe after an infrastructure failure; a training failure is "
            "recorded without rerun, and a failed seed is never replaced."
        )
    if rules.outcome_precedence != _OUTCOME_PRECEDENCE:
        raise EvidenceContractError(
            "Further-Investment Gate outcome precedence must be: "
            + ", ".join(_OUTCOME_PRECEDENCE)
            + "."
        )


def _validate_gate_seeds(rules: FurtherInvestmentGateRules) -> None:
    seeds = rules.gate_seeds
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise EvidenceContractError(
            "Gate seeds must be distinct non-negative integers."
        )
    if rules.qualification_run.seed in seeds:
        raise EvidenceContractError(
            "The Qualification Seed must not be part of the Gate Seed Set."
        )
    if not set(rules.qualification_run.evaluated_partitions) <= set(
        _QUALIFICATION_PARTITIONS
    ):
        raise EvidenceContractError(
            "The qualifying run must keep the Sealed Test Partition: it may "
            "evaluate only the train and validation partitions."
        )
    if not 1 <= rules.minimum_completed_seeds <= len(seeds):
        raise EvidenceContractError(
            "minimum_completed_seeds must be between 1 and the number of gate "
            "seeds."
        )
    if not 0 <= rules.maximum_dissenting_seeds < rules.minimum_completed_seeds:
        raise EvidenceContractError(
            "maximum_dissenting_seeds must be non-negative and smaller than "
            "minimum_completed_seeds."
        )


def _validate_primary_signal_rules(
    signals: tuple[GatePrimarySignalRule, ...],
) -> None:
    if tuple(signal.metric for signal in signals) != _PRIMARY_SIGNALS:
        raise EvidenceContractError(
            "Primary signal rules must cover exactly NLL and CRPS, in that order."
        )
    for signal in signals:
        if signal.difference not in _DIFFERENCE_FORMS:
            raise EvidenceContractError(
                f"Primary signal {signal.metric} must use an absolute or "
                "relative difference."
            )
        if signal.metric == "nll" and signal.difference != "absolute":
            raise EvidenceContractError(
                "NLL differences must be absolute: a relative NLL change has no "
                "stable meaning."
            )
        if signal.meaningful_improvement >= signal.clear_improvement:
            raise EvidenceContractError(
                f"meaningful_improvement must be smaller than clear_improvement "
                f"for {signal.metric}."
            )


def _validate_guardrail_rules(guardrails: tuple[GateGuardrailRule, ...]) -> None:
    if tuple(guardrail.metric for guardrail in guardrails) != tuple(
        _GUARDRAIL_STATISTICS
    ):
        raise EvidenceContractError(
            "Guardrail rules must cover exactly coverage, PIT, MAE, and RMSE, in "
            "that order."
        )
    for guardrail in guardrails:
        expected_statistic = _GUARDRAIL_STATISTICS[guardrail.metric]
        if guardrail.statistic != expected_statistic:
            raise EvidenceContractError(
                f"The {guardrail.metric} guardrail must use the "
                f"{expected_statistic} statistic."
            )
        if guardrail.difference not in _DIFFERENCE_FORMS:
            raise EvidenceContractError(
                f"The {guardrail.metric} guardrail must use an absolute or "
                "relative difference."
            )
        if (guardrail.metric == "coverage") != (guardrail.interval_level is not None):
            raise EvidenceContractError(
                "Only the coverage guardrail declares an interval_level, and it "
                "must declare one."
            )


def _named_mappings(
    mapping: dict[str, Any], key: str
) -> tuple[tuple[str, dict[str, Any]], ...]:
    value = _required_mapping(mapping, key)
    if not value or not all(
        isinstance(name, str) and isinstance(entry, dict)
        for name, entry in value.items()
    ):
        raise EvidenceContractError(
            f"Expected '{key}' to map metric names to rule mappings."
        )
    return tuple(value.items())


def _required_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceContractError(f"Expected '{key}' to be an integer.")
    return value


def _required_int_sequence(mapping: dict[str, Any], key: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(entry, int) and not isinstance(entry, bool) for entry in value
    ):
        raise EvidenceContractError(
            f"Expected '{key}' to be a non-empty list of integers."
        )
    return tuple(value)


def _required_positive_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise EvidenceContractError(f"Expected '{key}' to be a positive number.")
    return float(value)


def _optional_probability(mapping: dict[str, Any], key: str) -> float | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 < value < 1.0
    ):
        raise EvidenceContractError(f"Expected '{key}' to lie strictly in (0, 1).")
    return float(value)


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
