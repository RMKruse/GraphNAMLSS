import pytest

from graphnamlss.testing.scope_guards import assert_concept_demo_scope


@pytest.mark.parametrize(
    "claim",
    (
        "Synthetic demo metrics prove GraphNAMLSS outperforms baselines.",
        "GraphNAMLSS outperforms baselines based on synthetic demo metrics.",
        "The Concept Demonstrator supports Bayesian graph streams.",
        "The Concept Demo Bundle implements structural uncertainty.",
        "The Concept Demonstrator reports medical application results.",
        "The Concept Demo Bundle includes the GAMLSS and BAMLSS baselines.",
    ),
)
def test_scope_guard_rejects_premature_concept_demo_claims(claim):
    with pytest.raises(AssertionError, match="Concept Demonstrator scope violation"):
        assert_concept_demo_scope(claim)
