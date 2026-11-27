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


@pytest.mark.parametrize(
    "claim",
    (
        "The graph `mu` path reliably shows the spillover slowdown.",
        "The gated mu graph contribution consistently captures the spillover "
        "slowdown.",
        "The gated `mu` contrast is lower on every seed, so it shows the slowdown.",
        "Neighbor lag features expose the spillover as an expected slowdown "
        "through the location parameter.",
    ),
)
def test_scope_guard_rejects_claims_that_the_graph_mu_path_reliably_shows_the_slowdown(
    claim,
):
    with pytest.raises(AssertionError, match="Concept Demonstrator scope violation"):
        assert_concept_demo_scope(claim)


@pytest.mark.parametrize(
    "disclosure",
    (
        "Under the Tracer-Bullet Fit, the `mu` graph path is weak and "
        "seed-dependent.",
        "The graph `mu` path does not reliably show the spillover slowdown.",
        "The gated `mu` contrast carries little of the slowdown and reverses on "
        "some seeds.",
        "Upstream congestion makes the downstream target both slower and more "
        "variable.",
    ),
)
def test_scope_guard_accepts_honest_disclosure_of_the_weak_mu_path(disclosure):
    assert_concept_demo_scope(disclosure)
