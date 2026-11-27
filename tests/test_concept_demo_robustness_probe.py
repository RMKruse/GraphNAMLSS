import json
import subprocess
import sys

import pytest

from graphnamlss.demo_robustness import run_concept_demo_robustness_probe
from graphnamlss.testing.scope_guards import assert_concept_demo_scope


DECLARED_SEEDS = list(range(20250214, 20250224))


@pytest.fixture(scope="module")
def declared_probe(tmp_path_factory):
    output_directory = tmp_path_factory.mktemp("concept-demo-robustness")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphnamlss",
            "demo-robustness",
            "--output",
            str(output_directory),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return output_directory, json.loads(result.stdout)


def test_demo_robustness_command_writes_one_summary_for_the_declared_seed_set(
    declared_probe,
):
    output_directory, emitted = declared_probe

    summary_path = output_directory / "concept_demo_robustness_summary.json"
    assert json.loads(summary_path.read_text()) == emitted
    assert emitted["seed_set"] == "declared"
    assert emitted["seeds"] == DECLARED_SEEDS
    assert [record["seed"] for record in emitted["per_seed"]] == DECLARED_SEEDS

    for record in emitted["per_seed"]:
        assert record["status"] == "completed"
        bundle_directory = output_directory / record["bundle_directory"]
        assert (
            bundle_directory / "report" / "tracer_bullet_evaluation_report.md"
        ).exists()
        graph_run = bundle_directory / "runs" / "tracer-bullet-graph-namlss"
        assert (graph_run / "graph_contributions.json").exists()
        assert (graph_run / "predictions.json").exists()


def test_spillover_targets_and_gated_sigma_point_the_expected_way_on_every_declared_seed(
    declared_probe,
):
    _output_directory, emitted = declared_probe

    for record in emitted["per_seed"]:
        checks = record["checks"]
        location = checks["target_location"]
        variability = checks["target_variability"]
        gated_sigma = checks["gated_sigma_contribution"]

        assert location["spillover"] < location["routine"]
        assert variability["spillover"] > variability["routine"]
        assert gated_sigma["spillover"] > gated_sigma["routine"]

    aggregate_checks = emitted["aggregate"]["checks"]
    for name in ("target_location", "target_variability", "gated_sigma_contribution"):
        assert aggregate_checks[name]["held_on"] == DECLARED_SEEDS


def test_every_directional_check_reports_every_declared_seed_without_dropping_failures(
    declared_probe,
):
    _output_directory, emitted = declared_probe
    expected_spillover_direction = {
        "target_location": "lower",
        "target_variability": "higher",
        "gated_mu_contribution": "lower",
        "gated_sigma_contribution": "higher",
    }

    for record in emitted["per_seed"]:
        assert set(record["checks"]) == set(expected_spillover_direction)
        for name, direction in expected_spillover_direction.items():
            check = record["checks"][name]
            spillover_is_lower = check["spillover"] < check["routine"]
            assert check["held"] is (
                spillover_is_lower if direction == "lower" else not spillover_is_lower
            )

    aggregate = emitted["aggregate"]
    for name in expected_spillover_direction:
        check = aggregate["checks"][name]
        assert check["not_evaluated_on"] == []
        assert sorted(check["held_on"] + check["not_held_on"]) == DECLARED_SEEDS
        assert check["not_held_on"] == [
            record["seed"]
            for record in emitted["per_seed"]
            if not record["checks"][name]["held"]
        ]
    assert aggregate["signal_holds_on_every_seed"] is all(
        not check["not_held_on"] for check in aggregate["checks"].values()
    )


def test_gated_mu_share_of_slowdown_is_reported_per_seed_but_not_enforced(
    declared_probe,
):
    output_directory, emitted = declared_probe

    shares = []
    for record in emitted["per_seed"]:
        checks = record["checks"]
        share = record["gated_mu_share_of_slowdown"]
        graph_summary = json.loads(
            (
                output_directory
                / record["bundle_directory"]
                / "runs"
                / "tracer-bullet-graph-namlss"
                / "train_summary.json"
            ).read_text()
        )
        target_scale = graph_summary["scaling"]["target"]["scale"]
        standardized_slowdown = checks["target_location"]["difference"] / target_scale

        assert share * standardized_slowdown == pytest.approx(
            checks["gated_mu_contribution"]["difference"], abs=1e-5
        )
        assert (share > 0.0) is checks["gated_mu_contribution"]["held"]
        shares.append(share)

    assert emitted["aggregate"]["gated_mu_share_of_slowdown"] == {
        "min": min(shares),
        "max": max(shares),
    }
    assert "gated_mu_share_of_slowdown" in emitted["comparison"]
    assert "reported, not enforced" in emitted["comparison"]


def test_robustness_summary_states_it_is_a_teaching_sanity_check_outside_the_gate(
    declared_probe,
):
    _output_directory, emitted = declared_probe

    boundary = emitted["evidence_boundary"]
    assert "teaching sanity check" in boundary
    assert "cannot support the Further-Investment Gate" in boundary
    assert "not METR-LA Merit Probe evidence" in boundary
    assert_concept_demo_scope(boundary)


def test_rerunning_the_probe_on_the_same_seeds_writes_an_identical_bounded_summary(
    tmp_path,
):
    first = run_concept_demo_robustness_probe(tmp_path / "first", seeds=(7, 11))
    second = run_concept_demo_robustness_probe(tmp_path / "second", seeds=(7, 11))

    assert first == second
    assert (tmp_path / "first" / "concept_demo_robustness_summary.json").read_bytes() == (
        tmp_path / "second" / "concept_demo_robustness_summary.json"
    ).read_bytes()
    assert first["seed_set"] == "custom"
    assert first["seeds"] == [7, 11]
    assert [record["seed"] for record in first["per_seed"]] == [7, 11]
    for record in first["per_seed"]:
        for check in record["checks"].values():
            assert all(
                isinstance(value, (bool, int, float)) for value in check.values()
            )


def test_a_seed_whose_bundle_cannot_be_written_is_recorded_as_a_visible_failure(
    tmp_path,
):
    output_directory = tmp_path / "robustness"
    blocked_bundle = output_directory / "seeds" / "seed-11"
    blocked_bundle.parent.mkdir(parents=True)
    blocked_bundle.write_text("not a bundle directory\n")

    summary = run_concept_demo_robustness_probe(output_directory, seeds=(7, 11))

    completed, failed = summary["per_seed"]
    assert completed["seed"] == 7
    assert completed["status"] == "completed"
    assert failed["seed"] == 11
    assert failed["status"] == "failed"
    assert failed["bundle_directory"] == "seeds/seed-11"
    assert failed["error"]
    assert "checks" not in failed
    assert "gated_mu_share_of_slowdown" not in failed
    assert summary["aggregate"]["gated_mu_share_of_slowdown"] == {
        "min": completed["gated_mu_share_of_slowdown"],
        "max": completed["gated_mu_share_of_slowdown"],
    }

    for check in summary["aggregate"]["checks"].values():
        assert check["not_evaluated_on"] == [11]
        assert 11 not in check["held_on"] + check["not_held_on"]
    assert summary["aggregate"]["signal_holds_on_every_seed"] is False
    written = json.loads(
        (output_directory / "concept_demo_robustness_summary.json").read_text()
    )
    assert written == summary


@pytest.mark.parametrize("seeds", ((), (7, 11, 7)))
def test_probe_rejects_empty_or_repeated_seed_sets(tmp_path, seeds):
    with pytest.raises(ValueError, match="seed"):
        run_concept_demo_robustness_probe(tmp_path / "rejected", seeds=seeds)

    assert not (tmp_path / "rejected").exists()
