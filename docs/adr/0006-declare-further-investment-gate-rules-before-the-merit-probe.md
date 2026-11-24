# Declare the Further-Investment Gate rules before the Merit Probe

## Decision

The Further-Investment Gate's pass, fail and inconclusive rules and its Gate Seed Set are fixed now, before any real METR-LA comparison is seen, and are recorded in the `further_investment_gate.declared_rules` block of `configs/metr_la_merit_probe.yaml`. The gate evaluator reads them from there and invents no thresholds of its own.

**What is compared.** For each Completed Gate Seed, the paired difference Δ = Tracer-Bullet GraphNAMLSS − Graph-Ablated GraphNAMLSS on the test partition at the Fifteen-Minute Traffic Horizon. Negative Δ favours the candidate. NLL is compared as an absolute difference in nats per observation; CRPS, MAE and RMSE as differences relative to the reference; coverage as the change in |coverage − 0.90| for the central 90% interval; PIT as the change in the Kolmogorov–Smirnov distance between the PIT values and the uniform distribution.

**Seeds.** The Gate Seed Set is `{1, 2, 3, 4, 5}`. Seed `0` is the Qualification Seed: the qualifying run uses it to freeze the run recipe, evaluates only the train and validation partitions (Sealed Test Partition), and never counts toward the gate. A gate seed that fails is reported. It may be rerun with the identical seed and recipe after an infrastructure failure, is recorded without rerun after a training failure (divergence, NaN, no valid checkpoint), and is never replaced by another seed.

**Thresholds**, all on the mean paired Δ across Completed Gate Seeds:

| Signal | Clear Distributional Signal | Meaningful improvement | Serious Guardrail Degradation |
|---|---|---|---|
| NLL | ≤ −0.02 nats | ≤ −0.01 nats | — |
| CRPS | ≤ −2% | ≤ −1% | — |
| Coverage error | — | — | > +2 percentage points |
| PIT KS distance | — | — | > +0.02 |
| MAE, RMSE | — | — | > +2% |

A Clear Distributional Signal also needs every Completed Gate Seed but one to have Δ < 0. The other primary signal is *clearly worse* when it mirrors that rule: mean Δ ≥ +0.02 nats or ≥ +2%, with every Completed Gate Seed but one worse. Dispersion (sample standard deviation, minimum, maximum of each Δ) is reported for the human but is not part of the rule.

**Outcome**, decided in this order:

1. Fewer than 4 Completed Gate Seeds → **inconclusive**.
2. Neither NLL nor CRPS reaches a meaningful improvement → **fail**.
3. One primary signal is clear, the other is not clearly worse, and no guardrail shows Serious Guardrail Degradation → **pass**.
4. Anything else → **inconclusive**.

A Completed Gate Seed already carries its Local Shape Contributions, Graph Contributions and Graph Gate Diagnostics from the reloaded selected checkpoint, so a pass always has its interpretability artifacts. Secondary context models have no gate rule; they inform the human decision only.

**Integrity.** This ADR and the contract are committed before the qualifying run. Every Merit Probe run and the Merit Probe Evidence Bundle record the SHA-256 of the evidence contract they ran under, and the gate evaluator refuses a bundle whose recorded hash differs from the contract it is given.

## Rationale

- **One effect size, two primary scales.** For a calibrated Gaussian, CRPS scales linearly with σ and NLL shifts by log σ. A predictive distribution 2% sharper at equal calibration therefore gives about −2% CRPS and log(0.98) ≈ −0.020 nats. The NLL and CRPS thresholds describe the same effect. The local arm already sees its own lags at a 15-minute horizon, which makes it a strong reference, so a 5% bar risks failing a real but modest graph effect, and a 1% bar sits close to seed noise.
- **A guardrail may not lose more than a primary signal must gain**, so each guardrail tolerance matches the clear-signal magnitude.
- **Paired differences cancel units for NLL.** The `log(scale)` term from the Traffic Scaling Policy cancels in Δ NLL. Relative differences keep the other metrics unit-free.
- **All but one seed** guards against one noisy seed vetoing a real effect. Under no true effect, at least 4 of 5 seeds landing the right way happens about 19% of the time, and the mean threshold carries the magnitude. With five seeds, a dispersion-based rule would add a knob without adding robustness.
- **The fail boundary sits at half the clear threshold** so that a practically zero improvement reads as fail, not as an invitation to a diagnostic agenda.
- **Guardrails are judged against Graph-Ablated GraphNAMLSS, not absolute targets.** Miscalibration both arms share is not caused by the graph stream; it is a diagnostic note.
- **Only NLL and CRPS can supply the signal.** Coverage and PIT measure calibration, not skill.
- **A separate Qualification Seed and a Sealed Test Partition** keep the test set out of recipe decisions. Changing the seed alone would not do that, because every seed shares the same test partition.

## Considered options

- **Leave thresholds to the evaluator ticket.** Rejected: they would be set by whoever first sees evidence, which is post-hoc tuning by another name.
- **Formal significance tests over seeds.** Rejected: out of scope for this Merit Probe, and with five seeds they add little beyond a sign count and a mean threshold.
- **Absolute calibration targets** (for example, coverage within [85%, 95%]). Rejected: they gate on miscalibration both arms share.
- **Counting the qualifying run as a gate seed.** Rejected: the recipe was shaped by inspecting it.
- **A pass on coverage or PIT improvement alone.** Rejected, as above.

## Consequences

- `load_merit_probe_evidence_contract` parses and validates the rules into typed fields. It rejects incoherent rules, for example a Qualification Seed inside the Gate Seed Set, a qualifying run that reads the test partition, a meaningful threshold at or above the clear one, or a reordered outcome precedence. The pinned test fixes the declared values.
- `preferred_distributional_signals` is renamed `primary_distributional_signals`: only NLL and CRPS can supply a Clear Distributional Signal.
- The qualifying run proves its test-evaluation path on CI fixtures only; the first real test comparison appears in the multi-seed Merit Probe Evidence Bundle.
- Changing any rule after the qualifying run starts is a new decision that has to be recorded as such, and it invalidates the contract hash of every earlier run.
