# Use Graph-Ablated GraphNAMLSS as the Merit Probe reference

## Decision

The METR-LA Merit Probe Primary Comparison judges optimizer-trained Tracer-Bullet GraphNAMLSS against **Graph-Ablated GraphNAMLSS**, not against the closed-form Deterministic NAMLSS. The reference is trained through the same `optimizer_trained_graph_namlss_training` path, with the graph stream removed structurally: no graph stream, graph heads or Graph Contribution Gates for `mu` or `sigma`. Everything else is identical: local additive component architecture, `sigma` head, optimizer, stopping rule, epoch budget and hyperparameters. Neither arm gets its own tuning. Each arm selects its own checkpoint by its own validation NLL. The two arms form a Seed-Paired Comparison: each declared seed trains both from the same local-component initialization and data order, so local-component initialization must not depend on whether a graph stream exists. The Further-Investment Gate's rules are stated over per-seed paired differences.

This supersedes the clause of ADR 0004 that let the deterministic NAMLSS reference keep its deterministic fit.

## Rationale

The Further-Investment Gate tests whether graph-derived, parameter-specific contributions add distributional value. The closed-form reference has a ridge-linear `mu` and one constant `sigma`. The optimizer-trained candidate learns nonlinear local shapes and a `sigma` predictor as well as the graph contribution. NLL and CRPS are highly sensitive to scale, and traffic speed is heteroskedastic, so a gain over the closed-form reference could come entirely from the learned scale and local shapes. The gate could then pass while both Graph Contribution Gates sit near zero. Against a graph-ablated twin, a gain can only come from the graph stream.

## Considered options

- **Keep the closed-form reference.** Rejected: it confounds the graph contribution with heteroskedastic scale, nonlinear local effects and the fitting method.
- **Closed-form reference plus a `sigma` predictor.** Rejected: it still confounds nonlinear local shapes and the fitting method, and it needs a new fitting path anyway.
- **Gate against both references.** Rejected: two rule sets, and one of the two arms is still confounded.
- **Freeze the gates at zero instead of removing the graph stream.** Rejected: same function, but the reported parameter count includes an inert stream, and the provenance shows parameters that do nothing.

## Consequences

- The Deterministic NAMLSS stays a Concept Demonstrator model. In the Merit Probe it serves only as a training sanity floor in the qualifying run, never as gate or bundle evidence.
- Every model whose metrics enter the Merit Probe Evidence Bundle is optimizer-trained under the same recipe, including the secondary context models. This prevents the mirror-image confound: an optimizer-trained GraphNAMLSS shown next to a weakly fitted GNN.
- The evidence contract's `reference_model` changes from `deterministic_namlss` to `graph_ablated_graph_namlss`.
