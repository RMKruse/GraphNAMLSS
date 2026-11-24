# Require optimizer-trained GraphNAMLSS for the Merit Probe

## Decision

The METR-LA Merit Probe requires an optimizer-trained GraphNAMLSS candidate. The deterministic Tracer-Bullet Fit remains appropriate for the synthetic Concept Demonstrator because it makes the model decomposition and artifact contract runnable and inspectable, but it does not qualify as real-data evidence for the Further-Investment Gate.

The next implementation path is `optimizer_trained_graph_namlss_training`. That path must update the candidate model from the declared training objective, fit on the training partition, use validation evidence for model selection or stopping, leave the test partition untouched until evaluation, and emit enough training provenance to reproduce the fitted candidate. Naming an optimizer in configuration or repeating a fixed post-fit loss as training history does not satisfy this standard.

## Rationale

The current deterministic Tracer-Bullet Fit combines deterministic local fitting with residual graph fitting and fixed graph gates. That is a useful teaching simplification, but its relationship to the intended learned GraphNAMLSS architecture is too limited for a research-investment decision. Requiring optimizer-trained candidate artifacts makes the primary comparison interpretable and prevents an apparent metric signal from resting on an unclear model-training story.

## Consequences and limits

The Concept Demonstrator remains unblocked and must continue to disclose its deterministic Tracer-Bullet Fit. The deterministic NAMLSS reference may retain deterministic fitting because that is its intended baseline model; this decision concerns the GraphNAMLSS candidate used as Merit Probe evidence. *(Superseded by [ADR 0005](0005-use-graph-ablated-graphnamlss-as-merit-probe-reference.md): the Merit Probe reference is now Graph-Ablated GraphNAMLSS, trained by the candidate's recipe. Deterministic NAMLSS is used only as a training sanity floor.)* A METR-LA run produced by the demo fit cannot pass or fail the Further-Investment Gate.

This decision does not require PEMS-BAY replication, the full Primary Experiment Matrix, Bayesian graph streams, structural uncertainty, or medical applications. It does not select a specific optimizer, stopping rule, or hyperparameter budget; those belong to the follow-up implementation contract. It also makes no claim that optimizer-trained GraphNAMLSS will pass the Merit Probe.
