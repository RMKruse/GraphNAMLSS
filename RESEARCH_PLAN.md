# GraphNAMLSS Research Plan

This document explains the full research and development approach for GraphNAMLSS. It is meant for collaborators, supervisors, coauthors, and reviewers who need to understand the motivation, modeling choices, experiment sequence, and scope of the project without reading every issue or implementation detail.

## One-Minute Summary

GraphNAMLSS is a model family for distributional regression on graph-structured observations. It combines two ideas:

- NAMLSS-style local additive predictors, which keep feature effects interpretable.
- Graph neural network information, which lets each observation use neighborhood structure.

The research question is not only whether graph information improves prediction. The sharper question is whether graph information improves the predicted distribution, especially whether it helps location and scale differently.

The first implementation is deliberately narrow: a deterministic GraphNAMLSS tracer-bullet slice on METR-LA for 15-minute-ahead traffic speed prediction. Once the data pipeline, likelihood, metrics, and plots work, the project expands to the full architecture matrix, PEMS-BAY replication, Bayesian graph weights, and a spatial health application track.

## Core Motivation

NAMLSS and related GAMLSS-family models predict full distributional parameters such as location, scale, and shape. They are useful because they model more than a conditional mean and because their additive feature effects can be inspected.

However, standard NAMLSS assumes observations are conditionally independent given local covariates. That assumption is weak for graph-structured data:

- traffic sensors are connected by road networks
- regions are connected by adjacency, mobility, or referral patterns
- hospitals or catchment areas can influence each other through patient flow

Graph neural networks handle neighborhood structure, but common GNN regression models usually predict a point estimate or a simple Gaussian head. They rarely preserve NAMLSS-style local additive interpretability.

GraphNAMLSS fills this gap:

```text
local interpretability from NAMLSS
+ graph-aware distributional prediction from GNNs
= interpretable distributional graph regression
```

## Research Claims

The project is organized around four claims.

### 1. Distributional Calibration Claim

The main claim is that graph-aware predictors improve the quality of the predictive distribution, not just point accuracy.

Primary evidence:

- negative log-likelihood
- CRPS
- prediction interval coverage
- PIT histograms

Point accuracy is still reported through MAE and RMSE, but only as a guardrail. A result that improves RMSE but worsens calibration is not a successful GraphNAMLSS result.

### 2. Parameter-Specific Graph Effect

Graph information may help different distributional parameters in different ways.

For traffic:

- graph structure may improve expected speed or flow
- graph structure may also improve conditional uncertainty during congestion
- the graph contribution to `mu` and `sigma` may not be the same

For spatial health:

- graph structure may improve expected disease or hospital burden
- graph structure may also explain overdispersion and tail risk

This is tested by comparing graph contributions to location only, scale or dispersion only, and both.

### 3. Interpretable Graph-Aware Distributional Modeling

GraphNAMLSS should preserve the useful part of NAMLSS interpretability:

- local shape plots remain inspectable
- graph contribution maps show where neighborhood structure matters
- graph contribution gates show how strongly each distributional parameter uses graph information

This is the reason GraphNAMLSS is compared against a Distributional GNN Baseline. If a plain GNN with distributional heads performs similarly, GraphNAMLSS must justify itself through interpretability and parameter-specific insight.

### 4. Learned Graph Representation Beyond Simple Graph Features

The project compares learned graph representations against simpler graph-derived covariates.

This matters because a reviewer may ask:

> Why not just add neighbor averages to GAMLSS or BAMLSS?

The Graph-Feature Statistical Baseline answers that question directly.

## Model Family

GraphNAMLSS is the family name, not one architecture.

The local NAMLSS predictor for distributional parameter `k` is:

```text
eta_k(i) = beta_k0 + sum_j f_kj(x_ij)
```

GraphNAMLSS augments this with graph-derived information. The two main Graph Injection Strategy choices are:

1. Additive Graph Contribution
2. Graph-Aware Shape Function

### Additive Graph Contribution

This is the first implementation path.

```text
eta_k(i) =
  local_k(i)
  + alpha_k * centered_graph_k(i)
```

The local term contains Centered Local Shape Function components. The graph term is a Centered Graph Contribution. The scalar `alpha_k` is the Graph Contribution Gate for parameter `k`.

This design has a clean interpretation:

- local shape plots show local covariate effects
- graph contribution maps show structural effects
- graph gates show graph reliance by parameter

### Graph-Aware Shape Function

This branch lets graph context enter the shape functions themselves.

For feature `j` at node `i`:

```text
s_ij = feature-wise neighbor summary for feature j around node i
```

The graph-aware term can be neighbor-only:

```text
f_context,kj(s_ij)
```

or self-conditioned:

```text
f_context,kj(x_ij, s_ij)
```

Context Effect Centering is needed so the graph context effect does not absorb the ordinary local feature effect.

## Primary Experiment Matrix

The full Primary Experiment Matrix crosses three factors:

| Factor | Level 1 | Level 2 |
|---|---|---|
| Graph Representation Sharing | Shared graph representation | Per-parameter graph representations |
| Graph Injection Strategy | Additive graph contribution | Graph-aware shape functions |
| Graph Stream Self-Feature Access | Neighbor-focused | Self-inclusive |

This creates a 2x2x2 matrix.

Two comparison views are needed:

- Same-Recipe Comparison: same hidden dimensions and training recipe across variants.
- Parameter-Matched Check: model capacities adjusted so larger models do not win only because they are larger.

## Why Start With a Tracer-Bullet Slice

The full matrix is too much to implement first. The project starts with a Tracer-Bullet Slice so that the entire experimental path works before model breadth increases.

The first slice is:

```text
dataset: METR-LA
task: 15-minute-ahead traffic speed prediction
distribution: Gaussian location-scale
split: temporal train/validation/test
graph: dataset-provided physical traffic adjacency
model: deterministic shared-encoder additive GraphNAMLSS
encoder: GraphSAGE-style message passing
self-feature access: neighbor-focused
Bayesian weights: no
```

This choice is recorded in ADR 0001. The point is to validate:

- data loading
- lag construction
- target alignment
- graph loading
- missingness handling
- scaling
- Gaussian likelihood
- metric computation
- training loop
- shape plots
- graph contribution plots
- calibration plots

Only after this works should the project expand to the full matrix.

## Development Strategy

The implementation is a Self-Contained Experiment Package. This is recorded in ADR 0002.

The package owns:

- data pipeline
- model variants
- training loops
- evaluation
- plotting
- experiment configs
- tests

Prior NAMLSS or BANDS code may be reused only through External Code Adapter boundaries. The repo should not become a wrapper around another codebase.

## Implementation Milestone Sequence

The project should progress in this order:

1. Scaffold the package and Plain Pyproject Workflow.
2. Add Simple Experiment Config and dry-run command.
3. Create synthetic traffic fixture for fast tests.
4. Build Traffic Sensor-Time Observation pipeline.
5. Implement Temporal Traffic Split.
6. Build Fifteen-Minute Traffic Horizon.
7. Build Traffic Local Covariate Set.
8. Build Traffic Neighbor Covariate Set.
9. Implement Traffic Missingness Policy.
10. Implement Traffic Scaling Policy.
11. Load Dataset-Provided Traffic Adjacency.
12. Implement Positive Scale Link and Gaussian NLL.
13. Implement distributional metrics.
14. Implement Centered Local Shape Function components.
15. Train deterministic NAMLSS on synthetic traffic.
16. Train deterministic NAMLSS on METR-LA.
17. Implement the Tracer-Bullet Graph Encoder.
18. Implement Centered Graph Contribution and Graph Contribution Gate.
19. Train Tracer-Bullet GraphNAMLSS on synthetic traffic.
20. Train Tracer-Bullet GraphNAMLSS on METR-LA.
21. Add Distributional GNN Baseline.
22. Add Point GNN Reference.
23. Generate the Tracer-Bullet Interpretability Bundle.
24. Add PEMS-BAY replication.
25. Define Statistical Baseline adapter contracts.
26. Add Primary Experiment Matrix run definitions.
27. Add Bayesian Graph Stream.
28. Add Spatial Health Graph application track.
29. Add Structural Uncertainty Experiment.

GitHub issue #1 contains the PRD. Issues #2 through #26 break the first implementation phase into independently grabbable slices.

## Traffic Benchmark Design

The Traffic Network Benchmark is the first controlled benchmark because it is public, graph-native, and familiar to the GNN community.

### Datasets

The first dataset is METR-LA. It is smaller and better for the first end-to-end harness.

The second dataset is PEMS-BAY. It is used as a replication dataset after the METR-LA tracer-bullet result is stable.

### Observation Unit

The observation unit is a Traffic Sensor-Time Observation:

```text
one sensor
at one prediction time
with local lagged traffic features
with neighbor lagged traffic features
with one future target
```

This avoids making the first paper a full spatiotemporal sequence-modeling project.

### Prediction Horizon

The first target is the Fifteen-Minute Traffic Horizon:

```text
predict speed or flow 15 minutes ahead
```

Longer horizons and multi-horizon forecasting are later robustness extensions.

### Local Covariates

The Traffic Local Covariate Set contains:

- current speed or flow
- 5 minute lag
- 10 minute lag
- 15 minute lag
- 30 minute lag
- 60 minute lag
- time-of-day features
- day-of-week features
- weekend indicator

These are strong enough to make the local baseline credible, but still interpretable enough for local shape plots.

### Neighbor Covariates

The Traffic Neighbor Covariate Set contains neighboring-sensor lagged traffic values over the Physical Traffic Graph.

It does not include calendar features in the tracer-bullet slice. Calendar features are local time context, not graph context.

### Graph

The benchmark uses a Physical Traffic Graph with Dataset-Provided Traffic Adjacency first.

The Traffic Edge Weight Policy is:

- use dataset-provided edge weights when cleanly supported
- keep a documented unweighted fallback
- do not begin with learned or correlation-derived adjacency

This keeps the graph effect interpretable as known spatial structure rather than target-derived shortcut information.

### Split

The primary split is a Temporal Traffic Split:

```text
train: earliest time block
validation: later time block
test: final time block
```

The sensor graph remains fixed. Random sensor-time splits are avoided because they blur future-time generalization.

### Scaling

The Traffic Scaling Policy is:

- fit scalers on the training period only
- train on standardized targets and features if useful
- transform predicted means and scales back correctly when reporting original units
- label NLL and CRPS units clearly

For Gaussian location-scale:

```text
mu_original = mu_scaled * train_sd + train_mean
sigma_original = sigma_scaled * train_sd
```

### Missingness

The Traffic Missingness Policy is:

- mask missing targets from loss and metrics
- avoid future-looking imputation
- report missingness rates
- expose missingness indicators or metadata when imputation is used

## First Model: Tracer-Bullet GraphNAMLSS

The first GraphNAMLSS model is:

```text
eta_mu =
  beta_mu
  + sum_j centered(f_mu,j(x_j))
  + alpha_mu * centered(graph_mu(neighbors))

eta_sigma =
  beta_sigma
  + sum_j centered(f_sigma,j(x_j))
  + alpha_sigma * centered(graph_sigma(neighbors))

mu = eta_mu
sigma = softplus(eta_sigma) + epsilon
```

Important properties:

- deterministic first
- shared graph encoder
- GraphSAGE-style message passing
- additive graph contribution
- neighbor-focused graph stream
- positive scale link
- centered local and graph terms
- graph gates per parameter

This is not the final architecture. It is the first stable cell of the Primary Experiment Matrix.

## Baseline Strategy

The baseline suite has three purposes.

### 1. Local Distributional Baseline

Deterministic NAMLSS answers:

> Does graph information add value beyond local additive distributional regression?

### 2. Statistical Baselines

GAMLSS and BAMLSS answer:

> How does GraphNAMLSS compare with established statistical distributional regression?

Graph-Feature Statistical Baseline variants add simple neighbor-summary covariates. They answer:

> Does learned graph representation improve over simple graph-derived covariates?

### 3. Neural Graph Baselines

The Distributional GNN Baseline answers:

> Why not just use a non-additive GNN with distributional heads?

The Point GNN Reference answers:

> Does GraphNAMLSS remain competitive with common mean-only GNN practice on point accuracy?

The Point GNN Reference is not a peer distributional method. It is a guardrail.

## Evaluation Plan

The primary evaluation target is distributional quality.

Primary metrics:

- negative log-likelihood
- CRPS
- prediction interval coverage
- PIT histogram

Point-accuracy guardrails:

- MAE
- RMSE

Parameter-Specific Graph Effect checks:

- graph contribution to `mu` only
- graph contribution to `sigma` only
- graph contribution to both

The model should not be judged only by RMSE. If it improves RMSE but hurts calibration, it has not achieved the main research goal.

## Interpretability Plan

The first interpretability output set is the Tracer-Bullet Interpretability Bundle:

1. Local shape plots by distributional parameter.
2. Graph contribution maps over sensors by distributional parameter.
3. Learned graph contribution gates for `mu` and `sigma`.
4. Calibration plots.

Attention visualizations and GNNExplainer-style explanations are deferred until attention-based graph encoders or the full matrix are stable.

## Bayesian Scope

The first implementation is deterministic.

The next Bayesian milestone adds a Bayesian Graph Stream:

```text
loss = mean NLL + beta * KL[q(w | lambda) || p(w)]
```

In the first Bayesian milestone:

- graph-stream weights are variational
- local NAMLSS weights remain point-estimated
- the distributional family captures aleatoric uncertainty
- variational graph weights capture epistemic uncertainty in the graph stream

This keeps the Bayesian claim precise and testable.

## Structural Uncertainty Scope

Structural uncertainty is a later experiment.

The project should not claim structural uncertainty from ordinary edge dropout alone. A Structural Uncertainty Experiment must explicitly model or perturb graph topology.

Examples:

- missing-edge perturbation
- noisy adjacency perturbation
- stochastic edge model
- robustness curves over graph corruption levels

This should come after the deterministic and Bayesian graph-stream experiments work.

## Medical Application Track

The Medical Application Track is a secondary empirical track, not the first benchmark.

The target medical setting is a Spatial Health Graph:

```text
nodes: regions or hospital catchment areas
edges: geographic, mobility, referral, or adjacency relationships
target: overdispersed event counts
exposure: population, patient volume, or person-time
distribution: Negative Binomial or another exposure-aware count family
```

This track is useful for collaboration with medical GNN researchers because it connects the method to health burden, overdispersion, and spatial dependence.

It should follow the traffic benchmark because medical data access, preprocessing, and domain validation add risk before the core method is proven.

## What Is In Scope First

The first phase includes:

- package scaffold
- synthetic traffic fixture
- Simple Experiment Config
- Traffic Sensor-Time Observation pipeline
- Temporal Traffic Split
- Traffic Local Covariate Set
- Traffic Neighbor Covariate Set
- Traffic Missingness Policy
- Traffic Scaling Policy
- Dataset-Provided Traffic Adjacency
- Gaussian NLL and Positive Scale Link
- distributional metrics
- deterministic NAMLSS
- Tracer-Bullet GraphNAMLSS
- Distributional GNN Baseline
- Point GNN Reference
- Tracer-Bullet Interpretability Bundle

## What Is Explicitly Out of Scope First

The first phase does not include:

- full Bayesian Graph Stream
- Structural Uncertainty Experiment
- learned traffic adjacency
- correlation-derived traffic adjacency
- full spatiotemporal sequence modeling
- recurrent temporal models
- temporal GNNs
- multi-horizon output
- GAT attention visualizations
- GNNExplainer-style explanations
- Spatial Health Graph implementation
- complete GAMLSS or BAMLSS R package integration
- heavyweight experiment orchestration

## Risks and Mitigations

### Risk: The graph term relearns local effects

Mitigation:

- begin with a Neighbor-Focused Graph Stream
- center local and graph contributions
- use graph contribution gates
- compare self-inclusive access explicitly in the Primary Experiment Matrix

### Risk: More complex variants win only because they are larger

Mitigation:

- report Same-Recipe Comparison
- report Parameter-Matched Check

### Risk: Calibration metrics are unit-confused

Mitigation:

- follow Traffic Scaling Policy
- label standardized versus original units
- transform `mu` and `sigma` correctly

### Risk: Missingness leaks future information

Mitigation:

- mask missing targets
- avoid future-looking imputation
- test missingness behavior on synthetic data

### Risk: The project overclaims structural uncertainty

Mitigation:

- reserve structural uncertainty for explicit topology perturbation or topology modeling
- keep Bayesian Graph Stream and Structural Uncertainty Experiment separate

### Risk: The medical application distracts from method development

Mitigation:

- treat traffic as the controlled benchmark
- treat spatial health as the applied track after the traffic harness works

## Expected Research Narrative

The paper narrative should be:

1. Distributional regression models are interpretable but usually ignore graph structure.
2. GNNs use graph structure but usually do not provide interpretable distributional regression.
3. GraphNAMLSS combines local additive distributional predictors with graph-derived distributional effects.
4. The controlled traffic benchmark shows whether graph structure improves calibration and whether it helps location and scale differently.
5. Baselines show whether the benefit survives comparison against local NAMLSS, statistical distributional models, graph-feature statistical models, and non-additive GNNs.
6. Interpretability outputs show what the local and graph components are doing.
7. Later Bayesian and medical extensions expand the method without weakening the first result.

## Short Pitch

GraphNAMLSS extends neural additive distributional regression to graph-structured data. It preserves local shape-function interpretability while adding graph-aware contributions to distributional parameters. The first benchmark asks whether physical traffic-network structure improves calibrated 15-minute-ahead speed distributions on METR-LA, especially whether the graph helps conditional scale differently than location.

## Related Project Artifacts

- `CONTEXT.md`: project glossary and resolved terminology.
- `ARCHITECTURE.md`: technical architecture and experiment specification.
- `README.md`: concise public overview.
- `docs/adr/0001-start-with-deterministic-traffic-tracer-bullet.md`: sequencing decision.
- `docs/adr/0002-use-self-contained-experiment-package.md`: package boundary decision.
- GitHub issue #1: PRD for the self-contained experiment package.
- GitHub issues #2 through #26: implementation slices for the first phase.
