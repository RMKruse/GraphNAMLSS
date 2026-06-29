# GraphNAMLSS Architecture

This document describes the planned GraphNAMLSS architecture family and the first implementation slice. `CONTEXT.md` defines the project language; this file turns that language into an experiment and model specification.

## Base NAMLSS Predictor

For each distributional parameter `theta_k`, NAMLSS learns an additive predictor:

```text
eta_k(i) = beta_k0 + sum_j f_kj(x_ij)
theta_k(i) = link_k_inverse(eta_k(i))
```

The properties to preserve are:

- additive local feature contributions
- separate predictors per distributional parameter
- flexible distributional families
- shape-function plots that remain meaningful to inspect

Local feature contributions should be centered:

```text
local_k(i) = beta_k0 + sum_j centered(f_kj(x_ij))
```

This keeps intercepts, local effects, and graph effects from absorbing each other arbitrarily.

## GraphNAMLSS Family

GraphNAMLSS augments the local NAMLSS predictor with graph-derived information. The family has two graph injection strategies.

### Additive Graph Contribution

The graph stream produces a parameter-specific contribution that is added to the local predictor:

```text
eta_k(i) = local_k(i) + alpha_k * centered(graph_k(i))
```

`alpha_k` is a learned graph contribution gate. It stabilizes training by letting the model begin close to NAMLSS and gives a simple diagnostic for how much each distributional parameter uses graph information.

The first additive implementation uses:

- shared GraphSAGE-style graph encoder
- parameter-specific linear heads
- centered graph contributions
- per-parameter graph contribution gates
- deterministic graph weights

### Graph-Aware Shape Functions

The graph-aware shape-function branch lets local shape functions depend on feature-wise graph context:

```text
s_ij = neighbor_summary(feature_j around node_i)
```

The canonical form keeps the ordinary local effect separate from the context effect:

```text
eta_k(i) =
  beta_k0
  + sum_j centered(f_local,kj(x_ij))
  + sum_j centered(f_context,kj(x_ij, s_ij))
```

For neighbor-only context variants:

```text
f_context,kj = f_context,kj(s_ij)
```

For self-conditioned context variants:

```text
f_context,kj = f_context,kj(x_ij, s_ij)
```

Context effects should be centered or otherwise constrained so they do not absorb the local main effect.

## Primary Experiment Matrix

The primary matrix crosses three factors:

| Factor | Level 1 | Level 2 |
|---|---|---|
| Graph Representation Sharing | Shared graph representation | Per-parameter graph representations |
| Graph Injection Strategy | Additive graph contribution | Graph-aware shape functions |
| Graph Stream Self-Feature Access | Neighbor-focused | Self-inclusive |

The same-recipe comparison uses the same hidden dimensions and training recipe across variants. A parameter-matched check then adjusts capacities so wins are not explained only by larger models.

## Tracer-Bullet Slice

The first milestone is not the full matrix. It is an end-to-end deterministic traffic experiment:

```text
dataset = METR-LA
replication dataset = PEMS-BAY
target = 15-minute-ahead speed
distribution = Normal(mu, sigma)
split = temporal
graph = dataset-provided physical adjacency
model = shared-encoder additive GraphNAMLSS
graph encoder = GraphSAGE-style message passing
self-feature access = neighbor-focused
Bayesian weights = no
```

The tracer-bullet model is:

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

The tracer bullet validates the data pipeline, likelihood, split, scaling, missingness policy, metrics, and plots before adding the rest of the matrix.

## Traffic Data Contract

The traffic benchmark uses sensor-time observations.

Local covariates:

- current speed or flow
- 5 minute lag
- 10 minute lag
- 15 minute lag
- 30 minute lag
- 60 minute lag
- time-of-day features
- day-of-week features
- weekend indicator

Neighbor covariates:

- neighboring-sensor current speed or flow
- neighboring-sensor 5, 10, 15, 30, and 60 minute lags
- physical edge weights when supported cleanly

The graph stream should not receive calendar features in the tracer bullet.

Scaling:

- fit scalers on the training period only
- train on standardized targets and features if useful
- transform `mu` and `sigma` back correctly for original-unit reporting
- label NLL and CRPS units clearly if computed in standardized units

Missingness:

- follow standard METR-LA preprocessing where appropriate
- mask missing targets from the loss
- avoid future-looking imputation
- report missingness rate and imputation choices

Adjacency:

- use dataset-provided traffic adjacency first
- reconstruct distance-kernel adjacency only as a fallback or robustness check
- prefer edge weights when the chosen graph layer supports them cleanly
- keep an unweighted-edge fallback documented

## Baselines

The baseline suite has three roles.

Statistical distributional baselines:

- GAMLSS
- BAMLSS
- GAMLSS plus neighbor-summary graph covariates
- BAMLSS plus neighbor-summary graph covariates

Neural distributional baselines:

- deterministic NAMLSS
- distributional GNN with direct `mu` and `sigma` heads

Point-prediction reference:

- mean-only GNN evaluated with MAE and RMSE

The graph-feature statistical baselines are especially important. They test whether learned graph representations add value beyond simple neighbor-summary covariates.

## Training Objective

For the tracer-bullet Gaussian location-scale model:

```text
loss = mean Gaussian NLL over observed targets
```

Once the Bayesian graph stream is added:

```text
loss = mean NLL + beta * KL[q(w | lambda) || p(w)]
```

The KL term applies to graph-stream weights only in the first Bayesian milestone. The local NAMLSS stream remains point-estimated.

Structural uncertainty is not part of the tracer bullet or the first Bayesian milestone. It requires explicit topology perturbation or a graph-topology model.

## Evaluation

Primary distributional metrics:

- negative log-likelihood
- CRPS
- prediction interval coverage
- PIT histogram

Point-accuracy guardrails:

- MAE
- RMSE

Parameter-specific graph effect checks:

- graph contribution to `mu` only
- graph contribution to `sigma` only
- graph contribution to both

This tests whether graph structure improves location and scale differently.

## Interpretability Outputs

The tracer-bullet interpretability bundle contains:

- local shape plots by distributional parameter
- graph contribution maps over sensors by distributional parameter
- graph contribution gates `alpha_mu` and `alpha_sigma`
- calibration plots

Attention visualizations and GNNExplainer-style methods are deferred until attention-based graph encoders or the full matrix are stable.

## Implementation Milestones

1. Deterministic METR-LA tracer-bullet slice.
2. PEMS-BAY replication.
3. Full 2x2x2 primary matrix.
4. Parameter-matched check.
5. Bayesian graph stream.
6. Spatial Health Graph application track.
7. Structural uncertainty experiment.

The order is deliberate: first validate the experiment harness, then expand model breadth, then add Bayesian and structural uncertainty claims.
