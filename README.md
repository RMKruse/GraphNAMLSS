# GraphNAMLSS

GraphNAMLSS is a research project for distributional regression on graph-structured observations. The core idea is to preserve the local interpretability of NAMLSS-style additive predictors while adding graph-derived information to the distributional parameters.

For a detailed collaborator-facing explanation of the research and development plan, see `RESEARCH_PLAN.md`.

## Plain Pyproject Workflow

GraphNAMLSS uses a plain `pyproject.toml` workflow with editable installs. It does not require Poetry, uv, or another tool-specific environment manager.

From a clean checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

The repo includes `.python-version` with `3.12` so Python version managers use the PyTorch-compatible development runtime by default. If `python3.12` is not available, install Python 3.12 first rather than creating the environment with a newer interpreter.

The mandatory runtime dependencies stay minimal for package imports. PyYAML backs the Simple Experiment Config loader, and the `dev` extra installs the test runner plus PyTorch so graph-model tests run in the standard development environment.

## Guided Demo Path

The Guided Demo Path gives a Methodologically Sympathetic Colleague one command
and one primary report for understanding the Concept Demonstrator. From the
repository root with the development environment active, run:

```bash
graphnamlss demo --output outputs/concept-demo
```

The command writes the complete Concept Demo Bundle under
`outputs/concept-demo`. When it finishes, open the primary generated report at
`outputs/concept-demo/report/tracer_bullet_evaluation_report.md`.

Read that report in this order:

1. **Model Decomposition** — start with how the local additive predictor and
   graph contribution combine for each distributional parameter.
2. **Local Shape Contributions** — inspect
   `report/assets/local_shape_mu.svg` and
   `report/assets/local_shape_sigma.svg` to see the local additive effects for
   location and scale.
3. **Graph Contributions** — compare
   `report/assets/graph_contribution_mu.svg` and
   `report/assets/graph_contribution_sigma.svg`, then use
   `runs/tracer-bullet-graph-namlss/graph_contributions.json` for the underlying
   values.
4. **Graph Gate Diagnostics** — inspect
   `runs/tracer-bullet-graph-namlss/graph_gates.json` to see the
   parameter-specific graph reliance diagnostic.
5. **Metrics** — use the report table for the compact comparison and each
   model's `metrics.json` under `runs/` for its recorded values.
6. **Concept Demo Evidence Boundary** — finish with the limits on what the
   synthetic run can establish before interpreting any result.

The Concept Demo Bundle uses a synthetic Demo Fixture and a lightweight
deterministic Tracer-Bullet Fit so colleagues can inspect the full modeling idea
without downloading benchmark data. Synthetic demo metrics are sanity checks:
they show that the experiment contract, model paths, artifacts, and report run
end to end. They are not evidence that GraphNAMLSS outperforms baselines or
improves calibration on real traffic data.

This Concept Demonstrator is not the METR-LA Merit Probe. Its Tracer-Bullet Fit
does not satisfy the Merit Probe Training Standard, which requires an
optimizer-trained GraphNAMLSS evaluated on the real METR-LA benchmark. Use the
demo to judge whether the decomposition is coherent and inspectable, not to
make an empirical merit claim.

## Simple Experiment Config Dry Run

The first tracer-bullet experiment contract is a YAML-backed Simple Experiment Config. The sample config validates the METR-LA tracer setup without downloading METR-LA or starting training:

```bash
graphnamlss dry-run --config configs/tracer_bullet.yaml
```

The dry run prints the resolved dataset, temporal split, feature, model, optimizer, and output settings. It also creates the configured output directory and writes `resolved_config.json` there so the run settings are auditable.

## Synthetic Traffic Fixture

Fast tests can use `graphnamlss.testing.load_synthetic_traffic_fixture()` for a tiny deterministic traffic-style fixture. It contains multiple sensors, multiple five-minute time points, lagged local covariates, neighbor traffic summaries, missingness cases, and a small physical adjacency structure.

This fixture is test data, not a scientific benchmark. It exists to exercise splitting, lagging, adjacency handling, likelihood, model, metric, and smoke-run code paths without downloading METR-LA or using network access.

## Explicit PyG Dependency Policy

PyTorch and PyTorch Geometric dependencies stay explicit. They are optional for the initial package import, but graph experiments should install the PyG extra in the project environment:

```bash
python -m pip install -e ".[dev,pyg]"
```

The plain development path uses Python 3.12 and expects `torch>=2.5`. The PyG path adds `torch-geometric>=2.6`. If your platform needs a CPU- or CUDA-specific PyTorch wheel, install the matching PyTorch wheel first using the official PyTorch instructions, then install the editable PyG extra above. Record the exact Python, PyTorch, PyTorch Geometric, and accelerator versions with any experiment output.

The package is intentionally self-contained under `src/graphnamlss`. Reuse from existing NAMLSS or BANDS code should happen only through a narrow, documented external-code adapter boundary.

In standard NAMLSS, each distributional parameter has an additive predictor:

```text
eta_k(i) = beta_k0 + sum_j f_kj(x_ij)
```

GraphNAMLSS extends this family by letting graph information affect the predictors for parameters such as location and scale:

```text
eta_k(i) = local_NAMLSS_k(x_i) + graph_k(i)
```

The project treats GraphNAMLSS as a model family, not one fixed architecture. The goal is to compare how graph information should enter the distributional model and whether graph structure helps some distributional parameters more than others.

## Main Research Questions

1. Does graph information improve distributional calibration beyond a strong local NAMLSS baseline?
2. Does graph information help scale or dispersion differently than location or mean?
3. How much interpretability is preserved compared with a non-additive distributional GNN?
4. Is a shared graph representation sufficient, or do distributional parameters need separate graph representations?
5. Are simple graph-derived covariates enough, or is learned graph representation useful?

## Primary Experiment Matrix

The primary matrix crosses three experimental factors under one canonical base configuration:

```text
Graph Representation Sharing:
- shared graph representation
- per-parameter graph representations

Graph Injection Strategy:
- additive graph contribution
- graph-aware shape functions

Graph Stream Self-Feature Access:
- neighbor-focused graph information
- self-inclusive graph information
```

This gives a 2x2x2 comparison. The first report should include a same-recipe comparison and a parameter-matched check so larger per-parameter or self-inclusive models are not mistaken for inherently better architectures.

## First Milestone

The first implementation is intentionally narrower than the final matrix:

```text
dataset: METR-LA
task: 15-minute-ahead traffic speed prediction
distribution: Gaussian location-scale
split: temporal train/validation/test
graph: dataset-provided physical traffic adjacency
model: deterministic shared-encoder additive GraphNAMLSS
encoder: GraphSAGE-style message passing
```

This tracer-bullet slice validates the data pipeline, likelihood, metrics, contribution plots, and calibration outputs before adding Bayesian graph weights or the full primary matrix.

The first GraphNAMLSS cell is:

```text
eta_k(i,t) =
  intercept_k
  + sum_j centered_local_shape_kj(x_ij,t)
  + alpha_k * centered_graph_contribution_k(i,t)
```

where `alpha_k` is a learned per-parameter graph contribution gate. The scale parameter uses a positive link:

```text
sigma = softplus(eta_sigma) + epsilon
```

## Traffic Benchmark

The traffic benchmark starts with METR-LA and then replicates on PEMS-BAY.

### METR-LA Merit Probe Evidence Contract

The METR-LA Merit Probe is the first real-data evidence gate after the Concept
Demonstrator. Its machine-readable contract lives at
`configs/metr_la_merit_probe.yaml` and is validated by
`load_merit_probe_evidence_contract`.

The primary comparison is deterministic NAMLSS versus Tracer-Bullet GraphNAMLSS.
Distributional GNN is secondary distributional context; Point GNN is a
point-accuracy guardrail rather than a distributional peer. The
Further-Investment Gate looks first for an NLL or CRPS signal while treating
coverage, PIT, MAE, and RMSE as guardrails.

The Merit Probe candidate must come from an optimizer-trained GraphNAMLSS path.
The deterministic Tracer-Bullet Fit is limited to the Concept Demonstrator and
cannot supply evidence for the Further-Investment Gate. The named follow-up path
is `optimizer_trained_graph_namlss_training`; ADR 0004 records the rationale and
scope limits.

If that signal is unclear, the next step is diagnosis—not PEMS-BAY replication,
the full Primary Experiment Matrix, Bayesian graph streams, or medical
applications. PEMS-BAY becomes the Replication Gate only after a promising
METR-LA Merit Probe.

### PEMS-BAY Replication Dataset

The PEMS-BAY replication config lives at `configs/pems_bay.yaml` and uses the
same experiment schema, temporal split, covariate set, scaling policy,
missingness policy, and dataset-provided graph contract as the METR-LA tracer
config.

Required local artifacts:

- `data/raw/pems-bay/pems-bay.h5` for wide sensor readings.
- `data/raw/pems-bay/adj_mx_bay.pkl` for the dataset-provided DCRNN adjacency.

The local covariate set contains:

- current speed or flow
- 5, 10, 15, 30, and 60 minute lags
- time-of-day features
- day-of-week features
- weekend indicator

The neighbor covariate set contains neighboring-sensor traffic lags over the physical traffic graph. Calendar features are kept out of the graph stream at first so the graph contribution reflects neighboring traffic state rather than global time context.

Targets and features are scaled using statistics fit on the training period only. Means, scales, likelihoods, intervals, and calibration plots must be reported in clearly labeled standardized or original units, with original traffic units preferred for interpretability.

## Baselines

The baseline set includes both statistical and neural comparisons:

- deterministic NAMLSS
- GAMLSS
- BAMLSS
- GAMLSS plus graph-derived neighbor-summary covariates
- BAMLSS plus graph-derived neighbor-summary covariates
- distributional GNN with direct location and scale heads
- mean-only point GNN as a point-accuracy reference

The statistical graph-feature baselines are important because they test whether learned graph representations add value beyond simple neighbor summaries.

## Evaluation

The main claim is distributional calibration, not just point accuracy.

Primary metrics:

- negative log-likelihood
- CRPS
- prediction interval coverage
- PIT histogram

Guardrail metrics:

- MAE
- RMSE

Interpretability outputs for the tracer-bullet slice:

- local shape plots by distributional parameter
- graph contribution maps over sensors by distributional parameter
- learned graph contribution gates for location and scale
- calibration plots

## Bayesian and Structural Uncertainty Scope

The first implementation is deterministic. The next milestone adds a Bayesian graph stream by placing a variational posterior over graph-stream weights while keeping the local NAMLSS stream point-estimated.

Structural uncertainty is a later experiment. The project should not claim structural uncertainty from ordinary edge dropout alone. That claim is reserved for experiments that explicitly model or perturb graph topology.

## Medical Application Track

Traffic is the controlled benchmark. The medical application track should focus on spatial health graphs:

```text
nodes = regions or hospital catchment areas
edges = geographic, mobility, referral, or adjacency relationships
target = overdispersed event counts
exposure = population, patient volume, or person-time
distribution = Negative Binomial or another exposure-aware count family
```

This keeps the medical contribution aligned with graph neural networks in medicine while preserving a natural distributional-regression motivation.

## Architecture Decision Records

Architectural decisions are recorded in `docs/adr/`. The first ADR explains why the project starts with a deterministic METR-LA tracer-bullet slice before expanding to the full primary matrix and later Bayesian graph-stream experiments.
