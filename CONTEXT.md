# GraphNAMLSS

GraphNAMLSS is a research context for distributional regression on graph-structured observations while preserving the interpretability of neural additive distributional models.

## Language

**GraphNAMLSS**:
A model family that augments NAMLSS-style local additive predictors with graph-derived contributions to one or more distributional parameters.
_Avoid_: using GraphNAMLSS to mean only the shared-encoder architecture

**Shared-Encoder GraphNAMLSS**:
A GraphNAMLSS variant where one graph encoder produces node representations that are projected into parameter-specific graph contributions.
_Avoid_: full GraphNAMLSS, default GraphNAMLSS

**Per-Parameter GraphNAMLSS**:
A GraphNAMLSS variant where separate graph encoders or graph streams contribute to different distributional parameters.
_Avoid_: separate GraphNAMLSS

**Graph Representation Sharing**:
The experimental factor that determines whether distributional parameters receive graph information from one shared graph representation or from parameter-specific graph representations.
_Avoid_: graph sharing, shared GNN

**Graph Injection Strategy**:
The experimental factor that determines whether graph information enters as a separate additive contribution or inside the local shape functions.
_Avoid_: graph approach, graph method

**Additive Graph Contribution**:
A graph-derived term added to a distributional predictor alongside the local NAMLSS additive predictor.
_Avoid_: graph residual unless residualization is explicitly enforced

**Graph Contribution Gate**:
A learned per-parameter scalar that controls how strongly an additive graph contribution enters a distributional predictor.
_Avoid_: graph weight when referring to the scalar gate rather than graph edge weights

**Centered Graph Contribution**:
An additive graph contribution centered per distributional parameter so it does not absorb the predictor intercept.
_Avoid_: raw graph contribution

**Neighbor-Focused Graph Stream**:
An additive graph stream that emphasizes neighboring-node information while limiting direct reuse of the target node's own local features.
_Avoid_: neighbor-only graph stream unless self-information is completely removed

**Self-Inclusive Graph Stream**:
An additive graph stream that can use both the target node's own features and neighboring-node information.
_Avoid_: standard GNN unless self-feature access is the relevant distinction

**Graph Stream Self-Feature Access**:
The experimental factor that determines whether the additive graph stream can directly use the target node's own local features.
_Avoid_: self-loop setting when the modeling implication is broader than graph-loop construction

**Graph-Aware Shape Function**:
A local shape function that depends on both a node's own feature value and graph-derived context for that feature.
_Avoid_: graph shape function, contextual NAM term

**Feature-Wise Neighbor Summary**:
A graph-derived summary of neighboring nodes for one feature, such as a weighted neighbor mean or weighted neighbor standard deviation.
_Avoid_: graph embedding when interpreting graph-aware shape functions

**Shape Context Effect**:
The graph-aware shape-function component that models how a feature-wise neighbor summary changes a distributional predictor beyond the local shape function.
_Avoid_: local shape function when referring to graph-derived context

**Neighbor-Only Shape Context**:
A Shape Context Effect that depends on a feature-wise neighbor summary but not directly on the target node's own value for that feature.
_Avoid_: neighbor-only shape function

**Self-Conditioned Shape Context**:
A Shape Context Effect that depends on both the target node's own feature value and the corresponding feature-wise neighbor summary.
_Avoid_: 2D shape function unless the self/context axes are clear

**Context Effect Centering**:
The constraint or regularization that keeps a Shape Context Effect from absorbing the local main effect already represented by the local shape function.
_Avoid_: identifiability trick

**Canonical Base Configuration**:
The fixed experimental setup used to compare GraphNAMLSS variants while changing only the named experimental factors.
_Avoid_: default setup, baseline setup

**Primary Experiment Matrix**:
The first planned comparison grid crossing graph representation sharing, graph injection strategy, and graph stream self-feature access under one canonical base configuration.
_Avoid_: 2x2 matrix

**Same-Recipe Comparison**:
The primary comparison view where GraphNAMLSS variants use the same hidden dimensions and training recipe.
_Avoid_: fair comparison when parameter counts differ materially

**Parameter-Matched Check**:
A secondary comparison view where variant capacities are adjusted to roughly match parameter counts.
_Avoid_: capacity ablation unless the parameter budget is actually controlled

**Statistical Baseline**:
A non-neural distributional regression baseline used to compare GraphNAMLSS against established statistical models.
_Avoid_: classical baseline when the model family is distributional

**GAMLSS Baseline**:
A Statistical Baseline using generalized additive models for location, scale, and shape.
_Avoid_: NAMLSS baseline when the intended comparison is the established statistical model

**BAMLSS Baseline**:
A Statistical Baseline using Bayesian additive models for location, scale, and shape.
_Avoid_: Bayesian NAMLSS when the intended comparison is the established statistical model

**Graph-Feature Statistical Baseline**:
A Statistical Baseline that receives simple graph-derived covariates such as feature-wise neighbor summaries.
_Avoid_: graph baseline when no learned graph representation is used

**Distributional GNN Baseline**:
A non-additive neural graph baseline whose graph encoder predicts distributional parameters directly.
_Avoid_: GraphNAMLSS baseline because it does not preserve local additive shape functions

**Point GNN Reference**:
A graph neural network that predicts only the conditional mean and is used as a point-accuracy reference rather than a distributional baseline.
_Avoid_: distributional baseline

**Tracer-Bullet Slice**:
The first end-to-end implementation milestone that validates the data pipeline, likelihood, metrics, and plots with a minimal set of models before expanding to the full primary matrix.
_Avoid_: prototype when the end-to-end experiment contract is the important point

**Tracer-Bullet GraphNAMLSS**:
The first GraphNAMLSS implementation cell: shared graph representation, additive graph contribution, neighbor-focused graph stream, and deterministic graph weights before adding Bayesian graph weights.
_Avoid_: first model unless the exact cell is clear

**Tracer-Bullet Graph Encoder**:
The first graph encoder used in the tracer-bullet model, a simple GraphSAGE-style message-passing encoder before attention-based encoders are added.
_Avoid_: GAT-first encoder

**Implementation Milestone Sequence**:
The planned order of implementation: deterministic GraphNAMLSS first, Bayesian graph stream second, and structural uncertainty experiments after the core models work.
_Avoid_: Bayesian-first implementation plan

**Self-Contained Experiment Package**:
A local PyTorch/PyG package that owns the GraphNAMLSS data pipeline, models, training loops, evaluation, and plotting.
_Avoid_: wrapper repo

**External Code Adapter**:
A narrow reuse boundary for bringing ideas or small components from prior NAMLSS or BANDS code into the Self-Contained Experiment Package.
_Avoid_: copying external code without an explicit boundary

**Initial Package Structure**:
The first source layout for the Self-Contained Experiment Package, separating data, models, training, evaluation, experiments, configs, scripts, and tests.
_Avoid_: clever architecture before the experiment contract is stable

**Plain Pyproject Workflow**:
The dependency-management convention that uses `pyproject.toml` and editable installs without requiring a Poetry- or uv-specific workflow.
_Avoid_: tool-specific install path unless the project later standardizes on that tool

**Explicit PyG Dependency Policy**:
The installation convention that keeps PyTorch Geometric dependencies visible and documents the tested Torch/PyG installation path.
_Avoid_: hiding graph dependencies behind undocumented environment state

**Simple Experiment Config**:
A YAML-backed experiment configuration loaded into small typed Python objects for dataset, split, features, model, optimizer, and output settings.
_Avoid_: heavyweight orchestration framework for the tracer-bullet slice

**Traffic Network Benchmark**:
The first canonical benchmark domain for evaluating GraphNAMLSS on public node-level graph regression data with plausible distributional structure.
_Avoid_: traffic dataset unless referring to a specific dataset

**METR-LA Tracer Dataset**:
The first traffic dataset used to validate the tracer-bullet experiment harness.
_Avoid_: traffic benchmark when referring to the first implementation dataset

**PEMS-BAY Replication Dataset**:
The second traffic dataset used to check whether results from the METR-LA tracer dataset generalize to another public traffic network.
_Avoid_: first traffic dataset

**Traffic Location-Scale Outcome**:
A traffic speed or flow target modeled with a Gaussian distribution whose location and scale are both predicted.
_Avoid_: traffic distribution unless referring to a specific distribution family

**Positive Scale Link**:
The transformation that maps an unconstrained scale predictor to a strictly positive scale parameter using softplus plus a small floor.
_Avoid_: raw sigma output

**Traffic Sensor-Time Observation**:
A traffic benchmark observation defined by one sensor at one prediction time, using lagged traffic and calendar features to predict a future traffic outcome.
_Avoid_: spatiotemporal sequence unless a temporal sequence model is explicitly part of the architecture

**Fifteen-Minute Traffic Horizon**:
The first traffic prediction horizon, using current and past traffic features to predict traffic speed or flow 15 minutes ahead.
_Avoid_: multi-horizon target for the tracer-bullet slice

**Traffic Local Covariate Set**:
The interpretable local features for a Traffic Sensor-Time Observation, including recent traffic lags and calendar features.
_Avoid_: all available features unless they are explicitly part of the local covariate set

**Traffic Neighbor Covariate Set**:
The graph-stream traffic features drawn from neighboring sensors, using recent traffic lags over the physical traffic graph without calendar features.
_Avoid_: graph covariates when calendar or target-node features are included

**Traffic Missingness Policy**:
The benchmark rule for handling absent traffic readings without future leakage, including target masking, documented imputation, and missingness indicators when lag features require imputation.
_Avoid_: silent forward fill

**Traffic Scaling Policy**:
The benchmark rule for fitting normalization on the training period only and reporting distributional predictions in clearly labeled standardized or original traffic units.
_Avoid_: unlabeled scaled likelihood

**Physical Traffic Graph**:
A traffic sensor graph whose edges and weights come from road-network proximity or dataset-provided physical adjacency rather than learned target correlations.
_Avoid_: learned traffic graph for the canonical benchmark

**Dataset-Provided Traffic Adjacency**:
The standard physical traffic adjacency or distance matrix distributed with a traffic benchmark's preprocessing artifacts.
_Avoid_: reconstructed adjacency for the canonical benchmark unless the standard artifact is unavailable

**Traffic Edge Weight Policy**:
The rule for using dataset-provided traffic edge weights in graph aggregation when supported cleanly, with an unweighted-edge fallback if needed.
_Avoid_: ignoring edge weights without documenting the reason

**Temporal Traffic Split**:
A traffic benchmark split where earlier time blocks train the model and later time blocks validate and test it while the sensor graph remains fixed.
_Avoid_: random sensor-time split for the canonical benchmark

**Distributional Calibration Claim**:
The primary evaluation claim that graph-aware predictors improve the quality of the predictive distribution rather than only point accuracy.
_Avoid_: performance claim unless the metric target is specified

**Tracer-Bullet Interpretability Bundle**:
The first set of interpretability outputs: local shape plots, graph contribution maps, graph contribution gates, and calibration plots.
_Avoid_: full explanation suite

**Parameter-Specific Graph Effect**:
The hypothesis that graph information can improve some distributional parameters, such as scale or dispersion, differently than others, such as location or mean.
_Avoid_: graph helps uncertainty unless the affected parameter is specified

**Bayesian Graph Stream**:
A GraphNAMLSS graph stream whose weights are modeled variationally while the local additive NAMLSS stream remains point-estimated.
_Avoid_: Bayesian GraphNAMLSS unless the Bayesian scope is specified

**Structural Uncertainty Experiment**:
A later experiment that explicitly models or perturbs graph topology to assess uncertainty from missing, noisy, or incomplete edges.
_Avoid_: claiming structural uncertainty from ordinary graph regularization alone

**Centered Local Shape Function**:
A one-dimensional NAMLSS feature contribution centered so intercepts, local effects, and graph-derived effects remain interpretable.
_Avoid_: raw shape contribution

**Medical Application Track**:
A secondary empirical track that motivates GraphNAMLSS through medical graph learning problems where distributional uncertainty is substantively important.
_Avoid_: medical benchmark unless the dataset is fixed

**Spatial Health Graph**:
A medical application graph whose nodes are regions or care sites and whose edges represent geographic, mobility, referral, or adjacency relationships relevant to health burden.
_Avoid_: patient graph, medical graph when this more precise term applies

**Spatial Health Count Outcome**:
An overdispersed event count observed on a Spatial Health Graph together with an exposure term such as population, patient volume, or person-time.
_Avoid_: risk score, disease rate unless the modeling target is explicitly a rate

## Relationships

- **GraphNAMLSS** includes both **Shared-Encoder GraphNAMLSS** and **Per-Parameter GraphNAMLSS**
- **Shared-Encoder GraphNAMLSS** and **Per-Parameter GraphNAMLSS** are contrasted empirically rather than treated as separate research problems
- **Graph Representation Sharing** is the primary controlled contrast between **Shared-Encoder GraphNAMLSS** and **Per-Parameter GraphNAMLSS**
- **Graph Injection Strategy** includes **Additive Graph Contribution** and **Graph-Aware Shape Function**
- **Additive Graph Contribution** variants use a **Graph Contribution Gate** to stabilize and interpret graph reliance per distributional parameter
- **Additive Graph Contribution** variants use **Centered Graph Contribution** components so graph effects do not absorb intercepts
- **Additive Graph Contribution** variants distinguish **Neighbor-Focused Graph Stream** from **Self-Inclusive Graph Stream** through **Graph Stream Self-Feature Access**
- **Graph-Aware Shape Function** variants use **Feature-Wise Neighbor Summary** inputs before learned graph embeddings
- **Graph-Aware Shape Function** variants express graph context through a **Shape Context Effect** alongside the local shape function
- **Graph Stream Self-Feature Access** maps to **Neighbor-Only Shape Context** versus **Self-Conditioned Shape Context** in the graph-aware shape-function branch
- **Context Effect Centering** preserves the distinction between local shape functions and **Shape Context Effect** components
- **Graph Representation Sharing** and **Graph Injection Strategy** are separate experimental factors
- The **Primary Experiment Matrix** crosses **Graph Representation Sharing**, **Graph Injection Strategy**, and **Graph Stream Self-Feature Access** under one **Canonical Base Configuration**
- The **Primary Experiment Matrix** is evaluated first with a **Same-Recipe Comparison** and then checked with a **Parameter-Matched Check**
- GraphNAMLSS is compared against **Statistical Baseline** models, including the **GAMLSS Baseline** and **BAMLSS Baseline**
- **Graph-Feature Statistical Baseline** variants test whether learned graph representations improve over simple graph-derived covariates
- The **Distributional GNN Baseline** tests whether GraphNAMLSS preserves interpretability without giving up too much predictive distribution quality
- The **Point GNN Reference** checks point accuracy against common GNN practice but is not a peer distributional method
- The **Tracer-Bullet Slice** precedes the full **Primary Experiment Matrix**
- The **Tracer-Bullet Slice** uses **Tracer-Bullet GraphNAMLSS** as its first GraphNAMLSS variant
- The **Tracer-Bullet GraphNAMLSS** uses the **Tracer-Bullet Graph Encoder** before attention-based graph encoders
- The **Implementation Milestone Sequence** keeps variational inference out of the first tracer-bullet implementation
- GraphNAMLSS is implemented as a **Self-Contained Experiment Package**, with prior NAMLSS or BANDS reuse isolated behind **External Code Adapter** boundaries
- The **Self-Contained Experiment Package** begins with the **Initial Package Structure**
- The **Self-Contained Experiment Package** uses a **Plain Pyproject Workflow** and follows an **Explicit PyG Dependency Policy**
- The **Tracer-Bullet Slice** uses a **Simple Experiment Config** before adopting heavier experiment orchestration
- The **Traffic Network Benchmark** is the first **Canonical Base Configuration** domain
- The **METR-LA Tracer Dataset** anchors the first **Tracer-Bullet Slice**
- The **PEMS-BAY Replication Dataset** follows after the **METR-LA Tracer Dataset**
- The **Traffic Network Benchmark** first uses a **Traffic Location-Scale Outcome** before adding heavier-tailed or asymmetric traffic distributions
- **Traffic Location-Scale Outcome** models use a **Positive Scale Link** for the Gaussian scale parameter
- The **Traffic Network Benchmark** uses **Traffic Sensor-Time Observations** before introducing temporal graph sequence models
- The **Traffic Network Benchmark** begins with a **Fifteen-Minute Traffic Horizon** before adding longer or multi-horizon targets
- The **Traffic Network Benchmark** starts with a **Traffic Local Covariate Set** containing current, 5-minute, 10-minute, 15-minute, 30-minute, and 60-minute traffic lags plus calendar features
- The **Traffic Network Benchmark** starts with a **Traffic Neighbor Covariate Set** containing neighboring-sensor traffic lags but not calendar features
- The **Traffic Network Benchmark** follows a **Traffic Missingness Policy** that masks missing targets and prevents future leakage during feature imputation
- The **Traffic Network Benchmark** follows a **Traffic Scaling Policy** so means, scales, likelihoods, and intervals are interpreted in the correct units
- The **Traffic Network Benchmark** uses a **Physical Traffic Graph** before learned or correlation-based adjacency variants
- The **Traffic Network Benchmark** uses **Dataset-Provided Traffic Adjacency** before reconstructing adjacency from sensor distances
- The **Traffic Network Benchmark** follows a **Traffic Edge Weight Policy** for weighted versus unweighted message passing
- The **Traffic Network Benchmark** uses a **Temporal Traffic Split** as the primary generalization setting
- The **Distributional Calibration Claim** is evaluated primarily with NLL, CRPS, and interval/PIT calibration, with point accuracy as a guardrail
- The **Tracer-Bullet Interpretability Bundle** is produced before attention or post-hoc GNN explanation methods
- A **Parameter-Specific Graph Effect** is tested by allowing graph contributions to location/mean only, scale/dispersion only, or both
- The first Bayesian implementation uses a **Bayesian Graph Stream** while keeping the local NAMLSS stream point-estimated
- A **Structural Uncertainty Experiment** comes after the primary GraphNAMLSS variants are working
- Local NAMLSS predictors use **Centered Local Shape Function** components before adding graph-derived effects
- The **Medical Application Track** motivates applied relevance after the **Traffic Network Benchmark** establishes the controlled method comparison
- The **Medical Application Track** should focus on a **Spatial Health Graph** rather than molecular or generic patient-similarity graphs
- A **Spatial Health Count Outcome** should be modeled with exposure-aware count distributions such as Negative Binomial

## Example Dialogue

> **Dev:** "When we say **GraphNAMLSS**, are we committing to one shared graph encoder?"
> **Domain expert:** "No — **GraphNAMLSS** is the family; the shared-encoder and per-parameter variants are both members we compare."
>
> **Dev:** "If the per-parameter variant performs better, can we claim separate graph structure matters?"
> **Domain expert:** "Only after controlling for capacity — the main contrast is **Graph Representation Sharing**, not model size."
>
> **Dev:** "Are we only adding a graph term after the local NAMLSS predictor?"
> **Domain expert:** "No — we compare **Additive Graph Contribution** and **Graph-Aware Shape Function** as distinct **Graph Injection Strategy** choices."
>
> **Dev:** "How does the additive graph term enter the predictor?"
> **Domain expert:** "Use a **Graph Contribution Gate** per distributional parameter so graph reliance is learnable and inspectable."
>
> **Dev:** "Can the graph contribution carry the intercept?"
> **Domain expert:** "No — use **Centered Graph Contribution** components so positive or negative graph effects have a stable interpretation."
>
> **Dev:** "What graph context goes into a graph-aware shape function?"
> **Domain expert:** "Use a **Feature-Wise Neighbor Summary** so each shape function remains interpretable as a low-dimensional surface."
>
> **Dev:** "Can the additive graph stream see the node's own features?"
> **Domain expert:** "Compare both in the **Primary Experiment Matrix**, but make **Graph Stream Self-Feature Access** explicit so the graph term is not mistaken for purely neighborhood information."
>
> **Dev:** "What does self-feature access mean for graph-aware shape functions?"
> **Domain expert:** "It distinguishes **Neighbor-Only Shape Context** from **Self-Conditioned Shape Context**, with **Context Effect Centering** to protect the local shape functions."
>
> **Dev:** "Can we compare all primary architecture cells directly?"
> **Domain expert:** "Yes, if they share one **Canonical Base Configuration** so the only intended differences are the three experimental factors."
>
> **Dev:** "If a per-parameter model wins, do we know the architecture is better?"
> **Domain expert:** "Not from the **Same-Recipe Comparison** alone — use a **Parameter-Matched Check** to see whether capacity explains the win."
>
> **Dev:** "Should the Bayesian local baseline be Bayesian NAMLSS?"
> **Domain expert:** "No — compare against **GAMLSS Baseline** and **BAMLSS Baseline** as established statistical distributional models."
>
> **Dev:** "Do statistical baselines get graph information?"
> **Domain expert:** "Report both plain statistical baselines and **Graph-Feature Statistical Baseline** variants with neighbor-summary covariates."
>
> **Dev:** "Why not just use a GNN with a distributional head?"
> **Domain expert:** "That is the **Distributional GNN Baseline**; GraphNAMLSS must justify its additive interpretability against it."
>
> **Dev:** "Should we compare against a mean-only GNN?"
> **Domain expert:** "Yes, but only as the **Point GNN Reference**, because it does not address the distributional task."
>
> **Dev:** "Should we implement all eight primary matrix cells first?"
> **Domain expert:** "No — start with a **Tracer-Bullet Slice** to validate the experiment harness end to end."
>
> **Dev:** "Which GraphNAMLSS cell goes into the tracer-bullet slice?"
> **Domain expert:** "Use **Tracer-Bullet GraphNAMLSS**: shared representation, additive contribution, neighbor-focused stream, deterministic first."
>
> **Dev:** "Should the first graph encoder be GAT for attention visualizations?"
> **Domain expert:** "No — use the **Tracer-Bullet Graph Encoder** first, then add attention after the harness works."
>
> **Dev:** "Should Bayesian graph weights be in the first implementation?"
> **Domain expert:** "No — follow the **Implementation Milestone Sequence** so deterministic GraphNAMLSS validates the harness before Bayesian graph weights are added."
>
> **Dev:** "Should this repo wrap an older NAMLSS or BANDS codebase?"
> **Domain expert:** "No — make it a **Self-Contained Experiment Package** and use **External Code Adapter** boundaries only where reuse is genuinely worth it."
>
> **Dev:** "Should we design a sophisticated framework up front?"
> **Domain expert:** "No — start with the **Initial Package Structure** so the experiment contract stays visible."
>
> **Dev:** "Should setup depend on Poetry or uv?"
> **Domain expert:** "No — use a **Plain Pyproject Workflow** and document the **Explicit PyG Dependency Policy**."
>
> **Dev:** "Should we use Hydra, W&B, or Sacred from the start?"
> **Domain expert:** "No — begin with a **Simple Experiment Config** so the tracer-bullet remains transparent."
>
> **Dev:** "Should self-feature access wait for a secondary ablation?"
> **Domain expert:** "No — **Graph Stream Self-Feature Access** belongs in the **Primary Experiment Matrix**."
>
> **Dev:** "Should the first benchmark be traffic, molecules, energy, or medicine?"
> **Domain expert:** "Use the **Traffic Network Benchmark** first because it is public, graph-native, and already legible to GNN reviewers."
>
> **Dev:** "Which traffic dataset comes first?"
> **Domain expert:** "Use the **METR-LA Tracer Dataset** first, then use the **PEMS-BAY Replication Dataset** to check generalization."
>
> **Dev:** "Should the first traffic model use a flexible non-Gaussian family?"
> **Domain expert:** "No — begin with a **Traffic Location-Scale Outcome** so calibration improvements are easy to interpret."
>
> **Dev:** "Can the model output sigma directly?"
> **Domain expert:** "No — use a **Positive Scale Link** so Gaussian scale stays strictly positive and numerically stable."
>
> **Dev:** "Are we building a spatiotemporal GNN?"
> **Domain expert:** "Not first — the first traffic setup uses **Traffic Sensor-Time Observations** with lagged features and a spatial graph."
>
> **Dev:** "Should the first traffic model predict many future horizons?"
> **Domain expert:** "No — begin with the **Fifteen-Minute Traffic Horizon** and add longer horizons after the harness works."
>
> **Dev:** "What local features should the traffic NAMLSS stream see?"
> **Domain expert:** "Use the **Traffic Local Covariate Set**: recent lags plus calendar features, so local shape plots stay interpretable."
>
> **Dev:** "Should the graph stream also receive calendar features?"
> **Domain expert:** "Not first — use the **Traffic Neighbor Covariate Set** so the graph contribution reflects neighboring traffic state."
>
> **Dev:** "Can we just forward-fill missing sensor readings?"
> **Domain expert:** "Only within the **Traffic Missingness Policy** — never silently and never using future information across split boundaries."
>
> **Dev:** "Can we compute NLL on normalized targets and report it directly?"
> **Domain expert:** "Only if the **Traffic Scaling Policy** labels the units clearly; original-unit calibration is preferred for interpretability."
>
> **Dev:** "Should the first traffic graph be learned from speed correlations?"
> **Domain expert:** "No — use a **Physical Traffic Graph** first so the graph effect is not a hidden target-derived shortcut."
>
> **Dev:** "Should we reconstruct the traffic graph ourselves?"
> **Domain expert:** "No — use **Dataset-Provided Traffic Adjacency** first for comparability, then reconstruct only as a fallback or robustness check."
>
> **Dev:** "Should the graph encoder use traffic edge weights?"
> **Domain expert:** "Use the **Traffic Edge Weight Policy**: prefer dataset-provided weights when cleanly supported, with an unweighted fallback."
>
> **Dev:** "Can we randomly split sensor-time observations?"
> **Domain expert:** "Not for the canonical benchmark — use a **Temporal Traffic Split** so test performance reflects future-time generalization."
>
> **Dev:** "Should the main claim be lower RMSE?"
> **Domain expert:** "No — the main claim is the **Distributional Calibration Claim**; RMSE or MAE is a guardrail."
>
> **Dev:** "What interpretability outputs come first?"
> **Domain expert:** "Use the **Tracer-Bullet Interpretability Bundle** before adding attention or post-hoc GNN explanations."
>
> **Dev:** "Do we just show that graph information helps?"
> **Domain expert:** "No — test the **Parameter-Specific Graph Effect** by asking which distributional parameters actually need graph information."
>
> **Dev:** "Is the entire model Bayesian from the start?"
> **Domain expert:** "No — start with a **Bayesian Graph Stream** and keep the local additive stream point-estimated."
>
> **Dev:** "Can DropEdge alone justify a structural uncertainty claim?"
> **Domain expert:** "No — reserve that for a **Structural Uncertainty Experiment** that explicitly models or perturbs the graph topology."
>
> **Dev:** "Can local shape functions be uncentered?"
> **Domain expert:** "Not in the canonical implementation — use **Centered Local Shape Function** components so contribution plots remain meaningful."
>
> **Dev:** "Should medicine replace traffic as the first experiment?"
> **Domain expert:** "No — medicine is the **Medical Application Track**, while traffic remains the controlled first benchmark."
>
> **Dev:** "What kind of medical graph do we mean?"
> **Domain expert:** "A **Spatial Health Graph**: regions or care sites linked by geography, mobility, referral, or adjacency."
>
> **Dev:** "What should we predict in the medical application?"
> **Domain expert:** "A **Spatial Health Count Outcome**, because overdispersed counts make the distributional part of GraphNAMLSS substantively meaningful."

## Flagged Ambiguities

- "GraphNAMLSS" was used both for the model family and for the shared-encoder architecture — resolved: **GraphNAMLSS** names the family.
- "shared vs separate GNNs" could mean many architectural differences — resolved: the controlled factor is **Graph Representation Sharing**.
- "inject graph structure" was used for both adding a separate graph term and conditioning shape functions on graph context — resolved: the controlled factor is **Graph Injection Strategy**.
- "graph contribution weight" could mean edge weights, graph head weights, or a scalar predictor gate — resolved: use **Graph Contribution Gate** for the scalar.
- "graph contribution" could include arbitrary intercept offsets — resolved: use **Centered Graph Contribution** components.
- "GNN graph term" could mean neighbor-focused information or unrestricted local-feature reuse — resolved: treat **Graph Stream Self-Feature Access** as an explicit factor.
- "graph context for shape functions" could mean simple neighbor summaries or dense learned embeddings — resolved: start with **Feature-Wise Neighbor Summary**.
- "self-feature access" was clear for additive graph streams but ambiguous for graph-aware shape functions — resolved: distinguish **Neighbor-Only Shape Context** from **Self-Conditioned Shape Context**.
- "graph-aware shape function" could absorb ordinary local effects — resolved: use **Context Effect Centering** to preserve the local/context distinction.
- "2x2 comparison" excluded graph stream self-feature access — resolved: the **Primary Experiment Matrix** is a 2x2x2 comparison under one **Canonical Base Configuration**.
- "fair comparison" could mean same hyperparameters or same parameter count — resolved: report both a **Same-Recipe Comparison** and a **Parameter-Matched Check**.
- "Bayesian local baseline" could mean Bayesian NAMLSS or established Bayesian distributional regression — resolved: use **BAMLSS Baseline** for the Bayesian statistical comparison.
- "statistical baseline" could mean with or without graph-derived covariates — resolved: include plain and **Graph-Feature Statistical Baseline** variants.
- "GNN baseline" could mean point prediction or distributional prediction — resolved: use a **Distributional GNN Baseline** for the main neural graph comparison.
- "point GNN" could be mistaken for a distributional competitor — resolved: treat it as a **Point GNN Reference**.
- "first implementation" could mean the full primary matrix or a minimal end-to-end experiment — resolved: begin with a **Tracer-Bullet Slice**.
- "first GraphNAMLSS model" could refer to any primary matrix cell — resolved: use **Tracer-Bullet GraphNAMLSS**.
- "first graph encoder" could be GraphSAGE or GAT — resolved: use the **Tracer-Bullet Graph Encoder** first.
- "Bayesian-first implementation" would mix architecture debugging with variational inference debugging — resolved: follow the **Implementation Milestone Sequence**.
- "implementation repo" could mean a wrapper around existing NAMLSS/BANDS code or a local experiment package — resolved: build a **Self-Contained Experiment Package**.
- "package structure" could become over-abstracted early — resolved: begin with the **Initial Package Structure**.
- "dependency management" could require a tool-specific workflow — resolved: use a **Plain Pyproject Workflow** with an **Explicit PyG Dependency Policy**.
- "experiment config" could become a framework decision early — resolved: start with a **Simple Experiment Config**.
- "first traffic dataset" could mean METR-LA or PEMS-BAY — resolved: start with the **METR-LA Tracer Dataset** and then use the **PEMS-BAY Replication Dataset**.
- "first benchmark domain" was open across traffic, molecules, energy, and medicine — resolved: start with the **Traffic Network Benchmark**.
- "traffic response distribution" was open across Gaussian, heavy-tailed, and asymmetric families — resolved: begin with a **Traffic Location-Scale Outcome**.
- "Gaussian scale output" could be unconstrained or positive by construction — resolved: use a **Positive Scale Link**.
- "traffic modeling task" could mean full spatiotemporal sequence forecasting or graph-structured distributional regression — resolved: begin with **Traffic Sensor-Time Observations**.
- "traffic horizon" could mean next step, 15 minutes, or multi-horizon forecasting — resolved: start with the **Fifteen-Minute Traffic Horizon**.
- "traffic features" could mean every engineered signal or a controlled local set — resolved: begin with the **Traffic Local Covariate Set**.
- "graph-stream traffic features" could include calendar and local time context — resolved: begin with the **Traffic Neighbor Covariate Set**.
- "missing traffic readings" could be imputed in ways that leak future information — resolved: define and report a **Traffic Missingness Policy**.
- "traffic likelihood units" could be standardized or original — resolved: follow the **Traffic Scaling Policy** and label units explicitly.
- "traffic graph" could mean physical adjacency or learned correlation — resolved: begin with a **Physical Traffic Graph**.
- "physical traffic adjacency" could mean standard dataset artifacts or a reconstructed graph — resolved: use **Dataset-Provided Traffic Adjacency** first.
- "traffic edge weights" could be mandatory or ignored — resolved: follow the **Traffic Edge Weight Policy**.
- "traffic split" could mean random observations, unseen sensors, or future times — resolved: use a **Temporal Traffic Split** first.
- "performance" could mean point accuracy or distribution quality — resolved: prioritize the **Distributional Calibration Claim**.
- "interpretability outputs" could sprawl into attention and GNNExplainer early — resolved: start with the **Tracer-Bullet Interpretability Bundle**.
- "graph helps uncertainty" could mean several things — resolved: test the **Parameter-Specific Graph Effect** at the distributional-parameter level.
- "Bayesian GraphNAMLSS" could imply the whole model is Bayesian — resolved: the first implementation uses a **Bayesian Graph Stream**.
- "structural uncertainty" could be overclaimed from generic edge dropout — resolved: reserve it for a later **Structural Uncertainty Experiment**.
- "shape function contribution" could include arbitrary offsets — resolved: use **Centered Local Shape Function** components.
- "medical GNN motivation" could displace the method benchmark — resolved: medicine is the **Medical Application Track**, not the first canonical benchmark.
- "medical graph" could mean molecules, patients, anatomy, or regions — resolved: the **Medical Application Track** focuses on a **Spatial Health Graph**.
- "medical response variable" could mean risk score, rate, or count — resolved: target a **Spatial Health Count Outcome** with an exposure term.
