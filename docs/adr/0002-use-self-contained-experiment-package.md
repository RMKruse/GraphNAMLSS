# Use a self-contained experiment package

GraphNAMLSS will be implemented as a self-contained PyTorch/PyG experiment package rather than as a wrapper around an existing NAMLSS or BANDS codebase. This keeps the traffic data pipeline, model variants, training loops, evaluation, and plots reproducible inside this repo while still allowing narrow external-code adapters for reusable NAMLSS or BANDS components when the boundary is explicit.
