from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment config cannot be used."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    task: str
    adjacency: str
    artifacts: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    train: float
    validation: float
    test: float

    def boundaries(self) -> dict[str, list[float]]:
        train_end = _stable_boundary(self.train)
        validation_end = _stable_boundary(self.train + self.validation)
        return {
            "train": [0.0, train_end],
            "validation": [train_end, validation_end],
            "test": [validation_end, 1.0],
        }


@dataclass(frozen=True)
class FeatureConfig:
    local: tuple[str, ...]
    neighbor: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    family: str
    variant: str
    distribution: str
    graph_representation: str
    graph_injection: str
    graph_stream_self_feature_access: str
    graph_encoder: str


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    max_epochs: int


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    run_name: str


@dataclass(frozen=True)
class TemporalConfig:
    frequency_minutes: int
    horizon_minutes: int
    horizon_steps: int
    lag_minutes: tuple[int, ...]


@dataclass(frozen=True)
class ScalingConfig:
    mode: str
    fit_on: str
    target: str
    features: str


@dataclass(frozen=True)
class MissingnessConfig:
    target: str
    covariates: str
    future_looking_imputation: bool


@dataclass(frozen=True)
class GraphConfig:
    source: str
    format: str
    edge_weight: str
    directed: bool


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    split: SplitConfig
    features: FeatureConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    output: OutputConfig
    temporal: TemporalConfig | None = None
    scaling: ScalingConfig | None = None
    missingness: MissingnessConfig | None = None
    graph: GraphConfig | None = None

    def to_resolved_dict(self) -> dict[str, Any]:
        resolved = {
            "dataset": {
                "name": self.dataset.name,
                "task": self.dataset.task,
                "adjacency": self.dataset.adjacency,
                "artifacts": {
                    name: str(path) for name, path in self.dataset.artifacts.items()
                },
            },
            "split": {
                "strategy": self.split.strategy,
                "train": self.split.train,
                "validation": self.split.validation,
                "test": self.split.test,
                "boundaries": self.split.boundaries(),
            },
            "features": {
                "local": list(self.features.local),
                "neighbor": list(self.features.neighbor),
            },
            "model": {
                "family": self.model.family,
                "variant": self.model.variant,
                "distribution": self.model.distribution,
                "graph_representation": self.model.graph_representation,
                "graph_injection": self.model.graph_injection,
                "graph_stream_self_feature_access": self.model.graph_stream_self_feature_access,
                "graph_encoder": self.model.graph_encoder,
            },
            "optimizer": {
                "name": self.optimizer.name,
                "learning_rate": self.optimizer.learning_rate,
                "max_epochs": self.optimizer.max_epochs,
            },
            "output": {
                "directory": str(self.output.directory),
                "run_name": self.output.run_name,
            },
        }
        if self.temporal is not None:
            resolved["temporal"] = {
                "frequency_minutes": self.temporal.frequency_minutes,
                "horizon_minutes": self.temporal.horizon_minutes,
                "horizon_steps": self.temporal.horizon_steps,
                "lag_minutes": list(self.temporal.lag_minutes),
            }
        if self.scaling is not None:
            resolved["scaling"] = {
                "mode": self.scaling.mode,
                "fit_on": self.scaling.fit_on,
                "target": self.scaling.target,
                "features": self.scaling.features,
            }
        if self.missingness is not None:
            resolved["missingness"] = {
                "target": self.missingness.target,
                "covariates": self.missingness.covariates,
                "future_looking_imputation": (
                    self.missingness.future_looking_imputation
                ),
            }
        if self.graph is not None:
            resolved["graph"] = {
                "source": self.graph.source,
                "format": self.graph.format,
                "edge_weight": self.graph.edge_weight,
                "directed": self.graph.directed,
            }
        return resolved


def validate_experiment_artifacts(config: ExperimentConfig) -> None:
    missing_artifacts = {
        name: path for name, path in config.dataset.artifacts.items() if not path.exists()
    }
    if not missing_artifacts:
        return

    details = "\n".join(
        f"{name}: {path}" for name, path in sorted(missing_artifacts.items())
    )
    raise ConfigError(
        f"Missing required dataset artifacts for {config.dataset.name}:\n{details}"
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text())
    except OSError as exc:
        raise ConfigError(f"Could not read config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML config {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Experiment config must be a YAML mapping at the top level.")

    dataset = _required_mapping(raw, "dataset")
    split = _required_mapping(raw, "split")
    features = _required_mapping(raw, "features")
    model = _required_mapping(raw, "model")
    optimizer = _required_mapping(raw, "optimizer")
    output = _required_mapping(raw, "output")
    temporal = _optional_mapping(raw, "temporal")
    scaling = _optional_mapping(raw, "scaling")
    missingness = _optional_mapping(raw, "missingness")
    graph = _optional_mapping(raw, "graph")

    return ExperimentConfig(
        dataset=_load_dataset_config(config_path, dataset),
        split=_load_split_config(split),
        features=_load_feature_config(features),
        model=_load_model_config(model),
        optimizer=_load_optimizer_config(optimizer),
        output=_load_output_config(config_path, output),
        temporal=_load_temporal_config(temporal),
        scaling=_load_scaling_config(scaling),
        missingness=_load_missingness_config(missingness),
        graph=_load_graph_config(graph),
    )


def _load_dataset_config(config_path: Path, raw: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        name=_required_str(raw, "dataset.name"),
        task=_required_str(raw, "dataset.task"),
        adjacency=_required_str(raw, "dataset.adjacency"),
        artifacts=_optional_path_mapping(config_path, raw, "dataset.artifacts"),
    )


def _load_split_config(raw: dict[str, Any]) -> SplitConfig:
    return SplitConfig(
        strategy=_required_str(raw, "split.strategy"),
        train=_required_float(raw, "split.train"),
        validation=_required_float(raw, "split.validation"),
        test=_required_float(raw, "split.test"),
    )


def _load_feature_config(raw: dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        local=_required_str_tuple(raw, "features.local"),
        neighbor=_required_str_tuple(raw, "features.neighbor"),
    )


def _load_model_config(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        family=_required_str(raw, "model.family"),
        variant=_required_str(raw, "model.variant"),
        distribution=_required_str(raw, "model.distribution"),
        graph_representation=_required_str(raw, "model.graph_representation"),
        graph_injection=_required_str(raw, "model.graph_injection"),
        graph_stream_self_feature_access=_required_str(
            raw, "model.graph_stream_self_feature_access"
        ),
        graph_encoder=_required_str(raw, "model.graph_encoder"),
    )


def _load_optimizer_config(raw: dict[str, Any]) -> OptimizerConfig:
    return OptimizerConfig(
        name=_required_str(raw, "optimizer.name"),
        learning_rate=_required_float(raw, "optimizer.learning_rate"),
        max_epochs=_required_int(raw, "optimizer.max_epochs"),
    )


def _load_output_config(config_path: Path, raw: dict[str, Any]) -> OutputConfig:
    return OutputConfig(
        directory=_resolve_output_directory(
            config_path, _required_str(raw, "output.directory")
        ),
        run_name=_required_str(raw, "output.run_name"),
    )


def _load_temporal_config(raw: dict[str, Any] | None) -> TemporalConfig | None:
    if raw is None:
        return None
    return TemporalConfig(
        frequency_minutes=_required_int(raw, "temporal.frequency_minutes"),
        horizon_minutes=_required_int(raw, "temporal.horizon_minutes"),
        horizon_steps=_required_int(raw, "temporal.horizon_steps"),
        lag_minutes=_required_int_tuple(raw, "temporal.lag_minutes"),
    )


def _load_scaling_config(raw: dict[str, Any] | None) -> ScalingConfig | None:
    if raw is None:
        return None
    return ScalingConfig(
        mode=_required_str(raw, "scaling.mode"),
        fit_on=_required_str(raw, "scaling.fit_on"),
        target=_required_str(raw, "scaling.target"),
        features=_required_str(raw, "scaling.features"),
    )


def _load_missingness_config(raw: dict[str, Any] | None) -> MissingnessConfig | None:
    if raw is None:
        return None
    return MissingnessConfig(
        target=_required_str(raw, "missingness.target"),
        covariates=_required_str(raw, "missingness.covariates"),
        future_looking_imputation=_required_bool(
            raw, "missingness.future_looking_imputation"
        ),
    )


def _load_graph_config(raw: dict[str, Any] | None) -> GraphConfig | None:
    if raw is None:
        return None
    return GraphConfig(
        source=_required_str(raw, "graph.source"),
        format=_required_str(raw, "graph.format"),
        edge_weight=_required_str(raw, "graph.edge_weight"),
        directed=_required_bool(raw, "graph.directed"),
    )


def validate_experiment_config(config: ExperimentConfig) -> None:
    split_total = config.split.train + config.split.validation + config.split.test
    if abs(split_total - 1.0) > 1e-6:
        raise ConfigError(
            "split.train, split.validation, and split.test must sum to 1.0 "
            f"(got {split_total:g})."
        )
    if config.temporal is not None:
        expected_horizon = (
            config.temporal.frequency_minutes * config.temporal.horizon_steps
        )
        if expected_horizon != config.temporal.horizon_minutes:
            raise ConfigError(
                "temporal.horizon_steps must match temporal.horizon_minutes "
                "at temporal.frequency_minutes resolution."
            )


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required mapping: {key}.")
    return value


def _optional_mapping(raw: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"Expected mapping: {key}.")
    return value


def _required_str(raw: dict[str, Any], dotted_key: str) -> str:
    value = _lookup(raw, dotted_key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing required string: {dotted_key}.")
    return value


def _required_float(raw: dict[str, Any], dotted_key: str) -> float:
    value = _lookup(raw, dotted_key)
    if not isinstance(value, int | float):
        raise ConfigError(f"Missing required number: {dotted_key}.")
    return float(value)


def _required_int(raw: dict[str, Any], dotted_key: str) -> int:
    value = _lookup(raw, dotted_key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"Missing required integer: {dotted_key}.")
    return value


def _required_bool(raw: dict[str, Any], dotted_key: str) -> bool:
    value = _lookup(raw, dotted_key)
    if not isinstance(value, bool):
        raise ConfigError(f"Missing required boolean: {dotted_key}.")
    return value


def _required_str_tuple(raw: dict[str, Any], dotted_key: str) -> tuple[str, ...]:
    value = _lookup(raw, dotted_key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ConfigError(f"Missing required non-empty string list: {dotted_key}.")
    return tuple(value)


def _required_int_tuple(raw: dict[str, Any], dotted_key: str) -> tuple[int, ...]:
    value = _lookup(raw, dotted_key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ConfigError(f"Missing required non-empty integer list: {dotted_key}.")
    return tuple(value)


def _lookup(raw: dict[str, Any], dotted_key: str) -> Any:
    key = dotted_key.rsplit(".", maxsplit=1)[-1]
    return raw.get(key)


def _optional_path_mapping(
    config_path: Path, raw: dict[str, Any], dotted_key: str
) -> dict[str, Path]:
    value = _lookup(raw, dotted_key)
    if value is None:
        return {}
    if (
        not isinstance(value, dict)
        or not value
        or not all(
            isinstance(name, str)
            and name.strip()
            and isinstance(path, str)
            and path.strip()
            for name, path in value.items()
        )
    ):
        raise ConfigError(f"Expected non-empty path mapping: {dotted_key}.")
    return {
        name: _resolve_config_relative_path(config_path, path)
        for name, path in value.items()
    }


def _resolve_output_directory(config_path: Path, directory: str) -> Path:
    output_path = Path(directory).expanduser()
    if output_path.is_absolute():
        return output_path
    return (config_path.parent / output_path).resolve()


def _resolve_config_relative_path(config_path: Path, path: str) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    return (config_path.parent / resolved_path).resolve()


def _stable_boundary(value: float) -> float:
    return round(value, 10)
