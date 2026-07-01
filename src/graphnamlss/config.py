from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    train: float
    validation: float
    test: float


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
class ExperimentConfig:
    dataset: DatasetConfig
    split: SplitConfig
    features: FeatureConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    output: OutputConfig

    def to_resolved_dict(self) -> dict[str, Any]:
        return {
            "dataset": {
                "name": self.dataset.name,
                "task": self.dataset.task,
                "adjacency": self.dataset.adjacency,
            },
            "split": {
                "strategy": self.split.strategy,
                "train": self.split.train,
                "validation": self.split.validation,
                "test": self.split.test,
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

    return ExperimentConfig(
        dataset=DatasetConfig(
            name=_required_str(dataset, "dataset.name"),
            task=_required_str(dataset, "dataset.task"),
            adjacency=_required_str(dataset, "dataset.adjacency"),
        ),
        split=SplitConfig(
            strategy=_required_str(split, "split.strategy"),
            train=_required_float(split, "split.train"),
            validation=_required_float(split, "split.validation"),
            test=_required_float(split, "split.test"),
        ),
        features=FeatureConfig(
            local=_required_str_tuple(features, "features.local"),
            neighbor=_required_str_tuple(features, "features.neighbor"),
        ),
        model=ModelConfig(
            family=_required_str(model, "model.family"),
            variant=_required_str(model, "model.variant"),
            distribution=_required_str(model, "model.distribution"),
            graph_representation=_required_str(model, "model.graph_representation"),
            graph_injection=_required_str(model, "model.graph_injection"),
            graph_stream_self_feature_access=_required_str(
                model, "model.graph_stream_self_feature_access"
            ),
            graph_encoder=_required_str(model, "model.graph_encoder"),
        ),
        optimizer=OptimizerConfig(
            name=_required_str(optimizer, "optimizer.name"),
            learning_rate=_required_float(optimizer, "optimizer.learning_rate"),
            max_epochs=_required_int(optimizer, "optimizer.max_epochs"),
        ),
        output=OutputConfig(
            directory=_resolve_output_directory(
                config_path, _required_str(output, "output.directory")
            ),
            run_name=_required_str(output, "output.run_name"),
        ),
    )


def validate_experiment_config(config: ExperimentConfig) -> None:
    split_total = config.split.train + config.split.validation + config.split.test
    if abs(split_total - 1.0) > 1e-6:
        raise ConfigError(
            "split.train, split.validation, and split.test must sum to 1.0 "
            f"(got {split_total:g})."
        )


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required mapping: {key}.")
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


def _required_str_tuple(raw: dict[str, Any], dotted_key: str) -> tuple[str, ...]:
    value = _lookup(raw, dotted_key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ConfigError(f"Missing required non-empty string list: {dotted_key}.")
    return tuple(value)


def _lookup(raw: dict[str, Any], dotted_key: str) -> Any:
    key = dotted_key.rsplit(".", maxsplit=1)[-1]
    return raw.get(key)


def _resolve_output_directory(config_path: Path, directory: str) -> Path:
    output_path = Path(directory).expanduser()
    if output_path.is_absolute():
        return output_path
    return (config_path.parent / output_path).resolve()
