from graphnamlss.config import (
    ExperimentConfig,
    load_experiment_config,
    validate_experiment_config,
)


def test_sample_tracer_bullet_config_loads_as_typed_experiment_config():
    config = load_experiment_config("configs/tracer_bullet.yaml")
    validate_experiment_config(config)

    assert isinstance(config, ExperimentConfig)
    assert config.dataset.name == "METR-LA"
    assert config.dataset.task == "fifteen_minute_ahead_speed"
    assert config.split.strategy == "temporal"
    assert config.features.local[0] == "current_speed"
    assert config.model.variant == "tracer_bullet"
    assert config.optimizer.name == "adam"
    assert config.output.run_name == "tracer-bullet-dry-run"
