"""QLLM: modular quantum-classical hybrid language modeling in JAX/Flax.

The plugin architecture lets one training/eval pipeline run pure-classical
and hybrid models: quantum components are selected purely via config flags.
"""
from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    QuantumConfig,
    TrackingConfig,
    TrainConfig,
    from_dict,
    load_yaml,
    to_flat_dict,
)

__version__ = "0.17.0"

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "QuantumConfig",
    "TrackingConfig",
    "TrainConfig",
    "from_dict",
    "load_yaml",
    "to_flat_dict",
    "__version__",
]
