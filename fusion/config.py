"""Dataset-specific training configurations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Configuration:
    dataset: str
    dropout: float
    hsi_dilation: int
    lidar_dilation: int
    lidar_groups: int
    hsi_kernel: int
    lidar_kernel: int
    compact_head: bool
    decay_bound: float
    regional_strength: float
    local_strength: float
    operator: str
    learning_rate: float
    weight_decay: float
    smoothing: float
    class_weight: float
    sample_weight: float
    beta: float
    class_temperature: float
    regularization: bool
    ranking_weight: float = 0.1
    utility_weight: float = 0.03
    consistency_weight: float = 0.05
    sample_temperature: float = 0.1
    dimension: int = 64
    stride: int = 4


CONFIGURATIONS = {
    "muufl": Configuration(
        "muufl", 0.3, 1, 1, 0, 3, 3, False, 24.0, 4.0, 4.0,
        "torch", 0.001, 0.1, 0.05, 0.2, 0.6, 64.0, 0.1, False,
    ),
    "houston": Configuration(
        "houston", 0.1, 2, 12, 8, 5, 5, False, 24.0, 4.0, 4.0,
        "cuda", 0.001, 0.1, 0.05, 0.2, 0.2, 4.0, 0.075, True,
    ),
    "trento": Configuration(
        "trento", 0.1, 1, 4, 8, 3, 5, True, 96.0, 2.0, 0.25,
        "torch", 0.0018, 0.0001, 0.0, 0.3, 0.3, 64.0, 0.1, True,
        ranking_weight=0.05, utility_weight=0.06, consistency_weight=0.2,
    ),
}


def get_configuration(dataset):
    try:
        return CONFIGURATIONS[dataset.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown dataset: {dataset}") from error
