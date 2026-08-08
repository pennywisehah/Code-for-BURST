from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    seed: int = 42
    dataset: str = "mnist"
    data_dir: str = "data"
    download: bool = True
    iid: bool = False
    non_iid_alpha: float = 0.5
    num_clients: int = 20
    malicious_fraction: float = 0.2
    participation_fraction: float = 1.0
    aggregation: str = "fedavg"
    rounds: int = 30
    local_epochs: int = 2
    learning_rate: float = 0.01
    batch_size: int = 64
    eval_batch_size: int = 256
    momentum: float = 0.9
    weight_decay: float = 0.0005
    num_workers: int = 0
    device: str = "auto"
    attack: str = "sign_flip"
    attack_scale: float = 5.0
    trim_ratio: float = 0.2
    krum_byzantine: int | None = None
    aggregation_chunk_size: int = 1_000_000
    save_unlearning_history: bool = False
    unlearning_delta_t: int = 5
    output_dir: str = "runs"

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            values = json.load(file)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if value is not None:
                setattr(self, key, value)

    def validate(self) -> None:
        self.dataset = self.dataset.lower()
        if self.dataset not in {"mnist", "fashionmnist", "cifar10", "cifar100"}:
            raise ValueError(
                "dataset must be mnist, fashionmnist, cifar10, or cifar100"
            )
        if self.num_clients < 1:
            raise ValueError("num_clients must be at least 1")
        if not 0 <= self.malicious_fraction < 1:
            raise ValueError("malicious_fraction must be in [0, 1)")
        if not 0 < self.participation_fraction <= 1:
            raise ValueError("participation_fraction must be in (0, 1]")
        if self.aggregation not in {"fedavg", "krum", "trimmed_mean"}:
            raise ValueError("aggregation must be fedavg, krum, or trimmed_mean")
        if self.attack not in {"none", "sign_flip", "gaussian", "model_replacement"}:
            raise ValueError(
                "attack must be none, sign_flip, gaussian, or model_replacement"
            )
        if self.rounds < 1 or self.local_epochs < 1:
            raise ValueError("rounds and local_epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch_size and eval_batch_size must be positive")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if self.non_iid_alpha <= 0:
            raise ValueError("non_iid_alpha must be positive")
        if not 0 <= self.trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        if self.krum_byzantine is not None and self.krum_byzantine < 0:
            raise ValueError("krum_byzantine cannot be negative")
        if self.aggregation_chunk_size < 1:
            raise ValueError("aggregation_chunk_size must be positive")
        if self.unlearning_delta_t < 1:
            raise ValueError("unlearning_delta_t must be at least 1")
        if self.save_unlearning_history and self.aggregation != "fedavg":
            raise ValueError("FedEraser history currently requires FedAvg aggregation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
