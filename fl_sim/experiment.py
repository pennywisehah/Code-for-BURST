from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import torch

from .aggregation import aggregate
from .attacks import apply_attack
from .config import ExperimentConfig
from .data import load_federated_data
from .model import apply_update, build_model, evaluate, train_local
from .unlearning import FedEraserHistoryRecorder


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FederatedExperiment:
    def __init__(self, config: ExperimentConfig):
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.device = resolve_device(config.device)
        self.data = load_federated_data(
            dataset_name=config.dataset,
            data_dir=config.data_dir,
            num_clients=config.num_clients,
            iid=config.iid,
            non_iid_alpha=config.non_iid_alpha,
            download=config.download,
            seed=config.seed,
            partition=config.partition,
        )
        self.model = build_model(config.dataset)
        self.global_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }
        del self.model
        malicious_count = int(config.num_clients * config.malicious_fraction)
        ids = list(range(config.num_clients))
        self.rng.shuffle(ids)
        self.malicious_clients = set(ids[:malicious_count])
        self.attack_generator = torch.Generator().manual_seed(config.seed + 10_000)
        model_name = "CNN2" if config.dataset in {"mnist", "fashionmnist"} else "ResNet18"
        print(
            f"Dataset={config.dataset} | Model={model_name} | "
            f"Device={self.device} | Clients={config.num_clients} | "
            f"Partition={config.partition}"
        )
        if all(label is not None for label in self.data.client_labels):
            mapping = ", ".join(
                f"{client_id}->{label}"
                for client_id, label in enumerate(self.data.client_labels)
            )
            print(f"Client-to-label mapping: {mapping}")

    def _select_clients(self) -> list[int]:
        count = max(
            1, math.ceil(self.config.num_clients * self.config.participation_fraction)
        )
        return self.rng.sample(range(self.config.num_clients), count)

    def run(self) -> Path:
        run_dir = self._create_run_dir()
        history: list[dict[str, float | int | str]] = []
        unlearning_recorder = (
            FedEraserHistoryRecorder(
                self.global_state, self.config.unlearning_delta_t
            )
            if self.config.save_unlearning_history
            else None
        )
        for round_number in range(1, self.config.rounds + 1):
            selected = self._select_clients()
            updates = []
            sample_counts = []
            selected_malicious = 0
            for client_id in selected:
                update = train_local(
                    global_state=self.global_state,
                    dataset=self.data.clients[client_id],
                    model_name=self.config.dataset,
                    local_epochs=self.config.local_epochs,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    num_workers=self.config.num_workers,
                    device=self.device,
                    seed=self.config.seed + round_number * 100_000 + client_id,
                )
                if client_id in self.malicious_clients:
                    selected_malicious += 1
                    update = apply_attack(
                        update,
                        self.config.attack,
                        self.config.attack_scale,
                        len(selected),
                        self.attack_generator,
                    )
                updates.append(update)
                sample_counts.append(len(self.data.clients[client_id]))

            if unlearning_recorder is not None:
                unlearning_recorder.record_round(
                    round_number, selected, updates, sample_counts
                )

            assumed_byzantine = (
                self.config.krum_byzantine
                if self.config.krum_byzantine is not None
                else selected_malicious
            )
            aggregated = aggregate(
                method=self.config.aggregation,
                updates=updates,
                sample_counts=sample_counts,
                trim_ratio=self.config.trim_ratio,
                byzantine=assumed_byzantine,
                chunk_size=self.config.aggregation_chunk_size,
            )
            self.global_state = apply_update(self.global_state, aggregated)
            metrics = evaluate(
                state=self.global_state,
                dataset=self.data.test,
                model_name=self.config.dataset,
                batch_size=self.config.eval_batch_size,
                num_workers=self.config.num_workers,
                device=self.device,
            )
            record: dict[str, float | int | str] = {
                "round": round_number,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "selected_clients": len(selected),
                "malicious_clients": selected_malicious,
                "aggregation": self.config.aggregation,
                "dataset": self.config.dataset,
            }
            history.append(record)
            print(
                f"Round {round_number:03d}/{self.config.rounds} | "
                f"loss={metrics['loss']:.4f} | accuracy={metrics['accuracy']:.2%} | "
                f"clients={len(selected)} | malicious={selected_malicious}"
            )
        self._save_results(run_dir, history)
        if unlearning_recorder is not None:
            torch.save(
                unlearning_recorder.build_payload(
                    self.config.to_dict(), self.global_state
                ),
                run_dir / "federaser_history.pt",
            )
        print(f"Results saved to {run_dir}")
        return run_dir

    def _create_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir = Path(self.config.output_dir) / (
            f"{timestamp}-{self.config.dataset}-{self.config.aggregation}"
            f"-seed{self.config.seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _save_results(
        self, run_dir: Path, history: list[dict[str, float | int | str]]
    ) -> None:
        with (run_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(self.config.to_dict(), file, indent=2)
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        torch.save(
            {
                "dataset": self.config.dataset,
                "state_dict": self.global_state,
                "config": self.config.to_dict(),
            },
            run_dir / "model.pt",
        )
