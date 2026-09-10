from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from .aggregation import fedavg
from .data import load_federated_data, unique_dataset_labels
from .model import (
    StateDict,
    apply_update,
    evaluate,
    evaluate_per_class,
    train_local,
)


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


class FedEraserHistoryRecorder:
    """Accumulate each client's updates over delta_t-round calibration windows."""

    def __init__(self, initial_state: Mapping[str, torch.Tensor], delta_t: int):
        if delta_t < 1:
            raise ValueError("FedEraser delta_t must be at least 1.")
        self.delta_t = delta_t
        self.initial_state = _clone_state(initial_state)
        self.snapshots: list[dict] = []
        self._window_start_round: int | None = None
        self._last_round: int | None = None
        self._window_client_ids: list[int] = []
        self._window_updates: dict[int, dict[str, torch.Tensor]] = {}
        self._window_sample_counts: dict[int, int] = {}

    def _flush_window(self, end_round: int) -> None:
        if self._window_start_round is None or not self._window_updates:
            raise RuntimeError("Cannot flush an empty FedEraser history window.")
        client_ids = list(self._window_client_ids)
        self.snapshots.append(
            {
                "start_round": self._window_start_round,
                "round": int(end_round),
                "interval_round_count": (
                    int(end_round) - self._window_start_round + 1
                ),
                "client_ids": client_ids,
                "sample_counts": [
                    self._window_sample_counts[client_id]
                    for client_id in client_ids
                ],
                "updates": [
                    self._window_updates[client_id]
                    for client_id in client_ids
                ],
            }
        )
        self._window_start_round = None
        self._window_client_ids = []
        self._window_updates = {}
        self._window_sample_counts = {}

    def record_round(
        self,
        round_number: int,
        client_ids: Sequence[int],
        updates: Sequence[Mapping[str, torch.Tensor]],
        sample_counts: Sequence[int],
    ) -> None:
        if not (len(client_ids) == len(updates) == len(sample_counts)):
            raise ValueError("Client ids, updates, and sample counts must be aligned.")
        if self._last_round is not None and round_number != self._last_round + 1:
            raise ValueError("FedEraser history rounds must be consecutive.")
        if len(set(int(client_id) for client_id in client_ids)) != len(client_ids):
            raise ValueError("A client can appear at most once in each training round.")
        if self._window_start_round is None:
            self._window_start_round = int(round_number)

        for raw_client_id, update, raw_sample_count in zip(
            client_ids, updates, sample_counts
        ):
            client_id = int(raw_client_id)
            if client_id not in self._window_updates:
                self._window_client_ids.append(client_id)
                self._window_updates[client_id] = _clone_state(update)
            else:
                accumulated = self._window_updates[client_id]
                if tuple(accumulated) != tuple(update):
                    raise ValueError(
                        "A client's interval updates must have identical keys."
                    )
                for name, value in update.items():
                    accumulated[name].add_(value.detach().cpu())
            self._window_sample_counts[client_id] = int(raw_sample_count)

        self._last_round = int(round_number)
        if round_number % self.delta_t == 0:
            self._flush_window(round_number)

    def build_payload(
        self,
        config: Mapping,
        final_state: Mapping[str, torch.Tensor],
    ) -> dict:
        if self._last_round is None:
            raise RuntimeError("Cannot save an empty FedEraser history.")
        if self._window_start_round is not None:
            self._flush_window(self._last_round)
        return {
            "format_version": 2,
            "method": "federaser",
            "delta_t": self.delta_t,
            "history_update_semantics": "per_client_interval_sum",
            "config": dict(config),
            "initial_state": self.initial_state,
            "final_state": _clone_state(final_state),
            "snapshots": self.snapshots,
        }


def calibrate_update_direction(
    historical_update: Mapping[str, torch.Tensor],
    new_update: Mapping[str, torch.Tensor],
    norm_epsilon: float = 1e-12,
) -> StateDict:
    """Use each historical layer norm with the corresponding new update direction."""
    if tuple(historical_update) != tuple(new_update):
        raise ValueError("Historical and new updates must have identical keys.")
    calibrated: StateDict = OrderedDict()
    for name in historical_update:
        old_value = historical_update[name]
        new_value = new_update[name]
        old_norm = torch.linalg.vector_norm(old_value.float())
        new_norm = torch.linalg.vector_norm(new_value.float())
        if new_norm.item() <= norm_epsilon:
            calibrated[name] = torch.zeros_like(new_value)
        else:
            scale = (old_norm / new_norm).to(dtype=new_value.dtype)
            calibrated[name] = new_value * scale
    return calibrated


def remove_local_samples(dataset: Dataset, local_indices: Sequence[int]) -> Subset:
    """Return a client dataset with the requested local positions removed."""
    forgotten = {int(index) for index in local_indices}
    if any(index < 0 or index >= len(dataset) for index in forgotten):
        raise IndexError("A forget index is outside the requesting client's dataset.")
    retained = [index for index in range(len(dataset)) if index not in forgotten]
    if not retained:
        raise ValueError(
            "Partial-data unlearning removed the entire client dataset; use "
            "forget_all_client_data instead."
        )
    return Subset(dataset, retained)


def federaser_unlearn(
    history: Mapping,
    client_datasets: Sequence[Dataset],
    forget_client_id: int,
    forget_local_indices: Sequence[int] | None,
    forget_all_client_data: bool,
    calibration_local_epochs: int,
    calibration_learning_rate: float,
    batch_size: int,
    momentum: float,
    weight_decay: float,
    num_workers: int,
    device: torch.device,
    seed: int,
    reconstruct_without_forgetting: bool = False,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]]]:
    """Run FedEraser calibration for client-level or partial client-data removal."""
    if history.get("method") != "federaser":
        raise ValueError("The supplied history is not a FedEraser history file.")
    config = history["config"]
    if str(config.get("aggregation", "")).lower() != "fedavg":
        raise ValueError("This FedEraser baseline currently requires FedAvg history.")
    if not 0 <= forget_client_id < len(client_datasets):
        raise ValueError("forget_client_id is outside the configured client range.")
    if calibration_local_epochs < 1 or calibration_learning_rate <= 0:
        raise ValueError("Calibration epochs and learning rate must be positive.")
    if reconstruct_without_forgetting and (forget_all_client_data or forget_local_indices):
        raise ValueError(
            "Reconstruction without forgetting cannot include a forget request."
        )
    if forget_all_client_data and forget_local_indices:
        raise ValueError("Choose either full-client or partial-data unlearning, not both.")
    if (
        not reconstruct_without_forgetting
        and not forget_all_client_data
        and not forget_local_indices
    ):
        raise ValueError("Partial-data unlearning requires at least one local index.")

    retained_datasets = list(client_datasets)
    if not reconstruct_without_forgetting and not forget_all_client_data:
        retained_datasets[forget_client_id] = remove_local_samples(
            retained_datasets[forget_client_id], forget_local_indices or []
        )

    unlearned_state = _clone_state(history["initial_state"])
    calibration_history: list[dict[str, float | int]] = []
    dataset_name = str(config["dataset"])
    total_snapshots = len(history["snapshots"])
    total_start = time.perf_counter()

    for snapshot_number, snapshot in enumerate(history["snapshots"], start=1):
        snapshot_start = time.perf_counter()
        interval_start = int(snapshot.get("start_round", snapshot["round"]))
        interval_end = int(snapshot["round"])
        interval_round_count = int(
            snapshot.get(
                "interval_round_count", interval_end - interval_start + 1
            )
        )
        retained_client_ids = [
            int(client_id)
            for client_id in snapshot["client_ids"]
            if not (forget_all_client_data and int(client_id) == forget_client_id)
        ]
        print(
            f"[FedEraser {snapshot_number:02d}/{total_snapshots:02d}] "
            f"history_interval={interval_start}-{interval_end} | "
            f"retained_clients={len(retained_client_ids)}",
            flush=True,
        )
        calibrated_updates = []
        retained_counts = []
        historical_norm_sum = 0.0
        calibrated_norm_sum = 0.0

        retained_client_number = 0
        for client_id, historical_update in zip(
            snapshot["client_ids"], snapshot["updates"]
        ):
            client_id = int(client_id)
            if forget_all_client_data and client_id == forget_client_id:
                continue
            retained_client_number += 1
            retained_dataset = retained_datasets[client_id]
            client_start = time.perf_counter()
            new_update = train_local(
                global_state=unlearned_state,
                dataset=retained_dataset,
                model_name=dataset_name,
                local_epochs=calibration_local_epochs,
                learning_rate=calibration_learning_rate,
                batch_size=batch_size,
                momentum=momentum,
                weight_decay=weight_decay,
                num_workers=num_workers,
                device=device,
                seed=seed + snapshot_number * 100_000 + client_id,
            )
            calibrated = calibrate_update_direction(historical_update, new_update)
            calibrated_updates.append(calibrated)
            retained_counts.append(len(retained_dataset))
            historical_norm_sum += sum(
                torch.linalg.vector_norm(value.float()).item()
                for value in historical_update.values()
            )
            calibrated_norm_sum += sum(
                torch.linalg.vector_norm(value.float()).item()
                for value in calibrated.values()
            )
            print(
                f"  client {retained_client_number:02d}/"
                f"{len(retained_client_ids):02d} | id={client_id} | "
                f"samples={len(retained_dataset)} | "
                f"elapsed={time.perf_counter() - client_start:.1f}s",
                flush=True,
            )

        if not calibrated_updates:
            raise RuntimeError(
                f"No retained client updates remain at history round {snapshot['round']}."
            )
        aggregated = fedavg(calibrated_updates, retained_counts)
        unlearned_state = apply_update(unlearned_state, aggregated)
        calibration_history.append(
            {
                "history_round": interval_end,
                "history_interval_start": interval_start,
                "history_interval_end": interval_end,
                "interval_round_count": interval_round_count,
                "retained_clients": len(calibrated_updates),
                "retained_samples": sum(retained_counts),
                "historical_layer_norm_sum": historical_norm_sum,
                "calibrated_layer_norm_sum": calibrated_norm_sum,
            }
        )
        print(
            f"  snapshot complete | elapsed={time.perf_counter() - snapshot_start:.1f}s | "
            f"total_elapsed={time.perf_counter() - total_start:.1f}s",
            flush=True,
        )

    return unlearned_state, calibration_history


def _parse_indices(raw: str | None, file_path: str | None) -> list[int]:
    indices: list[int] = []
    if raw:
        indices.extend(int(value.strip()) for value in raw.split(",") if value.strip())
    if file_path:
        values = json.loads(Path(file_path).read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError("The forget-indices file must contain a JSON list.")
        indices.extend(int(value) for value in values)
    return sorted(set(indices))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FedEraser calibration for full-client or partial-data unlearning."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--forget-client-id", type=int, required=True)
    parser.add_argument(
        "--forget-local-indices",
        help="Comma-separated positions inside the requesting client's local dataset.",
    )
    parser.add_argument("--forget-indices-file", help="JSON list of local positions.")
    parser.add_argument("--forget-all-client-data", action="store_true")
    parser.add_argument("--calibration-local-epochs", type=int, default=1)
    parser.add_argument("--calibration-learning-rate", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=None
    )
    return parser


def main() -> None:
    # Imported lazily to avoid a cycle: experiment uses the history recorder above.
    from .experiment import resolve_device

    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    history_path = run_dir / "federaser_history.pt"
    if not history_path.exists():
        raise FileNotFoundError(
            f"{history_path} does not exist; train with --save-unlearning-history."
        )
    history = torch.load(history_path, map_location="cpu", weights_only=False)
    config = history["config"]
    download = bool(config["download"]) if args.download is None else args.download
    data = load_federated_data(
        dataset_name=config["dataset"],
        data_dir=config["data_dir"],
        num_clients=int(config["num_clients"]),
        iid=bool(config["iid"]),
        non_iid_alpha=float(config["non_iid_alpha"]),
        download=download,
        seed=int(config["seed"]),
        partition=str(config.get("partition", "auto")),
    )
    forget_indices = _parse_indices(
        args.forget_local_indices, args.forget_indices_file
    )
    if not 0 <= args.forget_client_id < len(data.clients):
        raise ValueError("--forget-client-id is outside the configured client range.")
    if args.forget_all_client_data and forget_indices:
        raise ValueError("Do not combine full-client and partial-data forget options.")
    if not args.forget_all_client_data and not forget_indices:
        raise ValueError("Partial-data unlearning requires forget indices.")
    requesting_client_dataset = data.clients[args.forget_client_id]
    requesting_client_labels = unique_dataset_labels(requesting_client_dataset)
    forgotten_dataset: Dataset = (
        requesting_client_dataset
        if args.forget_all_client_data
        else Subset(requesting_client_dataset, forget_indices)
    )
    device = resolve_device(args.device)
    calibration_lr = (
        float(config["learning_rate"])
        if args.calibration_learning_rate is None
        else args.calibration_learning_rate
    )
    unlearned_state, calibration_history = federaser_unlearn(
        history=history,
        client_datasets=data.clients,
        forget_client_id=args.forget_client_id,
        forget_local_indices=forget_indices,
        forget_all_client_data=args.forget_all_client_data,
        calibration_local_epochs=args.calibration_local_epochs,
        calibration_learning_rate=calibration_lr,
        batch_size=int(config["batch_size"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
        num_workers=int(config["num_workers"]),
        device=device,
        seed=int(config["seed"]) + 1_000_000,
    )

    original_metrics = evaluate(
        history["final_state"],
        data.test,
        config["dataset"],
        int(config["eval_batch_size"]),
        int(config["num_workers"]),
        device,
    )
    unlearned_metrics = evaluate(
        unlearned_state,
        data.test,
        config["dataset"],
        int(config["eval_batch_size"]),
        int(config["num_workers"]),
        device,
    )
    original_per_label = evaluate_per_class(
        history["final_state"],
        data.test,
        config["dataset"],
        data.num_classes,
        int(config["eval_batch_size"]),
        int(config["num_workers"]),
        device,
    )
    unlearned_per_label = evaluate_per_class(
        unlearned_state,
        data.test,
        config["dataset"],
        data.num_classes,
        int(config["eval_batch_size"]),
        int(config["num_workers"]),
        device,
    )
    per_label_metrics = {}
    for label in range(data.num_classes):
        original_accuracy = original_per_label[str(label)]["accuracy"]
        unlearned_accuracy = unlearned_per_label[str(label)]["accuracy"]
        per_label_metrics[str(label)] = {
            "sample_count": original_per_label[str(label)]["sample_count"],
            "original_accuracy": original_accuracy,
            "unlearned_accuracy": unlearned_accuracy,
            "accuracy_drop": (
                original_accuracy - unlearned_accuracy
                if original_accuracy is not None and unlearned_accuracy is not None
                else None
            ),
        }
    print(
        f"[Forget Set] Evaluating {len(forgotten_dataset)} forgotten samples...",
        flush=True,
    )
    original_forget_metrics = evaluate(
        history["final_state"],
        forgotten_dataset,
        config["dataset"],
        int(config["eval_batch_size"]),
        int(config["num_workers"]),
        device,
    )
    unlearned_forget_metrics = evaluate(
        unlearned_state,
        forgotten_dataset,
        config["dataset"],
        int(config["eval_batch_size"]),
        int(config["num_workers"]),
        device,
    )
    print(
        f"[Forget Set] Accuracy: "
        f"{original_forget_metrics['accuracy']:.2%} -> "
        f"{unlearned_forget_metrics['accuracy']:.2%}",
        flush=True,
    )
    forgotten_label = (
        requesting_client_labels[0] if len(requesting_client_labels) == 1 else None
    )
    if forgotten_label is not None:
        forgotten_label_metrics = per_label_metrics[str(forgotten_label)]
        print(
            f"[Forgotten Label {forgotten_label}] Test accuracy: "
            f"{forgotten_label_metrics['original_accuracy']:.2%} -> "
            f"{forgotten_label_metrics['unlearned_accuracy']:.2%} | "
            f"drop={forgotten_label_metrics['accuracy_drop']:.2%}",
            flush=True,
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir
        / "unlearning_runs"
        / f"{timestamp}-federaser-client{args.forget_client_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "method": "federaser",
        "source_run": str(run_dir.resolve()),
        "delta_t": int(history["delta_t"]),
        "history_update_semantics": history.get(
            "history_update_semantics", "single_retained_round"
        ),
        "forget_client_id": args.forget_client_id,
        "forget_mode": "full_client" if args.forget_all_client_data else "partial_data",
        "forgotten_local_indices": forget_indices,
        "forgotten_sample_count": (
            len(data.clients[args.forget_client_id])
            if args.forget_all_client_data
            else len(forget_indices)
        ),
        "calibration_local_epochs": args.calibration_local_epochs,
        "calibration_learning_rate": calibration_lr,
        "original_test_metrics": original_metrics,
        "unlearned_test_metrics": unlearned_metrics,
        "forgotten_set_metrics": {
            "sample_count": len(forgotten_dataset),
            "original_model": original_forget_metrics,
            "unlearned_model": unlearned_forget_metrics,
        },
        "requesting_client_labels": requesting_client_labels,
        "forgotten_label": forgotten_label,
        "per_label_test_metrics": per_label_metrics,
        "forgotten_label_metrics": (
            per_label_metrics[str(forgotten_label)]
            if forgotten_label is not None
            else None
        ),
        "calibration_history": calibration_history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save(
        {
            "dataset": config["dataset"],
            "state_dict": unlearned_state,
            "source_run": str(run_dir.resolve()),
            "unlearning": summary,
        },
        output_dir / "unlearned_model.pt",
    )
    completed_mode = "full-client" if args.forget_all_client_data else "partial-data"
    print(
        f"FedEraser {completed_mode} unlearning complete | "
        f"delta_t={history['delta_t']} | "
        f"snapshots={len(history['snapshots'])}"
    )
    print(
        f"Test accuracy: {original_metrics['accuracy']:.2%} -> "
        f"{unlearned_metrics['accuracy']:.2%}"
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
