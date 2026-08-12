from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets
from torchvision.utils import save_image

from fl_sim.aggregation import fedavg
from fl_sim.config import ExperimentConfig
from fl_sim.data import _transforms, dataset_targets, load_federated_data
from fl_sim.experiment import resolve_device
from fl_sim.model import (
    apply_update,
    build_model,
    evaluate,
    evaluate_per_class,
    train_local,
)
from fl_sim.pood import (
    collect_candidates,
    cosine_knn,
    denormalize_mnist,
    extract_features,
    load_proxy_model,
    optimize_universal_perturbation,
    same_label_selection,
    selected_feature_metrics,
)
from fl_sim.unlearning import FedEraserHistoryRecorder, federaser_unlearn


DEFAULT_DIRICHLET_ALPHA = 0.1


class PoisonedClientDataset(Dataset):
    """Append repeated POOD samples while retaining their exact local positions."""

    def __init__(
        self,
        benign_dataset: Dataset,
        poison_images: torch.Tensor,
        poison_labels: torch.Tensor,
        repeats: int,
    ) -> None:
        if len(poison_images) != len(poison_labels) or len(poison_images) < 1:
            raise ValueError("POOD images and labels must be aligned and non-empty.")
        if repeats < 1:
            raise ValueError("poison repeats must be at least 1.")
        self.benign_dataset = benign_dataset
        self.poison_images = poison_images.detach().cpu()
        self.poison_labels = poison_labels.detach().cpu().to(torch.long)
        self.repeats = repeats
        self.benign_count = len(benign_dataset)

    def __len__(self) -> int:
        return self.benign_count + len(self.poison_images) * self.repeats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < self.benign_count:
            return self.benign_dataset[index]
        poison_index = (index - self.benign_count) % len(self.poison_images)
        return self.poison_images[poison_index], int(self.poison_labels[poison_index])

    @property
    def poison_local_indices(self) -> list[int]:
        return list(range(self.benign_count, len(self)))


def _clone_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def latest_recursive_checkpoint(run_root: str | Path) -> Path:
    checkpoints = sorted(
        Path(run_root).rglob("model.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No model.pt found under {Path(run_root)!s}; "
            "pass --proxy-checkpoint explicitly."
        )
    return checkpoints[-1]


@torch.no_grad()
def choose_target_for_attack(
    model,
    dataset: Dataset,
    target_label: int,
    target_index: int | None,
    device: torch.device,
) -> tuple[int, torch.Tensor, int, int]:
    if target_index is not None:
        image, label = dataset[target_index]
        label = int(label)
        if label != target_label:
            raise ValueError(
                f"MNIST index {target_index} has label {label}, not target label "
                f"{target_label}."
            )
        prediction = int(model(image.unsqueeze(0).to(device)).argmax(dim=1).item())
        if prediction != label:
            raise ValueError(
                "The requested target is not correctly classified by the proxy model."
            )
        return target_index, image, label, prediction

    for index in range(len(dataset)):
        image, label = dataset[index]
        label = int(label)
        if label != target_label:
            continue
        prediction = int(model(image.unsqueeze(0).to(device)).argmax(dim=1).item())
        if prediction == label:
            return index, image, label, prediction
    raise RuntimeError(
        f"Could not find a correctly classified target with label {target_label}."
    )


@torch.no_grad()
def evaluate_target(
    state: Mapping[str, torch.Tensor],
    image: torch.Tensor,
    true_label: int,
    device: torch.device,
) -> dict[str, float | int | bool]:
    model = build_model("mnist").to(device)
    model.load_state_dict(state)
    model.eval()
    logits = model(image.unsqueeze(0).to(device))
    probabilities = F.softmax(logits, dim=1).squeeze(0)
    prediction = int(probabilities.argmax().item())
    sorted_probabilities = torch.sort(probabilities, descending=True).values
    return {
        "true_label": true_label,
        "prediction": prediction,
        "correct": prediction == true_label,
        "true_label_confidence": float(probabilities[true_label].item()),
        "predicted_confidence": float(probabilities[prediction].item()),
        "top1_top2_margin": float(
            (sorted_probabilities[0] - sorted_probabilities[1]).item()
        ),
        "cross_entropy": float(
            F.cross_entropy(logits, torch.tensor([true_label], device=device)).item()
        ),
    }


def compare_per_label(
    before: Mapping[str, Mapping[str, float | int | None]],
    after: Mapping[str, Mapping[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    if set(before) != set(after):
        raise ValueError("Before/after per-label metrics must contain identical labels.")
    comparison: dict[str, dict[str, float | int | None]] = {}
    for label in sorted(before, key=int):
        before_accuracy = before[label]["accuracy"]
        after_accuracy = after[label]["accuracy"]
        comparison[label] = {
            "sample_count": int(before[label]["sample_count"]),
            "before_accuracy": before_accuracy,
            "after_accuracy": after_accuracy,
            "accuracy_drop": (
                before_accuracy - after_accuracy
                if before_accuracy is not None and after_accuracy is not None
                else None
            ),
        }
    return comparison


def label_histogram(dataset: Dataset, num_classes: int = 10) -> list[int]:
    targets = dataset_targets(dataset)
    return torch.bincount(targets, minlength=num_classes).tolist()


def checkpoint_accuracy(checkpoint_path: Path) -> float | None:
    metrics_path = checkpoint_path.parent / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    if len(rows) < 2:
        return None
    return float(rows[-1].split(",")[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end POOD unlearning attack under Dirichlet Non-IID "
            "federated learning."
        )
    )
    parser.add_argument("--proxy-checkpoint")
    parser.add_argument("--proxy-run-root", default="runs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="pua_runs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target-label", type=int, default=3)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--candidate-count", type=int, default=1000)
    # With the target label excluded, k=20 can easily contain fewer than five
    # examples from any one of the remaining nine labels.  A larger default
    # keeps the paper's same-label POOD selection well-defined for p=5.
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--perturb-steps", type=int, default=200)
    parser.add_argument("--perturb-lr", type=float, default=0.01)
    parser.add_argument("--perturb-epsilon", type=float, default=0.3)
    parser.add_argument("--feature-batch-size", type=int, default=256)

    parser.add_argument("--malicious-client-id", type=int, default=6)
    parser.add_argument("--poison-repeats", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument(
        "--non-iid-alpha", type=float, default=DEFAULT_DIRICHLET_ALPHA
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--unlearning-delta-t", type=int, default=5)
    parser.add_argument("--calibration-local-epochs", type=int, default=1)
    parser.add_argument("--calibration-learning-rate", type=float)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.target_label <= 9:
        raise ValueError("target label must be between 0 and 9.")
    if not 0 <= args.malicious_client_id < args.num_clients:
        raise ValueError("malicious client id is outside the client range.")
    if args.num_clients < 2:
        raise ValueError("at least two clients are required.")
    if args.non_iid_alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    if args.candidate_count < 1 or args.feature_batch_size < 1:
        raise ValueError("candidate count and feature batch size must be positive.")
    if not 1 <= args.p <= args.k <= args.candidate_count:
        raise ValueError("require 1 <= p <= k <= candidate count.")
    if args.perturb_steps < 1 or args.perturb_lr <= 0:
        raise ValueError("perturbation steps and learning rate must be positive.")
    if not 0 < args.perturb_epsilon <= 1:
        raise ValueError("perturbation epsilon must be in (0, 1].")
    if args.poison_repeats < 1:
        raise ValueError("poison repeats must be at least 1.")
    if args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("rounds and local epochs must be positive.")
    if args.learning_rate <= 0 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("training learning rate and batch sizes must be positive.")
    if args.unlearning_delta_t < 1 or args.calibration_local_epochs < 1:
        raise ValueError("unlearning interval and calibration epochs must be positive.")


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = Path(args.output_dir) / (
        f"{timestamp}-pua-dirichlet-alpha{args.non_iid_alpha:g}-"
        f"client{args.malicious_client_id}-target{args.target_label}-seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    proxy_checkpoint = (
        Path(args.proxy_checkpoint)
        if args.proxy_checkpoint
        else latest_recursive_checkpoint(args.proxy_run_root)
    )
    proxy_model, _ = load_proxy_model(proxy_checkpoint, device)
    _, eval_transform = _transforms("mnist")
    mnist_test = datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=eval_transform,
        download=args.download,
    )
    qmnist_pool = datasets.QMNIST(
        root=args.data_dir,
        what="test50k",
        compat=True,
        transform=eval_transform,
        download=args.download,
    )
    target_index, target_image, target_label, proxy_prediction = (
        choose_target_for_attack(
            proxy_model,
            mnist_test,
            args.target_label,
            args.target_index,
            device,
        )
    )
    candidate_images, candidate_labels, candidate_source_indices = collect_candidates(
        qmnist_pool,
        excluded_label=target_label,
        candidate_count=args.candidate_count,
        seed=args.seed,
    )
    target_feature = extract_features(
        proxy_model,
        target_image.unsqueeze(0),
        args.feature_batch_size,
        device,
    )
    candidate_features = extract_features(
        proxy_model,
        candidate_images,
        args.feature_batch_size,
        device,
    )
    neighbor_positions, neighbor_similarities = cosine_knn(
        target_feature, candidate_features, args.k
    )
    neighbor_labels = candidate_labels[neighbor_positions]
    poison_label, selected_neighbor_positions = same_label_selection(
        neighbor_labels.tolist(), args.p
    )
    selected_candidate_positions = neighbor_positions[selected_neighbor_positions]
    selected_images = candidate_images[selected_candidate_positions]
    selected_labels = candidate_labels[selected_candidate_positions]
    selected_features = candidate_features[selected_candidate_positions]
    selection_metrics = selected_feature_metrics(target_feature, selected_features)
    (
        universal_perturbation,
        perturbed_images,
        perturbed_features,
        perturbation_metrics,
    ) = optimize_universal_perturbation(
        proxy_model,
        selected_images,
        target_feature,
        device,
        args.perturb_steps,
        args.perturb_lr,
        args.perturb_epsilon,
    )
    perturbed_feature_metrics = selected_feature_metrics(
        target_feature, perturbed_features
    )
    save_image(denormalize_mnist(target_image.unsqueeze(0)), run_dir / "target.png")
    save_image(
        denormalize_mnist(selected_images),
        run_dir / "selected_pood.png",
        nrow=args.p,
    )
    save_image(
        denormalize_mnist(perturbed_images),
        run_dir / "perturbed_pood.png",
        nrow=args.p,
    )

    config = ExperimentConfig(
        seed=args.seed,
        dataset="mnist",
        data_dir=args.data_dir,
        download=args.download,
        iid=False,
        partition="dirichlet",
        non_iid_alpha=args.non_iid_alpha,
        num_clients=args.num_clients,
        malicious_fraction=0.0,
        participation_fraction=1.0,
        aggregation="fedavg",
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        attack="none",
        save_unlearning_history=True,
        unlearning_delta_t=args.unlearning_delta_t,
        output_dir=str(args.output_dir),
    )
    config.validate()
    federated_data = load_federated_data(
        dataset_name="mnist",
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        iid=False,
        non_iid_alpha=args.non_iid_alpha,
        download=args.download,
        seed=args.seed,
        partition="dirichlet",
    )
    benign_distributions = {
        str(client_id): label_histogram(client)
        for client_id, client in enumerate(federated_data.clients)
    }
    benign_malicious_dataset = federated_data.clients[args.malicious_client_id]
    poisoned_dataset = PoisonedClientDataset(
        benign_malicious_dataset,
        perturbed_images,
        selected_labels,
        args.poison_repeats,
    )
    client_datasets: list[Dataset] = list(federated_data.clients)
    client_datasets[args.malicious_client_id] = poisoned_dataset
    poison_local_indices = poisoned_dataset.poison_local_indices

    initial_model = build_model("mnist")
    global_state = _clone_state(initial_model.state_dict())
    del initial_model
    recorder = FedEraserHistoryRecorder(global_state, args.unlearning_delta_t)
    round_metrics: list[dict[str, float | int]] = []
    print(
        f"PUA training | partition=dirichlet | alpha={args.non_iid_alpha:g} | "
        f"clients={args.num_clients} | device={device}",
        flush=True,
    )
    print(
        f"Target index={target_index}, label={target_label} | POOD label={poison_label} | "
        f"unique={args.p}, injected={len(poison_local_indices)} | "
        f"malicious_client={args.malicious_client_id}",
        flush=True,
    )
    for round_number in range(1, args.rounds + 1):
        selected_clients = list(range(args.num_clients))
        rng.shuffle(selected_clients)
        updates = []
        sample_counts = []
        for client_id in selected_clients:
            update = train_local(
                global_state=global_state,
                dataset=client_datasets[client_id],
                model_name="mnist",
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                device=device,
                seed=args.seed + round_number * 100_000 + client_id,
            )
            updates.append(update)
            sample_counts.append(len(client_datasets[client_id]))
        recorder.record_round(
            round_number, selected_clients, updates, sample_counts
        )
        global_state = apply_update(global_state, fedavg(updates, sample_counts))
        metrics = evaluate(
            global_state,
            federated_data.test,
            "mnist",
            args.eval_batch_size,
            args.num_workers,
            device,
        )
        target_round = evaluate_target(
            global_state, target_image, target_label, device
        )
        round_metrics.append(
            {
                "round": round_number,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "target_prediction": int(target_round["prediction"]),
                "target_true_label_confidence": float(
                    target_round["true_label_confidence"]
                ),
            }
        )
        print(
            f"Round {round_number:03d}/{args.rounds} | "
            f"loss={metrics['loss']:.4f} | accuracy={metrics['accuracy']:.2%} | "
            f"target_pred={target_round['prediction']} | "
            f"target_conf={target_round['true_label_confidence']:.2%}",
            flush=True,
        )

    history_config = config.to_dict()
    history_config["pua"] = {
        "malicious_client_id": args.malicious_client_id,
        "target_index": target_index,
        "target_label": target_label,
        "poison_label": poison_label,
        "unique_pood_samples": args.p,
        "poison_repeats": args.poison_repeats,
        "injected_poison_records": len(poison_local_indices),
    }
    history = recorder.build_payload(history_config, global_state)
    torch.save(history, run_dir / "federaser_history.pt")
    torch.save(
        {"dataset": "mnist", "state_dict": global_state, "config": history_config},
        run_dir / "model_before_unlearning.pt",
    )
    with (run_dir / "training_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=round_metrics[0].keys())
        writer.writeheader()
        writer.writerows(round_metrics)

    calibration_lr = (
        args.learning_rate
        if args.calibration_learning_rate is None
        else args.calibration_learning_rate
    )
    unlearned_state, calibration_history = federaser_unlearn(
        history=history,
        client_datasets=client_datasets,
        forget_client_id=args.malicious_client_id,
        forget_local_indices=poison_local_indices,
        forget_all_client_data=False,
        calibration_local_epochs=args.calibration_local_epochs,
        calibration_learning_rate=calibration_lr,
        batch_size=args.batch_size,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed + 1_000_000,
    )

    global_before = evaluate(
        global_state,
        federated_data.test,
        "mnist",
        args.eval_batch_size,
        args.num_workers,
        device,
    )
    global_after = evaluate(
        unlearned_state,
        federated_data.test,
        "mnist",
        args.eval_batch_size,
        args.num_workers,
        device,
    )
    unique_forget_dataset = TensorDataset(perturbed_images, selected_labels)
    forget_before = evaluate(
        global_state,
        unique_forget_dataset,
        "mnist",
        args.eval_batch_size,
        args.num_workers,
        device,
    )
    forget_after = evaluate(
        unlearned_state,
        unique_forget_dataset,
        "mnist",
        args.eval_batch_size,
        args.num_workers,
        device,
    )
    target_before = evaluate_target(global_state, target_image, target_label, device)
    target_after = evaluate_target(
        unlearned_state, target_image, target_label, device
    )
    per_label_before = evaluate_per_class(
        global_state,
        federated_data.test,
        "mnist",
        10,
        args.eval_batch_size,
        args.num_workers,
        device,
    )
    per_label_after = evaluate_per_class(
        unlearned_state,
        federated_data.test,
        "mnist",
        10,
        args.eval_batch_size,
        args.num_workers,
        device,
    )

    summary = {
        "method": "pua_pood_federaser",
        "threat_model": "limited-information POOD poisoning followed by deletion",
        "proxy": {
            "checkpoint": str(proxy_checkpoint.resolve()),
            "checkpoint_accuracy": checkpoint_accuracy(proxy_checkpoint),
            "device": str(device),
        },
        "federated_training": {
            "partition": "dirichlet",
            "non_iid_alpha": args.non_iid_alpha,
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "final_metrics": global_before,
            "client_label_histograms_before_poisoning": benign_distributions,
        },
        "target": {
            "dataset": "MNIST test",
            "source_index": target_index,
            "true_label": target_label,
            "proxy_prediction": proxy_prediction,
            "before_unlearning": target_before,
            "after_unlearning": target_after,
            "true_label_confidence_drop": (
                float(target_before["true_label_confidence"])
                - float(target_after["true_label_confidence"])
            ),
            "attack_success": bool(
                target_before["correct"] and not target_after["correct"]
            ),
        },
        "pood": {
            "candidate_dataset": "QMNIST test50k",
            "candidate_count": args.candidate_count,
            "target_label_excluded": target_label,
            "retrieval": {
                "method": "exact_cosine_knn",
                "paper_method": "HNSW approximate nearest neighbors",
                "note": (
                    "Exact search is used for the small candidate pool and returns "
                    "deterministic cosine nearest neighbors."
                ),
            },
            "k": args.k,
            "p": args.p,
            "selected_label": poison_label,
            "selected_qmnist_source_indices": [
                int(candidate_source_indices[position])
                for position in selected_candidate_positions.tolist()
            ],
            "neighbor_similarities": neighbor_similarities.tolist(),
            "selection_metrics": selection_metrics,
            "perturbation": {
                "steps": args.perturb_steps,
                "learning_rate": args.perturb_lr,
                "epsilon_linf_pixel_space": args.perturb_epsilon,
                **perturbation_metrics,
                "final_mean_target_feature_similarity": perturbed_feature_metrics[
                    "mean_target_feature_similarity"
                ],
            },
        },
        "poison_injection": {
            "malicious_client_id": args.malicious_client_id,
            "benign_client_sample_count": len(benign_malicious_dataset),
            "unique_pood_samples": args.p,
            "poison_repeats": args.poison_repeats,
            "injected_poison_records": len(poison_local_indices),
            "poison_local_index_start": poison_local_indices[0],
            "poison_local_index_end": poison_local_indices[-1],
        },
        "unlearning": {
            "method": "federaser_partial_data",
            "delta_t": args.unlearning_delta_t,
            "snapshot_count": len(history["snapshots"]),
            "calibration_local_epochs": args.calibration_local_epochs,
            "calibration_learning_rate": calibration_lr,
            "forgotten_injected_record_count": len(poison_local_indices),
            "global_metrics": {"before": global_before, "after": global_after},
            "unique_forget_set_metrics": {
                "sample_count": args.p,
                "before": forget_before,
                "after": forget_after,
            },
            "per_label_test_metrics": compare_per_label(
                per_label_before, per_label_after
            ),
            "calibration_history": calibration_history,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save(
        {
            "target_image": target_image,
            "target_label": target_label,
            "target_feature": target_feature.squeeze(0),
            "selected_images": selected_images,
            "selected_labels": selected_labels,
            "selected_features": selected_features,
            "perturbed_images": perturbed_images,
            "perturbed_features": perturbed_features,
            "universal_perturbation_pixel_space": universal_perturbation,
            "selected_candidate_positions": selected_candidate_positions,
            "poison_local_indices": torch.tensor(poison_local_indices),
        },
        run_dir / "pua_artifacts.pt",
    )
    torch.save(
        {
            "dataset": "mnist",
            "state_dict": unlearned_state,
            "source_run": str(run_dir.resolve()),
            "pua_summary": summary,
        },
        run_dir / "model_after_unlearning.pt",
    )
    print(
        f"PUA complete | global_acc={global_before['accuracy']:.2%} -> "
        f"{global_after['accuracy']:.2%} | target_pred="
        f"{target_before['prediction']} -> {target_after['prediction']} | "
        f"target_conf={target_before['true_label_confidence']:.2%} -> "
        f"{target_after['true_label_confidence']:.2%} | "
        f"attack_success={summary['target']['attack_success']}",
        flush=True,
    )
    print(f"Results saved to {run_dir}", flush=True)
    return run_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
