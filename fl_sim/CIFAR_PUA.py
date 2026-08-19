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

from .PUA import PoisonedClientDataset, compare_per_label, label_histogram
from .aggregation import fedavg
from .cifar_pood import (
    denormalize_cifar10,
    extract_features_and_predictions,
    load_cifar10_proxy,
    optimize_cifar_universal_perturbation,
    select_cifar100_pood,
    summarize_selected_features,
)
from .config import ExperimentConfig
from .data import _transforms, load_federated_data
from .experiment import resolve_device
from .model import apply_update, build_model, evaluate, evaluate_per_class, train_local
from .stl_pood import (
    build_stl10_cifar_transform,
    collect_stl10_candidates,
    select_stl10_pood,
)
from .unlearning import FedEraserHistoryRecorder, federaser_unlearn


DEFAULT_DIRICHLET_ALPHA = 0.1


def _retrieval_scores(
    target_feature: torch.Tensor,
    selected_features: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    if metric == "cosine":
        target = F.normalize(target_feature.float(), dim=1)
        selected = F.normalize(selected_features.float(), dim=1)
        return (selected @ target.T).squeeze(1)
    return torch.linalg.vector_norm(
        selected_features.float() - target_feature.float(), dim=1
    )


def _without_perturbation(
    selected_images: torch.Tensor,
    selected_features: torch.Tensor,
    target_feature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    mean_distance = torch.linalg.vector_norm(
        selected_features.float() - target_feature.float(), dim=1
    ).mean()
    distance = float(mean_distance.item())
    return (
        torch.zeros((1, 3, 32, 32), dtype=selected_images.dtype),
        selected_images.detach().cpu().clone(),
        selected_features.detach().cpu().clone(),
        {
            "initial_mean_feature_distance": distance,
            "final_mean_feature_distance": distance,
            "mean_feature_distance_reduction": 0.0,
            "perturbation_linf": 0.0,
            "perturbation_l2": 0.0,
            "last_optimization_loss": distance,
        },
    )


def _clone_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def latest_cifar10_checkpoint(run_root: str | Path) -> Path:
    matches: list[Path] = []
    for path in Path(run_root).rglob("model.pt"):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError, TypeError):
            continue
        if str(checkpoint.get("dataset", "")).lower() == "cifar10":
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No CIFAR-10 model.pt found under {Path(run_root)!s}; "
            "train one first or pass --proxy-checkpoint explicitly."
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def checkpoint_accuracy(checkpoint_path: Path) -> float | None:
    metrics_path = checkpoint_path.parent / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    if len(rows) < 2:
        return None
    try:
        return float(rows[-1].split(",")[2])
    except (IndexError, ValueError):
        return None


@torch.no_grad()
def choose_cifar10_target(
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
                f"CIFAR-10 index {target_index} has label {label}, not "
                f"target label {target_label}."
            )
        prediction = int(model(image.unsqueeze(0).to(device)).argmax(dim=1).item())
        if prediction != label:
            raise ValueError(
                "The requested target is not correctly classified by the proxy."
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
        f"No correctly classified CIFAR-10 target with label {target_label} found."
    )


def collect_cifar100_candidates(
    dataset: Dataset, candidate_count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 1 <= candidate_count <= len(dataset):
        raise ValueError("candidate count must be within the CIFAR-100 pool size.")
    positions = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[:candidate_count]
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for position in positions.tolist():
        image, label = dataset[position]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images), torch.tensor(labels), positions


@torch.no_grad()
def evaluate_target(
    state: Mapping[str, torch.Tensor],
    image: torch.Tensor,
    true_label: int,
    device: torch.device,
) -> dict[str, float | int | bool]:
    model = build_model("cifar10").to(device)
    model.load_state_dict(state)
    model.eval()
    logits = model(image.unsqueeze(0).to(device))
    probabilities = F.softmax(logits, dim=1).squeeze(0)
    prediction = int(probabilities.argmax().item())
    sorted_probabilities = probabilities.sort(descending=True).values
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CIFAR-10 federated POOD-unlearning attack using CIFAR-100 or "
            "STL-10 as the out-of-distribution candidate pool."
        )
    )
    parser.add_argument("--proxy-checkpoint")
    parser.add_argument("--proxy-run-root", default="runs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="cifar_pua_runs")
    parser.add_argument(
        "--pood-dataset",
        choices=["cifar100", "stl10"],
        default="cifar100",
        help="POOD candidate source; fl_sim.STL_PUA selects stl10 automatically.",
    )
    parser.add_argument(
        "--stl10-split",
        choices=["train", "test"],
        default="test",
        help="Labeled STL-10 split used as the POOD candidate pool.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target-label", type=int, default=3)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--candidate-count", type=int, default=5000)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument(
        "--retrieval-metric",
        choices=["cosine", "l2"],
        default="cosine",
        help="Exact Top-K feature retrieval metric; l2 uses raw Euclidean distance.",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=["optimized", "no_pood", "unperturbed", "random"],
        default="optimized",
        help=(
            "optimized runs the PUA baseline; no_pood performs calibration replay "
            "without injection or deletion; unperturbed injects retrieved raw POOD; "
            "random injects unperturbed samples from the same semantic/label group."
        ),
    )
    parser.add_argument("--perturb-steps", type=int, default=200)
    parser.add_argument("--perturb-lr", type=float, default=0.005)
    parser.add_argument("--perturb-epsilon", type=float, default=8 / 255)
    parser.add_argument("--feature-batch-size", type=int, default=256)

    parser.add_argument("--malicious-client-id", type=int, default=6)
    parser.add_argument("--poison-repeats", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument(
        "--non-iid-alpha", type=float, default=DEFAULT_DIRICHLET_ALPHA
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--unlearning-delta-t", type=int, default=5)
    parser.add_argument(
        "--keep-unlearning-history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Persist federaser_history.pt for later standalone replay. "
            "The integrated run uses the in-memory history and does not save it "
            "by default."
        ),
    )
    parser.add_argument("--calibration-local-epochs", type=int, default=1)
    parser.add_argument("--calibration-learning-rate", type=float)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.target_label <= 9:
        raise ValueError("target label must be between 0 and 9.")
    if args.num_clients < 2:
        raise ValueError("at least two clients are required.")
    if not 0 <= args.malicious_client_id < args.num_clients:
        raise ValueError("malicious client id is outside the client range.")
    if args.non_iid_alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    if args.candidate_count < 1 or args.feature_batch_size < 1:
        raise ValueError("candidate and feature batch counts must be positive.")
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
    pood_dataset_name = args.pood_dataset.lower()
    run_dir = Path(args.output_dir) / (
        f"{timestamp}-cifar10-{pood_dataset_name}-pua-"
        f"alpha{args.non_iid_alpha:g}-"
        f"client{args.malicious_client_id}-target{args.target_label}-seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    proxy_checkpoint = (
        Path(args.proxy_checkpoint)
        if args.proxy_checkpoint
        else latest_cifar10_checkpoint(args.proxy_run_root)
    )
    proxy_model, _ = load_cifar10_proxy(proxy_checkpoint, device)
    _, cifar10_eval_transform = _transforms("cifar10")
    cifar10_test = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=cifar10_eval_transform,
        download=args.download,
    )
    # Every POOD image is normalized for the CIFAR-10 proxy and later injected
    # into CIFAR-10 training. STL-10 is additionally resized from 96x96 to 32x32.
    if pood_dataset_name == "cifar100":
        pood_pool = datasets.CIFAR100(
            root=args.data_dir,
            train=False,
            transform=cifar10_eval_transform,
            download=args.download,
        )
        pood_pool_description = "CIFAR100 test"
    else:
        pood_pool = datasets.STL10(
            root=args.data_dir,
            split=args.stl10_split,
            transform=build_stl10_cifar_transform(),
            download=args.download,
        )
        pood_pool_description = f"STL10 {args.stl10_split}"
    target_index, target_image, target_label, proxy_prediction = (
        choose_cifar10_target(
            proxy_model,
            cifar10_test,
            args.target_label,
            args.target_index,
            device,
        )
    )
    if pood_dataset_name == "cifar100":
        candidate_images, candidate_source_labels, candidate_source_indices = (
            collect_cifar100_candidates(
                pood_pool, args.candidate_count, args.seed
            )
        )
        candidate_mapped_labels = None
    else:
        (
            candidate_images,
            candidate_source_labels,
            candidate_mapped_labels,
            candidate_source_indices,
        ) = collect_stl10_candidates(
            pood_pool,
            target_label,
            args.candidate_count,
            args.seed,
        )
    target_feature, _ = extract_features_and_predictions(
        proxy_model,
        target_image.unsqueeze(0),
        args.feature_batch_size,
        device,
    )
    candidate_features, candidate_pseudo_labels = (
        extract_features_and_predictions(
            proxy_model,
            candidate_images,
            args.feature_batch_size,
            device,
        )
    )
    if pood_dataset_name == "cifar100":
        (
            selected_source_label,
            poison_label,
            selected_candidate_positions,
            selected_similarities,
        ) = select_cifar100_pood(
            target_feature,
            candidate_features,
            candidate_source_labels,
            candidate_pseudo_labels,
            target_label,
            args.k,
            args.p,
            args.retrieval_metric,
        )
        selection_group = "cifar100_class_and_cifar10_proxy_prediction"
        label_strategy = "cifar10_proxy_pseudo_label"
    else:
        if candidate_mapped_labels is None:
            raise AssertionError("STL-10 candidates require mapped labels.")
        (
            selected_source_label,
            poison_label,
            selected_candidate_positions,
            selected_similarities,
        ) = select_stl10_pood(
            target_feature,
            candidate_features,
            candidate_source_labels,
            candidate_mapped_labels,
            args.k,
            args.p,
            args.retrieval_metric,
        )
        selection_group = "stl10_true_class_mapped_to_cifar10"
        label_strategy = "deterministic_stl10_to_cifar10_mapping"
    if args.experiment_mode == "random":
        if pood_dataset_name == "cifar100":
            same_group = torch.where(
                (candidate_source_labels == selected_source_label)
                & (candidate_pseudo_labels == poison_label)
            )[0]
        else:
            if candidate_mapped_labels is None:
                raise AssertionError("STL-10 candidates require mapped labels.")
            same_group = torch.where(
                (candidate_source_labels == selected_source_label)
                & (candidate_mapped_labels == poison_label)
            )[0]
        if len(same_group) < args.p:
            raise ValueError(
                "The selected semantic/label group is too small for random control."
            )
        order = torch.randperm(
            len(same_group),
            generator=torch.Generator().manual_seed(args.seed + 20_260_817),
        )
        selected_candidate_positions = same_group[order[: args.p]]
        selected_similarities = _retrieval_scores(
            target_feature,
            candidate_features[selected_candidate_positions],
            args.retrieval_metric,
        )

    source_class_name = pood_pool.classes[selected_source_label]
    selected_images = candidate_images[selected_candidate_positions]
    selected_source_labels = candidate_source_labels[selected_candidate_positions]
    selected_features = candidate_features[selected_candidate_positions]
    poison_labels = torch.full((args.p,), poison_label, dtype=torch.long)
    selection_metrics = summarize_selected_features(
        target_feature, selected_features
    )
    if args.experiment_mode in {"optimized", "no_pood"}:
        (
            universal_perturbation,
            perturbed_images,
            perturbed_features,
            perturbation_metrics,
        ) = optimize_cifar_universal_perturbation(
            proxy_model,
            selected_images,
            target_feature,
            device,
            args.perturb_steps,
            args.perturb_lr,
            args.perturb_epsilon,
        )
    else:
        (
            universal_perturbation,
            perturbed_images,
            perturbed_features,
            perturbation_metrics,
        ) = _without_perturbation(
            selected_images, selected_features, target_feature
        )
    perturbed_feature_metrics = summarize_selected_features(
        target_feature, perturbed_features
    )
    save_image(denormalize_cifar10(target_image.unsqueeze(0)), run_dir / "target.png")
    save_image(
        denormalize_cifar10(selected_images), run_dir / "selected_pood.png", nrow=args.p
    )
    save_image(
        denormalize_cifar10(perturbed_images),
        run_dir / "perturbed_pood.png",
        nrow=args.p,
    )

    config = ExperimentConfig(
        seed=args.seed,
        dataset="cifar10",
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
        dataset_name="cifar10",
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
    client_datasets: list[Dataset] = list(federated_data.clients)
    if args.experiment_mode == "no_pood":
        poison_local_indices: list[int] = []
    else:
        poisoned_dataset = PoisonedClientDataset(
            benign_malicious_dataset,
            perturbed_images,
            poison_labels,
            args.poison_repeats,
        )
        client_datasets[args.malicious_client_id] = poisoned_dataset
        poison_local_indices = poisoned_dataset.poison_local_indices

    initial_model = build_model("cifar10")
    global_state = _clone_state(initial_model.state_dict())
    del initial_model
    recorder = FedEraserHistoryRecorder(global_state, args.unlearning_delta_t)
    round_metrics: list[dict[str, float | int]] = []
    print(
        f"CIFAR PUA | train=CIFAR10 | POOD={pood_dataset_name.upper()} | "
        f"alpha={args.non_iid_alpha:g} | clients={args.num_clients} | "
        f"mode={args.experiment_mode} | retrieval={args.retrieval_metric} | "
        f"device={device}",
        flush=True,
    )
    print(
        f"target={cifar10_test.classes[target_label]}({target_label}) | "
        f"POOD={source_class_name}({selected_source_label}) | poison_label="
        f"{cifar10_test.classes[poison_label]}({poison_label}) | "
        f"unique={args.p} | injected={len(poison_local_indices)}",
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
                model_name="cifar10",
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
        recorder.record_round(round_number, selected_clients, updates, sample_counts)
        global_state = apply_update(global_state, fedavg(updates, sample_counts))
        metrics = evaluate(
            global_state,
            federated_data.test,
            "cifar10",
            args.eval_batch_size,
            args.num_workers,
            device,
        )
        target_round = evaluate_target(global_state, target_image, target_label, device)
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
            f"Round {round_number:03d}/{args.rounds} | loss={metrics['loss']:.4f} | "
            f"accuracy={metrics['accuracy']:.2%} | "
            f"target_pred={target_round['prediction']} | "
            f"target_conf={target_round['true_label_confidence']:.2%}",
            flush=True,
        )

    history_config = config.to_dict()
    history_config["cifar_pua"] = {
        "pood_dataset": pood_dataset_name,
        "malicious_client_id": args.malicious_client_id,
        "target_index": target_index,
        "target_label": target_label,
        "source_label": selected_source_label,
        "source_class": source_class_name,
        "label_strategy": label_strategy,
        "experiment_mode": args.experiment_mode,
        "retrieval_metric": args.retrieval_metric,
        "poison_label": poison_label,
        "unique_pood_samples": args.p,
        "poison_repeats": args.poison_repeats,
        "injected_poison_records": len(poison_local_indices),
    }
    history = recorder.build_payload(history_config, global_state)
    history_path = run_dir / "federaser_history.pt"
    if args.keep_unlearning_history:
        torch.save(history, history_path)
        print(f"FedEraser history saved to {history_path}", flush=True)
    else:
        print(
            "FedEraser history kept in memory only; "
            "use --keep-unlearning-history to persist it.",
            flush=True,
        )
    torch.save(
        {"dataset": "cifar10", "state_dict": global_state, "config": history_config},
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
        forget_local_indices=poison_local_indices or None,
        forget_all_client_data=False,
        calibration_local_epochs=args.calibration_local_epochs,
        calibration_learning_rate=calibration_lr,
        batch_size=args.batch_size,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed + 1_000_000,
        reconstruct_without_forgetting=args.experiment_mode == "no_pood",
    )
    global_before = evaluate(
        global_state, federated_data.test, "cifar10", args.eval_batch_size,
        args.num_workers, device
    )
    global_after = evaluate(
        unlearned_state, federated_data.test, "cifar10", args.eval_batch_size,
        args.num_workers, device
    )
    unique_forget_dataset = TensorDataset(perturbed_images, poison_labels)
    forget_before = evaluate(
        global_state, unique_forget_dataset, "cifar10", args.eval_batch_size,
        args.num_workers, device
    )
    forget_after = evaluate(
        unlearned_state, unique_forget_dataset, "cifar10", args.eval_batch_size,
        args.num_workers, device
    )
    target_before = evaluate_target(global_state, target_image, target_label, device)
    target_after = evaluate_target(unlearned_state, target_image, target_label, device)
    per_label_before = evaluate_per_class(
        global_state, federated_data.test, "cifar10", 10,
        args.eval_batch_size, args.num_workers, device
    )
    per_label_after = evaluate_per_class(
        unlearned_state, federated_data.test, "cifar10", 10,
        args.eval_batch_size, args.num_workers, device
    )

    selected_source_indices = [
        int(candidate_source_indices[position])
        for position in selected_candidate_positions.tolist()
    ]
    source_metadata = {
        "source_label_space": pood_dataset_name,
        "source_label": selected_source_label,
        "source_class": source_class_name,
        "label_strategy": label_strategy,
    }
    if pood_dataset_name == "cifar100":
        source_metadata.update(
            {
                "cifar100_source_label": selected_source_label,
                "cifar100_source_class": source_class_name,
                "selected_cifar100_source_indices": selected_source_indices,
            }
        )
    else:
        source_metadata.update(
            {
                "stl10_source_label": selected_source_label,
                "stl10_source_class": source_class_name,
                "selected_stl10_source_indices": selected_source_indices,
                "resize": "96x96_to_32x32",
                "unmapped_stl10_class_excluded": "monkey",
            }
        )
    score_key = (
        "selected_neighbor_similarities"
        if args.retrieval_metric == "cosine"
        else "selected_neighbor_distances"
    )
    summary = {
        "method": f"cifar10_{pood_dataset_name}_pua_federaser",
        "experiment_mode": args.experiment_mode,
        "datasets": {
            "federated_training": "CIFAR10",
            "target": "CIFAR10 test",
            "pood_candidate_pool": pood_pool_description,
            "candidate_normalization": "CIFAR10",
        },
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
            "source_index": target_index,
            "true_label": target_label,
            "class_name": cifar10_test.classes[target_label],
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
            "candidate_count": args.candidate_count,
            "retrieval": (
                f"exact_{args.retrieval_metric}_topk_in_"
                "resnet18_512d_feature_space"
            ),
            "retrieval_metric": args.retrieval_metric,
            "selection_group": selection_group,
            "k": args.k,
            "p": args.p,
            **source_metadata,
            "cifar10_poison_label": poison_label,
            "cifar10_poison_class": cifar10_test.classes[poison_label],
            "selected_source_indices": selected_source_indices,
            score_key: selected_similarities.tolist(),
            "selection_metrics": selection_metrics,
            "perturbation": {
                "enabled": args.experiment_mode in {"optimized", "no_pood"},
                "steps": (
                    args.perturb_steps
                    if args.experiment_mode in {"optimized", "no_pood"}
                    else 0
                ),
                "learning_rate": args.perturb_lr,
                "epsilon_linf_pixel_space": args.perturb_epsilon,
                **perturbation_metrics,
                "final_mean_target_feature_similarity": (
                    perturbed_feature_metrics["mean_target_feature_similarity"]
                ),
            },
        },
        "poison_injection": {
            "enabled": args.experiment_mode != "no_pood",
            "experiment_mode": args.experiment_mode,
            "malicious_client_id": args.malicious_client_id,
            "benign_client_sample_count": len(benign_malicious_dataset),
            "unique_pood_samples": args.p,
            "poison_repeats": args.poison_repeats,
            "injected_poison_records": len(poison_local_indices),
            "poison_local_index_start": (
                poison_local_indices[0] if poison_local_indices else None
            ),
            "poison_local_index_end": (
                poison_local_indices[-1] if poison_local_indices else None
            ),
        },
        "unlearning": {
            "method": (
                "federaser_reconstruction_without_forgetting"
                if args.experiment_mode == "no_pood"
                else "federaser_partial_data"
            ),
            "delta_t": args.unlearning_delta_t,
            "snapshot_count": len(history["snapshots"]),
            "history_saved_to_disk": args.keep_unlearning_history,
            "history_path": (
                str(history_path.resolve())
                if args.keep_unlearning_history
                else None
            ),
            "forgotten_injected_record_count": len(poison_local_indices),
            "global_metrics": {"before": global_before, "after": global_after},
            "unique_forget_set_metrics": {
                "sample_count": args.p,
                "is_probe_only": args.experiment_mode == "no_pood",
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
            "pood_dataset": pood_dataset_name,
            "selected_source_labels": selected_source_labels,
            "poison_labels": poison_labels,
            "selected_features": selected_features,
            "perturbed_images": perturbed_images,
            "perturbed_features": perturbed_features,
            "universal_perturbation_pixel_space": universal_perturbation,
            "poison_local_indices": torch.tensor(poison_local_indices),
        },
        run_dir / "cifar_pua_artifacts.pt",
    )
    torch.save(
        {
            "dataset": "cifar10",
            "state_dict": unlearned_state,
            "source_run": str(run_dir.resolve()),
            "cifar_pua_summary": summary,
        },
        run_dir / "model_after_unlearning.pt",
    )
    print(
        f"CIFAR PUA complete | global_acc={global_before['accuracy']:.2%} -> "
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
