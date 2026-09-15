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


def parse_target_indices(value: str) -> list[int]:
    """Parse a reproducible comma-separated target-index list."""
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "target indices must be comma-separated integers."
        ) from error
    if not indices:
        raise argparse.ArgumentTypeError("target indices cannot be empty.")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("target indices must be unique.")
    return indices


def parse_client_ids(value: str) -> list[int]:
    """Parse a unique comma-separated malicious-client list."""
    try:
        client_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "malicious client ids must be comma-separated integers."
        ) from error
    if not client_ids:
        raise argparse.ArgumentTypeError("malicious client ids cannot be empty.")
    if len(set(client_ids)) != len(client_ids):
        raise argparse.ArgumentTypeError("malicious client ids must be unique.")
    return client_ids


def resolve_malicious_client_ids(args: argparse.Namespace) -> list[int]:
    """Return plural ids when supplied, otherwise preserve the legacy option."""
    if args.malicious_client_ids is not None:
        return list(args.malicious_client_ids)
    return [int(args.malicious_client_id)]


def distribute_poison_record_positions(
    total_records: int, client_ids: list[int]
) -> dict[int, list[int]]:
    """Round-robin one fixed total POOD budget across malicious clients."""
    if total_records < len(client_ids) or not client_ids:
        raise ValueError(
            "The POOD budget must provide at least one record per malicious client."
        )
    return {
        client_id: list(range(position, total_records, len(client_ids)))
        for position, client_id in enumerate(client_ids)
    }


@torch.no_grad()
def choose_cifar10_targets(
    model,
    dataset: Dataset,
    target_label: int,
    target_count: int,
    target_index: int | None,
    target_indices: list[int] | None,
    device: torch.device,
) -> list[tuple[int, torch.Tensor, int, int]]:
    """Choose distinct proxy-correct targets without changing their class."""
    if target_indices is not None:
        return [
            choose_cifar10_target(model, dataset, target_label, index, device)
            for index in target_indices
        ]
    if target_index is not None:
        return [
            choose_cifar10_target(
                model, dataset, target_label, target_index, device
            )
        ]

    targets: list[tuple[int, torch.Tensor, int, int]] = []
    for index in range(len(dataset)):
        image, label = dataset[index]
        label = int(label)
        if label != target_label:
            continue
        prediction = int(model(image.unsqueeze(0).to(device)).argmax(dim=1).item())
        if prediction == label:
            targets.append((index, image, label, prediction))
            if len(targets) == target_count:
                return targets
    raise RuntimeError(
        f"Only found {len(targets)} proxy-correct CIFAR-10 targets with label "
        f"{target_label}; requested {target_count}."
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
    return evaluate_targets(
        state,
        image.unsqueeze(0),
        torch.tensor([true_label], dtype=torch.long),
        device,
    )[0]


@torch.no_grad()
def evaluate_targets(
    state: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    true_labels: torch.Tensor,
    device: torch.device,
) -> list[dict[str, float | int | bool]]:
    """Evaluate a target group with one model load and preserve per-sample metrics."""
    if images.ndim != 4 or len(images) != len(true_labels) or len(images) < 1:
        raise ValueError("Target images and labels must be aligned and non-empty.")
    model = build_model("cifar10").to(device)
    model.load_state_dict(state)
    model.eval()
    logits = model(images.to(device))
    probabilities = F.softmax(logits, dim=1)
    labels = true_labels.to(device=device, dtype=torch.long)
    losses = F.cross_entropy(logits, labels, reduction="none")
    evaluations: list[dict[str, float | int | bool]] = []
    for position in range(len(images)):
        sample_probabilities = probabilities[position]
        true_label = int(labels[position].item())
        prediction = int(sample_probabilities.argmax().item())
        sorted_probabilities = sample_probabilities.sort(descending=True).values
        evaluations.append(
            {
                "true_label": true_label,
                "prediction": prediction,
                "correct": prediction == true_label,
                "true_label_confidence": float(
                    sample_probabilities[true_label].item()
                ),
                "predicted_confidence": float(
                    sample_probabilities[prediction].item()
                ),
                "top1_top2_margin": float(
                    (sorted_probabilities[0] - sorted_probabilities[1]).item()
                ),
                "cross_entropy": float(losses[position].item()),
            }
        )
    return evaluations


def summarize_target_group(
    before: list[dict[str, float | int | bool]],
    after: list[dict[str, float | int | bool]],
) -> dict[str, float | int | None]:
    """Summarize the legacy success condition across multiple targets."""
    if len(before) != len(after) or not before:
        raise ValueError("Before/after target metrics must be aligned and non-empty.")
    target_count = len(before)
    correct_before = sum(bool(item["correct"]) for item in before)
    correct_after = sum(bool(item["correct"]) for item in after)
    success_count = sum(
        bool(before_item["correct"]) and not bool(after_item["correct"])
        for before_item, after_item in zip(before, after)
    )
    changed_count = sum(
        int(before_item["prediction"]) != int(after_item["prediction"])
        for before_item, after_item in zip(before, after)
    )
    mean_confidence_before = sum(
        float(item["true_label_confidence"]) for item in before
    ) / target_count
    mean_confidence_after = sum(
        float(item["true_label_confidence"]) for item in after
    ) / target_count
    return {
        "target_count": target_count,
        "correct_before_count": correct_before,
        "correct_after_count": correct_after,
        "accuracy_before": correct_before / target_count,
        "accuracy_after": correct_after / target_count,
        "success_definition": "correct_before_and_incorrect_after_unlearning",
        "success_count": success_count,
        "success_rate": success_count / target_count,
        "eligible_success_count": correct_before,
        "conditional_success_rate": (
            success_count / correct_before if correct_before else None
        ),
        "prediction_changed_count": changed_count,
        "prediction_changed_rate": changed_count / target_count,
        "mean_true_label_confidence_before": mean_confidence_before,
        "mean_true_label_confidence_after": mean_confidence_after,
        "mean_true_label_confidence_drop": (
            mean_confidence_before - mean_confidence_after
        ),
    }


def resolve_pood_training_label(
    label_mode: str,
    retrieved_group_label: int,
    target_label: int,
) -> int:
    """Resolve the class assigned to injected POOD samples during training."""
    if label_mode == "source":
        return retrieved_group_label
    if label_mode == "target":
        return target_label
    raise ValueError(f"Unsupported POOD training label mode: {label_mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CIFAR-10 federated POOD-unlearning simulation using CIFAR-100 or "
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
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu"
    )
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target-label", type=int, default=3)
    parser.add_argument("--target-index", type=int)
    parser.add_argument(
        "--target-count",
        type=int,
        default=1,
        help=(
            "Number m of distinct proxy-correct target samples. Each target "
            "independently runs the existing retrieval and perturbation flow."
        ),
    )
    parser.add_argument(
        "--target-indices",
        type=parse_target_indices,
        help=(
            "Optional comma-separated list of exact CIFAR-10 target indices; "
            "its length must equal --target-count."
        ),
    )
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
    parser.add_argument(
        "--pood-training-label",
        choices=["source", "target"],
        default="source",
        help=(
            "Label assigned to injected POOD samples during federated training. "
            "source preserves the selected POOD class mapping; target assigns "
            "the CIFAR-10 target sample label."
        ),
    )
    parser.add_argument("--perturb-steps", type=int, default=200)
    parser.add_argument("--perturb-lr", type=float, default=0.005)
    parser.add_argument("--perturb-epsilon", type=float, default=8 / 255)
    parser.add_argument("--feature-batch-size", type=int, default=256)

    parser.add_argument(
        "--malicious-client-id",
        type=int,
        default=6,
        help="Legacy single-client option; ignored when --malicious-client-ids is set.",
    )
    parser.add_argument(
        "--malicious-client-ids",
        type=parse_client_ids,
        help=(
            "Comma-separated malicious clients. The total repeated POOD record "
            "budget is distributed as evenly as possible across these clients."
        ),
    )
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
    if args.target_count < 1:
        raise ValueError("target count must be at least 1.")
    if args.target_index is not None and args.target_indices is not None:
        raise ValueError("Do not combine --target-index and --target-indices.")
    if args.target_index is not None and args.target_count != 1:
        raise ValueError("--target-index can only be used with --target-count 1.")
    if (
        args.target_indices is not None
        and len(args.target_indices) != args.target_count
    ):
        raise ValueError("--target-indices length must equal --target-count.")
    if args.num_clients < 2:
        raise ValueError("at least two clients are required.")
    malicious_client_ids = resolve_malicious_client_ids(args)
    if len(set(malicious_client_ids)) != len(malicious_client_ids):
        raise ValueError("Malicious client ids must be unique.")
    if any(
        client_id < 0 or client_id >= args.num_clients
        for client_id in malicious_client_ids
    ):
        raise ValueError("A malicious client id is outside the client range.")
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
    total_poison_records = args.target_count * args.p * args.poison_repeats
    if args.experiment_mode != "no_pood" and total_poison_records < len(
        malicious_client_ids
    ):
        raise ValueError(
            "The total POOD record budget must provide at least one record per "
            "malicious client."
        )
    if args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("rounds and local epochs must be positive.")
    if args.learning_rate <= 0 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("training learning rate and batch sizes must be positive.")
    if args.unlearning_delta_t < 1 or args.calibration_local_epochs < 1:
        raise ValueError("unlearning interval and calibration epochs must be positive.")


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    malicious_client_ids = resolve_malicious_client_ids(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    pood_dataset_name = args.pood_dataset.lower()
    target_count_suffix = f"-m{args.target_count}" if args.target_count > 1 else ""
    malicious_suffix = "-".join(str(client_id) for client_id in malicious_client_ids)
    run_dir = Path(args.output_dir) / (
        f"{timestamp}-cifar10-{pood_dataset_name}-pua-"
        f"alpha{args.non_iid_alpha:g}-"
        f"clients{malicious_suffix}-target{args.target_label}"
        f"{target_count_suffix}-seed{args.seed}"
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
    targets = choose_cifar10_targets(
        proxy_model,
        cifar10_test,
        args.target_label,
        args.target_count,
        args.target_index,
        args.target_indices,
        device,
    )
    target_indices = [item[0] for item in targets]
    target_images = torch.stack([item[1] for item in targets])
    target_labels = torch.tensor([item[2] for item in targets], dtype=torch.long)
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
            args.target_label,
            args.candidate_count,
            args.seed,
        )
    target_features, _ = extract_features_and_predictions(
        proxy_model,
        target_images,
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
        selection_group = "cifar100_class_and_cifar10_proxy_prediction"
        label_strategy = "cifar10_proxy_pseudo_label"
    else:
        if candidate_mapped_labels is None:
            raise AssertionError("STL-10 candidates require mapped labels.")
        selection_group = "stl10_true_class_mapped_to_cifar10"
        label_strategy = "deterministic_stl10_to_cifar10_mapping"

    score_key = (
        "selected_neighbor_similarities"
        if args.retrieval_metric == "cosine"
        else "selected_neighbor_distances"
    )
    target_records: list[dict] = []
    selected_image_groups: list[torch.Tensor] = []
    selected_source_label_groups: list[torch.Tensor] = []
    selected_feature_groups: list[torch.Tensor] = []
    perturbed_image_groups: list[torch.Tensor] = []
    perturbed_feature_groups: list[torch.Tensor] = []
    poison_label_groups: list[torch.Tensor] = []
    perturbation_groups: list[torch.Tensor] = []

    print(
        f"Preparing {args.target_count} target(s) serially | "
        f"p={args.p} per target | total_selected_pood="
        f"{args.target_count * args.p}",
        flush=True,
    )
    for target_number, (
        target_index,
        target_image,
        target_label,
        proxy_prediction,
    ) in enumerate(targets, start=1):
        target_feature = target_features[target_number - 1 : target_number]
        if pood_dataset_name == "cifar100":
            (
                selected_source_label,
                retrieved_group_label,
                selected_candidate_positions,
                selected_scores,
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
        else:
            if candidate_mapped_labels is None:
                raise AssertionError("STL-10 candidates require mapped labels.")
            (
                selected_source_label,
                retrieved_group_label,
                selected_candidate_positions,
                selected_scores,
            ) = select_stl10_pood(
                target_feature,
                candidate_features,
                candidate_source_labels,
                candidate_mapped_labels,
                args.k,
                args.p,
                args.retrieval_metric,
            )
        if args.experiment_mode == "random":
            if pood_dataset_name == "cifar100":
                same_group = torch.where(
                    (candidate_source_labels == selected_source_label)
                    & (candidate_pseudo_labels == retrieved_group_label)
                )[0]
            else:
                if candidate_mapped_labels is None:
                    raise AssertionError("STL-10 candidates require mapped labels.")
                same_group = torch.where(
                    (candidate_source_labels == selected_source_label)
                    & (candidate_mapped_labels == retrieved_group_label)
                )[0]
            if len(same_group) < args.p:
                raise ValueError(
                    "The selected semantic/label group is too small for random "
                    "control."
                )
            order = torch.randperm(
                len(same_group),
                generator=torch.Generator().manual_seed(
                    args.seed + 20_260_817 + target_number
                ),
            )
            selected_candidate_positions = same_group[order[: args.p]]
            selected_scores = _retrieval_scores(
                target_feature,
                candidate_features[selected_candidate_positions],
                args.retrieval_metric,
            )

        source_class_name = pood_pool.classes[selected_source_label]
        poison_training_label = resolve_pood_training_label(
            args.pood_training_label,
            retrieved_group_label,
            target_label,
        )
        selected_images_for_target = candidate_images[selected_candidate_positions]
        selected_source_labels_for_target = candidate_source_labels[
            selected_candidate_positions
        ]
        selected_features_for_target = candidate_features[
            selected_candidate_positions
        ]
        poison_labels_for_target = torch.full(
            (args.p,), poison_training_label, dtype=torch.long
        )
        selection_metrics = summarize_selected_features(
            target_feature, selected_features_for_target
        )
        if args.experiment_mode in {"optimized", "no_pood"}:
            (
                universal_perturbation,
                perturbed_images_for_target,
                perturbed_features_for_target,
                perturbation_metrics,
            ) = optimize_cifar_universal_perturbation(
                proxy_model,
                selected_images_for_target,
                target_feature,
                device,
                args.perturb_steps,
                args.perturb_lr,
                args.perturb_epsilon,
            )
        else:
            (
                universal_perturbation,
                perturbed_images_for_target,
                perturbed_features_for_target,
                perturbation_metrics,
            ) = _without_perturbation(
                selected_images_for_target,
                selected_features_for_target,
                target_feature,
            )
        perturbed_feature_metrics = summarize_selected_features(
            target_feature, perturbed_features_for_target
        )
        selected_source_indices = [
            int(candidate_source_indices[position])
            for position in selected_candidate_positions.tolist()
        ]
        target_output_dir = run_dir / f"target_{target_number:03d}_index_{target_index}"
        target_output_dir.mkdir()
        save_image(
            denormalize_cifar10(target_image.unsqueeze(0)),
            target_output_dir / "target.png",
        )
        save_image(
            denormalize_cifar10(selected_images_for_target),
            target_output_dir / "selected_pood.png",
            nrow=min(args.p, 10),
        )
        save_image(
            denormalize_cifar10(perturbed_images_for_target),
            target_output_dir / "perturbed_pood.png",
            nrow=min(args.p, 10),
        )

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
        target_records.append(
            {
                "target_number": target_number,
                "source_index": target_index,
                "true_label": target_label,
                "class_name": cifar10_test.classes[target_label],
                "proxy_prediction": proxy_prediction,
                "target_feature": target_feature.squeeze(0),
                "selected_candidate_positions": selected_candidate_positions,
                "selected_source_indices": selected_source_indices,
                "selected_scores": selected_scores,
                "source_metadata": source_metadata,
                "retrieved_group_label": retrieved_group_label,
                "poison_training_label": poison_training_label,
                "selection_metrics": selection_metrics,
                "perturbation_metrics": perturbation_metrics,
                "perturbed_feature_metrics": perturbed_feature_metrics,
            }
        )
        selected_image_groups.append(selected_images_for_target)
        selected_source_label_groups.append(selected_source_labels_for_target)
        selected_feature_groups.append(selected_features_for_target)
        perturbed_image_groups.append(perturbed_images_for_target)
        perturbed_feature_groups.append(perturbed_features_for_target)
        poison_label_groups.append(poison_labels_for_target)
        perturbation_groups.append(universal_perturbation)
        print(
            f"Target {target_number:03d}/{args.target_count:03d} | "
            f"index={target_index} | class={cifar10_test.classes[target_label]}"
            f"({target_label}) | POOD={source_class_name}({selected_source_label}) | "
            f"mapped_label={cifar10_test.classes[retrieved_group_label]}"
            f"({retrieved_group_label}) | training_label="
            f"{cifar10_test.classes[poison_training_label]}"
            f"({poison_training_label}) | optimized={args.p}",
            flush=True,
        )

    selected_images = torch.cat(selected_image_groups)
    selected_source_labels = torch.cat(selected_source_label_groups)
    selected_features = torch.cat(selected_feature_groups)
    perturbed_images = torch.cat(perturbed_image_groups)
    perturbed_features = torch.cat(perturbed_feature_groups)
    poison_labels = torch.cat(poison_label_groups)
    universal_perturbations = torch.cat(perturbation_groups)

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
    benign_malicious_datasets = {
        client_id: federated_data.clients[client_id]
        for client_id in malicious_client_ids
    }
    client_datasets: list[Dataset] = list(federated_data.clients)
    poison_local_indices_by_client: dict[int, list[int]] = {}
    if args.experiment_mode == "no_pood":
        pass
    else:
        total_poison_records = len(perturbed_images) * args.poison_repeats
        record_allocations = distribute_poison_record_positions(
            total_poison_records, malicious_client_ids
        )
        for client_id in malicious_client_ids:
            poisoned_dataset = PoisonedClientDataset(
                benign_malicious_datasets[client_id],
                perturbed_images,
                poison_labels,
                args.poison_repeats,
                poison_record_positions=record_allocations[client_id],
            )
            client_datasets[client_id] = poisoned_dataset
            poison_local_indices_by_client[client_id] = (
                poisoned_dataset.poison_local_indices
            )
    total_injected_records = sum(
        len(indices) for indices in poison_local_indices_by_client.values()
    )

    initial_model = build_model("cifar10")
    global_state = _clone_state(initial_model.state_dict())
    del initial_model
    recorder = FedEraserHistoryRecorder(global_state, args.unlearning_delta_t)
    round_metrics: list[dict[str, float | int]] = []
    print(
        f"CIFAR PUA | train=CIFAR10 | POOD={pood_dataset_name.upper()} | "
        f"alpha={args.non_iid_alpha:g} | clients={args.num_clients} | "
        f"targets={args.target_count} | p_per_target={args.p} | "
        f"mode={args.experiment_mode} | retrieval={args.retrieval_metric} | "
        f"training_label={args.pood_training_label} | "
        f"malicious_clients={malicious_client_ids} | "
        f"device={device}",
        flush=True,
    )
    print(
        f"target_class={cifar10_test.classes[args.target_label]}"
        f"({args.target_label}) | unique_pood={len(perturbed_images)} | "
        f"injected={total_injected_records} | allocation="
        f"{ {client_id: len(indices) for client_id, indices in poison_local_indices_by_client.items()} }",
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
        target_round = evaluate_targets(
            global_state, target_images, target_labels, device
        )
        target_correct_count = sum(bool(item["correct"]) for item in target_round)
        target_mean_confidence = sum(
            float(item["true_label_confidence"]) for item in target_round
        ) / len(target_round)
        round_record: dict[str, float | int] = {
            "round": round_number,
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "target_group_correct_count": target_correct_count,
            "target_group_accuracy": target_correct_count / len(target_round),
            "target_group_mean_true_label_confidence": target_mean_confidence,
        }
        if args.target_count == 1:
            round_record.update(
                {
                    "target_prediction": int(target_round[0]["prediction"]),
                    "target_true_label_confidence": float(
                        target_round[0]["true_label_confidence"]
                    ),
                }
            )
        round_metrics.append(round_record)
        print(
            f"Round {round_number:03d}/{args.rounds} | loss={metrics['loss']:.4f} | "
            f"accuracy={metrics['accuracy']:.2%} | "
            f"target_acc={target_correct_count / len(target_round):.2%} | "
            f"mean_target_conf={target_mean_confidence:.2%}",
            flush=True,
        )

    history_config = config.to_dict()
    history_config["cifar_pua"] = {
        "pood_dataset": pood_dataset_name,
        "malicious_client_id": malicious_client_ids[0],
        "malicious_client_ids": malicious_client_ids,
        "target_count": args.target_count,
        "target_indices": target_indices,
        "target_label": args.target_label,
        "label_strategy": label_strategy,
        "experiment_mode": args.experiment_mode,
        "retrieval_metric": args.retrieval_metric,
        "pood_training_label_mode": args.pood_training_label,
        "pood_samples_per_target": args.p,
        "unique_pood_samples": len(perturbed_images),
        "poison_repeats": args.poison_repeats,
        "injected_poison_records": total_injected_records,
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
        forget_client_id=malicious_client_ids[0],
        forget_local_indices=None,
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
        forget_requests=poison_local_indices_by_client or None,
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
    targets_before = evaluate_targets(
        global_state, target_images, target_labels, device
    )
    targets_after = evaluate_targets(
        unlearned_state, target_images, target_labels, device
    )
    target_group_metrics = summarize_target_group(targets_before, targets_after)
    per_label_before = evaluate_per_class(
        global_state, federated_data.test, "cifar10", 10,
        args.eval_batch_size, args.num_workers, device
    )
    per_label_after = evaluate_per_class(
        unlearned_state, federated_data.test, "cifar10", 10,
        args.eval_batch_size, args.num_workers, device
    )

    target_summaries: list[dict] = []
    pood_target_summaries: list[dict] = []
    for record, before, after in zip(
        target_records, targets_before, targets_after
    ):
        success = bool(before["correct"] and not after["correct"])
        target_summary = {
            "target_number": record["target_number"],
            "source_index": record["source_index"],
            "true_label": record["true_label"],
            "class_name": record["class_name"],
            "proxy_prediction": record["proxy_prediction"],
            "before_unlearning": before,
            "after_unlearning": after,
            "true_label_confidence_drop": (
                float(before["true_label_confidence"])
                - float(after["true_label_confidence"])
            ),
            "simulation_success": success,
        }
        if args.target_count == 1:
            target_summary["attack_success"] = success
        target_summaries.append(target_summary)
        retrieved_group_label = int(record["retrieved_group_label"])
        poison_training_label = int(record["poison_training_label"])
        pood_target_summaries.append(
            {
                "target_number": record["target_number"],
                "target_source_index": record["source_index"],
                **record["source_metadata"],
                "cifar10_retrieved_group_label": retrieved_group_label,
                "cifar10_retrieved_group_class": (
                    cifar10_test.classes[retrieved_group_label]
                ),
                "pood_training_label_mode": args.pood_training_label,
                "cifar10_poison_label": poison_training_label,
                "cifar10_poison_class": (
                    cifar10_test.classes[poison_training_label]
                ),
                "selected_source_indices": record["selected_source_indices"],
                score_key: record["selected_scores"].tolist(),
                "selection_metrics": record["selection_metrics"],
                "perturbation": {
                    "enabled": args.experiment_mode in {"optimized", "no_pood"},
                    "steps": (
                        args.perturb_steps
                        if args.experiment_mode in {"optimized", "no_pood"}
                        else 0
                    ),
                    "learning_rate": args.perturb_lr,
                    "epsilon_linf_pixel_space": args.perturb_epsilon,
                    **record["perturbation_metrics"],
                    "final_mean_target_feature_similarity": record[
                        "perturbed_feature_metrics"
                    ]["mean_target_feature_similarity"],
                },
            }
        )
    all_selected_source_indices = [
        source_index
        for item in pood_target_summaries
        for source_index in item["selected_source_indices"]
    ]
    pood_summary = {
        "candidate_count": args.candidate_count,
        "retrieval": (
            f"exact_{args.retrieval_metric}_topk_in_"
            "resnet18_512d_feature_space"
        ),
        "retrieval_metric": args.retrieval_metric,
        "selection_group": selection_group,
        "k": args.k,
        "p": args.p,
        "p_per_target": args.p,
        "target_count": args.target_count,
        "total_selected_pood_samples": len(selected_images),
        "distinct_source_sample_count": len(set(all_selected_source_indices)),
        "pood_training_label_mode": args.pood_training_label,
        "optimization_schedule": "serial_independent_per_target",
        "per_target": pood_target_summaries,
    }
    if args.target_count == 1:
        pood_summary.update(pood_target_summaries[0])
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
        "target": target_summaries[0],
        "targets": target_summaries,
        "target_group": target_group_metrics,
        "pood": pood_summary,
        "poison_injection": {
            "enabled": args.experiment_mode != "no_pood",
            "experiment_mode": args.experiment_mode,
            "training_label_mode": args.pood_training_label,
            "malicious_client_id": malicious_client_ids[0],
            "malicious_client_ids": malicious_client_ids,
            "malicious_client_count": len(malicious_client_ids),
            "benign_client_sample_count": len(
                benign_malicious_datasets[malicious_client_ids[0]]
            ),
            "benign_client_sample_counts": {
                str(client_id): len(benign_malicious_datasets[client_id])
                for client_id in malicious_client_ids
            },
            "target_count": args.target_count,
            "pood_samples_per_target": args.p,
            "selected_pood_records": len(perturbed_images),
            "distinct_source_sample_count": len(set(all_selected_source_indices)),
            "poison_repeats": args.poison_repeats,
            "injected_poison_records": total_injected_records,
            "injected_poison_records_per_client": {
                str(client_id): len(
                    poison_local_indices_by_client.get(client_id, [])
                )
                for client_id in malicious_client_ids
            },
            "poison_local_indices_per_client": {
                str(client_id): {
                    "start": (
                        poison_local_indices_by_client[client_id][0]
                        if poison_local_indices_by_client.get(client_id)
                        else None
                    ),
                    "end": (
                        poison_local_indices_by_client[client_id][-1]
                        if poison_local_indices_by_client.get(client_id)
                        else None
                    ),
                    "count": len(
                        poison_local_indices_by_client.get(client_id, [])
                    ),
                }
                for client_id in malicious_client_ids
            },
        },
        "unlearning": {
            "method": (
                "federaser_reconstruction_without_forgetting"
                if args.experiment_mode == "no_pood"
                else "federaser_partial_data"
            ),
            "delta_t": args.unlearning_delta_t,
            "forget_client_ids": malicious_client_ids,
            "forget_mode": (
                "none" if args.experiment_mode == "no_pood" else "partial_data"
            ),
            "history_update_semantics": history.get(
                "history_update_semantics", "single_retained_round"
            ),
            "snapshot_count": len(history["snapshots"]),
            "history_saved_to_disk": args.keep_unlearning_history,
            "history_path": (
                str(history_path.resolve())
                if args.keep_unlearning_history
                else None
            ),
            "forgotten_injected_record_count": total_injected_records,
            "forgotten_injected_records_per_client": {
                str(client_id): len(indices)
                for client_id, indices in poison_local_indices_by_client.items()
            },
            "global_metrics": {"before": global_before, "after": global_after},
            "unique_forget_set_metrics": {
                "sample_count": len(perturbed_images),
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
    artifacts = {
        "target_images": target_images,
        "target_labels": target_labels,
        "target_indices": torch.tensor(target_indices),
        "target_features": target_features,
        "selected_images": selected_images,
        "pood_dataset": pood_dataset_name,
        "selected_source_labels": selected_source_labels,
        "poison_labels": poison_labels,
        "selected_features": selected_features,
        "perturbed_images": perturbed_images,
        "perturbed_features": perturbed_features,
        "universal_perturbations_pixel_space": universal_perturbations,
        "pood_target_numbers": torch.arange(
            1, args.target_count + 1
        ).repeat_interleave(args.p),
        "poison_local_indices": (
            torch.tensor(poison_local_indices_by_client[malicious_client_ids[0]])
            if len(malicious_client_ids) == 1
            and malicious_client_ids[0] in poison_local_indices_by_client
            else None
        ),
        "poison_local_indices_by_client": {
            client_id: torch.tensor(indices)
            for client_id, indices in poison_local_indices_by_client.items()
        },
    }
    if args.target_count == 1:
        artifacts.update(
            {
                "target_image": target_images[0],
                "target_label": int(target_labels[0]),
                "target_feature": target_features[0],
                "universal_perturbation_pixel_space": universal_perturbations[0:1],
            }
        )
    torch.save(artifacts, run_dir / "cifar_pua_artifacts.pt")
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
        f"{global_after['accuracy']:.2%} | target_acc="
        f"{target_group_metrics['accuracy_before']:.2%} -> "
        f"{target_group_metrics['accuracy_after']:.2%} | "
        f"mean_target_conf="
        f"{target_group_metrics['mean_true_label_confidence_before']:.2%} -> "
        f"{target_group_metrics['mean_true_label_confidence_after']:.2%} | "
        f"success={target_group_metrics['success_count']}/"
        f"{target_group_metrics['target_count']} "
        f"({target_group_metrics['success_rate']:.2%})",
        flush=True,
    )
    print(f"Results saved to {run_dir}", flush=True)
    return run_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
