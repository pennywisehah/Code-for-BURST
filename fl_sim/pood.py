from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image

from .data import _transforms
from .experiment import resolve_device
from .model import CNN2, build_model


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def latest_checkpoint(run_root: str | Path = "runs") -> Path:
    checkpoints = sorted(Path(run_root).glob("*/model.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No model.pt found under {Path(run_root)!s}; pass --checkpoint explicitly."
        )
    return checkpoints[-1]


def load_proxy_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[CNN2, dict]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset_name = str(checkpoint.get("dataset", "")).lower()
    if dataset_name != "mnist":
        raise ValueError(
            f"POOD minimal validation requires an MNIST checkpoint, got {dataset_name!r}."
        )
    model = build_model("mnist")
    if not isinstance(model, CNN2):
        raise TypeError("MNIST proxy model must be CNN2.")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def extract_features(
    model: CNN2,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    batches = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        batches.append(model.extract_features(batch).cpu())
    return torch.cat(batches, dim=0)


@torch.no_grad()
def choose_target(
    model: CNN2,
    dataset,
    target_label: int,
    target_index: int | None,
    device: torch.device,
) -> tuple[int, torch.Tensor, int, int]:
    if target_index is not None:
        image, label = dataset[target_index]
        label = int(label)
        if label != target_label:
            raise ValueError(
                f"MNIST index {target_index} has label {label}, not --target-label "
                f"{target_label}."
            )
        prediction = int(model(image.unsqueeze(0).to(device)).argmax(dim=1).item())
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
        f"Could not find a correctly classified MNIST sample with label {target_label}."
    )


def collect_candidates(
    dataset,
    excluded_label: int,
    candidate_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    images: list[torch.Tensor] = []
    labels: list[int] = []
    source_indices: list[int] = []
    for tensor_index in order:
        source_index = int(tensor_index.item())
        image, label = dataset[source_index]
        label = int(label)
        if label == excluded_label:
            continue
        images.append(image)
        labels.append(label)
        source_indices.append(source_index)
        if len(images) == candidate_count:
            break
    if len(images) != candidate_count:
        raise ValueError(
            f"Only found {len(images)} candidates after excluding label "
            f"{excluded_label}; requested {candidate_count}."
        )
    candidate_labels = torch.tensor(labels, dtype=torch.long)
    if torch.any(candidate_labels == excluded_label):
        raise AssertionError("Target-label samples leaked into the QMNIST candidate pool.")
    return (
        torch.stack(images),
        candidate_labels,
        torch.tensor(source_indices, dtype=torch.long),
    )


def cosine_knn(
    target_feature: torch.Tensor,
    candidate_features: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_feature.shape != (1, candidate_features.shape[1]):
        raise ValueError("Target and candidate feature dimensions do not match.")
    if not 1 <= k <= len(candidate_features):
        raise ValueError("k must be between 1 and the number of candidates.")
    target_normalized = F.normalize(target_feature.float(), dim=1)
    candidate_normalized = F.normalize(candidate_features.float(), dim=1)
    similarities = candidate_normalized @ target_normalized.T
    values, indices = torch.topk(similarities.squeeze(1), k=k, largest=True)
    return indices, values


def same_label_selection(
    neighbor_labels: Sequence[int], p: int
) -> tuple[int, list[int]]:
    if p < 1:
        raise ValueError("p must be at least 1.")
    counts = Counter(int(label) for label in neighbor_labels)
    max_count = max(counts.values())
    selected_label = next(
        int(label) for label in neighbor_labels if counts[int(label)] == max_count
    )
    selected_positions = [
        index
        for index, label in enumerate(neighbor_labels)
        if int(label) == selected_label
    ][:p]
    if len(selected_positions) < p:
        raise ValueError(
            f"The largest same-label group in the kNN result has only "
            f"{len(selected_positions)} samples; increase --k or reduce --p."
        )
    return selected_label, selected_positions


def selected_feature_metrics(
    target_feature: torch.Tensor,
    selected_features: torch.Tensor,
) -> dict[str, float]:
    """Compute cosine-similarity metrics for the final p POOD candidates."""
    if target_feature.ndim != 2 or target_feature.shape[0] != 1:
        raise ValueError("target_feature must have shape [1, feature_dimension].")
    if selected_features.ndim != 2 or selected_features.shape[0] < 1:
        raise ValueError(
            "selected_features must have shape [p, feature_dimension] with p >= 1."
        )
    if target_feature.shape[1] != selected_features.shape[1]:
        raise ValueError("Target and selected feature dimensions do not match.")

    target_normalized = F.normalize(target_feature.float(), dim=1)
    selected_normalized = F.normalize(selected_features.float(), dim=1)

    target_similarities = (selected_normalized @ target_normalized.T).squeeze(1)
    mean_target_similarity = float(target_similarities.mean().item())

    if len(selected_normalized) == 1:
        mean_internal_consistency = 1.0
    else:
        pairwise_similarities = selected_normalized @ selected_normalized.T
        upper_triangle = torch.triu_indices(
            len(selected_normalized), len(selected_normalized), offset=1
        )
        mean_internal_consistency = float(
            pairwise_similarities[
                upper_triangle[0], upper_triangle[1]
            ].mean().item()
        )

    return {
        "mean_target_feature_similarity": mean_target_similarity,
        "mean_candidate_internal_consistency": mean_internal_consistency,
    }


def add_pixel_perturbation(
    normalized_images: torch.Tensor,
    perturbation: torch.Tensor,
) -> torch.Tensor:
    """Add one shared perturbation in [0, 1] pixel space and renormalize."""
    pixel_images = normalized_images * MNIST_STD + MNIST_MEAN
    perturbed_pixels = (pixel_images + perturbation).clamp(0.0, 1.0)
    return (perturbed_pixels - MNIST_MEAN) / MNIST_STD


def optimize_universal_perturbation(
    model: CNN2,
    selected_images: torch.Tensor,
    target_feature: torch.Tensor,
    device: torch.device,
    steps: int,
    learning_rate: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Optimize a shared L-infinity-bounded perturbation for p POOD samples."""
    if selected_images.ndim != 4 or selected_images.shape[0] < 1:
        raise ValueError("selected_images must have shape [p, channels, height, width].")
    if target_feature.ndim != 2 or target_feature.shape[0] != 1:
        raise ValueError("target_feature must have shape [1, feature_dimension].")
    if steps < 1:
        raise ValueError("Perturbation steps must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("Perturbation learning rate must be positive.")
    if not 0 < epsilon <= 1:
        raise ValueError("Perturbation epsilon must be in (0, 1].")

    images = selected_images.to(device)
    target = target_feature.detach().to(device)
    perturbation = torch.zeros(
        (1, *images.shape[1:]), dtype=images.dtype, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([perturbation], lr=learning_rate)

    with torch.no_grad():
        initial_features = model.extract_features(images)
        initial_distance = torch.linalg.vector_norm(
            initial_features - target, dim=1
        ).mean()

    final_loss = initial_distance
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        perturbed_images = add_pixel_perturbation(images, perturbation)
        perturbed_features = model.extract_features(perturbed_images)
        # The mean differs from the PDF's sum only by the constant factor 1 / p.
        loss = torch.linalg.vector_norm(perturbed_features - target, dim=1).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            perturbation.clamp_(-epsilon, epsilon)
        final_loss = loss.detach()

    with torch.no_grad():
        perturbed_images = add_pixel_perturbation(images, perturbation)
        perturbed_features = model.extract_features(perturbed_images)
        final_distance = torch.linalg.vector_norm(
            perturbed_features - target, dim=1
        ).mean()
        perturbation_linf = perturbation.detach().abs().max()
        perturbation_l2 = torch.linalg.vector_norm(perturbation.detach())

    metrics = {
        "initial_mean_feature_distance": float(initial_distance.item()),
        "final_mean_feature_distance": float(final_distance.item()),
        "mean_feature_distance_reduction": float(
            (initial_distance - final_distance).item()
        ),
        "perturbation_linf": float(perturbation_linf.item()),
        "perturbation_l2": float(perturbation_l2.item()),
        "last_optimization_loss": float(final_loss.item()),
    }
    return (
        perturbation.detach().cpu(),
        perturbed_images.detach().cpu(),
        perturbed_features.detach().cpu(),
        metrics,
    )


def denormalize_mnist(images: torch.Tensor) -> torch.Tensor:
    return (images * MNIST_STD + MNIST_MEAN).clamp(0.0, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct a minimal MNIST/QMNIST POOD nearest-neighbor set."
    )
    parser.add_argument("--checkpoint", help="Path to an MNIST model.pt checkpoint.")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="pood_runs")
    parser.add_argument("--target-label", type=int, default=3)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--p", type=int, default=5)
    parser.add_argument("--perturb-steps", type=int, default=200)
    parser.add_argument("--perturb-lr", type=float, default=0.01)
    parser.add_argument(
        "--perturb-epsilon",
        type=float,
        default=0.3,
        help="L-infinity perturbation budget in the original [0, 1] pixel space.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.target_label <= 9:
        raise ValueError("--target-label must be between 0 and 9.")
    if args.candidate_count < 1 or args.batch_size < 1:
        raise ValueError("--candidate-count and --batch-size must be positive.")
    if args.k < args.p:
        raise ValueError("--k must be greater than or equal to --p.")
    if args.perturb_steps < 1 or args.perturb_lr <= 0:
        raise ValueError("--perturb-steps and --perturb-lr must be positive.")
    if not 0 < args.perturb_epsilon <= 1:
        raise ValueError("--perturb-epsilon must be in (0, 1].")

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.run_root)
    )
    device = resolve_device(args.device)
    model, checkpoint = load_proxy_model(checkpoint_path, device)
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

    target_index, target_image, target_label, target_prediction = choose_target(
        model, mnist_test, args.target_label, args.target_index, device
    )
    candidate_images, candidate_labels, candidate_source_indices = collect_candidates(
        qmnist_pool, target_label, args.candidate_count, args.seed
    )
    target_feature = extract_features(
        model, target_image.unsqueeze(0), args.batch_size, device
    )
    candidate_features = extract_features(
        model, candidate_images, args.batch_size, device
    )
    neighbor_positions, neighbor_similarities = cosine_knn(
        target_feature, candidate_features, args.k
    )
    neighbor_labels = candidate_labels[neighbor_positions]
    selected_label, selected_neighbor_positions = same_label_selection(
        neighbor_labels.tolist(), args.p
    )
    selected_candidate_positions = neighbor_positions[selected_neighbor_positions]
    selected_features = candidate_features[selected_candidate_positions]
    selection_metrics = selected_feature_metrics(target_feature, selected_features)
    (
        universal_perturbation,
        perturbed_selected_images,
        perturbed_selected_features,
        perturbation_metrics,
    ) = optimize_universal_perturbation(
        model=model,
        selected_images=candidate_images[selected_candidate_positions],
        target_feature=target_feature,
        device=device,
        steps=args.perturb_steps,
        learning_rate=args.perturb_lr,
        epsilon=args.perturb_epsilon,
    )
    perturbed_selection_metrics = selected_feature_metrics(
        target_feature, perturbed_selected_features
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = Path(args.output_dir) / (
        f"{timestamp}-mnist-qmnist-target{target_label}-seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    neighbor_records = []
    for rank, (candidate_position, similarity) in enumerate(
        zip(neighbor_positions.tolist(), neighbor_similarities.tolist()), start=1
    ):
        neighbor_records.append(
            {
                "rank": rank,
                "candidate_position": candidate_position,
                "qmnist_source_index": int(candidate_source_indices[candidate_position]),
                "label": int(candidate_labels[candidate_position]),
                "cosine_similarity": float(similarity),
                "selected_for_pood": candidate_position
                in selected_candidate_positions.tolist(),
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_accuracy": None,
        "proxy_device": str(device),
        "feature_dimension": int(target_feature.shape[1]),
        "target": {
            "dataset": "MNIST test",
            "source_index": target_index,
            "label": target_label,
            "prediction": target_prediction,
        },
        "candidate_pool": {
            "dataset": "QMNIST test50k",
            "count": args.candidate_count,
            "excluded_label": target_label,
            "excluded_label_count_after_filter": int(
                (candidate_labels == target_label).sum().item()
            ),
        },
        "knn": {"metric": "cosine_similarity", "k": args.k},
        "same_label_selection": {
            "p": args.p,
            "selected_label": selected_label,
            "selected_count": len(selected_candidate_positions),
            **selection_metrics,
        },
        "universal_perturbation": {
            "objective": "mean_l2_feature_distance",
            "optimizer": "adam",
            "steps": args.perturb_steps,
            "learning_rate": args.perturb_lr,
            "epsilon_linf_pixel_space": args.perturb_epsilon,
            **perturbation_metrics,
            "final_mean_target_feature_similarity": perturbed_selection_metrics[
                "mean_target_feature_similarity"
            ],
            "target_feature_similarity_improvement": (
                perturbed_selection_metrics["mean_target_feature_similarity"]
                - selection_metrics["mean_target_feature_similarity"]
            ),
            "final_mean_candidate_internal_consistency": perturbed_selection_metrics[
                "mean_candidate_internal_consistency"
            ],
        },
        "neighbors": neighbor_records,
    }
    metrics_path = checkpoint_path.parent / "metrics.csv"
    if metrics_path.exists():
        lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > 1:
            summary["checkpoint_accuracy"] = float(lines[-1].split(",")[2])

    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    torch.save(
        {
            "target_image": target_image,
            "target_label": target_label,
            "target_feature": target_feature.squeeze(0),
            "candidate_images": candidate_images,
            "candidate_labels": candidate_labels,
            "candidate_source_indices": candidate_source_indices,
            "candidate_features": candidate_features,
            "neighbor_positions": neighbor_positions,
            "neighbor_similarities": neighbor_similarities,
            "selected_candidate_positions": selected_candidate_positions,
            "selected_images": candidate_images[selected_candidate_positions],
            "selected_labels": candidate_labels[selected_candidate_positions],
            "selected_features": selected_features,
            "universal_perturbation_pixel_space": universal_perturbation,
            "perturbed_selected_images": perturbed_selected_images,
            "perturbed_selected_features": perturbed_selected_features,
            "perturbation_metrics": perturbation_metrics,
            "selected_features": candidate_features[selected_candidate_positions],
        },
        run_dir / "pood_knn.pt",
    )

    target_visual = denormalize_mnist(target_image.unsqueeze(0))
    neighbor_visuals = denormalize_mnist(candidate_images[neighbor_positions])
    selected_visuals = denormalize_mnist(candidate_images[selected_candidate_positions])
    perturbed_visuals = denormalize_mnist(perturbed_selected_images)
    save_image(target_visual, run_dir / "target.png")
    save_image(
        torch.cat([target_visual, neighbor_visuals]),
        run_dir / "knn_neighbors.png",
        nrow=min(7, args.k + 1),
        padding=2,
    )
    save_image(
        torch.cat([target_visual, selected_visuals]),
        run_dir / "selected_pood.png",
        nrow=args.p + 1,
        padding=2,
    )
    save_image(
        torch.cat([target_visual, perturbed_visuals]),
        run_dir / "perturbed_pood.png",
        nrow=args.p + 1,
        padding=2,
    )

    print(
        f"Target MNIST index={target_index}, label={target_label}, "
        f"prediction={target_prediction}"
    )
    print(
        f"QMNIST candidates={args.candidate_count}, excluded_label={target_label}, "
        f"k={args.k}, p={args.p}, selected_label={selected_label}"
    )
    print(
        "POOD metrics | "
        f"mean_target_feature_similarity="
        f"{selection_metrics['mean_target_feature_similarity']:.6f} | "
        f"mean_candidate_internal_consistency="
        f"{selection_metrics['mean_candidate_internal_consistency']:.6f}"
    )
    print(
        "Universal perturbation | "
        f"mean_feature_distance="
        f"{perturbation_metrics['initial_mean_feature_distance']:.6f}->"
        f"{perturbation_metrics['final_mean_feature_distance']:.6f} | "
        f"mean_target_similarity="
        f"{selection_metrics['mean_target_feature_similarity']:.6f}->"
        f"{perturbed_selection_metrics['mean_target_feature_similarity']:.6f} | "
        f"linf={perturbation_metrics['perturbation_linf']:.6f}"
    )
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()