from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .model import build_model, extract_model_features
from .pood import feature_knn, selected_feature_metrics


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _channel_tensor(
    values: tuple[float, float, float], reference: torch.Tensor
) -> torch.Tensor:
    return reference.new_tensor(values).view(1, 3, 1, 1)


def load_cifar10_proxy(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset_name = str(checkpoint.get("dataset", "")).lower()
    if dataset_name != "cifar10":
        raise ValueError(
            "CIFAR POOD requires a CIFAR-10 checkpoint, "
            f"got {dataset_name!r}."
        )
    if "state_dict" not in checkpoint:
        raise ValueError("The proxy checkpoint has no state_dict.")
    model = build_model("cifar10")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def extract_features_and_predictions(
    model: nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        feature_batches.append(extract_model_features(model, batch).cpu())
        prediction_batches.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(feature_batches), torch.cat(prediction_batches)


def select_cifar100_pood(
    target_feature: torch.Tensor,
    candidate_features: torch.Tensor,
    cifar100_labels: torch.Tensor,
    cifar10_pseudo_labels: torch.Tensor,
    target_label: int,
    k: int,
    p: int,
    retrieval_metric: str = "cosine",
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Select a coherent CIFAR-100 class with one non-target CIFAR-10 label.

    CIFAR-100 labels cannot be used directly to train a 10-class model.  We first
    remove candidates that the proxy assigns to the target class, retrieve top-k
    by feature similarity, and group neighbors by
    (CIFAR-100 ground-truth class, CIFAR-10 proxy prediction).  The largest group
    supplies p POOD samples and its proxy prediction becomes the poison label.
    """
    if not (
        len(candidate_features)
        == len(cifar100_labels)
        == len(cifar10_pseudo_labels)
    ):
        raise ValueError("Candidate features and labels must have equal lengths.")
    if p < 1 or k < p:
        raise ValueError("require 1 <= p <= k.")
    eligible_positions = torch.where(cifar10_pseudo_labels != target_label)[0]
    if len(eligible_positions) < k:
        raise ValueError(
            f"Only {len(eligible_positions)} non-target candidates remain; "
            f"cannot retrieve k={k}."
        )
    local_positions, scores = feature_knn(
        target_feature,
        candidate_features[eligible_positions],
        k,
        metric=retrieval_metric,
    )
    neighbor_positions = eligible_positions[local_positions]
    keys = [
        (int(cifar100_labels[position]), int(cifar10_pseudo_labels[position]))
        for position in neighbor_positions.tolist()
    ]
    counts = Counter(keys)
    largest_count = max(counts.values())
    selected_key = next(key for key in keys if counts[key] == largest_count)
    selected_neighbor_offsets = [
        offset for offset, key in enumerate(keys) if key == selected_key
    ][:p]
    if len(selected_neighbor_offsets) < p:
        raise ValueError(
            "The largest (CIFAR-100 class, CIFAR-10 pseudo-label) group in "
            f"top-k has only {len(selected_neighbor_offsets)} samples; "
            "increase --k or reduce --p."
        )
    selected_offsets = torch.tensor(selected_neighbor_offsets, dtype=torch.long)
    return (
        selected_key[0],
        selected_key[1],
        neighbor_positions[selected_offsets],
        scores[selected_offsets],
    )


def add_cifar10_pixel_perturbation(
    normalized_images: torch.Tensor, perturbation: torch.Tensor
) -> torch.Tensor:
    """Apply a shared perturbation in RGB [0, 1] space and renormalize."""
    mean = _channel_tensor(CIFAR10_MEAN, normalized_images)
    std = _channel_tensor(CIFAR10_STD, normalized_images)
    pixel_images = normalized_images * std + mean
    perturbed_pixels = (pixel_images + perturbation).clamp(0.0, 1.0)
    return (perturbed_pixels - mean) / std


def denormalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    mean = _channel_tensor(CIFAR10_MEAN, images)
    std = _channel_tensor(CIFAR10_STD, images)
    return (images * std + mean).clamp(0.0, 1.0)


def optimize_cifar_universal_perturbation(
    model: nn.Module,
    selected_images: torch.Tensor,
    target_feature: torch.Tensor,
    device: torch.device,
    steps: int,
    learning_rate: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    if selected_images.ndim != 4 or selected_images.shape[1:] != (3, 32, 32):
        raise ValueError("CIFAR POOD images must have shape [p, 3, 32, 32].")
    if target_feature.ndim != 2 or target_feature.shape[0] != 1:
        raise ValueError("target_feature must have shape [1, feature_dimension].")
    if steps < 1 or learning_rate <= 0 or not 0 < epsilon <= 1:
        raise ValueError("Invalid perturbation optimization parameters.")

    images = selected_images.to(device)
    target = target_feature.detach().to(device)
    perturbation = torch.zeros(
        (1, 3, 32, 32), dtype=images.dtype, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([perturbation], lr=learning_rate)
    with torch.no_grad():
        initial_features = extract_model_features(model, images)
        initial_distance = torch.linalg.vector_norm(
            initial_features - target, dim=1
        ).mean()

    final_loss = initial_distance
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        perturbed_images = add_cifar10_pixel_perturbation(images, perturbation)
        perturbed_features = extract_model_features(model, perturbed_images)
        loss = torch.linalg.vector_norm(perturbed_features - target, dim=1).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            perturbation.clamp_(-epsilon, epsilon)
        final_loss = loss.detach()

    with torch.no_grad():
        perturbed_images = add_cifar10_pixel_perturbation(images, perturbation)
        perturbed_features = extract_model_features(model, perturbed_images)
        final_distance = torch.linalg.vector_norm(
            perturbed_features - target, dim=1
        ).mean()

    metrics = {
        "initial_mean_feature_distance": float(initial_distance.item()),
        "final_mean_feature_distance": float(final_distance.item()),
        "mean_feature_distance_reduction": float(
            (initial_distance - final_distance).item()
        ),
        "perturbation_linf": float(perturbation.detach().abs().max().item()),
        "perturbation_l2": float(
            torch.linalg.vector_norm(perturbation.detach()).item()
        ),
        "last_optimization_loss": float(final_loss.item()),
    }
    return (
        perturbation.detach().cpu(),
        perturbed_images.detach().cpu(),
        perturbed_features.detach().cpu(),
        metrics,
    )


def summarize_selected_features(
    target_feature: torch.Tensor, selected_features: torch.Tensor
) -> Mapping[str, float]:
    return selected_feature_metrics(target_feature, selected_features)
