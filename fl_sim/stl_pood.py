from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .cifar_pood import CIFAR10_MEAN, CIFAR10_STD
from .pood import feature_knn, same_label_selection


# Torchvision STL-10 labels:
# 0 airplane, 1 bird, 2 car, 3 cat, 4 deer,
# 5 dog, 6 horse, 7 monkey, 8 ship, 9 truck.
# CIFAR-10 has no monkey class; CIFAR-10 frog has no STL-10 counterpart.
STL10_TO_CIFAR10: Mapping[int, int] = {
    0: 0,  # airplane -> airplane
    1: 2,  # bird -> bird
    2: 1,  # car -> automobile
    3: 3,  # cat -> cat
    4: 4,  # deer -> deer
    5: 5,  # dog -> dog
    6: 7,  # horse -> horse
    8: 8,  # ship -> ship
    9: 9,  # truck -> truck
}


def build_stl10_cifar_transform() -> transforms.Compose:
    """Resize STL-10 to CIFAR resolution and apply CIFAR-10 normalization."""
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def collect_stl10_candidates(
    dataset: Dataset,
    target_label: int,
    candidate_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect mapped STL-10 candidates outside the CIFAR-10 target class.

    Monkey samples are excluded because CIFAR-10 has no corresponding output.
    A mapped class equal to the target is also excluded to retain the original
    semantic-conflict POOD setting.
    """
    if candidate_count < 1:
        raise ValueError("candidate count must be positive.")
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )
    images: list[torch.Tensor] = []
    source_labels: list[int] = []
    mapped_labels: list[int] = []
    source_indices: list[int] = []
    for tensor_index in order:
        source_index = int(tensor_index.item())
        image, raw_source_label = dataset[source_index]
        source_label = int(raw_source_label)
        mapped_label = STL10_TO_CIFAR10.get(source_label)
        if mapped_label is None or mapped_label == target_label:
            continue
        images.append(image)
        source_labels.append(source_label)
        mapped_labels.append(mapped_label)
        source_indices.append(source_index)
        if len(images) == candidate_count:
            break
    if len(images) != candidate_count:
        raise ValueError(
            f"Only found {len(images)} eligible STL-10 candidates after "
            f"excluding unmapped and target-label samples; requested "
            f"{candidate_count}."
        )
    return (
        torch.stack(images),
        torch.tensor(source_labels, dtype=torch.long),
        torch.tensor(mapped_labels, dtype=torch.long),
        torch.tensor(source_indices, dtype=torch.long),
    )


def select_stl10_pood(
    target_feature: torch.Tensor,
    candidate_features: torch.Tensor,
    stl10_source_labels: torch.Tensor,
    cifar10_mapped_labels: torch.Tensor,
    k: int,
    p: int,
    retrieval_metric: str = "cosine",
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Select the nearest coherent STL-10 class using deterministic mapping."""
    if not (
        len(candidate_features)
        == len(stl10_source_labels)
        == len(cifar10_mapped_labels)
    ):
        raise ValueError("Candidate features and labels must have equal lengths.")
    if not 1 <= p <= k <= len(candidate_features):
        raise ValueError("require 1 <= p <= k <= candidate count.")
    neighbor_positions, scores = feature_knn(
        target_feature, candidate_features, k, metric=retrieval_metric
    )
    poison_label, selected_neighbor_offsets = same_label_selection(
        cifar10_mapped_labels[neighbor_positions].tolist(), p
    )
    selected_offsets = torch.tensor(selected_neighbor_offsets, dtype=torch.long)
    selected_positions = neighbor_positions[selected_offsets]
    selected_source_labels = stl10_source_labels[selected_positions]
    if torch.unique(selected_source_labels).numel() != 1:
        raise AssertionError(
            "One-to-one STL-10 mapping should yield one coherent source class."
        )
    return (
        int(selected_source_labels[0].item()),
        poison_label,
        selected_positions,
        scores[selected_offsets],
    )
