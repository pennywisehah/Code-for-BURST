from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class FederatedData:
    clients: list[Subset]
    test: Dataset
    num_classes: int
    input_channels: int
    client_labels: list[int | None]


_DATASET_META = {
    "mnist": (datasets.MNIST, 10, 1),
    "fashionmnist": (datasets.FashionMNIST, 10, 1),
    "cifar10": (datasets.CIFAR10, 10, 3),
    "cifar100": (datasets.CIFAR100, 100, 3),
}


def _transforms(dataset_name: str) -> tuple[transforms.Compose, transforms.Compose]:
    if dataset_name == "mnist":
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        common = transforms.Compose([transforms.ToTensor(), normalize])
        return common, common
    if dataset_name == "fashionmnist":
        normalize = transforms.Normalize((0.2860,), (0.3530,))
        common = transforms.Compose([transforms.ToTensor(), normalize])
        return common, common
    if dataset_name == "cifar10":
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        )
    else:
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
    train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test = transforms.Compose([transforms.ToTensor(), normalize])
    return train, test


def _iid_partition(
    sample_count: int, num_clients: int, generator: torch.Generator
) -> list[list[int]]:
    shuffled = torch.randperm(sample_count, generator=generator).tolist()
    return [shuffled[client_id::num_clients] for client_id in range(num_clients)]


def _dirichlet_partition(
    targets: torch.Tensor,
    num_clients: int,
    num_classes: int,
    alpha: float,
    generator: torch.Generator,
) -> list[list[int]]:
    # Retry rare pathological draws that leave a client without data.
    for _ in range(100):
        partitions: list[list[int]] = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_indices = torch.where(targets == class_id)[0]
            order = torch.randperm(len(class_indices), generator=generator)
            class_indices = class_indices[order]
            concentration = torch.full((num_clients,), alpha)
            proportions = torch.distributions.Dirichlet(concentration).sample()
            raw_counts = proportions * len(class_indices)
            counts = torch.floor(raw_counts).to(torch.int64)
            remainder = len(class_indices) - int(counts.sum())
            if remainder:
                fractions = raw_counts - counts
                for index in torch.argsort(fractions, descending=True)[:remainder]:
                    counts[index] += 1
            start = 0
            for client_id, count in enumerate(counts.tolist()):
                partitions[client_id].extend(
                    class_indices[start : start + count].tolist()
                )
                start += count
        if min(map(len, partitions)) > 0:
            for partition in partitions:
                order = torch.randperm(len(partition), generator=generator).tolist()
                partition[:] = [partition[index] for index in order]
            return partitions
    raise RuntimeError(
        "Could not create non-empty Dirichlet partitions; increase non_iid_alpha "
        "or reduce num_clients."
    )


def _label_per_client_partition(
    targets: torch.Tensor,
    num_clients: int,
    num_classes: int,
    generator: torch.Generator,
) -> list[list[int]]:
    """Assign every class to exactly one client with client_id == class_id."""
    if num_clients != num_classes:
        raise ValueError(
            "label_per_client requires num_clients to equal the number of classes"
        )
    partitions: list[list[int]] = []
    for label in range(num_classes):
        label_indices = torch.where(targets == label)[0]
        if len(label_indices) == 0:
            raise ValueError(f"Dataset has no training samples for label {label}")
        order = torch.randperm(len(label_indices), generator=generator)
        partitions.append(label_indices[order].tolist())
    return partitions


def dataset_targets(dataset: Dataset) -> torch.Tensor:
    """Return labels without applying image transforms when metadata is available."""
    if isinstance(dataset, Subset):
        parent_targets = dataset_targets(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return parent_targets[indices]
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets, dtype=torch.long)
    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        return torch.as_tensor(dataset.tensors[1], dtype=torch.long)
    return torch.tensor([int(dataset[index][1]) for index in range(len(dataset))])


def unique_dataset_labels(dataset: Dataset) -> list[int]:
    return sorted(
        int(label) for label in torch.unique(dataset_targets(dataset)).tolist()
    )


def load_federated_data(
    dataset_name: str,
    data_dir: str,
    num_clients: int,
    iid: bool,
    non_iid_alpha: float,
    download: bool,
    seed: int,
    partition: str | None = None,
) -> FederatedData:
    dataset_name = dataset_name.lower()
    dataset_class, num_classes, input_channels = _DATASET_META[dataset_name]
    train_transform, test_transform = _transforms(dataset_name)
    root = str(Path(data_dir).expanduser())
    train_set = dataset_class(
        root=root, train=True, transform=train_transform, download=download
    )
    test_set = dataset_class(
        root=root, train=False, transform=test_transform, download=download
    )
    targets = torch.as_tensor(train_set.targets, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    resolved_partition = (partition or "auto").lower()
    if resolved_partition == "auto":
        resolved_partition = "iid" if iid else "dirichlet"
    if resolved_partition == "iid":
        indices = _iid_partition(len(train_set), num_clients, generator)
        client_labels: list[int | None] = [None] * num_clients
    elif resolved_partition == "dirichlet":
        torch.manual_seed(seed)
        indices = _dirichlet_partition(
            targets, num_clients, num_classes, non_iid_alpha, generator
        )
        client_labels = [None] * num_clients
    elif resolved_partition == "label_per_client":
        indices = _label_per_client_partition(
            targets, num_clients, num_classes, generator
        )
        client_labels = list(range(num_classes))
    else:
        raise ValueError(
            "partition must be auto, iid, dirichlet, or label_per_client"
        )
    clients = [Subset(train_set, client_indices) for client_indices in indices]
    return FederatedData(
        clients, test_set, num_classes, input_channels, client_labels
    )
