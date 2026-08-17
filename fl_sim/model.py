from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet

StateDict = OrderedDict[str, torch.Tensor]


class CNN2(nn.Module):
    """Two-convolution network used for MNIST and FashionMNIST."""

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(inputs)
        features = self.classifier[3](features)
        return self.classifier[4](features)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the 128-dimensional penultimate representation."""
        features = self.features(inputs)
        features = self.classifier[0](features)
        features = self.classifier[1](features)
        return self.classifier[2](features)


def extract_model_features(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Return the representation immediately before the classification head."""
    if isinstance(model, CNN2):
        return model.extract_features(inputs)
    if isinstance(model, ResNet):
        features = model.conv1(inputs)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)
        features = model.layer1(features)
        features = model.layer2(features)
        features = model.layer3(features)
        features = model.layer4(features)
        features = model.avgpool(features)
        return torch.flatten(features, 1)
    raise TypeError(
        f"Feature extraction is not implemented for {type(model).__name__}."
    )


def build_model(dataset_name: str) -> nn.Module:
    dataset_name = dataset_name.lower()
    if dataset_name in {"mnist", "fashionmnist"}:
        return CNN2(input_channels=1, num_classes=10)
    if dataset_name in {"cifar10", "cifar100"}:
        model = resnet18(
            weights=None, num_classes=10 if dataset_name == "cifar10" else 100
        )
        # CIFAR images are 32x32, so use the standard CIFAR ResNet stem.
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def floating_state_dict(model: nn.Module) -> StateDict:
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    )


def model_update(
    local_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
) -> StateDict:
    return OrderedDict(
        (name, local_state[name].detach().cpu() - global_state[name])
        for name in global_state
        if global_state[name].is_floating_point()
    )


def apply_update(
    global_state: Mapping[str, torch.Tensor], update: Mapping[str, torch.Tensor]
) -> StateDict:
    result: StateDict = OrderedDict()
    for name, value in global_state.items():
        if value.is_floating_point():
            result[name] = value + update[name].to(dtype=value.dtype)
        else:
            result[name] = value.clone()
    return result


def train_local(
    global_state: Mapping[str, torch.Tensor],
    dataset,
    model_name: str,
    local_epochs: int,
    learning_rate: float,
    batch_size: int,
    momentum: float,
    weight_decay: float,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> StateDict:
    # Also controls stochastic model layers (for example Dropout), not only shuffling.
    torch.manual_seed(seed)
    model = build_model(model_name).to(device)
    model.load_state_dict(global_state)
    model.train()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
    update = model_update(model.state_dict(), global_state)
    del model
    return update


@torch.no_grad()
def evaluate(
    state: Mapping[str, torch.Tensor],
    dataset,
    model_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, float]:
    model = build_model(model_name).to(device)
    model.load_state_dict(state)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        total_loss += criterion(logits, targets).item()
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate_per_class(
    state: Mapping[str, torch.Tensor],
    dataset,
    model_name: str,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, dict[str, float | int | None]]:
    """Evaluate class-wise accuracy in one pass over a labelled dataset."""
    model = build_model(model_name).to(device)
    model.load_state_dict(state)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    counts = torch.zeros(num_classes, dtype=torch.long)
    correct = torch.zeros(num_classes, dtype=torch.long)
    for inputs, targets in loader:
        predictions = model(inputs.to(device, non_blocking=True)).argmax(dim=1).cpu()
        targets = targets.cpu()
        counts += torch.bincount(targets, minlength=num_classes)
        correct += torch.bincount(
            targets[predictions == targets], minlength=num_classes
        )
    result: dict[str, dict[str, float | int | None]] = {}
    for label in range(num_classes):
        sample_count = int(counts[label].item())
        result[str(label)] = {
            "sample_count": sample_count,
            "accuracy": (
                float(correct[label].item() / sample_count)
                if sample_count
                else None
            ),
        }
    return result
