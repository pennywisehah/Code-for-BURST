from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch

StateDict = OrderedDict[str, torch.Tensor]


def _validate(updates: list[Mapping[str, torch.Tensor]]) -> None:
    if not updates:
        raise ValueError("updates cannot be empty")
    keys = tuple(updates[0])
    if any(tuple(update) != keys for update in updates[1:]):
        raise ValueError("all updates must contain identical parameter keys")


def fedavg(
    updates: list[Mapping[str, torch.Tensor]], sample_counts: list[int]
) -> StateDict:
    _validate(updates)
    if len(updates) != len(sample_counts):
        raise ValueError("updates and sample_counts must be aligned")
    total = sum(sample_counts)
    if total <= 0:
        raise ValueError("sample counts must sum to a positive number")
    result: StateDict = OrderedDict()
    for name in updates[0]:
        value = torch.zeros_like(updates[0][name])
        for update, count in zip(updates, sample_counts):
            value.add_(update[name], alpha=count / total)
        result[name] = value
    return result


def _squared_distance(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    return sum(
        torch.sum((left[name].float() - right[name].float()).square()).item()
        for name in left
    )


def krum(updates: list[Mapping[str, torch.Tensor]], byzantine: int) -> StateDict:
    _validate(updates)
    count = len(updates)
    if byzantine < 0:
        raise ValueError("byzantine cannot be negative")
    if count < 2 * byzantine + 3:
        raise ValueError(
            f"Krum requires n >= 2f + 3, but received n={count}, f={byzantine}"
        )
    neighbors = count - byzantine - 2
    scores = []
    for index, candidate in enumerate(updates):
        distances = sorted(
            _squared_distance(candidate, other)
            for other_index, other in enumerate(updates)
            if other_index != index
        )
        scores.append(sum(distances[:neighbors]))
    selected = updates[min(range(count), key=scores.__getitem__)]
    return OrderedDict((name, value.clone()) for name, value in selected.items())


def trimmed_mean(
    updates: list[Mapping[str, torch.Tensor]],
    trim_ratio: float,
    chunk_size: int = 1_000_000,
) -> StateDict:
    _validate(updates)
    if not 0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    trim_count = int(len(updates) * trim_ratio)
    if len(updates) - 2 * trim_count < 1:
        raise ValueError("trim_ratio removes every update")
    result: StateDict = OrderedDict()
    for name, reference in updates[0].items():
        flat_result = torch.empty(reference.numel(), dtype=reference.dtype)
        for start in range(0, reference.numel(), chunk_size):
            stop = min(start + chunk_size, reference.numel())
            stacked = torch.stack(
                [update[name].reshape(-1)[start:stop] for update in updates]
            )
            ordered, _ = torch.sort(stacked, dim=0)
            kept = ordered[
                trim_count : len(updates) - trim_count if trim_count else len(updates)
            ]
            flat_result[start:stop] = kept.mean(dim=0)
        result[name] = flat_result.reshape(reference.shape)
    return result


def aggregate(
    method: str,
    updates: list[Mapping[str, torch.Tensor]],
    sample_counts: list[int],
    trim_ratio: float,
    byzantine: int,
    chunk_size: int,
) -> StateDict:
    if method == "fedavg":
        return fedavg(updates, sample_counts)
    if method == "krum":
        return krum(updates, byzantine)
    if method == "trimmed_mean":
        return trimmed_mean(updates, trim_ratio, chunk_size)
    raise ValueError(f"Unknown aggregation method: {method}")
