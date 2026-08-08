from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch

StateDict = OrderedDict[str, torch.Tensor]


def apply_attack(
    update: Mapping[str, torch.Tensor],
    attack: str,
    attack_scale: float,
    total_clients: int,
    generator: torch.Generator,
) -> StateDict:
    if attack == "none":
        return OrderedDict((name, value.clone()) for name, value in update.items())
    if attack == "sign_flip":
        return OrderedDict(
            (name, value * -attack_scale) for name, value in update.items()
        )
    if attack == "gaussian":
        return OrderedDict(
            (
                name,
                torch.randn(
                    value.shape, dtype=value.dtype, generator=generator
                )
                * attack_scale,
            )
            for name, value in update.items()
        )
    if attack == "model_replacement":
        factor = attack_scale * total_clients
        return OrderedDict((name, value * factor) for name, value in update.items())
    raise ValueError(f"Unknown attack: {attack}")
