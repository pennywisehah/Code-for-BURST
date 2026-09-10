"""Run the STL-10 POOD simulation with target-aligned training labels.

The retrieval, perturbation, federated training, evaluation, and FedEraser
pipeline are shared with ``fl_sim.STL_PUA``. The treatment change is limited
to assigning the target sample's CIFAR-10 label to every injected POOD image.
"""

from __future__ import annotations

import argparse

from .CIFAR_PUA import run
from .STL_PUA import build_stl_parser


def build_target_label_parser() -> argparse.ArgumentParser:
    parser = build_stl_parser()
    parser.description = (
        "CIFAR-10/STL-10 POOD simulation where injected POOD samples use "
        "the selected CIFAR-10 target label during training."
    )
    parser.set_defaults(
        pood_training_label="target",
        output_dir="stl_pua_target_label_runs",
    )
    return parser


def main() -> None:
    run(build_target_label_parser().parse_args())


if __name__ == "__main__":
    main()
