"""Run the serial multi-target CIFAR-10/STL-10 POOD simulation."""

from __future__ import annotations

import argparse

from .CIFAR_PUA import run
from .STL_PUA import build_stl_parser


def build_group_parser() -> argparse.ArgumentParser:
    parser = build_stl_parser()
    parser.description = (
        "Serial multi-target CIFAR-10/STL-10 POOD simulation. Each target "
        "independently retrieves p POOD samples and optimizes its own universal "
        "perturbation before all POOD groups are merged for training and forgetting."
    )
    parser.set_defaults(
        target_count=10,
        output_dir="stl_pua_group_runs",
    )
    return parser


def main() -> None:
    run(build_group_parser().parse_args())


if __name__ == "__main__":
    main()
