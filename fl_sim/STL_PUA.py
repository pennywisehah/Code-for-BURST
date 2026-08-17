from __future__ import annotations

import argparse

from .CIFAR_PUA import build_parser, run


def build_stl_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = (
        "CIFAR-10 federated POOD-unlearning attack using labeled STL-10 "
        "as the out-of-distribution candidate pool."
    )
    parser.set_defaults(pood_dataset="stl10", output_dir="stl_pua_runs")
    return parser


def main() -> None:
    run(build_stl_parser().parse_args())


if __name__ == "__main__":
    main()
