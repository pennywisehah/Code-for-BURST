from __future__ import annotations

import argparse

from .config import ExperimentConfig
from .experiment import FederatedExperiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configurable federated learning and Byzantine attack simulator."
    )
    parser.add_argument("--config", help="Path to a JSON configuration file.")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashionmnist", "cifar10", "cifar100"],
    )
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--iid", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--aggregation",
        choices=["fedavg", "krum", "trimmed_mean"],
        help="Server aggregation method.",
    )
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--malicious-fraction", type=float)
    parser.add_argument("--participation-fraction", type=float)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--attack",
        choices=["none", "sign_flip", "gaussian", "model_replacement"],
    )
    parser.add_argument("--attack-scale", type=float)
    parser.add_argument("--trim-ratio", type=float)
    parser.add_argument("--krum-byzantine", type=int)
    parser.add_argument("--aggregation-chunk-size", type=int)
    parser.add_argument(
        "--save-unlearning-history",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--unlearning-delta-t", type=int)
    parser.add_argument("--non-iid-alpha", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = (
        ExperimentConfig.from_json(args.config)
        if args.config
        else ExperimentConfig()
    )
    config.update(
        {key: value for key, value in vars(args).items() if key != "config"}
    )
    FederatedExperiment(config).run()
