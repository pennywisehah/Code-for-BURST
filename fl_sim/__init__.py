"""A small, dependency-free federated learning simulation framework."""

from .config import ExperimentConfig

__all__ = ["ExperimentConfig", "FederatedExperiment"]


def __getattr__(name: str):
    if name == "FederatedExperiment":
        from .experiment import FederatedExperiment

        return FederatedExperiment
    raise AttributeError(name)
