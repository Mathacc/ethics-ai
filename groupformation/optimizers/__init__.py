"""Optimizer registry shared by the CLI and the comparison harness."""

from __future__ import annotations

from .base import Optimizer

METHODS = ("random", "local_search", "sa", "ilp")


def get_optimizer(method: str) -> Optimizer:
    if method == "random":
        from .random_baseline import RandomBaseline

        return RandomBaseline()
    if method == "local_search":
        from .local_search import LocalSearch

        return LocalSearch()
    if method == "sa":
        from .local_search import SimulatedAnnealing

        return SimulatedAnnealing()
    if method == "ilp":
        from .ilp import ILPOptimizer

        return ILPOptimizer()
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
