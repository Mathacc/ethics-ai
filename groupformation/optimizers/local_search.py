"""Local search (hill climbing + simulated annealing).

See tasks/epic-4-algorithms/S4.3-local-search.md and S4.4-simulated-annealing.md.

Both methods start from a feasible random assignment and repeatedly propose swapping two
participants in *different* groups. Swaps preserve group sizes, so feasibility is invariant.
They differ only in the acceptance rule:

  * LocalSearch       — accept strictly improving swaps (greedy hill climbing); stop on a
                        no-improvement plateau or the iteration cap.
  * SimulatedAnnealing — also accept worsening swaps with probability exp(-Δ/T), with T
                        cooled geometrically; returns the best assignment ever seen.

Each optimizer records ``self.history`` (cost per iteration) for convergence plots (S5.3)
and ``self.final_cost``.
"""

from __future__ import annotations

import math

import numpy as np

from ..constraints import feasible_random_assignment, group_capacities, is_feasible
from ..models import Assignment, Participant
from ..objectives import IncrementalEvaluator
from .base import Optimizer

_EPS = 1e-12


def _propose(ids: list[str], mapping: dict[str, str], rng: np.random.Generator, max_tries: int = 32):
    """Pick two participants currently in different groups."""
    n = len(ids)
    for _ in range(max_tries):
        i, j = rng.integers(0, n), rng.integers(0, n)
        a, b = ids[i], ids[j]
        if mapping[a] != mapping[b]:
            return a, b
    return ids[i], ids[j]  # fallback (delta_swap will be 0 if same group)


def _start(
    participants: list[Participant], config, initial: Assignment | None = None
) -> tuple[IncrementalEvaluator, list[str]]:
    rng = np.random.default_rng(config.seed)
    ids = [p.id for p in participants]
    caps = group_capacities(len(participants), config)
    if initial is not None and is_feasible(initial, ids, caps)[0]:
        start = initial
    else:
        start = feasible_random_assignment(ids, caps, rng)
    return IncrementalEvaluator(participants, config.weights, start), ids


class LocalSearch(Optimizer):
    """Swap-based greedy hill climbing."""

    def solve(self, participants: list[Participant], config, initial: Assignment | None = None) -> Assignment:
        params = config.local_search or {}
        max_iter = int(params.get("max_iterations", 5000))
        patience = int(params.get("no_improvement_patience", 500))

        rng = np.random.default_rng(config.seed + 1)  # distinct stream from the start builder
        ev, ids = _start(participants, config, initial)

        current = ev.cost()
        self.history = [current]
        since_improve = 0

        for _ in range(max_iter):
            a, b = _propose(ids, ev.mapping, rng)
            delta = ev.delta_swap(a, b)
            if delta < -_EPS:
                ev.apply_swap(a, b)
                current += delta
                since_improve = 0
            else:
                since_improve += 1
            self.history.append(current)
            if since_improve >= patience:
                break

        self.final_cost = ev.cost()  # exact recompute (guards float drift)
        return ev.assignment()


class SimulatedAnnealing(Optimizer):
    """Hill climbing with probabilistic uphill moves; returns the best assignment seen."""

    def solve(self, participants: list[Participant], config, initial: Assignment | None = None) -> Assignment:
        params = config.simulated_annealing or {}
        temp = float(params.get("initial_temp", 1.0))
        cooling = float(params.get("cooling", 0.995))
        max_iter = int(params.get("max_iterations", 10000))

        rng = np.random.default_rng(config.seed + 1)
        ev, ids = _start(participants, config, initial)

        current = ev.cost()
        best_cost = current
        best_map = dict(ev.mapping)
        self.history = [current]

        for _ in range(max_iter):
            a, b = _propose(ids, ev.mapping, rng)
            delta = ev.delta_swap(a, b)
            accept = delta < 0 or (temp > _EPS and rng.random() < math.exp(-delta / temp))
            if accept:
                ev.apply_swap(a, b)
                current += delta
                if current < best_cost:
                    best_cost = current
                    best_map = dict(ev.mapping)
            self.history.append(current)
            temp *= cooling

        self.final_cost = best_cost
        return Assignment(best_map)
