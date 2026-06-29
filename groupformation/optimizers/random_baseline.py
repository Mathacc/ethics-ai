"""Random baseline — see tasks/epic-4-algorithms/S4.1-random-baseline.md.

A seeded random assignment that respects the hard constraints. This is the reference
point every optimized method is compared against (S5.2).
"""

from __future__ import annotations

import numpy as np

from ..constraints import feasible_random_assignment, group_capacities
from ..models import Assignment, Participant
from .base import Optimizer


class RandomBaseline(Optimizer):
    def solve(self, participants: list[Participant], config, initial: Assignment | None = None) -> Assignment:
        # A random baseline ignores any warm start by design.
        rng = np.random.default_rng(config.seed)
        capacities = group_capacities(len(participants), config)
        ids = [p.id for p in participants]
        return feasible_random_assignment(ids, capacities, rng)
