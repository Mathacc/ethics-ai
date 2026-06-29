"""Optimizer interface — shared by all methods in Epic 4."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Assignment, Participant


class Optimizer(ABC):
    @abstractmethod
    def solve(self, participants: list[Participant], config, initial: Assignment | None = None) -> Assignment:
        """Return a feasible Assignment.

        ``initial`` is an optional warm-start assignment (used by the robustness study to
        re-optimize from a previous solution). Methods that don't support warm starts
        ignore it.
        """
        raise NotImplementedError
