"""Hard constraints — see tasks/epic-3-constraints/S3.1-hard-constraints.md.

Hard constraints that must never be violated:
  * fixed group sizes (balanced partition of n across the groups),
  * each participant assigned to exactly one group,
  * no orphans and no unknown participants/groups.

Note: ``Assignment`` maps participant_id -> group_id, so "exactly one group per
participant" is structural — a participant cannot appear twice. We still verify
coverage (no missing/unknown ids) here.
"""

from __future__ import annotations

import numpy as np

from .models import Assignment


def group_capacities(n: int, config) -> dict[str, int]:
    """Return ``{group_id: capacity}`` summing to ``n``.

    Uses ``config.n_groups`` if set, otherwise derives the group count from
    ``config.group_size``. Sizes are balanced: they differ by at most one.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if getattr(config, "n_groups", None):
        k = int(config.n_groups)
    elif getattr(config, "group_size", None):
        k = max(1, round(n / int(config.group_size)))
    else:
        raise ValueError("config must set either n_groups or group_size")
    if k > n:
        raise ValueError(f"cannot form {k} groups from {n} participants")

    base, rem = divmod(n, k)
    return {f"g{i}": base + (1 if i < rem else 0) for i in range(k)}


def is_feasible(
    assignment: Assignment,
    participant_ids: list[str],
    capacities: dict[str, int],
) -> tuple[bool, list[str]]:
    """Return ``(feasible, violations)`` with human-readable violation messages."""
    violations: list[str] = []
    ids = set(participant_ids)
    assigned = set(assignment.participant_to_group)

    for pid in sorted(ids - assigned):
        violations.append(f"participant {pid} is not assigned")
    for pid in sorted(assigned - ids):
        violations.append(f"unknown participant {pid} in assignment")

    groups = assignment.groups()
    for gid in sorted(set(groups) - set(capacities)):
        violations.append(f"unknown group {gid} in assignment")
    for gid, cap in capacities.items():
        size = len(groups.get(gid, []))
        if size != cap:
            violations.append(f"group {gid} has {size} members, expected {cap}")

    return (len(violations) == 0, violations)


def feasible_random_assignment(
    participant_ids: list[str],
    capacities: dict[str, int],
    rng: np.random.Generator,
) -> Assignment:
    """Build a feasible assignment by shuffling participants and filling groups to capacity."""
    if sum(capacities.values()) != len(participant_ids):
        raise ValueError("capacities must sum to the number of participants")

    order = list(participant_ids)
    rng.shuffle(order)

    mapping: dict[str, str] = {}
    cursor = 0
    for gid, cap in capacities.items():
        for pid in order[cursor : cursor + cap]:
            mapping[pid] = gid
        cursor += cap
    return Assignment(mapping)
