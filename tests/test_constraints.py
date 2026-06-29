"""Tests for S3.1 (hard constraints) and S4.1 (random baseline)."""

import numpy as np
import pytest

from groupformation.config import Config
from groupformation.constraints import (
    feasible_random_assignment,
    group_capacities,
    is_feasible,
)
from groupformation.data.generator import generate
from groupformation.models import Assignment
from groupformation.optimizers.random_baseline import RandomBaseline


def test_capacities_sum_to_n_even():
    config = Config(group_size=6, n_groups=None)
    caps = group_capacities(60, config)
    assert sum(caps.values()) == 60
    assert set(caps.values()) == {6}


def test_capacities_balanced_with_remainder():
    config = Config(group_size=None, n_groups=7)
    caps = group_capacities(60, config)
    assert sum(caps.values()) == 60
    # Sizes differ by at most one.
    assert max(caps.values()) - min(caps.values()) <= 1


def test_feasible_assignment_passes_checks():
    ids = [f"p{i}" for i in range(12)]
    config = Config(group_size=None, n_groups=4)
    caps = group_capacities(12, config)
    a = feasible_random_assignment(ids, caps, np.random.default_rng(0))
    feasible, violations = is_feasible(a, ids, caps)
    assert feasible, violations


def test_detects_missing_and_wrong_size():
    ids = [f"p{i}" for i in range(6)]
    caps = {"g0": 3, "g1": 3}
    # p5 unassigned, g0 overfull.
    bad = Assignment({"p0": "g0", "p1": "g0", "p2": "g0", "p3": "g0", "p4": "g1"})
    feasible, violations = is_feasible(bad, ids, caps)
    assert not feasible
    assert any("p5 is not assigned" in v for v in violations)
    assert any("g0 has 4" in v for v in violations)


def test_random_baseline_feasible_and_reproducible():
    participants = generate(60, seed=42, scenario="balanced")
    config = Config(group_size=6, n_groups=None, seed=123)
    a1 = RandomBaseline().solve(participants, config)
    a2 = RandomBaseline().solve(participants, config)

    caps = group_capacities(60, config)
    feasible, violations = is_feasible(a1, [p.id for p in participants], caps)
    assert feasible, violations
    assert a1.participant_to_group == a2.participant_to_group  # seeded → identical


def test_capacities_reject_too_many_groups():
    config = Config(group_size=None, n_groups=10)
    with pytest.raises(ValueError):
        group_capacities(5, config)
