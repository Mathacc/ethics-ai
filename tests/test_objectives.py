"""Tests for S3.2 (soft constraints) and S4.2 (weighted cost + incremental delta)."""

import numpy as np

from groupformation.constraints import feasible_random_assignment, group_capacities
from groupformation.config import Config
from groupformation.data.generator import generate
from groupformation.models import Assignment, Participant
from groupformation.objectives import (
    IncrementalEvaluator,
    cost,
    diversity,
    index,
    preference_penalty,
    skill_balance,
)

WEIGHTS = {"skill_balance": 1.0, "diversity": 1.0, "preference": 0.5}


def _p(pid, coding, design, nat, likes=None, dislikes=None):
    prefs = {}
    if likes:
        prefs["likes"] = likes
    if dislikes:
        prefs["dislikes"] = dislikes
    return Participant(
        id=pid,
        skills={"coding": coding, "design": design},
        experience=0,
        diversity_attrs={"nationality": nat},
        preferences=prefs,
    )


def test_skill_balance_perfect_vs_worst():
    # Balanced: each group has one high + one low coder.
    by_id = index([_p("a", 1.0, 0.5, "IT"), _p("b", 0.0, 0.5, "IT"), _p("c", 1.0, 0.5, "IT"), _p("d", 0.0, 0.5, "IT")])
    balanced = Assignment({"a": "g0", "b": "g0", "c": "g1", "d": "g1"})
    skewed = Assignment({"a": "g0", "c": "g0", "b": "g1", "d": "g1"})
    assert skill_balance(balanced, by_id) < skill_balance(skewed, by_id)
    assert skill_balance(balanced, by_id) == 0.0  # identical group means


def test_diversity_even_vs_segregated():
    by_id = index([_p("a", 0.5, 0.5, "IT"), _p("b", 0.5, 0.5, "DE"), _p("c", 0.5, 0.5, "IT"), _p("d", 0.5, 0.5, "DE")])
    even = Assignment({"a": "g0", "b": "g0", "c": "g1", "d": "g1"})        # each group IT+DE
    segregated = Assignment({"a": "g0", "c": "g0", "b": "g1", "d": "g1"})  # g0 all IT, g1 all DE
    assert diversity(even, by_id) < diversity(segregated, by_id)
    assert diversity(even, by_id) == 0.0


def test_preference_likes_and_dislikes():
    by_id = index([_p("a", 0.5, 0.5, "IT", likes=["b"], dislikes=["c"]), _p("b", 0.5, 0.5, "IT"), _p("c", 0.5, 0.5, "IT"), _p("d", 0.5, 0.5, "IT")])
    good = Assignment({"a": "g0", "b": "g0", "c": "g1", "d": "g1"})  # likes b together, avoids c
    bad = Assignment({"a": "g0", "c": "g0", "b": "g1", "d": "g1"})   # opposite
    assert preference_penalty(good, by_id) == 0.0
    assert preference_penalty(bad, by_id) == 1.0


def test_no_preferences_zero_penalty():
    by_id = index([_p("a", 0.5, 0.5, "IT"), _p("b", 0.5, 0.5, "IT")])
    a = Assignment({"a": "g0", "b": "g1"})
    assert preference_penalty(a, by_id) == 0.0


def test_weights_change_cost():
    ps = generate(40, seed=1, scenario="skewed-skill")
    by_id = index(ps)
    a = feasible_random_assignment([p.id for p in ps], group_capacities(40, Config(group_size=5)), np.random.default_rng(0))
    c1 = cost(a, by_id, {"skill_balance": 1.0, "diversity": 0.0, "preference": 0.0})
    c2 = cost(a, by_id, {"skill_balance": 0.0, "diversity": 1.0, "preference": 0.0})
    assert c1 != c2


def test_incremental_matches_full_cost():
    ps = generate(48, seed=3, scenario="balanced")
    by_id = index(ps)
    caps = group_capacities(48, Config(group_size=6))
    a = feasible_random_assignment([p.id for p in ps], caps, np.random.default_rng(5))
    ev = IncrementalEvaluator(ps, WEIGHTS, a)
    assert abs(ev.cost() - cost(a, by_id, WEIGHTS)) < 1e-9


def test_delta_swap_matches_full_recompute():
    ps = generate(48, seed=4, scenario="skewed-skill")
    by_id = index(ps)
    caps = group_capacities(48, Config(group_size=6))
    a = feasible_random_assignment([p.id for p in ps], caps, np.random.default_rng(7))
    ev = IncrementalEvaluator(ps, WEIGHTS, a)
    rng = np.random.default_rng(11)
    ids = [p.id for p in ps]

    for _ in range(200):
        x, y = rng.choice(ids, size=2, replace=False)
        before = ev.cost()
        delta = ev.delta_swap(x, y)
        # independent full recompute of the swapped assignment
        m = dict(ev.mapping)
        m[x], m[y] = m[y], m[x]
        after = cost(Assignment(m), by_id, WEIGHTS)
        assert abs(delta - (after - before)) < 1e-9, (x, y, delta, after - before)


def test_apply_swap_keeps_caches_consistent():
    ps = generate(36, seed=6, scenario="minority-underrepresented")
    by_id = index(ps)
    caps = group_capacities(36, Config(group_size=6))
    a = feasible_random_assignment([p.id for p in ps], caps, np.random.default_rng(9))
    ev = IncrementalEvaluator(ps, WEIGHTS, a)
    rng = np.random.default_rng(13)
    ids = [p.id for p in ps]

    for _ in range(50):
        x, y = rng.choice(ids, size=2, replace=False)
        ev.apply_swap(x, y)
        assert abs(ev.cost() - cost(ev.assignment(), by_id, WEIGHTS)) < 1e-9
