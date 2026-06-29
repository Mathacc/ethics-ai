"""Tests for S7.1 (explanations) and S7.2 (rendering)."""

import json

from groupformation import explain
from groupformation.config import Config
from groupformation.data.generator import generate
from groupformation.models import Assignment, Participant
from groupformation.objectives import IncrementalEvaluator, cost, index
from groupformation.optimizers.local_search import LocalSearch

WEIGHTS = {"skill_balance": 1.0, "diversity": 1.0, "preference": 0.5}


def _p(pid, coding, nat, likes=None, dislikes=None):
    prefs = {}
    if likes:
        prefs["likes"] = likes
    if dislikes:
        prefs["dislikes"] = dislikes
    return Participant(id=pid, skills={"coding": coding}, experience=0, diversity_attrs={"nationality": nat}, preferences=prefs)


def test_preference_factor_reports_satisfaction():
    ps = [_p("a", 0.5, "IT", likes=["b"], dislikes=["c"]), _p("b", 0.5, "IT"), _p("c", 0.5, "IT"), _p("d", 0.5, "IT")]
    by_id = index(ps)
    good = Assignment({"a": "g0", "b": "g0", "c": "g1", "d": "g1"})
    factor = explain._preference_factor(good, by_id, "a")
    assert factor.score == 0.0  # both prefs satisfied
    assert "2/2 satisfied" in factor.detail


def test_counterfactual_is_faithful_to_cost():
    # On a random assignment, an improving swap must actually lower the cost by best_delta.
    ps = generate(36, seed=42, scenario="skewed-skill")
    by_id = index(ps)
    config = Config(group_size=6, seed=1, weights=WEIGHTS)
    from groupformation.constraints import feasible_random_assignment, group_capacities
    import numpy as np

    a = feasible_random_assignment([p.id for p in ps], group_capacities(36, config), np.random.default_rng(0))
    ev = IncrementalEvaluator(ps, WEIGHTS, a)

    for pid in list(by_id)[:10]:
        e = explain.explain_participant(a, by_id, ev, pid)
        if not e.well_placed:
            # apply the proposed swap on a fresh evaluator and check the cost drop
            ev2 = IncrementalEvaluator(ps, WEIGHTS, a)
            assert abs(ev2.delta_swap(pid, e.best_swap_partner) - e.best_delta) < 1e-9
            before = cost(a, by_id, WEIGHTS)
            m = dict(a.participant_to_group)
            m[pid], m[e.best_swap_partner] = m[e.best_swap_partner], m[pid]
            assert cost(Assignment(m), by_id, WEIGHTS) < before


def test_optimized_assignment_mostly_well_placed():
    ps = generate(36, seed=42, scenario="skewed-skill")
    config = Config(group_size=6, seed=1, weights=WEIGHTS,
                    local_search={"max_iterations": 5000, "no_improvement_patience": 1000})
    a = LocalSearch().solve(ps, config)
    result = explain.explain_assignment(a, ps, WEIGHTS)
    # a converged local optimum should leave most participants with no improving swap
    assert result["well_placed_rate"] > 0.7


def test_explain_assignment_serializable_and_complete():
    ps = generate(24, seed=7, scenario="balanced")
    config = Config(group_size=6, seed=1, weights=WEIGHTS)
    a = LocalSearch().solve(ps, config)
    result = explain.explain_assignment(a, ps, WEIGHTS)

    assert len(result["per_participant"]) == 24
    assert len(result["groups"]) == 4
    json.dumps(result)  # must be JSON-serializable


def test_render_participant_text():
    ps = [_p("a", 0.9, "IT", likes=["b"]), _p("b", 0.1, "DE"), _p("c", 0.5, "IT"), _p("d", 0.5, "DE")]
    by_id = index(ps)
    a = Assignment({"a": "g0", "b": "g0", "c": "g1", "d": "g1"})
    ev = IncrementalEvaluator(ps, WEIGHTS, a)
    text = explain.render_participant(explain.explain_participant(a, by_id, ev, "a"))
    assert "Participant a → group g0" in text
    assert "Preferences:" in text and "Skill:" in text
