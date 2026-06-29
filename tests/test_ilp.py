"""Tests for S4.5 (ILP optimizer via OR-Tools)."""

from groupformation.config import Config
from groupformation.constraints import group_capacities, is_feasible
from groupformation.data.generator import generate
from groupformation.objectives import cost, index
from groupformation.optimizers.ilp import ILPOptimizer
from groupformation.optimizers.local_search import LocalSearch
from groupformation.optimizers.random_baseline import RandomBaseline

WEIGHTS = {"skill_balance": 1.0, "diversity": 1.0, "preference": 0.5}


def _config(seed=1):
    return Config(
        group_size=6,
        n_groups=None,
        seed=seed,
        weights=WEIGHTS,
        local_search={"max_iterations": 4000, "no_improvement_patience": 800},
        ilp={"time_limit_seconds": 10},
    )


def _setup(n=24, scenario="skewed-skill"):
    ps = generate(n, seed=42, scenario=scenario)
    return ps, _config(), index(ps)


def test_ilp_feasible_and_solved():
    ps, config, _ = _setup()
    caps = group_capacities(len(ps), config)
    opt = ILPOptimizer()
    a = opt.solve(ps, config)
    feasible, violations = is_feasible(a, [p.id for p in ps], caps)
    assert feasible, violations
    assert opt.status in ("OPTIMAL", "FEASIBLE")


def test_ilp_beats_random_baseline():
    ps, config, by_id = _setup()
    a_ilp = ILPOptimizer().solve(ps, config)
    a_rand = RandomBaseline().solve(ps, config)
    assert cost(a_ilp, by_id, WEIGHTS) < cost(a_rand, by_id, WEIGHTS)


def test_ilp_competitive_with_local_search():
    # ILP optimizes a linear surrogate, so it need not strictly beat local search on the
    # real cost — but on a small instance it should be in the same ballpark.
    ps, config, by_id = _setup()
    c_ilp = cost(ILPOptimizer().solve(ps, config), by_id, WEIGHTS)
    c_ls = cost(LocalSearch().solve(ps, config), by_id, WEIGHTS)
    assert c_ilp <= c_ls * 1.25


def test_ilp_respects_disabled_terms():
    # With only preference weight, ILP should drive the preference penalty very low.
    ps = generate(24, seed=7, scenario="balanced")
    by_id = index(ps)
    config = _config()
    config.weights = {"skill_balance": 0.0, "diversity": 0.0, "preference": 1.0}
    from groupformation.objectives import preference_penalty

    a = ILPOptimizer().solve(ps, config)
    assert preference_penalty(a, by_id) < 0.2
