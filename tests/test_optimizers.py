"""Tests for S4.3 (local search) and S4.4 (simulated annealing)."""

from groupformation.config import Config
from groupformation.constraints import group_capacities, is_feasible
from groupformation.data.generator import generate
from groupformation.objectives import cost, index
from groupformation.optimizers.local_search import LocalSearch, SimulatedAnnealing
from groupformation.optimizers.random_baseline import RandomBaseline

WEIGHTS = {"skill_balance": 1.0, "diversity": 1.0, "preference": 0.5}


def _config(seed=1):
    return Config(
        group_size=6,
        n_groups=None,
        seed=seed,
        weights=WEIGHTS,
        local_search={"max_iterations": 4000, "no_improvement_patience": 800},
        simulated_annealing={"initial_temp": 0.5, "cooling": 0.999, "max_iterations": 6000},
    )


def _setup(seed=1, scenario="skewed-skill", n=60):
    ps = generate(n, seed=42, scenario=scenario)
    return ps, _config(seed), index(ps)


def test_local_search_feasible_and_improves():
    ps, config, by_id = _setup()
    caps = group_capacities(len(ps), config)
    ls = LocalSearch()
    a = ls.solve(ps, config)

    feasible, violations = is_feasible(a, [p.id for p in ps], caps)
    assert feasible, violations
    # final cost must not exceed the starting (random) cost
    assert ls.final_cost <= ls.history[0] + 1e-9
    # history's reported final matches an exact recompute
    assert abs(ls.final_cost - cost(a, by_id, WEIGHTS)) < 1e-9


def test_local_search_beats_random_baseline():
    ps, config, by_id = _setup()
    base = RandomBaseline().solve(ps, config)
    opt = LocalSearch().solve(ps, config)
    assert cost(opt, by_id, WEIGHTS) < cost(base, by_id, WEIGHTS)


def test_local_search_reproducible():
    ps, config, _ = _setup()
    a1 = LocalSearch().solve(ps, config)
    a2 = LocalSearch().solve(ps, config)
    assert a1.participant_to_group == a2.participant_to_group


def test_simulated_annealing_feasible_and_beats_random():
    ps, config, by_id = _setup()
    caps = group_capacities(len(ps), config)
    sa = SimulatedAnnealing()
    a = sa.solve(ps, config)

    feasible, violations = is_feasible(a, [p.id for p in ps], caps)
    assert feasible, violations
    assert cost(a, by_id, WEIGHTS) < cost(RandomBaseline().solve(ps, config), by_id, WEIGHTS)
    # SA returns the best seen → its reported best matches the returned assignment
    assert abs(sa.final_cost - cost(a, by_id, WEIGHTS)) < 1e-9


def test_simulated_annealing_reproducible():
    ps, config, _ = _setup()
    a1 = SimulatedAnnealing().solve(ps, config)
    a2 = SimulatedAnnealing().solve(ps, config)
    assert a1.participant_to_group == a2.participant_to_group
