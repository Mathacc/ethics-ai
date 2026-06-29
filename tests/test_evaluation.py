"""Tests for S5.1 (metrics), S5.2 (harness), and S5.3 (plots)."""

import os

from groupformation import evaluation, metrics, plots
from groupformation.config import Config
from groupformation.data.generator import generate
from groupformation.models import Assignment, Participant

WEIGHTS = {"skill_balance": 1.0, "diversity": 1.0, "preference": 0.5}


def _p(pid, coding, nat, exp=0, likes=None):
    return Participant(
        id=pid,
        skills={"coding": coding},
        experience=exp,
        diversity_attrs={"nationality": nat},
        preferences={"likes": likes} if likes else {},
    )


# ----- S5.1 metrics -----
def test_metrics_balanced_is_fair():
    by_id = [_p("a", 1.0, "IT", 0), _p("b", 0.0, "DE", 4), _p("c", 1.0, "IT", 0), _p("d", 0.0, "DE", 4)]
    balanced = Assignment({"a": "g0", "b": "g0", "c": "g1", "d": "g1"})
    m = metrics.evaluate(balanced, by_id)
    assert m["skill_imbalance"] == 0.0
    assert m["experience_imbalance"] == 0.0
    assert m["diversity_imbalance"] == 0.0


def test_metrics_segregated_is_unfair():
    by_id = [_p("a", 1.0, "IT"), _p("b", 0.0, "DE"), _p("c", 1.0, "IT"), _p("d", 0.0, "DE")]
    segregated = Assignment({"a": "g0", "c": "g0", "b": "g1", "d": "g1"})
    m = metrics.evaluate(segregated, by_id)
    assert m["skill_imbalance"] > 0.0
    assert m["diversity_imbalance"] > 0.0


def test_preference_satisfaction_rate():
    by_id = [_p("a", 0.5, "IT", likes=["b"]), _p("b", 0.5, "IT"), _p("c", 0.5, "IT"), _p("d", 0.5, "IT")]
    good = Assignment({"a": "g0", "b": "g0", "c": "g1", "d": "g1"})
    bad = Assignment({"a": "g0", "c": "g0", "b": "g1", "d": "g1"})
    assert metrics.evaluate(good, by_id)["preference_satisfaction"] == 1.0
    assert metrics.evaluate(bad, by_id)["preference_satisfaction"] == 0.0


# ----- S5.2 harness -----
def test_run_comparison_shape_and_files(tmp_path):
    ps = generate(36, seed=42, scenario="skewed-skill")
    config = Config(group_size=6, seed=1, weights=WEIGHTS,
                    local_search={"max_iterations": 1500, "no_improvement_patience": 400})
    methods = ["random", "local_search", "sa"]
    raw, agg, histories, rep = evaluation.run_comparison(ps, config, methods, seeds=3, out_dir=str(tmp_path))

    assert len(raw) == len(methods) * 3
    assert set(raw["method"]) == set(methods)
    for col in ["cost", "skill_imbalance", "preference_satisfaction", "runtime_s"]:
        assert col in raw.columns
    assert set(rep) == set(methods)
    assert set(histories) <= {"local_search", "sa"}
    assert os.path.exists(tmp_path / "results_raw.csv")
    assert os.path.exists(tmp_path / "results_agg.csv")


def test_optimized_methods_beat_random_on_average(tmp_path):
    ps = generate(36, seed=42, scenario="skewed-skill")
    config = Config(group_size=6, seed=1, weights=WEIGHTS,
                    local_search={"max_iterations": 2000, "no_improvement_patience": 500})
    raw, *_ = evaluation.run_comparison(ps, config, ["random", "local_search", "sa"], seeds=3, out_dir=str(tmp_path))
    mean_cost = raw.groupby("method")["cost"].mean()
    assert mean_cost["local_search"] < mean_cost["random"]
    assert mean_cost["sa"] < mean_cost["random"]


# ----- S5.3 plots -----
def test_plots_write_files(tmp_path):
    ps = generate(24, seed=42, scenario="balanced")
    config = Config(group_size=6, seed=1, weights=WEIGHTS,
                    local_search={"max_iterations": 1000, "no_improvement_patience": 300})
    raw, _agg, histories, rep = evaluation.run_comparison(ps, config, ["random", "local_search", "sa"], seeds=2, out_dir=str(tmp_path))

    p1 = plots.plot_method_comparison(raw, str(tmp_path / "cmp.png"))
    p2 = plots.plot_skill_distribution(rep, ps, str(tmp_path / "dist.png"))
    p3 = plots.plot_convergence(histories, str(tmp_path / "conv.png"))
    for p in (p1, p2, p3):
        assert os.path.exists(p) and os.path.getsize(p) > 0
