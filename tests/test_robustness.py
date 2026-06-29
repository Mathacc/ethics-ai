"""Tests for S6.1 (perturbations) and S6.2 (stability metrics + study)."""

import os

import numpy as np

from groupformation import plots, robustness
from groupformation.config import Config
from groupformation.constraints import group_capacities, is_feasible
from groupformation.models import Assignment

WEIGHTS = {"skill_balance": 1.0, "diversity": 1.0, "preference": 0.5}


def _config(seed=1):
    return Config(group_size=6, seed=seed, weights=WEIGHTS,
                  local_search={"max_iterations": 1500, "no_improvement_patience": 400})


def _participants(n=36, scenario="skewed-skill"):
    from groupformation.data.generator import generate

    return generate(n, seed=42, scenario=scenario)


def test_perturb_remove():
    ps = _participants(36)
    kept, removed = robustness.perturb_remove(ps, 6, np.random.default_rng(0))
    assert len(kept) == 30
    assert len(removed) == 6
    assert removed.isdisjoint({p.id for p in kept})


def test_perturb_add_no_id_collision():
    ps = _participants(36)
    aug, added = robustness.perturb_add(ps, 6, seed=1, scenario="balanced")
    assert len(aug) == 42
    assert len(added) == 6
    assert added.isdisjoint({p.id for p in ps})


def test_repair_to_capacity_is_feasible():
    ps = _participants(30)  # 5 groups of 6
    ids = [p.id for p in ps]
    caps = group_capacities(30, _config())
    # a deliberately broken mapping: everyone in g0, some unassigned
    broken = {pid: "g0" for pid in ids[:20]}
    repaired = robustness.repair_to_capacity(broken, ids, caps, np.random.default_rng(3))
    feasible, violations = is_feasible(repaired, ids, caps)
    assert feasible, violations


def test_group_change_and_pair_metrics():
    old = Assignment({"a": "g0", "b": "g0", "c": "g1", "d": "g1"})
    same = Assignment({"a": "g0", "b": "g0", "c": "g1", "d": "g1"})
    relabeled = Assignment({"a": "g1", "b": "g1", "c": "g0", "d": "g0"})  # same partition, swapped labels
    moved = Assignment({"a": "g0", "b": "g1", "c": "g0", "d": "g1"})
    common = {"a", "b", "c", "d"}

    assert robustness.group_change_rate(old, same, common) == 0.0
    # label-invariant: relabeling alone is not churn
    assert robustness.group_change_rate(old, relabeled, common) == 0.0
    assert robustness.pair_preservation(old, relabeled, common) == 1.0
    # a real move changes things
    assert robustness.group_change_rate(old, moved, common) > 0.0
    assert robustness.pair_preservation(old, moved, common) < 1.0


def test_run_robustness_shape_and_files(tmp_path):
    ps = _participants(36)
    df = robustness.run_robustness(
        ps, _config(), "local_search", ["remove", "add"], ks=[1, 3], n_trials=2,
        scenario="balanced", out_dir=str(tmp_path),
    )
    # 2 perturbations * 2 ks * 2 trials * 2 starts (warm+cold for local_search)
    assert len(df) == 2 * 2 * 2 * 2
    assert set(df["start"]) == {"cold", "warm"}
    assert "group_change_rate" in df.columns and "d_skill_imbalance" in df.columns
    assert (df["group_change_rate"].between(0.0, 1.0)).all()
    assert os.path.exists(tmp_path / "robustness_raw.csv")


def test_warm_start_churns_less_than_cold(tmp_path):
    # Warm-starting from the previous solution should, on average, change fewer assignments.
    ps = _participants(48)
    df = robustness.run_robustness(
        ps, _config(), "local_search", ["remove"], ks=[3], n_trials=4,
        out_dir=str(tmp_path),
    )
    means = df.groupby("start")["group_change_rate"].mean()
    assert means["warm"] <= means["cold"] + 1e-9


def test_plot_robustness_writes_file(tmp_path):
    ps = _participants(36)
    df = robustness.run_robustness(
        ps, _config(), "local_search", ["remove", "add"], ks=[1, 3], n_trials=2,
        out_dir=str(tmp_path),
    )
    out = plots.plot_robustness(df, str(tmp_path / "rob.png"))
    assert os.path.exists(out) and os.path.getsize(out) > 0
