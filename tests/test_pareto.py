"""Tests for S9.1 (Pareto front of fairness trade-offs)."""

import os

from groupformation import pareto, plots
from groupformation.config import Config
from groupformation.data.generator import generate

WEIGHTS = {"skill_balance": 1.0, "diversity": 1.0, "preference": 0.5}


def _config():
    return Config(group_size=6, seed=1, weights=WEIGHTS,
                  local_search={"max_iterations": 2000, "no_improvement_patience": 500})


def test_run_pareto_shape_and_front(tmp_path):
    ps = generate(48, seed=42, scenario="skewed-skill")
    df = pareto.run_pareto(ps, _config(), "local_search", steps=7, out_dir=str(tmp_path))
    assert len(df) == 7
    assert {"alpha_skill", "skill_imbalance", "preference_satisfaction", "pareto_optimal"} <= set(df.columns)
    assert df["pareto_optimal"].sum() >= 1
    assert os.path.exists(tmp_path / "pareto.csv")


def test_trade_off_direction():
    # More weight on skill balance should not produce worse skill_imbalance than all-preference.
    ps = generate(48, seed=42, scenario="skewed-skill")
    df = pareto.run_pareto(ps, _config(), "local_search", steps=6, out_dir=str(__import__("tempfile").mkdtemp()))
    skill_focused = df.loc[df["alpha_skill"].idxmax(), "skill_imbalance"]
    pref_focused = df.loc[df["alpha_skill"].idxmin(), "skill_imbalance"]
    assert skill_focused <= pref_focused + 1e-9


def test_pareto_mask_logic():
    # point B dominates A; C is on the front (best preference)
    mask = pareto._pareto_mask(
        minimize=[0.5, 0.3, 0.8],          # skill_imbalance
        maximize=[0.6, 0.6, 0.9],          # preference_satisfaction
    )
    assert mask == [False, True, True]


def test_plot_pareto_writes_file(tmp_path):
    ps = generate(36, seed=42, scenario="skewed-skill")
    df = pareto.run_pareto(ps, _config(), "local_search", steps=6, out_dir=str(tmp_path))
    out = plots.plot_pareto(df, str(tmp_path / "pareto.png"))
    assert os.path.exists(out) and os.path.getsize(out) > 0
