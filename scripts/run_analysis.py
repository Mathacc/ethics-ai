"""Reproducible analysis pipeline for the report.

Generates the canonical dataset and every figure / results table under outputs/.
Run from the repo root:  python scripts/run_analysis.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow running from anywhere

from groupformation import evaluation, pareto, plots, robustness
from groupformation.config import Config
from groupformation.data import scenarios
from groupformation.data.generator import generate

# ---- Canonical experiment configuration (cited in the report) ----
N = 60
SCENARIO = "skewed-skill"
DATA_SEED = 42
EVAL_SEEDS = 5
METHODS = ["random", "local_search", "sa", "ilp"]
DATA_PATH = "data/participants.csv"


def main() -> None:
    base = Config.load("config.yaml")
    base = replace(base, ilp={"time_limit_seconds": 10})  # keep the sweep tractable

    print(f"[1/4] Generating dataset: n={N}, scenario={SCENARIO}, seed={DATA_SEED}")
    participants = generate(N, seed=DATA_SEED, scenario=SCENARIO)
    scenarios.save(participants, DATA_PATH)

    print(f"[2/4] Method comparison: {METHODS} x {EVAL_SEEDS} seeds")
    raw, _agg, histories, representative = evaluation.run_comparison(participants, base, METHODS, EVAL_SEEDS)
    plots.plot_method_comparison(raw, "outputs/fig_method_comparison.png")
    plots.plot_skill_distribution(representative, participants, "outputs/fig_skill_distribution.png")
    plots.plot_convergence(histories, "outputs/fig_convergence.png")
    print(evaluation.summary_table(raw).to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n[3/4] Robustness study (local_search, add/remove, k=1,3,6)")
    rob = robustness.run_robustness(participants, base, "local_search", ["remove", "add"], [1, 3, 6], n_trials=5)
    plots.plot_robustness(rob, "outputs/fig_robustness.png")

    print("[4/4] Pareto sweep (skill vs preference)")
    par = pareto.run_pareto(participants, base, "local_search", steps=11)
    plots.plot_pareto(par, "outputs/fig_pareto.png")

    print("\nDone. Artifacts in outputs/:")
    for f in (
        "results_raw.csv", "results_agg.csv", "robustness_raw.csv", "pareto.csv",
        "fig_method_comparison.png", "fig_skill_distribution.png", "fig_convergence.png",
        "fig_robustness.png", "fig_pareto.png",
    ):
        print(f"  outputs/{f}")


if __name__ == "__main__":
    main()
