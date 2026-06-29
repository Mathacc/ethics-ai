"""Comparison harness — see tasks/epic-5-fairness-evaluation/S5.2-comparison-harness.md.

Runs each method across many seeds on a fixed dataset, computes fairness metrics (S5.1)
and the cost (S4.2), and aggregates into results tables saved under ``outputs/``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import pandas as pd

from . import metrics
from .models import Assignment, Participant
from .objectives import cost, index
from .optimizers import get_optimizer

CONVERGENCE_METHODS = ("local_search", "sa")


def run_comparison(
    participants: list[Participant],
    base_config,
    methods: list[str],
    seeds: int,
    out_dir: str = "outputs",
):
    """Return ``(raw_df, agg_df, histories, representative)`` and write CSVs to ``out_dir``.

    * ``raw_df``   — one row per (method, seed) with runtime, cost, and metrics.
    * ``agg_df``   — mean/std/min/max per method.
    * ``histories``— cost trajectory (first seed) for convergence-capable methods.
    * ``representative`` — one assignment per method (first seed) for distribution plots.
    """
    by_id = index(participants)
    seed_list = [base_config.seed + i for i in range(seeds)]
    rows: list[dict] = []
    histories: dict[str, list[float]] = {}
    representative: dict[str, Assignment] = {}

    for method in methods:
        for si, s in enumerate(seed_list):
            cfg = replace(base_config, seed=s)
            opt = get_optimizer(method)
            t0 = perf_counter()
            assignment = opt.solve(participants, cfg)
            runtime = perf_counter() - t0

            row = {
                "method": method,
                "seed": s,
                "runtime_s": runtime,
                "cost": cost(assignment, by_id, base_config.weights),
                **metrics.evaluate(assignment, by_id),
            }
            rows.append(row)

            if si == 0:
                representative[method] = assignment
                if method in CONVERGENCE_METHODS and hasattr(opt, "history"):
                    histories[method] = opt.history

    raw = pd.DataFrame(rows)
    agg = raw.drop(columns=["seed"]).groupby("method").agg(["mean", "std", "min", "max"])

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw.to_csv(f"{out_dir}/results_raw.csv", index=False)
    agg.to_csv(f"{out_dir}/results_agg.csv")
    return raw, agg, histories, representative


def summary_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Compact mean-per-method view for printing."""
    cols = ["cost", "skill_imbalance", "experience_imbalance", "diversity_imbalance", "preference_satisfaction", "runtime_s"]
    return raw.groupby("method")[cols].mean().sort_values("cost")
