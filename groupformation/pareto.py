"""Pareto front of fairness trade-offs — see tasks/epic-9-stretch/S9.1-pareto-front.md.

Competing soft objectives pull in different directions: maximizing skill balance can hurt
preference satisfaction and vice-versa. Sweeping the weight ratio between two objectives
and plotting the achieved (real) metrics reveals the trade-off curve; the non-dominated
points form the Pareto front.

Here we sweep skill-balance vs preference weight (diversity weight held at 0) and report
the achieved ``skill_imbalance`` (minimize) and ``preference_satisfaction`` (maximize).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .models import Participant
from .optimizers import get_optimizer


def run_pareto(
    participants: list[Participant],
    base_config,
    method: str = "local_search",
    steps: int = 11,
    out_dir: str = "outputs",
) -> pd.DataFrame:
    """Sweep the skill-vs-preference weight ratio and tag non-dominated solutions."""
    rows: list[dict] = []
    for alpha in np.linspace(0.0, 1.0, steps):
        weights = {"skill_balance": float(alpha), "preference": float(1.0 - alpha), "diversity": 0.0}
        cfg = replace(base_config, weights=weights)
        sol = get_optimizer(method).solve(participants, cfg)
        m = metrics.evaluate(sol, participants)
        rows.append(
            {
                "alpha_skill": round(float(alpha), 3),
                "skill_imbalance": m["skill_imbalance"],
                "preference_satisfaction": m["preference_satisfaction"],
            }
        )

    df = pd.DataFrame(rows)
    df["pareto_optimal"] = _pareto_mask(df["skill_imbalance"].values, df["preference_satisfaction"].values)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_dir}/pareto.csv", index=False)
    return df


def _pareto_mask(minimize, maximize) -> list[bool]:
    """Non-dominated mask: minimize ``minimize``, maximize ``maximize``."""
    minimize = np.asarray(minimize, dtype=float)
    maximize = np.asarray(maximize, dtype=float)
    mask = []
    for i in range(len(minimize)):
        dominated = np.any(
            (minimize <= minimize[i])
            & (maximize >= maximize[i])
            & ((minimize < minimize[i]) | (maximize > maximize[i]))
        )
        mask.append(not bool(dominated))
    return mask
