"""Matplotlib figures saved to outputs/ — see tasks/epic-5-fairness-evaluation/S5.3-plots.md.

Uses the non-interactive Agg backend so figures render headless (no display, no notebook).
Figures regenerate from the result tables / representative assignments produced by the
comparison harness (evaluation.py).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .models import Assignment, Participant  # noqa: E402

# Metrics to chart; True = lower is better (annotated in titles).
_METRICS = {
    "cost": True,
    "skill_imbalance": True,
    "experience_imbalance": True,
    "diversity_imbalance": True,
    "preference_satisfaction": False,
}


def plot_method_comparison(raw_df, path: str) -> str:
    """Grouped bar chart (mean ± std across seeds) per metric, one subplot each."""
    methods = list(raw_df["method"].unique())
    keys = list(_METRICS)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 4.2))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        means = [raw_df[raw_df.method == m][key].mean() for m in methods]
        stds = [raw_df[raw_df.method == m][key].std(ddof=0) for m in methods]
        ax.bar(methods, means, yerr=stds, capsize=4, color="#4C72B0")
        arrow = "↓ better" if _METRICS[key] else "↑ better"
        ax.set_title(f"{key}\n({arrow})", fontsize=10)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Method comparison (mean ± std across seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_convergence(histories: dict[str, list[float]], path: str) -> str:
    """Cost vs iteration for convergence-capable methods."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method, hist in histories.items():
        ax.plot(range(len(hist)), hist, label=method, linewidth=1.2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost (lower is better)")
    ax.set_title("Optimizer convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pareto(df, path: str) -> str:
    """Scatter of the weight sweep with the Pareto front highlighted (S9.1)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(df["skill_imbalance"], df["preference_satisfaction"], c="#BBBBBB", label="weight settings", zorder=2)

    front = df[df["pareto_optimal"]].sort_values("skill_imbalance")
    ax.plot(front["skill_imbalance"], front["preference_satisfaction"], "o-", color="#C44E52", label="Pareto front", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(f"{r['alpha_skill']:.1f}", (r["skill_imbalance"], r["preference_satisfaction"]), fontsize=7, alpha=0.7)

    ax.set_xlabel("skill_imbalance (↓ better)")
    ax.set_ylabel("preference_satisfaction (↑ better)")
    ax.set_title("Fairness trade-off: skill balance vs preference\n(labels = skill weight α)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_robustness(df, path: str) -> str:
    """Group-change rate and fairness delta vs perturbation size k, by perturbation/start."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    for (perturb, start), g in df.groupby(["perturbation", "start"]):
        agg = g.groupby("k")["group_change_rate"].mean()
        ax1.plot(agg.index, agg.values, marker="o", label=f"{perturb}/{start}")
        aggd = g.groupby("k")["d_skill_imbalance"].mean()
        ax2.plot(aggd.index, aggd.values, marker="o", label=f"{perturb}/{start}")

    ax1.set_xlabel("k (participants changed)")
    ax1.set_ylabel("group-change rate")
    ax1.set_title("Assignment churn vs perturbation size")
    ax1.legend(fontsize=8)

    ax2.axhline(0.0, color="grey", linewidth=0.8)
    ax2.set_xlabel("k (participants changed)")
    ax2.set_ylabel("Δ skill_imbalance (after − before)")
    ax2.set_title("Fairness drift vs perturbation size")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_skill_distribution(assignments: dict[str, Assignment], participants: list[Participant], path: str) -> str:
    """Box plot of per-group mean skill under each method's representative assignment."""
    by_id = {p.id: p for p in participants}
    data, labels = [], []
    for method, assignment in assignments.items():
        per_group = [
            float(np.mean([np.mean(list(by_id[m].skills.values())) for m in members]))
            for members in assignment.groups().values()
        ]
        data.append(per_group)
        labels.append(method)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("per-group mean skill")
    ax.set_title("Spread of group skill means (tighter = more balanced)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
