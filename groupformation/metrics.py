"""Fairness metrics — see tasks/epic-5-fairness-evaluation/S5.1-fairness-metrics.md.

Computed independently of the cost function (objectives.py) to avoid circular evaluation:
the optimizers minimize ``cost``; these metrics judge the *result* on its own terms.

Returned keys (all lower = fairer, except ``preference_satisfaction`` where higher = better):
  * skill_imbalance        — mean across skills of the variance of per-group skill means.
  * experience_imbalance   — variance of per-group mean experience.
  * diversity_imbalance    — mean total-variation distance of group category mix vs global.
  * preference_satisfaction — fraction of preference relations honored (rate, higher better).
"""

from __future__ import annotations

import numpy as np

from .models import Assignment, Participant


def evaluate(assignment: Assignment, participants) -> dict[str, float]:
    by_id = participants if isinstance(participants, dict) else {p.id: p for p in participants}
    groups = assignment.groups()
    return {
        "skill_imbalance": _skill_imbalance(groups, by_id),
        "experience_imbalance": _experience_imbalance(groups, by_id),
        "diversity_imbalance": _diversity_imbalance(groups, by_id),
        "preference_satisfaction": _preference_satisfaction(assignment, by_id),
    }


def _skill_names(by_id: dict[str, Participant]) -> list[str]:
    names: set[str] = set()
    for p in by_id.values():
        names.update(p.skills)
    return sorted(names)


def _skill_imbalance(groups, by_id) -> float:
    names = _skill_names(by_id)
    if not names:
        return 0.0
    means = np.array(
        [[np.mean([by_id[m].skills.get(s, 0.0) for m in members]) for s in names] for members in groups.values()]
    )
    return float(np.mean(means.var(axis=0)))


def _experience_imbalance(groups, by_id) -> float:
    means = np.array([np.mean([by_id[m].experience for m in members]) for members in groups.values()])
    return float(means.var())


def _diversity_imbalance(groups, by_id) -> float:
    attrs: set[str] = set()
    for p in by_id.values():
        attrs.update(p.diversity_attrs)
    if not attrs:
        return 0.0
    tvs: list[float] = []
    for attr in sorted(attrs):
        global_dist = _dist([p.diversity_attrs.get(attr) for p in by_id.values()])
        for members in groups.values():
            local = _dist([by_id[m].diversity_attrs.get(attr) for m in members])
            tvs.append(0.5 * sum(abs(local.get(k, 0.0) - global_dist.get(k, 0.0)) for k in set(local) | set(global_dist)))
    return float(np.mean(tvs))


def _preference_satisfaction(assignment: Assignment, by_id) -> float:
    group_of = assignment.participant_to_group
    total = satisfied = 0
    for u, p in by_id.items():
        for v in p.preferences.get("likes", []):
            if v in by_id and v != u:
                total += 1
                satisfied += group_of[u] == group_of[v]
        for v in p.preferences.get("dislikes", []):
            if v in by_id and v != u:
                total += 1
                satisfied += group_of[u] != group_of[v]
    return 1.0 if total == 0 else satisfied / total


def _dist(values) -> dict[str, float]:
    counts: dict[str, int] = {}
    n = 0
    for v in values:
        if v is None:
            continue
        counts[v] = counts.get(v, 0) + 1
        n += 1
    return {k: c / n for k, c in counts.items()} if n else {}
