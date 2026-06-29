"""Robustness analysis — see tasks/epic-6-robustness/.

S6.1 — perturb the participant set (add/remove), re-solve, measure assignment churn.
S6.2 — stability metrics (group-change rate, label-invariant pair preservation) and the
       change in fairness metrics, comparing cold-start vs warm-start re-solving.

Group labels are arbitrary, so the group-change rate aligns the new groups to the old
ones by maximum membership overlap before comparing. ``pair_preservation`` is fully
label-invariant (it compares co-membership of participant pairs) and corroborates it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import metrics
from .constraints import group_capacities
from .data.generator import generate
from .models import Assignment, Participant
from .optimizers import get_optimizer

WARM_CAPABLE = ("local_search", "sa", "ilp")


# --------------------------------------------------------------------- perturbations
def perturb_remove(participants: list[Participant], k: int, rng: np.random.Generator):
    """Remove ``k`` random participants. Returns (kept, removed_ids)."""
    idx = rng.choice(len(participants), size=k, replace=False)
    removed = {participants[i].id for i in idx}
    kept = [p for p in participants if p.id not in removed]
    return kept, removed


def perturb_add(participants: list[Participant], k: int, seed: int, scenario: str):
    """Add ``k`` freshly generated participants with non-colliding ids. Returns (augmented, added_ids)."""
    existing = {p.id for p in participants}
    batch = generate(k, seed=seed, scenario=scenario)
    idmap = {p.id: f"x{i:03d}" for i, p in enumerate(batch)}
    # ensure no collision with existing ids
    assert not (set(idmap.values()) & existing), "generated ids collided with existing"
    added: list[Participant] = []
    for p in batch:
        prefs = {kind: [idmap[t] for t in lst if t in idmap] for kind, lst in p.preferences.items()}
        added.append(replace(p, id=idmap[p.id], preferences={k: v for k, v in prefs.items() if v}))
    return participants + added, set(idmap.values())


def repair_to_capacity(
    base_mapping: dict[str, str], ids: list[str], capacities: dict[str, int], rng: np.random.Generator
) -> Assignment:
    """Adapt a (possibly stale) mapping into a feasible assignment for the new id set.

    Keeps participants in their existing group where valid, then rebalances overfull/empty
    slots — the warm-start initializer.
    """
    groups: dict[str, list[str]] = {g: [] for g in capacities}
    floating: list[str] = []
    for pid in ids:
        g = base_mapping.get(pid)
        if g in groups:
            groups[g].append(pid)
        else:
            floating.append(pid)

    for g, cap in capacities.items():
        if len(groups[g]) > cap:
            rng.shuffle(groups[g])
            floating.extend(groups[g][cap:])
            groups[g] = groups[g][:cap]

    rng.shuffle(floating)
    cursor = 0
    for g, cap in capacities.items():
        while len(groups[g]) < cap and cursor < len(floating):
            groups[g].append(floating[cursor])
            cursor += 1

    return Assignment({pid: g for g, members in groups.items() for pid in members})


# --------------------------------------------------------------------- stability metrics
def group_change_rate(old: Assignment, new: Assignment, common: set[str]) -> float:
    """Fraction of common participants who changed group, after aligning group labels."""
    if not common:
        return 0.0
    relabel = _align_labels(old, new, common)
    changed = sum(1 for pid in common if relabel.get(new.participant_to_group[pid]) != old.participant_to_group[pid])
    return changed / len(common)


def pair_preservation(old: Assignment, new: Assignment, common: set[str]) -> float:
    """Label-invariant: fraction of common pairs whose same-group status is preserved."""
    members = sorted(common)
    if len(members) < 2:
        return 1.0
    agree = total = 0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            same_old = old.participant_to_group[a] == old.participant_to_group[b]
            same_new = new.participant_to_group[a] == new.participant_to_group[b]
            agree += same_old == same_new
            total += 1
    return agree / total


def _align_labels(old: Assignment, new: Assignment, common: set[str]) -> dict[str, str]:
    """Greedily map new group ids -> old group ids by maximum membership overlap."""
    old_groups: dict[str, set[str]] = {}
    new_groups: dict[str, set[str]] = {}
    for pid in common:
        old_groups.setdefault(old.participant_to_group[pid], set()).add(pid)
        new_groups.setdefault(new.participant_to_group[pid], set()).add(pid)

    overlaps = sorted(
        ((len(o & n), ng, og) for og, o in old_groups.items() for ng, n in new_groups.items()),
        reverse=True,
    )
    relabel: dict[str, str] = {}
    used_old: set[str] = set()
    for _, ng, og in overlaps:
        if ng not in relabel and og not in used_old:
            relabel[ng] = og
            used_old.add(og)
    # any unmatched new group keeps its own label (counts as a change)
    return relabel


# --------------------------------------------------------------------- study driver
def run_robustness(
    participants: list[Participant],
    base_config,
    method: str,
    perturbations: list[str],
    ks: list[int],
    n_trials: int,
    scenario: str = "balanced",
    out_dir: str = "outputs",
) -> pd.DataFrame:
    """Run the perturbation study and return a tidy DataFrame (also written to CSV)."""
    base_assignment = get_optimizer(method).solve(participants, base_config)
    base_ids = [p.id for p in participants]
    base_metrics = metrics.evaluate(base_assignment, participants)
    starts = ("cold", "warm") if method in WARM_CAPABLE else ("cold",)

    rows: list[dict] = []
    for perturb in perturbations:
        for k in ks:
            for trial in range(n_trials):
                rng = np.random.default_rng(base_config.seed + 1000 * trial + k)
                if perturb == "remove":
                    new_ps, _ = perturb_remove(participants, k, rng)
                    common = set(p.id for p in new_ps)
                else:
                    new_ps, _ = perturb_add(participants, k, seed=base_config.seed + trial, scenario=scenario)
                    common = set(base_ids)

                caps = group_capacities(len(new_ps), base_config)
                cfg = replace(base_config, seed=base_config.seed + trial)
                new_map = {p.id: base_assignment.participant_to_group.get(p.id) for p in new_ps}

                for start in starts:
                    initial = repair_to_capacity(new_map, [p.id for p in new_ps], caps, rng) if start == "warm" else None
                    sol = get_optimizer(method).solve(new_ps, cfg, initial=initial)
                    after = metrics.evaluate(sol, new_ps)
                    rows.append(
                        {
                            "perturbation": perturb,
                            "k": k,
                            "trial": trial,
                            "start": start,
                            "group_change_rate": group_change_rate(base_assignment, sol, common),
                            "pair_preservation": pair_preservation(base_assignment, sol, common),
                            **{f"d_{m}": after[m] - base_metrics[m] for m in base_metrics},
                        }
                    )

    df = pd.DataFrame(rows)
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_dir}/robustness_raw.csv", index=False)
    return df
