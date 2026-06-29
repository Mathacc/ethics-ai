"""Explainability — see tasks/epic-7-explainability/.

S7.1 — explain *why* each participant sits in their group, faithfully to the cost model:
  * counterfactual — the best feasible move is a swap (group sizes are fixed); we find the
    swap that most reduces cost using the same ``delta_swap`` the optimizer uses. If no
    swap improves cost, the participant is "well-placed".
  * per-factor breakdown — concrete, data-derived notes on preferences, diversity, and
    skill, so explanations reflect the actual soft constraints (not post-hoc invention).

S7.2 — render the structured explanations as human-readable text / JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .models import Assignment, Participant
from .objectives import IncrementalEvaluator

_EPS = 1e-9


@dataclass
class Factor:
    name: str          # 'preferences' | 'diversity' | 'skill'
    detail: str        # human-readable explanation
    score: float = 0.0  # term-specific signal (e.g. unsatisfied fraction / deviation)


@dataclass
class ParticipantExplanation:
    participant_id: str
    group_id: str
    well_placed: bool
    best_alternative_group: str | None
    best_swap_partner: str | None
    best_delta: float          # cost change of the best swap (negative = improvement)
    factors: list[Factor] = field(default_factory=list)


def explain_participant(
    assignment: Assignment, by_id: dict[str, Participant], ev: IncrementalEvaluator, pid: str
) -> ParticipantExplanation:
    cur_group = ev.mapping[pid]

    # counterfactual: best improving swap with a member of another group
    best_delta, best_partner = 0.0, None
    for other in by_id:
        if other == pid or ev.mapping[other] == cur_group:
            continue
        d = ev.delta_swap(pid, other)
        if d < best_delta - _EPS:
            best_delta, best_partner = d, other

    best_group = ev.mapping[best_partner] if best_partner is not None else None
    return ParticipantExplanation(
        participant_id=pid,
        group_id=cur_group,
        well_placed=best_partner is None,
        best_alternative_group=best_group,
        best_swap_partner=best_partner,
        best_delta=best_delta,
        factors=[
            _preference_factor(assignment, by_id, pid),
            _diversity_factor(assignment, by_id, pid),
            _skill_factor(assignment, by_id, pid),
        ],
    )


def explain_assignment(
    assignment: Assignment, participants: list[Participant], weights: dict[str, float]
) -> dict:
    by_id = {p.id: p for p in participants}
    ev = IncrementalEvaluator(participants, weights, assignment)
    per_participant = [explain_participant(assignment, by_id, ev, p.id) for p in participants]
    return {
        "per_participant": [asdict(e) for e in per_participant],
        "groups": _group_summaries(assignment, by_id),
        "well_placed_rate": float(np.mean([e.well_placed for e in per_participant])),
    }


# ------------------------------------------------------------------- factors
def _preference_factor(assignment, by_id, pid) -> Factor:
    p = by_id[pid]
    g = assignment.participant_to_group
    notes, sat, total = [], 0, 0
    for v in p.preferences.get("likes", []):
        if v in by_id and v != pid:
            total += 1
            if g[v] == g[pid]:
                sat += 1
                notes.append(f"grouped with liked {v}")
            else:
                notes.append(f"NOT grouped with liked {v}")
    for v in p.preferences.get("dislikes", []):
        if v in by_id and v != pid:
            total += 1
            if g[v] != g[pid]:
                sat += 1
                notes.append(f"kept apart from disliked {v}")
            else:
                notes.append(f"grouped with disliked {v}")
    if total == 0:
        return Factor("preferences", "no preferences expressed", 0.0)
    return Factor("preferences", f"{sat}/{total} satisfied — " + "; ".join(notes), (total - sat) / total)


def _diversity_factor(assignment, by_id, pid) -> Factor:
    p = by_id[pid]
    members = assignment.groups()[assignment.participant_to_group[pid]]
    n = len(by_id)
    notes, dev_total = [], 0.0
    for attr, cat in p.diversity_attrs.items():
        local = sum(by_id[m].diversity_attrs.get(attr) == cat for m in members) / len(members)
        glob = sum(q.diversity_attrs.get(attr) == cat for q in by_id.values()) / n
        dev_total += abs(local - glob)
        rel = "over-represented" if local > glob + 0.05 else "under-represented" if local < glob - 0.05 else "balanced"
        notes.append(f"{attr}={cat} {local:.0%} in group vs {glob:.0%} overall ({rel})")
    return Factor("diversity", "; ".join(notes), dev_total)


def _skill_factor(assignment, by_id, pid) -> Factor:
    p = by_id[pid]
    members = assignment.groups()[assignment.participant_to_group[pid]]
    p_mean = float(np.mean(list(p.skills.values()))) if p.skills else 0.0
    group_mean = float(np.mean([np.mean(list(by_id[m].skills.values())) for m in members]))
    global_mean = float(np.mean([np.mean(list(q.skills.values())) for q in by_id.values()]))
    rel = "raises" if p_mean > group_mean else "lowers" if p_mean < group_mean else "matches"
    detail = f"mean skill {p_mean:.2f} {rel} group mean {group_mean:.2f} (overall {global_mean:.2f})"
    return Factor("skill", detail, abs(p_mean - group_mean))


def _group_summaries(assignment, by_id) -> list[dict]:
    out = []
    for gid, members in sorted(assignment.groups().items()):
        skills = [np.mean(list(by_id[m].skills.values())) for m in members]
        nat = {}
        for m in members:
            c = by_id[m].diversity_attrs.get("nationality")
            nat[c] = nat.get(c, 0) + 1
        out.append(
            {
                "group_id": gid,
                "size": len(members),
                "mean_skill": round(float(np.mean(skills)), 3),
                "experience_mean": round(float(np.mean([by_id[m].experience for m in members])), 2),
                "nationality_mix": dict(sorted(nat.items())),
            }
        )
    return out


# ------------------------------------------------------------------- rendering (S7.2)
def render_participant(e: ParticipantExplanation) -> str:
    lines = [f"Participant {e.participant_id} → group {e.group_id}"]
    if e.well_placed:
        lines.append("  • Placement: well-placed (no beneficial swap found)")
    else:
        lines.append(
            f"  • Placement: could improve by swapping with {e.best_swap_partner} "
            f"(group {e.best_alternative_group}, Δcost {e.best_delta:+.4f})"
        )
    for f in e.factors:
        lines.append(f"  • {f.name.capitalize()}: {f.detail}")
    return "\n".join(lines)


def render_group_summary(summary: dict) -> str:
    return (
        f"Group {summary['group_id']}: size {summary['size']}, "
        f"mean skill {summary['mean_skill']}, exp {summary['experience_mean']}, "
        f"nationalities {summary['nationality_mix']}"
    )
