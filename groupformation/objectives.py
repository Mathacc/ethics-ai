"""Soft constraints + weighted cost.

See tasks/epic-3-constraints/S3.2-soft-constraints.md
and tasks/epic-4-algorithms/S4.2-cost-function.md.

All soft-constraint terms are expressed as **penalties in [0, 1]** (lower = better) so a
single weighted sum (``cost``) can be minimized by every optimizer.

  * skill_balance  — variance of per-group skill means across groups (normalized by the
                     max variance 0.25 of values in [0, 1]).
  * diversity      — mean total-variation distance between each group's category mix and
                     the global mix, per diversity attribute.
  * preference     — fraction of preference relations *not* honored (a "like" not grouped
                     together, or a "dislike" grouped together).

``cost`` is the source of truth. ``IncrementalEvaluator`` provides an O(changed-groups)
``delta_swap`` for the local-search inner loop, verified against ``cost`` in tests.
"""

from __future__ import annotations

import numpy as np

from .models import Assignment, Participant

_MAX_VAR = 0.25  # max variance of values in [0, 1] → normalizer for skill balance


# --------------------------------------------------------------------------- helpers
def index(participants: list[Participant]) -> dict[str, Participant]:
    return {p.id: p for p in participants}


def _skill_names(by_id: dict[str, Participant]) -> list[str]:
    names: set[str] = set()
    for p in by_id.values():
        names.update(p.skills)
    return sorted(names)


def _attr_names(by_id: dict[str, Participant]) -> list[str]:
    names: set[str] = set()
    for p in by_id.values():
        names.update(p.diversity_attrs)
    return sorted(names)


def _relations(by_id: dict[str, Participant]) -> list[tuple[str, str, str]]:
    """Directed preference relations (u, v, kind) for existing, non-self targets."""
    rels: list[tuple[str, str, str]] = []
    for u, p in by_id.items():
        for kind in ("likes", "dislikes"):
            for v in p.preferences.get(kind, []):
                if v in by_id and v != u:
                    rels.append((u, v, "like" if kind == "likes" else "dislike"))
    return rels


# --------------------------------------------------------------- standalone penalties
def skill_balance(assignment: Assignment, by_id: dict[str, Participant]) -> float:
    names = _skill_names(by_id)
    groups = assignment.groups()
    means = np.array(
        [[np.mean([by_id[m].skills.get(s, 0.0) for m in members]) for s in names] for members in groups.values()]
    )  # shape (K, J)
    return float(np.clip(np.mean(means.var(axis=0)) / _MAX_VAR, 0.0, 1.0))


def diversity(assignment: Assignment, by_id: dict[str, Participant]) -> float:
    attrs = _attr_names(by_id)
    if not attrs:
        return 0.0
    groups = assignment.groups()
    penalties: list[float] = []
    for attr in attrs:
        global_dist = _category_dist([p.diversity_attrs.get(attr) for p in by_id.values()])
        for members in groups.values():
            local = _category_dist([by_id[m].diversity_attrs.get(attr) for m in members])
            penalties.append(_total_variation(local, global_dist))
    return float(np.mean(penalties))


def preference_penalty(assignment: Assignment, by_id: dict[str, Participant]) -> float:
    rels = _relations(by_id)
    if not rels:
        return 0.0
    group_of = assignment.participant_to_group
    satisfied = sum(1 for u, v, kind in rels if _relation_satisfied(kind, group_of[u], group_of[v]))
    return 1.0 - satisfied / len(rels)


def cost(assignment: Assignment, by_id: dict[str, Participant], weights: dict[str, float]) -> float:
    """Weighted aggregate the optimizers minimize (source of truth)."""
    return (
        weights.get("skill_balance", 0.0) * skill_balance(assignment, by_id)
        + weights.get("diversity", 0.0) * diversity(assignment, by_id)
        + weights.get("preference", 0.0) * preference_penalty(assignment, by_id)
    )


def _category_dist(values) -> dict[str, float]:
    counts: dict[str, int] = {}
    n = 0
    for v in values:
        if v is None:
            continue
        counts[v] = counts.get(v, 0) + 1
        n += 1
    return {k: c / n for k, c in counts.items()} if n else {}


def _total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _relation_satisfied(kind: str, g_u: str, g_v: str) -> bool:
    return (g_u == g_v) if kind == "like" else (g_u != g_v)


# ----------------------------------------------------------- incremental evaluator
class IncrementalEvaluator:
    """Stateful cost evaluator bound to an assignment, with fast single-swap deltas.

    Maintains per-group skill sums and category counts plus a running satisfied-relations
    count, so ``delta_swap`` only touches the two affected groups (and relations incident
    to the two swapped participants).
    """

    def __init__(self, participants: list[Participant], weights: dict[str, float], assignment: Assignment):
        self.by_id = index(participants)
        self.weights = weights
        self.skill_names = _skill_names(self.by_id)
        self.attr_names = _attr_names(self.by_id)
        self.mapping: dict[str, str] = dict(assignment.participant_to_group)

        # skill vectors per participant
        self.skill_vec = {
            pid: np.array([p.skills.get(s, 0.0) for s in self.skill_names]) for pid, p in self.by_id.items()
        }
        # global category distributions per attribute (constant)
        self.global_dist = {
            attr: _category_dist([p.diversity_attrs.get(attr) for p in self.by_id.values()])
            for attr in self.attr_names
        }
        # preference relations + incidence index
        self.rels = _relations(self.by_id)
        self.total_rels = len(self.rels)
        self._incident: dict[str, list[int]] = {}
        for i, (u, v, _) in enumerate(self.rels):
            self._incident.setdefault(u, []).append(i)
            self._incident.setdefault(v, []).append(i)

        self._build_caches()

    # -- caches -----------------------------------------------------------------
    def _build_caches(self) -> None:
        groups = Assignment(self.mapping).groups()
        self.gids = sorted(groups)
        self.size = {g: len(m) for g, m in groups.items()}
        self.skill_sum = {
            g: np.sum([self.skill_vec[m] for m in members], axis=0) for g, members in groups.items()
        }
        self.cat_count = {
            attr: {g: _counter([self.by_id[m].diversity_attrs.get(attr) for m in members]) for g, members in groups.items()}
            for attr in self.attr_names
        }
        self.satisfied = sum(
            1 for u, v, kind in self.rels if _relation_satisfied(kind, self.mapping[u], self.mapping[v])
        )

    # -- full cost from caches --------------------------------------------------
    def cost(self) -> float:
        return (
            self.weights.get("skill_balance", 0.0) * self._skill_penalty(self.skill_sum)
            + self.weights.get("diversity", 0.0) * self._diversity_penalty(self.cat_count)
            + self.weights.get("preference", 0.0) * self._preference_penalty(self.satisfied)
        )

    def _skill_penalty(self, skill_sum: dict[str, np.ndarray]) -> float:
        means = np.array([skill_sum[g] / self.size[g] for g in self.gids])
        return float(np.clip(np.mean(means.var(axis=0)) / _MAX_VAR, 0.0, 1.0))

    def _diversity_penalty(self, cat_count: dict[str, dict[str, dict]]) -> float:
        if not self.attr_names:
            return 0.0
        tvs: list[float] = []
        for attr in self.attr_names:
            gdist = self.global_dist[attr]
            for g in self.gids:
                local = {k: c / self.size[g] for k, c in cat_count[attr][g].items()}
                tvs.append(_total_variation(local, gdist))
        return float(np.mean(tvs))

    def _preference_penalty(self, satisfied: int) -> float:
        return 0.0 if self.total_rels == 0 else 1.0 - satisfied / self.total_rels

    # -- delta for swapping a and b (different groups) --------------------------
    def delta_swap(self, a: str, b: str) -> float:
        ga, gb = self.mapping[a], self.mapping[b]
        if ga == gb:
            return 0.0
        w = self.weights

        # skill: only ga, gb sums change
        new_sum = dict(self.skill_sum)
        new_sum[ga] = self.skill_sum[ga] - self.skill_vec[a] + self.skill_vec[b]
        new_sum[gb] = self.skill_sum[gb] - self.skill_vec[b] + self.skill_vec[a]
        d_skill = self._skill_penalty(new_sum) - self._skill_penalty(self.skill_sum)

        # diversity: only ga, gb counts change
        d_div = 0.0
        if self.attr_names:
            denom = len(self.attr_names) * len(self.gids)
            for attr in self.attr_names:
                ca, cb = self.by_id[a].diversity_attrs.get(attr), self.by_id[b].diversity_attrs.get(attr)
                old_ga = self.cat_count[attr][ga]
                old_gb = self.cat_count[attr][gb]
                new_ga = _shift(old_ga, remove=ca, add=cb)
                new_gb = _shift(old_gb, remove=cb, add=ca)
                d_div += (
                    self._tv_from_counts(new_ga, ga, attr) - self._tv_from_counts(old_ga, ga, attr)
                    + self._tv_from_counts(new_gb, gb, attr) - self._tv_from_counts(old_gb, gb, attr)
                )
            d_div /= denom

        # preference: only relations incident to a or b change
        d_pref = 0.0
        if self.total_rels:
            affected = set(self._incident.get(a, [])) | set(self._incident.get(b, []))
            new_map = self.mapping  # read with overrides for a, b
            dsat = 0
            for i in affected:
                u, v, kind = self.rels[i]
                gu_old, gv_old = new_map[u], new_map[v]
                gu_new = gb if u == a else ga if u == b else gu_old
                gv_new = gb if v == a else ga if v == b else gv_old
                dsat += int(_relation_satisfied(kind, gu_new, gv_new)) - int(
                    _relation_satisfied(kind, gu_old, gv_old)
                )
            d_pref = -dsat / self.total_rels

        return w.get("skill_balance", 0.0) * d_skill + w.get("diversity", 0.0) * d_div + w.get("preference", 0.0) * d_pref

    def apply_swap(self, a: str, b: str) -> None:
        ga, gb = self.mapping[a], self.mapping[b]
        if ga == gb:
            return
        # update satisfied via the same incident logic before mutating mapping
        affected = set(self._incident.get(a, [])) | set(self._incident.get(b, []))
        for i in affected:
            u, v, kind = self.rels[i]
            self.satisfied -= int(_relation_satisfied(kind, self.mapping[u], self.mapping[v]))
        # skill + diversity caches
        self.skill_sum[ga] = self.skill_sum[ga] - self.skill_vec[a] + self.skill_vec[b]
        self.skill_sum[gb] = self.skill_sum[gb] - self.skill_vec[b] + self.skill_vec[a]
        for attr in self.attr_names:
            ca, cb = self.by_id[a].diversity_attrs.get(attr), self.by_id[b].diversity_attrs.get(attr)
            self.cat_count[attr][ga] = _shift(self.cat_count[attr][ga], remove=ca, add=cb)
            self.cat_count[attr][gb] = _shift(self.cat_count[attr][gb], remove=cb, add=ca)
        # mapping
        self.mapping[a], self.mapping[b] = gb, ga
        for i in affected:
            u, v, kind = self.rels[i]
            self.satisfied += int(_relation_satisfied(kind, self.mapping[u], self.mapping[v]))

    def assignment(self) -> Assignment:
        return Assignment(dict(self.mapping))

    def _tv_from_counts(self, counts: dict[str, int], gid: str, attr: str) -> float:
        local = {k: c / self.size[gid] for k, c in counts.items()}
        return _total_variation(local, self.global_dist[attr])


def _counter(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        if v is None:
            continue
        out[v] = out.get(v, 0) + 1
    return out


def _shift(counts: dict[str, int], *, remove: str | None, add: str | None) -> dict[str, int]:
    out = dict(counts)
    if remove is not None:
        out[remove] = out.get(remove, 0) - 1
        if out[remove] <= 0:
            out.pop(remove, None)
    if add is not None:
        out[add] = out.get(add, 0) + 1
    return out
