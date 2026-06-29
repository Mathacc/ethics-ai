"""ILP optimizer via OR-Tools CP-SAT — see tasks/epic-4-algorithms/S4.5-ilp-optimizer.md.

Decision variables ``x[p, g] in {0, 1}`` (participant p assigned to group g) with hard
constraints (one group per participant; fixed group sizes). The soft objective is a
**linear surrogate** of the heuristics' cost:

  * preference — exact: same-group indicators via AND-linearization (z = x_u AND x_v),
                 penalty = fraction of unsatisfied relations.
  * diversity  — exact (up to integer rounding of targets): per-group total-variation
                 distance via L1 deviation variables.
  * skill      — APPROXIMATION: the real cost minimizes variance of group skill means;
                 CP-SAT cannot express that linearly, so we minimize the L1 deviation of
                 each group's skill total from its balanced target. Same intent (balance),
                 different norm.

All methods are evaluated on the *real* metrics (S5.1) afterward, so this surrogate is
only used to drive the solver. Term coefficients are normalized so the three penalties
stay roughly in [0, 1] before applying the configured weights.
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from ..constraints import group_capacities
from ..models import Assignment, Participant
from ..objectives import _attr_names, _relations, _skill_names, index
from .base import Optimizer

_SCALE = 10**6      # integer scaling for the objective
_SKILL_SCALE = 1000  # float skills -> integers


class ILPOptimizer(Optimizer):
    def solve(self, participants: list[Participant], config, initial: Assignment | None = None) -> Assignment:
        by_id = index(participants)
        ids = [p.id for p in participants]
        n = len(ids)
        caps = group_capacities(n, config)
        gids = list(caps)
        weights = config.weights

        model = cp_model.CpModel()
        x = {(p, g): model.NewBoolVar(f"x_{p}_{g}") for p in ids for g in gids}

        # Hard constraints.
        for p in ids:
            model.Add(sum(x[p, g] for g in gids) == 1)
        for g in gids:
            model.Add(sum(x[p, g] for p in ids) == caps[g])

        # Optional warm start (solver hint).
        if initial is not None:
            for p, g in initial.participant_to_group.items():
                if (p, g) in x:
                    model.AddHint(x[p, g], 1)

        obj_vars: list = []
        obj_coeffs: list[int] = []

        self._add_skill_terms(model, x, by_id, ids, gids, caps, weights, obj_vars, obj_coeffs)
        self._add_diversity_terms(model, x, by_id, ids, gids, caps, weights, obj_vars, obj_coeffs)
        self._add_preference_terms(model, x, by_id, gids, weights, obj_vars, obj_coeffs)

        if obj_vars:
            model.Minimize(cp_model.LinearExpr.weighted_sum(obj_vars, obj_coeffs))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float((config.ilp or {}).get("time_limit_seconds", 30))
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        self.status = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"ILP found no feasible solution (status={self.status})")

        mapping = {p: g for p in ids for g in gids if solver.Value(x[p, g]) == 1}
        return Assignment(mapping)

    # -- skill balance (L1 surrogate for variance) ------------------------------
    def _add_skill_terms(self, model, x, by_id, ids, gids, caps, weights, obj_vars, obj_coeffs):
        w = weights.get("skill_balance", 0.0)
        if w <= 0:
            return
        names = _skill_names(by_id)
        skill_int = {p: [round(by_id[p].skills.get(s, 0.0) * _SKILL_SCALE) for s in names] for p in ids}
        total = [sum(skill_int[p][j] for p in ids) for j in range(len(names))]
        coeff = round(w * _SCALE / (len(gids) * max(1, len(names)) * _SKILL_SCALE))
        if coeff <= 0:
            return
        for g in gids:
            for j in range(len(names)):
                group_sum = sum(skill_int[p][j] * x[p, g] for p in ids)
                target = round(total[j] * caps[g] / len(ids))
                dev = model.NewIntVar(0, len(ids) * _SKILL_SCALE, f"sdev_{g}_{j}")
                model.Add(dev >= group_sum - target)
                model.Add(dev >= target - group_sum)
                obj_vars.append(dev)
                obj_coeffs.append(coeff)

    # -- diversity (exact TV via L1) --------------------------------------------
    def _add_diversity_terms(self, model, x, by_id, ids, gids, caps, weights, obj_vars, obj_coeffs):
        w = weights.get("diversity", 0.0)
        attrs = _attr_names(by_id)
        if w <= 0 or not attrs:
            return
        base = w * _SCALE * 0.5 / (len(attrs) * len(gids))
        for attr in attrs:
            cats = sorted({by_id[p].diversity_attrs.get(attr) for p in ids} - {None})
            global_count = {c: sum(by_id[p].diversity_attrs.get(attr) == c for p in ids) for c in cats}
            for g in gids:
                coeff = round(base / caps[g])
                if coeff <= 0:
                    continue
                for c in cats:
                    members_c = [x[p, g] for p in ids if by_id[p].diversity_attrs.get(attr) == c]
                    count = sum(members_c) if members_c else 0
                    target = round(global_count[c] * caps[g] / len(ids))
                    dev = model.NewIntVar(0, caps[g], f"ddev_{attr}_{g}_{c}")
                    model.Add(dev >= count - target)
                    model.Add(dev >= target - count)
                    obj_vars.append(dev)
                    obj_coeffs.append(coeff)

    # -- preferences (exact same-group AND-linearization) -----------------------
    def _add_preference_terms(self, model, x, by_id, gids, weights, obj_vars, obj_coeffs):
        w = weights.get("preference", 0.0)
        rels = _relations(by_id)
        if w <= 0 or not rels:
            return
        c = w * _SCALE / len(rels)
        # net coefficient per unordered pair: like rewards same (-c), dislike penalizes (+c)
        pair_coeff: dict[frozenset, float] = {}
        for u, v, kind in rels:
            key = frozenset((u, v))
            pair_coeff[key] = pair_coeff.get(key, 0.0) + (-c if kind == "like" else c)

        for pair, net in pair_coeff.items():
            coeff = round(net)
            if coeff == 0:
                continue
            u, v = tuple(pair)
            for g in gids:
                z = model.NewBoolVar(f"same_{u}_{v}_{g}")
                model.Add(z <= x[u, g])
                model.Add(z <= x[v, g])
                model.Add(z >= x[u, g] + x[v, g] - 1)
                obj_vars.append(z)
                obj_coeffs.append(coeff)
