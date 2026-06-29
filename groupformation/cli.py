"""Command-line interface.

Run with:  python -m groupformation.cli <command> [options]

Commands (target UX — see PLAN.md):
  generate    Generate a synthetic participant dataset
  solve       Solve grouping with one method
  evaluate    Compare methods across seeds (writes results + plots)
  robustness  Add/remove participants and measure stability
  explain     Produce human-readable explanations for an assignment
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="groupform", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a synthetic dataset")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--scenario", default="balanced")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("-o", "--output", required=True)

    s = sub.add_parser("solve", help="solve grouping with one method")
    s.add_argument("--data", required=True)
    s.add_argument("--method", choices=["random", "local_search", "sa", "ilp"], required=True)
    s.add_argument("--config", default="config.yaml")

    e = sub.add_parser("evaluate", help="compare methods across seeds")
    e.add_argument("--data", required=True)
    e.add_argument("--methods", default="random,local_search,sa,ilp")
    e.add_argument("--seeds", type=int, default=20)
    e.add_argument("--config", default="config.yaml")

    r = sub.add_parser("robustness", help="measure stability under perturbation")
    r.add_argument("--data", required=True)
    r.add_argument("--method", default="local_search")
    r.add_argument("--perturb", default="add,remove")
    r.add_argument("--k", default="1,3,6", help="comma-separated perturbation sizes")
    r.add_argument("--trials", type=int, default=5)
    r.add_argument("--scenario", default="balanced", help="scenario for added participants")
    r.add_argument("--config", default="config.yaml")

    pa = sub.add_parser("pareto", help="sweep weights to map the fairness trade-off")
    pa.add_argument("--data", required=True)
    pa.add_argument("--method", default="local_search")
    pa.add_argument("--steps", type=int, default=11)
    pa.add_argument("--config", default="config.yaml")

    x = sub.add_parser("explain", help="explain an assignment")
    x.add_argument("--assignment", required=True, help="assignment JSON written by `solve`")
    x.add_argument("--data", required=True, help="participants dataset used for the assignment")
    x.add_argument("--participant", default=None, help="explain a single participant id")
    x.add_argument("--config", default="config.yaml")

    return parser


def _cmd_generate(args) -> int:
    from collections import Counter

    from .data import scenarios
    from .data.generator import generate

    participants = generate(args.n, seed=args.seed, scenario=args.scenario)
    scenarios.save(participants, args.output)

    exp = Counter(p.experience for p in participants)
    nat = Counter(p.diversity_attrs.get("nationality") for p in participants)
    with_prefs = sum(1 for p in participants if p.preferences)
    print(f"Generated {len(participants)} participants (scenario={args.scenario}, seed={args.seed})")
    print(f"  saved to: {args.output}")
    print(f"  experience levels: {dict(sorted(exp.items()))}")
    print(f"  nationalities: {dict(sorted(nat.items()))}")
    print(f"  with preferences: {with_prefs}")
    return 0


def _build_optimizer(method: str):
    from .optimizers import get_optimizer

    try:
        return get_optimizer(method)
    except ValueError as exc:
        raise SystemExit(str(exc))


def _cmd_solve(args) -> int:
    import json

    from .config import Config
    from .constraints import group_capacities, is_feasible
    from .data import scenarios
    from .objectives import (
        cost,
        diversity,
        index,
        preference_penalty,
        skill_balance,
    )

    config = Config.load(args.config)
    participants = scenarios.load(args.data)
    optimizer = _build_optimizer(args.method)

    assignment = optimizer.solve(participants, config)

    ids = [p.id for p in participants]
    capacities = group_capacities(len(participants), config)
    feasible, violations = is_feasible(assignment, ids, capacities)

    by_id = index(participants)
    total = cost(assignment, by_id, config.weights)

    out = "outputs/assign.json"
    with open(out, "w") as f:
        json.dump({"method": args.method, "assignment": assignment.participant_to_group}, f, indent=2)

    sizes = {gid: len(m) for gid, m in sorted(assignment.groups().items())}
    print(f"Solved with method={args.method} on {len(participants)} participants")
    print(f"  groups: {len(capacities)}  sizes: {sizes}")
    print(f"  feasible: {feasible}")
    for v in violations:
        print(f"    ! {v}")
    if hasattr(optimizer, "history") and optimizer.history:
        print(f"  iterations: {len(optimizer.history) - 1}  (start cost {optimizer.history[0]:.4f})")
    if hasattr(optimizer, "status"):
        print(f"  solver status: {optimizer.status}")
    print(f"  cost: {total:.4f}  (lower is better)")
    print(f"    skill_balance: {skill_balance(assignment, by_id):.4f}")
    print(f"    diversity:     {diversity(assignment, by_id):.4f}")
    print(f"    preference:    {preference_penalty(assignment, by_id):.4f}")
    print(f"  saved assignment to: {out}")
    return 0 if feasible else 1


def _cmd_evaluate(args) -> int:
    from . import evaluation, plots
    from .config import Config
    from .data import scenarios

    config = Config.load(args.config)
    participants = scenarios.load(args.data)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"Evaluating {methods} over {args.seeds} seed(s) on {len(participants)} participants...")
    raw, _agg, histories, representative = evaluation.run_comparison(participants, config, methods, args.seeds)

    figs = [
        plots.plot_method_comparison(raw, "outputs/fig_method_comparison.png"),
        plots.plot_skill_distribution(representative, participants, "outputs/fig_skill_distribution.png"),
    ]
    if histories:
        figs.append(plots.plot_convergence(histories, "outputs/fig_convergence.png"))

    print("\nMean results per method (sorted by cost):")
    print(evaluation.summary_table(raw).to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nWrote: outputs/results_raw.csv, outputs/results_agg.csv")
    for f in figs:
        print(f"       {f}")
    return 0


def _cmd_robustness(args) -> int:
    from . import plots, robustness
    from .config import Config
    from .data import scenarios

    config = Config.load(args.config)
    participants = scenarios.load(args.data)
    perturbations = [p.strip() for p in args.perturb.split(",") if p.strip()]
    ks = [int(k) for k in args.k.split(",") if k.strip()]

    print(f"Robustness study: method={args.method} perturb={perturbations} k={ks} trials={args.trials}")
    df = robustness.run_robustness(
        participants, config, args.method, perturbations, ks, args.trials, scenario=args.scenario
    )
    fig = plots.plot_robustness(df, "outputs/fig_robustness.png")

    summary = df.groupby(["perturbation", "start", "k"])[["group_change_rate", "pair_preservation", "d_skill_imbalance"]].mean()
    print("\nMean stability by perturbation / start / k:")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote: outputs/robustness_raw.csv\n       {fig}")
    return 0


def _cmd_explain(args) -> int:
    import json

    from . import explain
    from .config import Config
    from .data import scenarios
    from .models import Assignment

    config = Config.load(args.config)
    participants = scenarios.load(args.data)
    with open(args.assignment) as f:
        payload = json.load(f)
    assignment = Assignment(payload["assignment"])

    if args.participant:
        by_id = {p.id: p for p in participants}
        if args.participant not in by_id:
            raise SystemExit(f"unknown participant {args.participant!r}")
        from .objectives import IncrementalEvaluator

        ev = IncrementalEvaluator(participants, config.weights, assignment)
        print(explain.render_participant(explain.explain_participant(assignment, by_id, ev, args.participant)))
        return 0

    result = explain.explain_assignment(assignment, participants, config.weights)
    out = "outputs/explanations.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Well-placed participants: {result['well_placed_rate']:.0%}\n")
    print("Group summaries:")
    for s in result["groups"]:
        print("  " + explain.render_group_summary(s))
    print(f"\nFull per-participant explanations written to: {out}")
    print("Tip: re-run with --participant <id> for a single detailed explanation.")
    return 0


def _cmd_pareto(args) -> int:
    from . import pareto, plots
    from .config import Config
    from .data import scenarios

    config = Config.load(args.config)
    participants = scenarios.load(args.data)
    print(f"Pareto sweep: method={args.method} steps={args.steps}")
    df = pareto.run_pareto(participants, config, args.method, args.steps)
    fig = plots.plot_pareto(df, "outputs/fig_pareto.png")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote: outputs/pareto.csv\n       {fig}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "solve":
        return _cmd_solve(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "robustness":
        return _cmd_robustness(args)
    if args.command == "explain":
        return _cmd_explain(args)
    if args.command == "pareto":
        return _cmd_pareto(args)
    print(f"[stub] command={args.command} args={vars(args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
