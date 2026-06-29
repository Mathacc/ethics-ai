# Fair and Constraint-Aware Group Formation — Project Plan

> AI-based decision-support system for forming fair, balanced groups in collaborative
> environments (e.g., Erasmus+ youth exchanges). Models group formation as a
> constraint-based optimization problem, compares a random baseline against optimized
> methods, and evaluates fairness, robustness, and explainability.

**Authors:** Anastasiia Skryzhadlovska, Matheus De Almeida Cesar Carneiro
**Course context:** Ethics & AI — University of Bologna

---

## Decisions

| Topic | Decision |
|---|---|
| Language | Python 3.11+ |
| Optimizers | **Two first-class methods**: (1) swap-based **local search** (hill-climbing / simulated annealing), (2) **ILP** via OR-Tools — compared head-to-head |
| Interface | **CLI** driven, results as **saved matplotlib plots** (PNG). No Streamlit, no Jupyter notebooks |
| Core libs | `numpy`, `pandas`, `matplotlib`, `ortools`, `pydantic`/`dataclasses`, `pytest` |
| Reproducibility | Global random seed in all generators and optimizers |

---

## Architecture

```
src/groupformation/
  models.py         # Participant, Group, Assignment, Config (dataclasses/pydantic)
  data/
    generator.py    # synthetic participant generator (seeded)
    scenarios.py    # preset scenarios (balanced, skewed-skill, minority-underrep)
  constraints.py    # hard-constraint feasibility checks
  objectives.py     # soft constraints as scorable penalty/utility terms + weighted cost
  optimizers/
    base.py         # Optimizer interface
    random_baseline.py
    local_search.py # hill climbing + simulated annealing
    ilp.py          # OR-Tools CP-SAT / ILP model
  metrics.py        # fairness metrics
  robustness.py     # add/remove perturbations + stability metrics
  explain.py        # per-assignment explanations
  plots.py          # matplotlib figures -> PNG
  cli.py            # argparse/typer entrypoint: generate / solve / evaluate / robustness
tests/
data/               # generated datasets (csv/json)
outputs/            # plots + result tables
```

CLI verbs (target UX):
```
groupform generate --n 60 --groups 10 --seed 42 --scenario skewed-skill -o data/p.csv
groupform solve    --data data/p.csv --method local_search --config config.yaml
groupform solve    --data data/p.csv --method ilp
groupform evaluate --data data/p.csv --methods random,local_search,ilp --seeds 20
groupform robustness --data data/p.csv --method local_search --perturb add,remove
groupform explain  --assignment outputs/assign.json
```

---

## Backlog — Epics, Stories, Tasks

Story points: Fibonacci (1,2,3,5,8). **Total ≈ 84 pts.**

### EPIC 1 — Project setup & data model — 8 pts
- **S1.1 Repo scaffold + tooling** *(2)*
  - [ ] `pyproject.toml`, deps, package layout
  - [ ] lint (ruff) + format (black) + `pytest` in CI
  - [ ] README with run instructions
- **S1.2 Participant & Group models** *(3)*
  - [ ] `Participant`: id, skills vector, experience level, diversity attrs, preferences
  - [ ] `Group`, `Assignment` types + validation
- **S1.3 Config schema** *(3)*
  - [ ] group count/size, hard/soft constraint toggles, fairness weights, seed
  - [ ] load from YAML/JSON + defaults

### EPIC 2 — Synthetic dataset generation — 8 pts
- **S2.1 Participant generator** *(5)*
  - [ ] tunable distributions (skills, experience, diversity), seeded/reproducible
  - [ ] preference generation (optional likes/dislikes)
- **S2.2 Scenarios + persistence** *(3)*
  - [ ] presets: balanced, skewed-skill, minority-underrepresented
  - [ ] save/load CSV + JSON

### EPIC 3 — Constraint modeling — 8 pts
- **S3.1 Hard constraints + feasibility checker** *(5)*
  - [ ] fixed group sizes, unique assignment of each participant
  - [ ] `is_feasible(assignment)` validator
- **S3.2 Soft constraints as objectives** *(3)*
  - [ ] skill balance, diversity, preference satisfaction → penalty/utility terms

### EPIC 4 — Algorithms — 18 pts
- **S4.1 Random baseline** *(2)* — respects hard constraints
- **S4.2 Weighted cost function** *(3)* — aggregate soft constraints into one score
- **S4.3 Local search optimizer** *(5)* — swap-based hill climbing
- **S4.4 Simulated annealing variant** *(3)* — temperature schedule, escape local optima
- **S4.5 ILP optimizer (OR-Tools)** *(5)* — CP-SAT model encoding hard + linearized soft constraints

### EPIC 5 — Fairness evaluation — 13 pts
- **S5.1 Fairness metrics** *(5)* — variance in skill distribution across groups, experience balance, diversity spread, preference-satisfaction rate
- **S5.2 Comparison harness** *(5)* — run N seeds across all methods, aggregate, stats summary table
- **S5.3 Plots (PNG)** *(3)* — metric distributions, convergence curves, method comparison bars

### EPIC 6 — Robustness analysis — 8 pts
- **S6.1 Perturbations** *(5)* — add/remove participants, re-solve, measure assignment churn
- **S6.2 Stability metrics + plots** *(3)* — % participants changing group, fairness delta

### EPIC 7 — Explainability — 8 pts
- **S7.1 Per-assignment explanations** *(5)* — factors driving each placement, marginal score contribution
- **S7.2 Human-readable output** *(3)* — text/JSON summaries via CLI

### EPIC 8 — Ethics & final report — 10 pts
- **S8.1 Ethics analysis** *(5)* — fairness definitions, inclusion, transparency, risks/limitations of automation
- **S8.2 Final report** *(5)* — methodology + results + figures

### EPIC 9 — Stretch — 3 pts
- **S9.1 Pareto front of fairness trade-offs** *(3)* — visualize skill-balance vs preference-satisfaction tension

---

## Sprint plan (2 contributors)

| Sprint | Focus | Stories |
|---|---|---|
| 1 | Foundation | E1, E2 |
| 2 | Modeling + baseline | E3, S4.1, S4.2 |
| 3 | Optimizers | S4.3, S4.4, S4.5 |
| 4 | Evaluation + robustness | E5, E6 |
| 5 | Explainability + ethics + report | E7, E8, E9 |

---

## Definition of Done
- Code typed where reasonable, formatted, lint-clean.
- Every metric/optimizer covered by a `pytest` unit test.
- All experiments reproducible from a seed via a single CLI command.
- Plots regenerate into `outputs/` from raw result tables.
- Report claims trace back to a committed result file.
