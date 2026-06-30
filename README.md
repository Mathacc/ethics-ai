# Fair and Constraint-Aware Group Formation

## What this is

Splitting people into groups — students into project teams, participants into an
Erasmus+ youth exchange — sounds trivial but rarely is. Groups must satisfy **hard
rules** (fixed sizes, everyone assigned exactly once) while balancing competing
**soft goals**: even skill levels, mixed experience, diverse backgrounds, and
respected preferences. Doing this by hand is slow, opaque, and easy to bias.

This project is an **AI decision-support prototype** that treats group formation as a
**constrained optimization problem** and asks a sharper question than "can we automate
it?": *when we do, is the result fair, stable, and explainable?*

## How it works

- **Generates** synthetic participant pools under controllable scenarios (balanced,
  skewed-skill, minority-underrepresented), so methods can be compared on known ground.
- **Solves** the assignment with four approaches, head-to-head:
  - a **random baseline** (respects only the hard rules),
  - **local search** (swap-based hill climbing),
  - **simulated annealing** (escapes local optima),
  - an **ILP solver** (OR-Tools, exact optimization).
- **Evaluates** each method on **fairness metrics** (skill variance across groups,
  experience balance, diversity spread, preference-satisfaction rate), averaged over
  many seeds.
- **Stress-tests** results for **robustness** — add or remove participants, re-solve,
  and measure how much the assignment churns.
- **Explains** individual placements: why participant *X* landed in their group and
  what each factor contributed.

The aim isn't just a better optimizer — it's a transparent way to see the **trade-offs**
(e.g. skill balance vs. honoring preferences) that any automated grouping forces, which
is the ethics question at the heart of the course this was built for.

CLI-driven; results are saved as matplotlib PNG plots. See [`PLAN.md`](PLAN.md) for the
full plan and [`tasks/`](tasks/) for the per-story backlog.

## Setup (pyenv + venv)

Requires Python **3.11.9** (pinned in `.python-version`).

```bash
pyenv install -s 3.11.9          # if not already installed
pyenv local 3.11.9               # uses .python-version
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Tests
pytest

# CLI
python -m groupformation.cli generate --n 60 --scenario skewed-skill --seed 42 -o data/p.json
python -m groupformation.cli solve    --data data/p.json --method local_search --config config.yaml
python -m groupformation.cli evaluate --data data/p.json --methods random,local_search,sa,ilp --seeds 20
python -m groupformation.cli robustness --data data/p.json --method local_search --perturb add,remove
python -m groupformation.cli pareto   --data data/p.json --method local_search --steps 11
python -m groupformation.cli explain  --assignment outputs/assign.json --data data/p.json --participant p007
```

## Full analysis + report

```bash
# Regenerate every figure and results table into outputs/
python scripts/run_analysis.py

# Build the LaTeX report (report/report.pdf)
cd report && pdflatex report.tex && pdflatex report.tex
```

## Layout

```
groupformation/        package (models, constraints, objectives, optimizers, metrics, plots, cli)
tasks/                 backlog: one folder per epic, one .md per story
data/                  generated datasets (git-ignored)
outputs/               plots + result tables (git-ignored)
config.yaml            run configuration
requirements.txt       dependencies
```
