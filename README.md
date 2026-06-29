# Fair and Constraint-Aware Group Formation

AI-based decision-support prototype that assigns participants to groups as a
constraint-based optimization problem, comparing a random baseline against local search,
simulated annealing, and an ILP solver — with fairness metrics, robustness analysis, and
simple explainability.

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
