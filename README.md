# Multilayer Graph Learning for Energy-Efficient User Association

This repository contains the code, generated experiment outputs, MATLAB figure
scripts, and IEEE manuscript workspace for the paper draft:

**Multilayer Graph Learning for Energy-Efficient User Association and Cell
Switch-Off in Dense Wireless Networks**

The study formulates energy-aware user association as a constrained
multilayer graph-learning problem. Each wireless snapshot is represented as a
UE-level multilayer graph with separate relation layers for association
competition, interference similarity, load-pressure similarity, and temporal
traffic similarity. Fixed-fusion and attention-weighted multilayer GCN models
are compared against conventional user-association heuristics, flat learning,
aggregated graph learning, and a finite-search reference.

## Repository Layout

```text
configs/              Experiment configurations
data/                 Data notes and optional generated-data location
matlab_figures/       MATLAB scripts and CSVs for paper figures
overleaf/             IEEE two-column manuscript source and final figures
results/              Generated experiment outputs used in the manuscript
scripts/              Experiment, sweep, runtime, and figure-generation scripts
src/mlua/             Simulator, graph construction, models, repair, and metrics
tests/                Smoke tests
```

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Reproducibility Check

Run the smoke experiment and tests:

```bash
python scripts/run_experiment.py --config configs/smoke.json --output results/smoke
pytest
```

## Main Experiments

The manuscript uses the saved outputs under `results/`. To regenerate the main
experiment families, run:

```bash
python scripts/run_multiseed.py --config configs/multiseed_initial.json --output results/multiseed_initial
python scripts/run_experiment.py --config configs/ablation_initial.json --output results/ablation_initial
python scripts/run_result_sweeps.py
python scripts/run_surface_sweep.py
python scripts/run_runtime_benchmark.py
python scripts/make_result_figures.py
```

The most important generated files are:

- `results/multiseed_initial/summary_mean.csv`
- `results/multiseed_initial/summary_std.csv`
- `results/paper_sweeps/sweep_summary.csv`
- `results/surface_sweep/surface_summary.csv`
- `results/runtime_benchmark/runtime_summary.csv`
- `overleaf/figures/results/*.png`
- `matlab_figures/results/data/*.csv`

## Manuscript

The IEEE two-column manuscript source is in `overleaf/main.tex`. The compiled
draft PDF is included as `overleaf/main.pdf`.

If LaTeX is installed, compile from the `overleaf/` folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## MATLAB Figures

Editable MATLAB figure scripts are stored under `matlab_figures/results/`.
Each script reads CSV data from `matlab_figures/results/data/` and recreates
the corresponding paper figure. Running the scripts in MATLAB will also allow
export to native `.fig` files.

## Data and Reproducibility Notes

The current experiments use reproducible synthetic dense-cellular snapshots.
The simulator creates UE positions, BS positions, channel gains, traffic
demands, multilayer graph supports, finite-search reference labels, and
policy-evaluation metrics from the configuration files and random seeds.
Aggregate outputs and figure data are included in `results/` and
`matlab_figures/results/data/`.

## License

The code is released under the MIT License. The manuscript draft and figures
are provided for academic reproducibility and paper review.
