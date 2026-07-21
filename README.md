# Data-Driven Inverse Optimal Control for ADAS Trajectory Planning

Research code for a master's thesis on data-driven inverse optimal control for lane-change and trajectory-planning behavior. The repository focuses on learning high-level objectives from demonstrations, fitting dynamics models, and comparing a data-driven IOC pipeline against a classical IOC baseline.

This public version is scoped to the standalone research components: model learning, bilevel optimization, synthetic validation, and benchmark comparisons.

## Highlights

- data-driven IOC pipeline with learned dynamics and learned high-level objective weights
- classical IOC benchmark for side-by-side comparison
- synthetic validation scripts and figure generation for qualitative inspection
- modular planner-tuning utilities for bilevel experiments

## Why this project matters

The core thesis question is whether lane-change behavior can be explained and reused through a learned high-level objective rather than only through hand-tuned cost functions. In this repository, that question is explored through:

- trajectory feature design for inverse optimal control
- Koopman-style dynamics learning with polynomial and neural lifts
- bilevel planner tuning against learned objective weights
- comparison against a classical linear IOC baseline

This makes the repository useful both as thesis evidence and as a concrete portfolio project for controls, autonomy, and machine-learning interviews.

## What is in the public repo

- research code for objective learning, dynamics learning, and planner tuning
- synthetic validation and benchmark scripts that run without proprietary datasets
- curated figures under `docs/figures/` for quick qualitative inspection
- a standalone end-to-end synthetic IOC example under `examples/`

## Current evidence

The included synthetic validation workflow in `ioc/DDIOC/validate_dnn.py` compares polynomial and neural Koopman lifts on held-out trajectories. In the current public run, the DNN lift achieved lower one-step and rollout error than the polynomial lift, and the generated figures are kept under `docs/figures/`.

## Repository layout

- `ioc/DDIOC/`: data-driven IOC pipeline, dynamics learning, objective learning, validation, and planner tuning
- `ioc/LIOC/`: classical IOC benchmark implementation
- `examples/`: runnable public examples that showcase the thesis ideas end to end
- `docs/figures/`: curated public figures suitable for reports or a project page
- `outputs/`: local run artifacts generated during experiments

## Main components

`ioc/DDIOC/pipeline.py`
End-to-end learning pipeline for Koopman-style dynamics learning and shared high-level objective estimation.

`ioc/DDIOC/hlo_learning.py`
Utilities for learning, loading, and saving high-level objective weights.

`ioc/DDIOC/tune_qp_planner.py`
Bilevel optimization utilities for tuning planner parameters against a learned high-level objective. The script supports generic JSON-based planner parameter updates and optional external planner or MPC hooks.

`ioc/DDIOC/validate_dnn.py`
Synthetic validation script for comparing polynomial and neural Koopman lifts. Figures are written under `outputs/figures/validate_dnn/`.

`ioc/LIOC/linear_ioc_bilevel.py`
Classical bilevel IOC benchmark for comparison against the data-driven method.

`examples/synthetic_ioc_demo.py`
Standalone synthetic demonstration of learned HLO recovery, LQR planner tuning, and comparison against a classical IOC-style baseline.

## Quick start

Create a local environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Typical entry points:

```bash
python3 -m ioc.DDIOC.pipeline --help
python3 -m ioc.DDIOC.tune_qp_planner --help
python3 -m ioc.LIOC.linear_ioc_bilevel --help
python3 -m ioc.DDIOC.validate_dnn
python3 examples/synthetic_ioc_demo.py
```

Core dependencies are listed in `requirements.txt`, including PyTorch for the DNN Koopman-lift workflow. Optional optimization workflows also use packages such as `cvxpy`, `scikit-optimize`, and `pymoo`.

## Suggested first look

- run `python3 -m ioc.DDIOC.validate_dnn` for a self-contained validation example
- run `python3 examples/synthetic_ioc_demo.py` for a compact end-to-end synthetic IOC example
- inspect `ioc/DDIOC/pipeline.py` for the end-to-end learning flow
- inspect `ioc/LIOC/linear_ioc_bilevel.py` for the classical benchmark formulation

## Public scope

- this repo contains the research core and standalone evaluation code
- it does not include non-public datasets, private preprocessing packages, or production planner binaries
- some dataset-extraction entry points expect optional helper modules to be supplied separately if that workflow is needed

## License

This repository is released under the MIT License. See `LICENSE`.

## Notes

- `outputs/` is intended for local run artifacts and is git-ignored by default.
- `docs/figures/` stores curated sample figures for reports or a project page.
- `examples/` contains runnable showcase scripts that are intended to be read by other engineers and researchers.