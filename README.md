# CityFlow-branch

This repository contains the CityFlow simulator, a browser-based replay frontend, and a set of Python modules for traffic-signal experimentation. This README focuses on what is implemented in the codebase today, rather than on future research proposals or roadmap items.

## Overview

CityFlow-branch is a fork of the CityFlow project and includes:

- the compiled C++ simulation engine and Python interface,
- frontend visualization and replay tooling,
- documentation and installation guides,
- a structured Python feature package under `cityflow_features/`,
- and a collection of case-study scripts under `cases-study/`.

The repository mixes official simulator components with experimental Python modules for signal-control research, mock RL environments, and scenario-driven evaluation.

## Repository structure

- `src/` — C++ engine source and simulation logic
- `frontend/` — browser-based replay and visualization helper files
- `simulation/` — traffic-visualization utilities and generated assets
- `cityflow_features/` — Python feature modules for signal observations, reward functions, action spaces, training environments, and evaluation helpers
- `cases-study/` — scenario generation and experiment scripts
- `tests/` — test scaffolding and C++/Python test configuration
- `docs/` — Sphinx documentation structure
- `HOW_TO_RUN.md` — installation and usage instructions
- `setup.py`, `CMakeLists.txt`, `Dockerfile` — build and packaging configuration

## Implemented simulator and tooling

### CityFlow core

The repository includes the standard CityFlow build layout and installation flow:

- `CMakeLists.txt` configures the native build
- `setup.py` installs the Python package
- `Dockerfile` provides a containerized build path
- `HOW_TO_RUN.md` documents how to install and use the simulator

This repo also includes the expected CityFlow Python interface pattern, where a `cityflow.Engine` object can be created and advanced with `next_step()`.

### Frontend and replay tooling

The frontend includes the replay viewer workflow:

- `frontend/index.html` is the main browser-based replay UI
- `frontend/README.md` explains how to view a replay and download example replay files
- `simulation/Readme.md` describes a separate traffic intersection visualizer for `roadnet.json` and `replay.txt` inputs

These tools are present in the repository and are part of the implemented visualization surface.

### Simulation visualizer

`simulation/Readme.md` describes a visualization tool with features such as:

- accurate road geometry and lane markings,
- signal display by lane and direction,
- vehicle rendering and stable colors,
- high-resolution static export,
- vector PDF export,
- optional MP4 export via ffmpeg,
- and a blitted rendering pipeline designed for performance.

This is implemented as a documented visual analysis utility rather than a general-purpose traffic controller.

## Implemented traffic-signal feature modules

The `cityflow_features/` package is a Python pipeline for signal-control experimentation.

### 1. Signal characterization

`cityflow_features/signal_characterization.py`

This module records per-intersection signal state, including:

- phase index,
- phase duration,
- vehicles in the north/south direction,
- vehicles in the east/west direction,
- queue ratio,
- average phase duration.

It exposes a signal-state history and summary metrics such as total phase records, signal intersections tracked, and average phases per intersection.

### 2. Action space definitions

`cityflow_features/action_space.py`

This module defines action encodings for traffic signal control:

- `discrete_phase`
- `switch_hold`
- `phase_duration`

It includes:

- validation logic,
- sample generation,
- action masking,
- mode conversion helpers,
- and metrics for action-space usage.

### 3. Training environment

`cityflow_features/training_env.py`

This module provides a Gym-style environment interface with:

- `reset()` and `step()`,
- min-green enforcement,
- per-intersection phase bookkeeping,
- total reward accumulation,
- and per-intersection info reporting.

Its default simulator is `MockTrafficSimulator`, which models simplified stochastic arrivals and departures for two-direction queues. This is a self-contained mock environment rather than a calibrated CityFlow traffic benchmark.

### 4. Reward calculation

`cityflow_features/reward_calculator.py`

This module computes a multi-objective reward based on:

- queue pressure,
- waiting-time penalty,
- throughput gain,
- and fairness penalty.

It is designed as an interpretable RL signal and tracks per-component values and cumulative episode return. It should be understood as a modeled reward function, not as a validated real-world traffic metric system.

### 5. Traffic control algorithms

`cityflow_features/traffic_control_algorithm.py`

This module defines non-MARL signal controllers:

- `fixed_time_control`
- `adaptive_control`
- `optimized_control`

The controller records decisions and exposes algorithm metrics such as total decisions and average phase duration. These are simple heuristic rules implemented in Python.

### 6. Additional feature modules

The package also includes helper modules for related components:

- `congestion_state_space.py` — congestion-state representations
- `object_closure_parameters.py` — safety/constraint parameters
- `v2x_communication.py` — simplified vehicle-to-infrastructure communication simulation
- `v2v_communication.py` — simplified vehicle-to-vehicle communication simulation
- `scale_mid.py` and `scale_large.py` — scale configuration definitions
- `multi_sim.py` — multi-simulator orchestration and parallel evaluation
- `panel.py` — metrics dashboard and JSON save/load panel
- `run.py` — command-line runner for the training pipeline and smoke tests

## Implemented experiment and scenario code

The `cases-study/` directory contains a substantial set of scripts used for scenario modeling and structured experimentation.

### Scenario and matrix tools

- `scenario_matrix.py` builds a scenario matrix across controller combinations, communication severities, natural conditions, and demand patterns.
- It supports multi-seed runs and aggregate summary metrics.
- It can print comparison tables and export CSV results.

### Adaptive execution study

`adaptive_multi_execution.py` contains:

- a signal-state machine with green/yellow/all-red phases,
- an adaptive controller with online parameter tuning,
- neighbor snapshot exchange,
- queue/arrival simulation,
- and multi-episode parallel execution.

### Related case-study scripts

These scripts are present in the repo and represent implemented research-oriented code:

- `scenario_all.py`
- `scenario_advanced.py`
- `scenario_extensions.py`
- `scenario_ramps.py`
- `synthetic_dataset.py`
- `synth_to_pred.py`
- `analytical_surrogate.py`
- `bayesian_active_learning.py`
- `trust_region_active_learning.py`
- `trust_region_decay.py`

These components demonstrate modeling, simulation, and experimentation work in Python, but they are not automatically validated end-to-end benchmarks for real-world traffic performance.

## Evaluation and reporting utilities

`cityflow_features/multi_sim.py` implements:

- `MultiSimulator` for routing state/actions across child simulators,
- `ParallelEvaluator` for running independent episodes and aggregating results,
- and helper routines for comparing action modes.

`cityflow_features/panel.py` records snapshots from the training environment and provides:

- a rolling metrics history,
- summary formatting,
- panel comparisons,
- rendering helpers,
- and JSON persistence for recovery.

These utilities are implemented and usable as experiment support tooling.

## Documentation and test scaffolding

The repository includes:

- `HOW_TO_RUN.md` for installation and usage,
- `docs/` for project documentation structure,
- `tests/CMakeLists.txt` and test directories for C++ and Python test configuration,
- and the frontend README for replay functionality.

## Important interpretation

The repository contains both the real CityFlow simulator and a number of Python-only experimental modules. The implemented code should be interpreted as follows:

- The core CityFlow engine and frontend are part of the repository.
- The `cityflow_features` pipeline is a self-contained traffic-signal experimentation stack implemented in Python.
- The default environment used by those feature modules is a mock simulator, not a validated benchmark of real CityFlow roads.
- The case-study scripts are implemented experiment drivers and models, not guaranteed deployment-ready traffic-control solutions.
- The presence of a metric, scenario, or experiment script does not by itself establish that the approach is proven to outperform other controllers.

## Quick usage examples

### Python package install

```bash
pip install .
```

### Run a smoke test

```bash
python cityflow_features/run.py --smoke-test
```

### Run a training-environment demo

```bash
python cityflow_features/run.py --training-env --num-intersections 5 --steps 50 --seed 42
```

### Run case-study scenario matrix (depending on the script’s intended use)

```bash
python cases-study/scenario_matrix.py --seeds 3 --steps 80
```

## Summary

This repository is not just a simulator; it also contains a Python feature environment, case-study experimentation, and multiple signal-control abstractions. The implemented codebase demonstrates a full stack of simulation, visualization, mock RL training, scenario modeling, and evaluation tooling, but it should be read as a collection of implemented components with varying maturity and validation status.
