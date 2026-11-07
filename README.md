# CityFlow-branch

**CityFlow-branch** is a fork of the [CityFlow traffic simulator](https://github.com/cityflow-project/CityFlow) that combines the CityFlow simulation engine with additional Python modules for traffic-signal experimentation, scenario evaluation, synthetic dataset generation, and stress testing.

This README documents the functionality currently implemented in the repository. It focuses on the existing codebase rather than presenting future research proposals or unimplemented roadmap items.

**Repository:**  
https://github.com/210041258/CityFlow-branch

**Feature package:**  
https://github.com/210041258/CityFlow-branch/tree/master/cityflow_features

---

## Overview

The repository contains:

- The CityFlow C++ simulation engine and Python interface.
- Browser-based replay and traffic-visualization tooling.
- Installation and usage documentation.
- A structured Python feature package under `cityflow_features/`.
- Experimental traffic-signal control and evaluation modules.
- Case-study and scenario-generation scripts under `cases-study/`.
- Synthetic dataset and stress-testing tools under `cityflow_features/dataset_/`.

The repository therefore combines the original CityFlow simulator with additional experimental Python components for signal control, mock reinforcement-learning environments, scenario-based evaluation, communication modeling, and synthetic data generation.

---

## Repository Structure

```text
CityFlow-branch/
├── src/                       # CityFlow C++ simulation engine
├── frontend/                  # Browser-based replay and visualization
├── simulation/                # Traffic visualization utilities and assets
├── cityflow_features/         # Python traffic-signal feature modules
│   └── dataset_/              # Dataset generation and stress-testing tools
├── cases-study/               # Scenario and experiment scripts
├── tests/                     # Test scaffolding
├── docs/                      # Sphinx documentation
├── HOW_TO_RUN.md              # Installation and usage instructions
├── setup.py                   # Python package configuration
├── CMakeLists.txt             # Native build configuration
└── Dockerfile                 # Containerized build configuration
```

---

# Implemented Simulator and Tooling

## CityFlow Core

The repository retains the standard CityFlow build and Python interface.

Key components include:

- `CMakeLists.txt` — native C++ build configuration.
- `setup.py` — Python package installation.
- `Dockerfile` — containerized build path.
- `HOW_TO_RUN.md` — installation and usage instructions.
- `src/` — core simulation-engine implementation.

The Python interface follows the standard CityFlow pattern, including creation of a `cityflow.Engine` instance and advancing the simulation with methods such as `next_step()`.

---

## Frontend and Replay Tooling

The repository also contains browser-based replay and traffic-visualization utilities.

### Frontend

- `frontend/index.html` — main browser-based replay interface.
- `frontend/README.md` — instructions for viewing replay data and using example replay files.

### Simulation Visualizer

- `simulation/Readme.md` — documentation for the traffic-intersection visualization utility.
- Supports visualization workflows based on `roadnet.json` and `replay.txt` inputs.

> **Note:** The visualization components are intended as replay and traffic-analysis utilities rather than a general-purpose traffic-control system.

---

# Traffic-Signal Feature Package

The `cityflow_features/` directory contains Python modules for traffic-signal experimentation, simulation, evaluation, and control.

## 1. Signal Characterization

### `signal_characterization.py`

Provides signal-state and traffic-condition characterization at intersections.

Implemented functionality includes:

- Signal phase index and duration.
- Vehicle counts in N/S and E/W directions.
- Queue-related measurements.
- Average phase duration.
- Signal-state history.
- Summary statistics and metrics.

---

## 2. Action-Space Definitions

### `action_space.py`

Defines action representations for traffic-signal control.

Supported concepts include:

- `discrete_phase`
- `switch_hold`
- `phase_duration`
- Action masking.
- Action-mode conversion helpers.
- Action usage and decision metrics.

---

## 3. Training Environment

### `training_env.py`

Provides a Gym-style environment interface for traffic-signal experimentation.

The environment includes:

- `reset()` and `step()` interfaces.
- Minimum-green-time enforcement.
- Per-intersection phase bookkeeping.
- Action processing.
- State and reward integration.

The default simulator is `MockTrafficSimulator`, which models simplified stochastic vehicle arrivals and departures for two-direction queues.

> **Important:** The default mock simulator is a self-contained experimental environment. It should not be interpreted as a calibrated CityFlow benchmark or a validated model of real-world traffic.

---

## 4. Reward Calculation

### `reward_calculator.py`

Implements a multi-objective reward model using traffic-related quantities such as:

- Queue pressure.
- Waiting-time penalties.
- Throughput improvements.
- Fairness penalties.
- Per-component reward tracking.
- Cumulative episode returns.

> **Note:** These rewards are implemented experimental metrics. Their presence in the repository does not imply validation against real-world traffic-control objectives.

---

## 5. Traffic-Control Algorithms

### `traffic_control_algorithm.py`

Provides non-MARL traffic-signal control strategies, including:

- `fixed_time_control`
- `adaptive_control`
- `optimized_control`

The module also records controller decisions and exposes metrics such as:

- Total decisions.
- Average phase duration.
- Control-policy activity.

---

## 6. Additional Feature Modules

The package also includes supporting modules for traffic-state representation, communication, scaling, evaluation, and execution.

| Module | Purpose |
|---|---|
| `congestion_state_space.py` | Congestion-state representations |
| `object_closure_parameters.py` | Safety and constraint parameters |
| `v2x_communication.py` | Simplified vehicle-to-infrastructure communication |
| `v2v_communication.py` | Simplified vehicle-to-vehicle communication |
| `scale_mid.py` | Medium-scale configuration definitions |
| `scale_large.py` | Large-scale configuration definitions |
| `multi_sim.py` | Multi-simulator orchestration and parallel evaluation |
| `panel.py` | Metrics dashboard and JSON persistence |
| `run.py` | Command-line runner and smoke-test utilities |

---

# Dataset Generation and Stress Testing

## `cityflow_features/dataset_/`

The `dataset_` directory contains the scripts and generated outputs used for synthetic traffic/stress-test dataset generation.

The directory currently includes:

```text
dataset_/
├── api_generator_v3.py
├── readme.txt
├── stress_v31.csv
├── stress_v31.parquet
└── stress_v31.schema.json
```

### Dataset Generator

#### `api_generator_v3.py`

This is the primary dataset-generation script.

It can generate synthetic network-traffic/datagram data, apply simulated attack and failure conditions, and export the resulting dataset in multiple formats.

Example:

```bash
cd cityflow_features/dataset_
python api_generator_v3.py --export all --out stress_v31 --seed 1337
```

### Generated Files

The generator can produce:

- `stress_v31.csv` — CSV representation of the generated dataset.
- `stress_v31.parquet` — columnar Parquet representation.
- `stress_v31.schema.json` — schema describing the generated dataset.

### Dataset Statistics

The generation workflow records statistics such as:

- Valid and dropped datagrams.
- Simulated attacks.
- IDS alerts.
- Nonce replays.
- Sequence gaps.
- Signature failures.
- Backoffs.
- Cascade windows.
- Total signal switches.
- Final queue.
- Average reward.
- Total spilled traffic.

The repository's `readme.txt` also records the generation command and resulting dataset statistics for the current `stress_v31` dataset.

> **Important:** These datasets are synthetic stress-test outputs generated by the repository's tooling. They should not automatically be interpreted as measurements collected from a real-world traffic or V2X deployment.

---

# Experiment and Scenario Code

## `cases-study/`

The `cases-study/` directory contains scenario-generation, simulation, analytical, and experimental scripts.

## Scenario Matrix

### `scenario_matrix.py`

Builds structured scenario combinations across different experimental dimensions, including:

- Controller combinations.
- Communication severity.
- Environmental or natural conditions.
- Traffic-demand patterns.
- Multiple random seeds.

It can also aggregate results and export summary data.

---

## Adaptive Multi-Execution

### `adaptive_multi_execution.py`

Contains experimental components for multi-episode and adaptive signal-control execution, including:

- Signal-state transitions.
- Green/yellow/all-red state handling.
- Adaptive controller behavior.
- Online parameter tuning.
- Neighbor-state exchange.
- Parallel multi-episode execution.

---

## Related Case-Study Modules

Additional scripts include:

```text
scenario_all.py
scenario_advanced.py
scenario_extensions.py
scenario_ramps.py

synthetic_dataset.py
synth_to_pred.py

analytical_surrogate.py
bayesian_active_learning.py
trust_region_active_learning.py
trust_region_decay.py
```

These scripts support different forms of scenario modeling, synthetic data generation, surrogate modeling, and experimental evaluation.

> **Note:** These modules represent implemented modeling and experimentation code. Their presence does not imply that the resulting methods have been validated as real-world traffic-control systems or proven to outperform established controllers.

---

# Evaluation and Reporting Utilities

## `multi_sim.py`

Provides multi-simulation and evaluation functionality, including concepts such as:

- `MultiSimulator` for coordinating child simulators.
- `ParallelEvaluator` for running independent episodes.
- Aggregation of simulation results.
- Comparison of different action modes.

---

## `panel.py`

Provides metrics and reporting utilities for the experimental environment.

Functionality includes:

- Recording environment snapshots.
- Maintaining rolling metric histories.
- Formatting summaries.
- Comparing panels or experiment results.
- Rendering metrics.
- Saving and loading JSON state for recovery.

---

# Quick Usage

## Install the Python Package

```bash
pip install .
```

---

## Run a Smoke Test

```bash
python cityflow_features/run.py --smoke-test
```

---

## Run the Training Environment

For example:

```bash
python cityflow_features/run.py \
    --training-env \
    --num-intersections 5 \
    --steps 50 \
    --seed 42
```

---

## Generate the Stress-Test Dataset

```bash
cd cityflow_features/dataset_
python api_generator_v3.py \
    --export all \
    --out stress_v31 \
    --seed 1337
```

This generates the dataset exports and associated schema/metadata files.

---

## Run a Case-Study Scenario Matrix

```bash
python cases-study/scenario_matrix.py \
    --seeds 3 \
    --steps 80
```

---

# Important Interpretation and Disclaimer

This repository contains both the original CityFlow simulation infrastructure and additional experimental Python modules. These components should be interpreted according to their implementation and validation status.

1. **CityFlow Core**  
   The core CityFlow simulator, C++ engine, Python interface, and associated frontend tooling are included in the repository.

2. **`cityflow_features/`**  
   This directory contains a self-contained Python experimentation stack for traffic-signal states, actions, rewards, environments, controllers, communication models, and evaluation.

3. **Experimental Environments**  
   Some modules use simplified or mock simulation models. These are useful for experimentation and software testing but should not automatically be considered calibrated CityFlow or real-world traffic models.

4. **Dataset Generation**  
   The `dataset_` tools generate synthetic stress-test and network-traffic data. Generated statistics describe the behavior of the implemented generator and should not be interpreted as real-world measurements.

5. **Case Studies**  
   The scripts under `cases-study/` implement scenario modeling and experimental workflows. They are not, by themselves, evidence of deployment readiness or superiority over existing traffic-control approaches.

6. **Experimental Metrics**  
   Metrics such as queue length, waiting time, reward, throughput, attacks, IDS alerts, and communication failures are implemented for experimentation. Their presence does not by itself establish scientific or real-world validation.

---

# Repository Status

The repository is best understood as a combination of:

- A working CityFlow simulation foundation.
- Python-based traffic-signal experimentation tools.
- Mock and simplified environments for development and testing.
- Scenario and controller experimentation.
- Synthetic dataset and stress-test generation.
- Evaluation and reporting utilities.

The distinction between **implemented functionality**, **experimental models**, and **validated real-world performance** is intentional and should be maintained when interpreting results from this repository.

---

## Related Resources

- **CityFlow project:** https://github.com/cityflow-project/CityFlow
- **This fork:** https://github.com/210041258/CityFlow-branch
- **`cityflow_features/`:** https://github.com/210041258/CityFlow-branch/tree/master/cityflow_features
- **Dataset generator:** `cityflow_features/dataset_/api_generator_v3.py`

---
