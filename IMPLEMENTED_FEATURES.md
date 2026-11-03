# Implemented Features in CityFlow-branch

This inventory describes code and documentation present in the repository. It distinguishes runnable or implemented components from demonstrations and simplified models; it does not treat roadmap ideas as completed work or make claims about traffic-control effectiveness.

## Repository and simulator

- The repository retains CityFlow's C++ microscopic traffic-simulation engine, Python interface, build configuration, examples, tests, and browser-based replay frontend.
- `HOW_TO_RUN.md` documents Docker/source installation, use of the Python engine API, and replay visualization. These are repository instructions; this inventory does not imply that every documented setup was tested here.
- The frontend can display CityFlow replay files. `simulation/Readme.md` also describes a separate visualization utility for `roadnet.json` and `replay.txt` data.

## Traffic-signal feature modules

The `cityflow_features/` package contains Python modules for a signal-control pipeline:

- **Observations — `signal_characterization.py`:** records each intersection's phase, phase duration, north–south/east–west vehicle counts, queue ratio, and average recorded duration. It also reports counts of recorded signal states and tracked intersections.
- **Actions — `action_space.py`:** defines `discrete_phase`, `switch_hold`, and `phase_duration` action encodings, with sampling, validation, action masks, conversion helpers, and counters. The `phase_duration` encoding carries a duration value, but the current environment/simulator path resolves actions to phase indices; duration-aware execution is not established by these modules.
- **Environment — `training_env.py`:** provides a Gymnasium-style `reset`/`step` interface, per-intersection phase bookkeeping, a minimum-green hold rule, episode truncation, and per-intersection information. Its default backend is `MockTrafficSimulator`, a stochastic two-approach model—not the CityFlow engine.
- **Mock traffic dynamics:** `MockTrafficSimulator` approximates arrivals and departures for north–south and east–west queues and tracks cleared and stopped vehicles. It is included for testing and bootstrapping, not as evidence of calibrated microscopic traffic behavior.
- **Reward — `reward_calculator.py`:** computes a weighted scalar from queue pressure, stopped-vehicle penalty, cleared-vehicle throughput gain, and an approach-fairness penalty. It exposes component and episode-return metrics. These are reward signals, not a complete independent evaluation suite for travel time or trip completion.
- **Controllers — `traffic_control_algorithm.py`:** implements fixed-time, adaptive, and optimized phase-selection rules based on time, directional vehicle counts, waiting counts, and congestion. The module records decisions and average phase duration.
- **Additional modules:** `congestion_state_space.py` provides congestion-state handling; `object_closure_parameters.py` contains safety-constraint parameters; `v2x_communication.py` and `v2v_communication.py` provide lightweight communication/risk-event models; and `scale_mid.py` / `scale_large.py` define scale configurations.

`cityflow_features/run.py` offers `--training-env` and `--smoke-test` entry points. The training loop samples random actions for demonstration; it does not implement or train a DQN, PPO, or other learned policy. A `CityFlowTrafficSimulator` adapter class is present in that file, but the command-line training path constructs the mock backend by default. The adapter's lane-direction and cleared-vehicle values are documented as simplified mappings/proxies, so they should not be read as validated CityFlow metrics.

## Experiment and analysis scripts

The `cases-study/` directory contains standalone experimental scripts and modeled scenarios:

- `traffic_control_algorithm.py` supplies the shared intersection/car-following model, controller registry, episode simulation, and traffic-control metrics used by the case-study drivers.
- `scenario_matrix.py` assembles controller combinations and crosses them with demand, communication-severity, and natural-condition scenarios. It can run multiple seeds, aggregate mean/standard deviation/normal-approximation 95% confidence intervals for its tracked metrics, print comparison tables, and export CSV. Its code describes a modeled scenario workflow; those outputs are not automatically results from the CityFlow microscopic engine.
- `adaptive_multi_execution.py` defines per-intersection adaptive controllers, a signal state machine with green/yellow/all-red phases, a neighbor snapshot bus, queue-based metrics, and a process-pool episode runner. It uses the case-study intersection/car-following model.
- `scenario_all.py`, `scenario_extensions.py`, `scenario_advanced.py`, and `scenario_ramps.py` define or combine scenario configurations for the case-study simulations.
- `analytical_surrogate.py`, `bayesian_active_learning.py`, `trust_region_active_learning.py`, `trust_region_decay.py`, `synthetic_dataset.py`, and `synth_to_pred.py` contain surrogate-model, active-learning, synthetic-data, and prediction scripts. Their presence identifies implemented research code, but does not by itself establish the validity or reproducibility of reported results.

The `cases-study/readme.md` reports example MAE values for Global EI, Fixed TuRBO, and Decayed TuRBO. Those figures are claims in that README; this inventory does not independently reproduce or verify them.

## Evaluation and reporting utilities

- `cityflow_features/multi_sim.py` provides `MultiSimulator`, which routes intersection actions and state queries across child simulators, and `ParallelEvaluator`, which runs independent environment episodes in a thread pool and aggregates episode returns.
- `cityflow_features/panel.py` records bounded history snapshots from the environment, action space, signal metrics, and reward metrics; it supports text summaries, flattening, panel comparison, and JSON save/load.
- `scenario_matrix.py` and `adaptive_multi_execution.py` also contain their own reporting and multi-seed aggregation paths. These are separate experiment utilities, not a single shared experiment-record format.

## Scope and interpretation

The repository contains a mix of the CityFlow simulator, visualization/build/test assets, standalone Python feature modules, mock-environment demonstrations, and case-study models. The existence of a module, CLI option, smoke test, scenario name, or metric field should not be interpreted as evidence that the feature is connected to the microscopic engine, has been benchmarked under controlled conditions, or improves real traffic outcomes. In particular, the `cityflow_features` training path currently demonstrates a random-action loop on a mock simulator, while the case-study scenario drivers use their own modeled simulation workflow.
