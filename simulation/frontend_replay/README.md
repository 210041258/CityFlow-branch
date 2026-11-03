# Frontend Replay: CityFlow Simulation Visualizer

This module provides a lightweight replay and visualization layer for CityFlow traffic simulations. It is designed to help researchers and developers inspect how vehicles move through a road network, how traffic lights behave, and how route-level decisions affect the observed traffic pattern.

## Purpose

The frontend replay tools turn raw simulation data into a readable, inspectable visual trace. Instead of only looking at logs or numeric outputs, you can view the motion of vehicles over time, inspect network geometry, and check how signal phases interact with the flow.

This is useful for:
- debugging traffic generation and vehicle routing
- validating simulation assumptions
- comparing different signal control policies
- explaining traffic patterns in a visual form
- producing figures and replay clips for reports or presentations

## Scope of the module

The replay system typically includes:
- road and intersection geometry
- lane boundaries and lane widths
- traffic-light positions and stop-line logic
- vehicle positions and movement traces
- phase-aware visualization of traffic signal states
- summary output for network structure and simulation states

## Typical workflow

1. Generate or load a network and flow dataset.
2. Run CityFlow simulation or replay the current simulation output.
3. Parse the generated motion/state data.
4. Render the network and vehicles in the frontend replay tool.
5. Inspect timing, congestion, and signal behavior frame by frame.

## Main files

- `replay_simu.py` — core replay engine and simulation visualization logic
- `currnet.py` — current simulation parser and metrics utilities
- `visu.py` — visualization helpers for network rendering
- `output.txt`, `growth.txt`, and other generated assets — derived simulation summaries for analysis or reference

## Benefits in the branch

This folder is valuable in the repository because it connects the model and the human-readable output:

- It makes traffic behavior easier to understand.
- It reduces the gap between raw simulation data and visual interpretation.
- It supports debugging of road definitions and signal timing.
- It offers a practical interface for evaluating scenario changes.
- It helps bridge the research side and the demonstration side of the project.

## Expected use cases

This module is well suited for:
- algorithm validation in traffic signal control
- visual inspection of grid-based simulations
- replaying a small scenario to detect routing issues
- preparing demo videos or screenshots for publication
- analyzing queue buildup and vehicle persistence across frames

## Notes

The replay tools are most effective when the simulation data includes consistent geometry, valid route definitions, and expected traffic-light metadata. Missing or malformed inputs may be skipped or reported as errors, so it is important to verify that the generated network files and flow data are structurally valid before using the visualizer.

## Minimal requirements

Typical dependencies include:
- Python 3
- NumPy
- Matplotlib
- optional convenience libraries such as `tabulate` for console summaries
- the CityFlow runtime and compatible simulation data

## Example usage

A common flow is:

```bash
python replay_simu.py
```

or, depending on the exact script and scenario, run one of the current simulation scripts that loads the generated replay and displays the visual output.

## Summary

The frontend replay module is the branch’s practical window into the simulation. It turns abstract traffic data into an explainable, inspectable visual representation, making it easier to validate logic, compare scenarios, and communicate results clearly.
