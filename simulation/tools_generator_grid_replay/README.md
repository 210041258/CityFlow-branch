# Grid Traffic Replay: Theory and Practical Benefits

## Purpose

This folder demonstrates a grid-traffic replay using `roadnet_3_4.json` and `flow_3_4.json`. The road-network file describes intersections, roads, lanes, and signal phases; the flow file describes vehicle properties, routes, and injection intervals. `visu.py` parses these inputs, builds map geometry, advances vehicle movement, and visualizes traffic and signal states.

## Core ideas

Traffic simulation separates the **network model** from **traffic demand**. The network represents connectivity and geometry, while flow records specify where and when vehicles enter and which routes they follow. Keeping these inputs separate makes scenarios easier to inspect and vary.

The visualizer turns the JSON definitions into a map with road surfaces, lane boundaries, intersection outlines, stop-line positions, vehicles, and signal colors. This connects the configuration to an understandable visual result and supports review of routes, geometry, and signal mappings.

## Practical benefits

- Inspect intersections, road edges, lane widths, routes, vehicle properties, and signal phases together.
- Review whether the configured flow routes match the road IDs in the network.
- Visualize vehicle movement and signal states for demonstration, teaching, and discussion.
- Generate static PNG/PDF frames and optionally an MP4 animation.
- Reproduce the bundled scenario from the same road-network and flow inputs.
- Review scenario inputs before running larger experiments with CityFlow.

## Run

From this directory:

```bash
python -m pip install numpy matplotlib
python visu.py --roadnet roadnet_3_4.json --flow flow_3_4.json
```

Optional `tabulate` provides formatted console tables; MP4 export requires FFmpeg:

```bash
python -m pip install tabulate
python visu.py --roadnet roadnet_3_4.json --flow flow_3_4.json \
  --duration 60 --dt 0.1 --save --output replay
```

Use `python visu.py --help` to see the available command-line options.

## References

SocialLight: Distributed Cooperation Learning towardsNetwork-Wide Traffic Signal Control
https://dl.acm.org/doi/epdf/10.5555/3545946.3598809

CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario
https://dl.acm.org/doi/pdf/10.1145/3308558.3314139

IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control
https://doi.org/10.1145/3219819.3220096

Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control
https://doi.org/10.1109/TITS.2019.2913345


CoLight: Learning Network-level Cooperation for Traffic Signal Control
https://doi.org/10.1145/3357384.3357902

Learning Phase Competition for Traffic Signal Control
https://proceedings.mlr.press/v97/zheng19d.html

Coordinated Deep Reinforcement Learners for Traffic Light Control
https://arxiv.org/abs/1609.01120


Modeling and Simulation of Urban Traffic Networks with Realistic Flow Patterns
https://doi.org/10.1016/j.simpat.2018.04.002

Reinforcement Learning Methods for Adaptive Traffic Signal Control: A Review
https://doi.org/10.1109/OJITS.2022.3157413

A Deep Learning Framework for Real-Time Urban Traffic Signal Control
https://doi.org/10.1109/ITSC.2022.9914867
