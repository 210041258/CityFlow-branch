# Grid Traffic Replay: Theory and Practical Value

## Purpose

This folder demonstrates how a CityFlow road-network description and traffic-flow definitions can be converted into a visual replay. `roadnet_3_4.json` describes intersections, roads, lanes, and signal phases; `flow_3_4.json` describes vehicle types, routes, and injection intervals. `visu.py` parses these inputs, derives map geometry, advances a simplified traffic model, and renders vehicles and signal colors.

## Core ideas

Urban traffic simulation separates the **network model** from **traffic demand**. The road network defines connectivity and geometry, while flow records define where and when vehicles enter and which routes they follow. Keeping these inputs separate makes it easier to inspect a scenario and compare demand patterns without changing the map.

The visualizer maps network coordinates to road surfaces, lane boundaries, intersection outlines, and stop-line positions. It then animates vehicle positions along route edges and displays signal phases. This makes the structure and behavior of a scenario easier to communicate than raw JSON alone, and can help identify malformed routes, unexpected geometry, or signal-mapping mistakes.

## Practical benefits

- **Scenario inspection:** view intersections, road edges, lane widths, routes, vehicle properties, and signal phases together.
- **Debugging support:** check that flow routes correspond to road IDs and inspect how lane geometry and signal locations are rendered.
- **Repeatable demonstrations:** use the bundled network and flow inputs to reproduce the same visualization setup.
- **Communication:** generate a static PNG/PDF frame and optionally an MP4 animation for reports, teaching, or design discussions.
- **Experiment preparation:** use the visualization to review input assumptions before running larger experiments in CityFlow.

## Run

From this directory, install the required Python packages and launch the visualizer:

```bash
python -m pip install numpy matplotlib
python visu.py --roadnet roadnet_3_4.json --flow flow_3_4.json
```

Optional `tabulate` adds formatted console tables. MP4 export also requires FFmpeg:

```bash
python -m pip install tabulate
python visu.py --roadnet roadnet_3_4.json --flow flow_3_4.json --duration 60 --dt 0.1 --save --output replay
```

The main options are `--roadnet`, `--flow`, `--output`, `--duration`, `--dt`, `--interval`, and `--save`. Use `python visu.py --help` for the script's command-line help.

## Scope and limitations

The simulator inside `visu.py` is a lightweight visualization model with simplified vehicle movement, flow spawning, and signal handling. It is not the CityFlow engine, does not reproduce all CityFlow traffic dynamics, and is not calibrated to real-world traffic observations. Treat its output as an illustration of the provided inputs, not as measured performance or a validated prediction. For research conclusions, run the intended scenario in CityFlow, document configuration and seeds, and compare against appropriate baselines.

## References

- Zhang, H. et al. “CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario.” *The World Wide Web Conference*, 2019. https://doi.org/10.1145/3308558.3314139
- Lopez, P. A. et al. “Microscopic Traffic Simulation using SUMO.” *IEEE Intelligent Transportation Systems Conference*, 2018. https://doi.org/10.1109/ITSC.2018.8569938
- Wei, H. et al. “PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network.” *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2019. https://doi.org/10.1145/3292500.3330949
- Wei, H. et al. “CoLight: Learning Network-level Cooperation for Traffic Signal Control.” *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*, 2019. https://doi.org/10.1145/3357384.3357902
- Wei, H. et al. “IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control.” *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2018. https://doi.org/10.1145/3219819.3220096
