# 9×9 Grid Scenario: Network Maps and Concurrent Flows

This folder contains a 9×9 grid experiment: a CityFlow-format road network and turn-flow definitions, plus two separate Python experiment scripts and previously generated outputs. The term “concurrent flows” here refers to the multiple route streams described in the flow JSON; it does not mean that the scripts are distributed or multi-process simulations.

## Files in this scenario

- [`roadnet_9_9_turn.json`](roadnet_9_9_turn.json) — CityFlow network geometry and road/intersection connectivity for the 9×9 scenario.
- [`flow_9_9_turn.json`](flow_9_9_turn.json) — route-flow entries using the network's road IDs. The entries include horizontal and vertical routes, with `startTime: 0`, `endTime: -1`, and a configured spawn interval (about 5.11 seconds in this file). In a CityFlow run, these definitions describe traffic streams that can be active over overlapping time periods.
- [`TRAFI.py`](TRAFI.py) — custom traffic-flow simulator. It includes IDM, Gipps, and simplified safe car-following options, as well as a small, hard-coded intersection/road mapping. It is not the CityFlow engine and should not be assumed to model the full 9×9 network faithfully.
- [`TRAFI-Simulation.py`](TRAFI-Simulation.py) — custom state generator for normal, congestion, and emergency scenarios. It reads a flow file and writes scenario state and report artifacts.
- [`output_states/`](output_states/) — checked-in example scenario reports, state files, plots, and a comparative report.
- [`expirments/`](expirments/) — saved experiment commands, logs, plots, and result files (directory name preserved as stored in the repository).
- [`v2x_event_effect_trafi_py/`](v2x_event_effect_trafi_py/) — V2X-related TRAFI result artifacts.

## Other grid maps in the repository

The numbered example folders contain different grid-size scenarios. Compare each road-network JSON with its corresponding flow JSON; road IDs in a flow route must exist in the selected network.

| Scenario | Road network | Flow file |
| --- | --- | --- |
| 9×9 | [`examples/9-9/roadnet_9_9_turn.json`](roadnet_9_9_turn.json) | [`examples/9-9/flow_9_9_turn.json`](flow_9_9_turn.json) |
| 13×13 | [`examples/13-13/roadnet_13_13_turn.json`](../13-13/roadnet_13_13_turn.json) | [`examples/13-13/flow_13_13_turn.json`](../13-13/flow_13_13_turn.json) |
| 17×17 | [`examples/17-17/roadnet_17_17_turn.json`](../17-17/roadnet_17_17_turn.json) | [`examples/17-17/flow_17_17_turn.json`](../17-17/flow_17_17_turn.json) |
| 24×24 | [`examples/24-24/roadnet_24_24_turn.json`](../24-24/roadnet_24_24_turn.json) | [`examples/24-24/flow_24_24_turn.json`](../24-24/flow_24_24_turn.json) |
| 30×30 | See [`simulation-tools/tools_generator_grid_replay/_v2x_grid_data_34_3030/grid_30_30/project/`](../../simulation-tools/tools_generator_grid_replay/_v2x_grid_data_34_3030/grid_30_30/project/) for the generated road network and flow inputs. The `examples/30-30/` folder contains related scripts and experiment artifacts. | Same project folder; use the matching 30×30 flow file. |

The repository also documents generator settings for the 9×9 through 30×30 scenarios in [`simulation-tools/tools_generator_grid_replay/_v2x_grid_data_34_3030/readme.md`](../../simulation-tools/tools_generator_grid_replay/_v2x_grid_data_34_3030/readme.md). The maps are distinct inputs, not interchangeable files: changing grid size requires using the matching network and flow together.

## Run the custom scenario-state generator

From this directory, install the Python packages used by the script:

```bash
python -m pip install numpy matplotlib
```

Run both generated hard scenarios into a new output directory to avoid overwriting the checked-in sample artifacts:

```bash
python TRAFI-Simulation.py \
  --flow-file flow_9_9_turn.json \
  --scenario both \
  --duration 600 \
  --dt 0.5 \
  --output-dir output_states_local
```

Use `--scenario congestion` or `--scenario emergency` to run only one scenario. The script's supported options are `--flow-file`, `--output-dir`, `--duration`, `--dt`, `--road-length`, and `--scenario`; run `python TRAFI-Simulation.py --help` for the current CLI.

The output directory may contain scenario state JSON, summary reports, analysis PNGs, and—when both scenarios are generated—a comparative JSON report. Results can vary because the script uses randomized starting positions, speeds, and signal offsets; the checked-in reports are example artifacts, not guaranteed reproduction targets.

## What “concurrent” means in the flow file

Each route entry identifies a vehicle type, a road sequence, and timing parameters. In the 9×9 flow file, the streams begin at time zero and use a recurring interval with an open-ended end time (`-1` in the input). Multiple horizontal and vertical streams can therefore overlap during a CityFlow simulation. Use `flow_9_9_turn.json` with `roadnet_9_9_turn.json` when running the CityFlow engine.

The custom scripts do not necessarily preserve CityFlow's flow semantics. In particular, `TRAFI.py` loads flow JSON through a hard-coded local Windows path in `load_flow_data`, despite exposing a `--flow` CLI argument, and its intersection mapping is only a small fixed mapping. Do not rely on its `--flow` option as a portable way to select the checked-in 9×9 file without correcting that implementation. The scenario generator uses a simplified road/vehicle model; its results are not equivalent to CityFlow engine results.

## Interpreting the checked-in scenario reports

The report JSON files describe outputs from the custom state generator. For example, the checked-in peak-congestion report records a 600-second run and the emergency report records a 1,200-second run; these are different configurations and should not be compared as a controlled head-to-head experiment. The checked-in report also records zero completed vehicles in both cases. Treat reported metrics and generated conclusions as exploratory until the simulator's route handling, signal mapping, metric definitions, and run configuration have been validated.

## Research context

This folder contains scenario data and custom simulation/analysis code. Its presence does not establish a faithful CityFlow engine experiment or an implementation of the SocialLight learning algorithm. For the project’s references and implementation-status notes, see the [repository README](../../README.md).
