# Grid Scenario Examples

This directory contains sample CityFlow scenarios at several grid sizes. You can also generate your own road network and traffic flows with the grid-scenario generator. This guide walks through generating, checking, and using those files.

## Generate a scenario

Run commands from the repository root. This creates a 3-row by 4-column grid and writes the road network and flow files into the current directory:

```bash
python tools/generator/generate_grid_scenario.py 3 4
```

By default, the output files are named `roadnet_<rows>_<columns>.json` and `flow_<rows>_<columns>.json`. With `--turn`, the generator adds `_turn` to both names and includes turning routes.

To keep generated files in `examples/generated`, create that directory first—the generator does not create missing directories—and provide output filenames if desired:

```bash
mkdir -p examples/generated
python tools/generator/generate_grid_scenario.py 3 4 \
  --dir examples/generated \
  --roadnetFile roadnet_3_4.json \
  --flowFile flow_3_4.json
```

The first two positional arguments are the number of rows and columns. Use `--help` to see the full command-line interface:

```bash
python tools/generator/generate_grid_scenario.py --help
```

## Configure the network and demand

The generator supports the following settings. Unless noted otherwise, these are the defaults in the script.

### Network layout

- `--rowDistance` — spacing between intersections along east-west roads, in meters (`300`).
- `--columnDistance` — spacing between intersections along south-north roads, in meters (`300`).
- `--intersectionWidth` — intersection width, in meters (`30`).
- `--numLeftLanes`, `--numStraightLanes`, `--numRightLanes` — lane counts by movement (`1` each).
- `--laneMaxSpeed` — maximum lane speed, in meters per second (`16.67`).
- `--tlPlan` — include the predefined traffic-light plan.

### Traffic demand and vehicle properties

- `--turn` — add turning routes as well as straight-through routes.
- `--interval` — time between vehicles in each generated flow, in seconds (`2.0`).
- `--vehLen`, `--vehWidth` — vehicle length and width, in meters (`5.0` and `2.0`).
- `--vehMaxPosAcc`, `--vehMaxNegAcc` — maximum acceleration and deceleration (`2.0` and `4.5`).
- `--vehUsualPosAcc`, `--vehUsualNegAcc` — usual acceleration and deceleration (`2.0` and `4.5`).
- `--vehMinGap` — minimum gap between vehicles (`2.5`).
- `--vehMaxSpeed` — maximum vehicle speed, in meters per second (`16.67`).
- `--vehHeadwayTime` — vehicle headway time (`1.5`).
- `--dir` — output directory (`./`).
- `--roadnetFile`, `--flowFile` — output filenames; defaults are based on the row and column counts and whether `--turn` is enabled.

For example, generate a grid with two straight lanes, a signal plan, and turning routes:

```bash
mkdir -p examples/generated
python tools/generator/generate_grid_scenario.py 3 4 \
  --numStraightLanes 2 --tlPlan --turn \
  --dir examples/generated
```

## Check the generated files

Before running a simulation, verify that:

1. The road-network and flow JSON files exist in the selected output directory.
2. The grid dimensions, intersection spacing, lane counts, and signal plan match your intended scenario.
3. Flow routes reference roads in the generated network.
4. The vehicle properties and flow interval are appropriate for the demand you want to test.
5. Your CityFlow configuration references the generated road-network and flow files.

The generator writes road-network and flow inputs only. It does not create a complete simulation configuration or run CityFlow.

## Run a simulation

Create or update a CityFlow configuration to point to the generated files, then follow the repository's [getting-started and run instructions](../HOW_TO_RUN.md). For example, if you generated files under `examples/generated`, make sure the paths in your configuration resolve from the directory where you launch the simulation.

## Inspect and compare results

Review replay output, vehicle counts, travel-time or queue metrics, and signal behavior. When comparing scenarios, keep network and simulation settings consistent where possible. Record the generator options and any random seeds used by your experiment so the comparison can be reproduced.

## Included example scenarios

The directories `3-4`, `9-9`, `13-13`, `17-17`, `24-24`, and `30-30` contain example scenarios with progressively larger grids. Before running one, inspect its configuration and input files to confirm that the road network, flows, and file paths match your intended setup. Larger grids may take more time and memory to simulate, so choose a scenario that fits your available resources. For a fair comparison, keep the simulation settings consistent and record the scenario and configuration used for each experiment.

## Generator source

The generator implementation is [`tools/generator/generate_grid_scenario.py`](../tools/generator/generate_grid_scenario.py). Road-network construction is provided by `tools/generator/generate_json_from_grid.py`.
