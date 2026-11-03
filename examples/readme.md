# Grid Scenario Examples

This directory contains CityFlow road-network and traffic-flow examples at several grid sizes, plus Python traffic-flow experiments. This guide distinguishes those artifacts: the JSON grid scenarios are CityFlow inputs, while the `TRAFI.py` and `TRAFI-Simulation.py` scripts implement separate, simplified Python experiments. Their hard-coded scenario settings should not be read as calibrated real-world traffic ratios.

## Generate a CityFlow grid scenario

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

## What the included Python scenarios actually model

The grid-size folders contain `TRAFI.py` and/or `TRAFI-Simulation.py` scripts, generated CityFlow-style roadnet/flow JSON files, and saved outputs. The Python scripts are simplified traffic models with hard-coded scenario settings; they are not documented as calibrated against real-world traffic counts or as using a measured vehicle-class mix.

In the `TRAFI-Simulation.py` scenario runner, the same nominal settings appear in the grid folders:

| Scenario | Vehicles per flow (script setting) | Congestion factor | Additional scripted behavior |
| --- | ---: | ---: | --- |
| Congestion | 15 | 0.5 | Attempts to create a traffic jam at the central intersection |
| Emergency | 8 | 1.0 | Adds 2 emergency vehicles |
| Normal (fallback case) | 5 | 1.0 | No additional event specified |

These are simulator parameters, not traffic composition ratios (for example, they do not establish a real-world percentage of cars, buses, or emergency vehicles). The scripts' congestion metric is also a model-defined quantity: running vehicles divided by a capacity assumption of 20 vehicles per road, capped at 1.0.

The JSON `flow_<grid>_turn.json` files describe generated routes and vehicle parameters. Their `interval` and `maxSpeed` values vary across the checked-in grid files; inspect the specific JSON rather than assuming every grid uses the generator's default demand. The Python execution logs also show that some runs could not find the requested flow file and then used sample data or initialized no vehicles. Treat saved output files as run artifacts, not validation that a particular grid's JSON flow was successfully consumed.

To inspect a scenario, check its script, flow JSON, roadnet JSON, and any run instructions or output logs together. From a scenario directory, the checked-in command examples use patterns such as:

```bash
python TRAFI-Simulation.py --flow-file flow_9_9_turn.json --duration 600 --dt 0.5
python TRAFI.py --flow flow_9_9_turn.json --steps 500 --dt 0.5 --model IDM
```

Use the filename and options appropriate to that directory, and verify that the script reports loading the intended flow file. If it reports a missing flow file or switches to sample data, the run is not exercising that grid's stored demand. The scripts use Python packages such as NumPy and Matplotlib; install them in your Python environment if they are not already available.

## Check generated CityFlow files

Before running a CityFlow simulation, verify that:

1. The road-network and flow JSON files exist in the selected output directory.
2. The grid dimensions, intersection spacing, lane counts, and signal plan match your intended scenario.
3. Flow routes reference roads in the generated network.
4. The vehicle properties and flow interval are appropriate for the demand you want to test.
5. Your CityFlow configuration references the generated road-network and flow files.

The grid generator writes road-network and flow inputs only. It does not create a complete simulation configuration or run CityFlow.

## Run a CityFlow simulation

Create or update a CityFlow configuration to point to the generated files, then follow the repository's [getting-started and run instructions](../HOW_TO_RUN.md). Make sure the paths in your configuration resolve from the directory where you launch the simulation.

## Included example scenarios

The directories `3-4`, `9-9`, `13-13`, `17-17`, `24-24`, and `30-30` contain examples at different grid sizes. The `3-4` folder also includes a nested `Traffic Flow Simulator (trafic)` Python example; the larger grid folders include Python traffic-flow scripts, route/road-network data, and saved run artifacts. The grid dimensions indicate scenario scale, not a calibrated demand level or a traffic ratio. Larger road-network JSON files can take substantially more disk space, memory, and time to inspect or simulate; for example, the checked-in 24-by-24 roadnet is tens of megabytes. Start with a smaller folder, confirm the Python script loads the intended files, and scale up as your machine permits.

For comparisons, use the same flow file, model, time step, duration, and other settings when possible. Record whether each run used the actual stored flow JSON or fallback sample data. Do not compare saved results as if they were controlled benchmark results unless the inputs and successful loading have been verified.

## Generator source

The generator implementation is [`tools/generator/generate_grid_scenario.py`](../tools/generator/generate_grid_scenario.py). Road-network construction is provided by `tools/generator/generate_json_from_grid.py`.
