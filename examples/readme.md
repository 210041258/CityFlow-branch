# Grid Scenario Examples

The examples and grid generator provide progressively larger CityFlow scenarios for testing network definitions, traffic demand, signal plans, and simulation behavior. Use a staged workflow: start with a small grid, validate the generated files, run the simulation, then scale or vary one parameter at a time.

## Stage 1: Generate a small scenario

From the repository root, generate a 3-row by 4-column grid. The generator writes a road-network JSON and a matching flow JSON to the current directory by default:

```bash
python tools/generator/generate_grid_scenario.py 3 4
```

To specify output names and location explicitly:

```bash
python tools/generator/generate_grid_scenario.py 3 4 \
  --dir ./examples/generated \
  --roadnetFile roadnet_3_4.json \
  --flowFile flow_3_4.json
```

The positional arguments are the number of rows and columns. The generated default filenames are `roadnet_<rows>_<columns>.json` and `flow_<rows>_<columns>.json`; turn-demand filenames include `_turn` when `--turn` is used.

## Stage 2: Choose network and traffic assumptions

The generator supports configuring geometry, lanes, vehicle properties, route patterns, and vehicle injection rate. Begin with defaults, then change only the settings relevant to the experiment.

### Network layout

- `--rowDistance` — spacing between intersections along east-west roads; default `300`.
- `--columnDistance` — spacing between intersections along south-north roads; default `300`.
- `--intersectionWidth` — intersection width; default `30`.
- `--numLeftLanes`, `--numStraightLanes`, `--numRightLanes` — lane counts by movement; each defaults to `1`.
- `--laneMaxSpeed` — maximum lane speed in meters per second; default `16.67`.
- `--tlPlan` — generate a predefined traffic-light plan.

### Demand and vehicle model

- `--turn` — generate turning routes instead of only straight-through routes.
- `--interval` — time in seconds between vehicles on each generated flow; default `2.0`.
- `--vehLen`, `--vehWidth` — vehicle dimensions in meters; defaults `5.0` and `2.0`.
- `--vehMaxPosAcc`, `--vehMaxNegAcc` — maximum acceleration and deceleration; defaults `2.0` and `4.5`.
- `--vehUsualPosAcc`, `--vehUsualNegAcc` — usual acceleration and deceleration; defaults `2.0` and `4.5`.
- `--vehMinGap` — minimum following gap; default `2.5`.
- `--vehMaxSpeed` — vehicle maximum speed in meters per second; default `16.67`.
- `--vehHeadwayTime` — vehicle headway time; default `1.5`.

For example, generate a scenario with two straight lanes, a predefined signal plan, and turning flows:

```bash
python tools/generator/generate_grid_scenario.py 3 4 \
  --numStraightLanes 2 --tlPlan --turn \
  --dir ./examples/generated
```

## Stage 3: Validate generated files

Before running a simulation, check that:

1. The road-network and flow files were written to the selected `--dir`.
2. The network dimensions, intersection spacing, lane counts, and signal plan match the experiment.
3. Each flow route references roads present in the generated network.
4. Vehicle parameters and flow `interval` are appropriate for the intended demand level.
5. The simulation configuration points to these generated files.

The generator creates the road network and flow inputs; it does not create a complete simulation configuration or run CityFlow by itself.

## Stage 4: Run CityFlow

Create or update a CityFlow configuration that references the generated road-network and flow files. For configuration format and simulation commands, follow the repository's [getting-started documentation](../docs/source/start.rst). Set a seed and simulation interval for reproducibility, and enable replay output if you plan to inspect the run visually.

## Stage 5: Inspect and compare results

Review replay output, vehicle counts, travel-time or queue metrics, and signal behavior. Compare scenarios using the same network and simulation settings where possible. For a useful experiment, change one factor at a time—for example, lane count, signal plan, route pattern, or flow interval—and record the configuration and seed alongside the results.

## Included example scenarios

Directories such as `3-4`, `9-9`, `13-13`, `17-17`, `24-24`, and `30-30` provide scenarios at different grid sizes. Inspect each directory's configuration and data files before running; larger grids can require more computation. The root `config.json`, `roadnet.json`, and `flow.json` are additional example inputs.

## Generator reference

The implementation is [`tools/generator/generate_grid_scenario.py`](../tools/generator/generate_grid_scenario.py). Run `python tools/generator/generate_grid_scenario.py --help` from the repository root to review the available command-line options.
