# Grid Flow Simulator and Visualizer

This folder contains a Python tool for visualizing a small CityFlow-style grid and generating a flow-based traffic animation:

- `3_4_tool_simul_single_with_multi.py` reads a road-network JSON and flow JSON, runs a lightweight Python traffic simulation, and displays the resulting animation.
- `roadnet_3_4.json` and `flow_3_4.json` are the example input files.
- `_v2x_grid_data_34_3030/` contains additional generated grid scenarios and related project data. Those files are not guaranteed to be compatible with this script.
- `Figure_1_v2v_grid.png`, `Figure_2_v2v_grid_.png`, and `_single_agent_actions_output_.txt` are accompanying artifacts.

## Requirements

Use Python 3 and install the required packages:

```bash
python -m pip install numpy matplotlib
```

`tabulate` is optional and only affects console table formatting:

```bash
python -m pip install tabulate
```

To export an MP4, install the `ffmpeg` executable and make sure it is on your `PATH`. The `tqdm` import is optional and is not used by the current script.

## Run the example

From the repository root, provide input paths explicitly. The script's defaults are local Windows paths and are unlikely to exist on another machine.

```bash
python simulation-tools/tools_generator_grid_replay/3_4_tool_simul_single_with_multi.py \
  --roadnet simulation-tools/tools_generator_grid_replay/roadnet_3_4.json \
  --flow simulation-tools/tools_generator_grid_replay/flow_3_4.json
```

To export the animation as `traffic_simulation_flow.mp4` in the current working directory:

```bash
python simulation-tools/tools_generator_grid_replay/3_4_tool_simul_single_with_multi.py \
  --roadnet simulation-tools/tools_generator_grid_replay/roadnet_3_4.json \
  --flow simulation-tools/tools_generator_grid_replay/flow_3_4.json \
  --save
```

See all options with:

```bash
python simulation-tools/tools_generator_grid_replay/3_4_tool_simul_single_with_multi.py --help
```

| Option | Default | Description |
| --- | --- | --- |
| `--roadnet` | Local Windows path | Road-network JSON input. |
| `--flow` | Local Windows path | Flow JSON input. |
| `--output` | `frame` | Prefix for the first-frame PNG and PDF exports. |
| `--save` | Off | Export the animation to `traffic_simulation_flow.mp4` using ffmpeg. |
| `--interval` | `50` ms | Display delay between animation frames. |
| `--duration` | `60.0` seconds | Simulated duration. |
| `--dt` | `0.1` seconds | Simulation time step; total frames are `int(duration / dt)`. |

For example, `--output results/grid` writes `results/grid_000.png` and `results/grid_000.pdf`. Create the `results` directory first if it does not exist. The MP4 output name is fixed in the script and does not use the `--output` prefix.

## Inputs and outputs

The road-network parser expects the legacy top-level `intersections` and `roads` JSON structure. Intersection entries need an `id` and `point`; ordinary intersections also need `width`. Road entries need `id`, `points`, `lanes`, `startIntersection`, and `endIntersection`. Lane entries need a `width`.

The flow input is a JSON array. Each flow entry is expected to provide `vehicle`, `route`, `interval`, `startTime`, and `endTime`. The simulator uses vehicle dimensions and maximum speed from `vehicle` and road IDs from `route`.

On startup, the tool prints a network summary and creates a snapshot of the initial state:

- `<output>_000.png` — PNG snapshot.
- `<output>_000.pdf` — PDF snapshot.
- `traffic_simulation_flow.mp4` — optional animation export when `--save` is used.

The interactive animation is shown through Matplotlib. The initial snapshot is taken before the simulation advances, so it may not contain vehicles that are scheduled to spawn later.

## Scope and limitations

This is a custom, simplified Python traffic simulator and visualizer, not a call to the CityFlow engine. It is not a faithful implementation of the CityFlow paper or a multi-agent reinforcement-learning controller. In particular, the current code uses simplified vehicle motion and signal logic, models only one signalized intersection (hard-coded as `intersection_1_1` in parts of the simulator), assumes boundary coordinates at ±300, and chooses a middle lane when spawning vehicles. It does not model detailed car-following, lane changes, collisions, or network-wide signal coordination. Results should be treated as illustrative, not as validated traffic-engine or research results.

The script's geometry and traffic-light mapping are tailored to the included 3-by-4 example. Other grids under `_v2x_grid_data_34_3030/` may use different formats or topology and may need code changes before use. For general CityFlow engine simulation and grid generation, see [`tools/generator/readme.md`](../../../tools/generator/readme.md) and the [CityFlow documentation](https://cityflow.readthedocs.io/).

## Research references

These papers are relevant background, but this script does not implement their learning methods:

1. *SocialLight: Distributed Cooperation Learning towards Network-Wide Traffic Signal Control.* [ACM Digital Library](https://dl.acm.org/doi/epdf/10.5555/3545946.3598809).
2. *CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario.* [ACM Digital Library](https://dl.acm.org/doi/pdf/10.1145/3308558.3314139).
