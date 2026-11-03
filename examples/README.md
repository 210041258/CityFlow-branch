# Traffic Simulation Examples

This directory collects CityFlow-format network and flow data, runnable small examples, larger grid scenarios, and saved experiment artifacts. Pair each road-network file with its matching flow file: routes refer to road IDs, so a flow file from one grid should not be assumed to work with another network.

## Contents

### Shared sample inputs

- [`config.json`](config.json) — CityFlow configuration for the top-level sample `roadnet.json` and `flow.json`; it enables replay saving.
- [`roadnet.json`](roadnet.json) and [`flow.json`](flow.json) — small shared network and traffic-flow sample referenced by that configuration.

### Grid scenarios

| Folder | Scenario assets and notes |
| --- | --- |
| [`3-4/`](3-4/) | Small example with a `cityflow/` subfolder containing Python engine examples, plus simulation and TRAFI-related files. See [`3-4/cityflow/Readme.md`](3-4/cityflow/Readme.md). |
| [`9-9/`](9-9/) | Turn-flow road network, TRAFI scripts, scenario outputs, and V2X-related artifacts. See [`9-9/readme.md`](9-9/readme.md) for network/flow pairing, run notes, and limitations. |
| [`13-13/`](13-13/) | Larger turn-flow grid with scripts and saved analysis outputs. |
| [`17-17/`](17-17/) | Larger turn-flow grid with scripts and saved analysis outputs. |
| [`24-24/`](24-24/) | Larger turn-flow grid with scripts and saved analysis outputs. |
| [`30-30/`](30-30/) | Large scenario scripts and experiment artifacts; consult the files in the folder for the matching network and flow inputs. |

Many scenario folders contain generated reports, plots, and replay/statistics artifacts. These are sample outputs, not necessarily fresh results from the commands below.

## Run the top-level CityFlow sample

Install or build the CityFlow Python package for your environment, then run from the repository root so the relative paths in `examples/config.json` resolve as intended:

```python
import cityflow

engine = cityflow.Engine("examples/config.json", thread_num=1)
for _ in range(100):
    engine.next_step()
```

The configuration sets `saveReplay` to `true`, with replay output paths under `examples/`. Check the generated output location and existing files before running if you want to preserve earlier results.

A more guided 3×4 engine example is available in [`3-4/cityflow/basic_run_cityflow_simulation.py`](3-4/cityflow/basic_run_cityflow_simulation.py). From the repository root:

```bash
python examples/3-4/cityflow/basic_run_cityflow_simulation.py
```

That example uses the `cityflow` Python module and the 3×4 roadnet/flow files located beside the script.

## Custom experiment scripts

Some folders contain custom TRAFI simulators or scenario generators in addition to CityFlow data. These are separate implementations and should not be treated as equivalent to running the CityFlow engine. Read the folder-specific README and inspect each script's options and input paths before use. In particular, see [`9-9/readme.md`](9-9/readme.md) for important notes on its custom simulator, reported results, and flow behavior.

## Reproducibility

- Record the exact network, flow, configuration, simulator, command, package versions, and random seed used for each experiment.
- Use matching road-network and flow files for each grid size.
- Keep new generated outputs separate from checked-in examples when possible.
- Treat checked-in plots and reports as artifacts unless you independently reproduce and validate their runs.

For repository-level setup and the project map, see the [repository README](../README.md). For generating new grid roadnets and flows, see [`../tools/generator/readme.md`](../tools/generator/readme.md).
