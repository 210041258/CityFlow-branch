# Grid Traffic Replay and Visualization

This folder contains a self-contained traffic visualization experiment for the `roadnet_3_4.json` road network and its matching `flow_3_4.json` traffic demand. The Python script `visu.py` parses the network and flow definitions, computes lane and intersection geometry, simulates vehicles moving along configured routes, and displays a traffic-light-controlled replay.

## Benefits

- **Understand network structure:** prints a summary of intersections, roads, lane counts, lane widths, and signal positions.
- **Inspect traffic demand:** reads the flow file's routes, vehicle properties, spawn intervals, and active time ranges.
- **Visualize movement:** renders vehicles on lane paths and shows signal colors during the replay.
- **Create shareable artifacts:** exports a static initial frame as PNG and PDF, and can save an animation as MP4 when FFmpeg is installed.
- **Experiment without a full CityFlow run:** provides a lightweight, Python-based visualization path for inspecting the bundled example inputs.
- **Reproducible inputs:** the included road network and flow files make it possible to repeat the visualization with the same scenario definitions.

This folder is an example and visualization aid. Its Python traffic simulator uses simplified movement and signal logic; it is not a replacement for CityFlow's engine or a calibrated model of real-world traffic. Results should not be treated as real-world measurements or validated operational predictions.

## Files

- `visu.py` — parser, geometry builder, simplified traffic simulator, and Matplotlib visualization.
- `roadnet_3_4.json` — CityFlow-format road-network definition.
- `flow_3_4.json` — vehicle routes, vehicle properties, and flow timing.
- `grid_data/` — related grid scenario data.
- `frame_000.png` and `frame_000.pdf` — example static visualization outputs, when present.
- `Figure_1.png`, `image.png`, and screenshots — supporting visual references.
- `output.txt` — an example console summary from a run; it may reflect an earlier environment or script version.

## Requirements

- Python 3
- `numpy`
- `matplotlib`
- Optional: `tabulate` for formatted console tables
- Optional: FFmpeg for MP4 export

Install the Python dependencies with:

```bash
python -m pip install numpy matplotlib tabulate
```

## Run

From this directory:

```bash
python visu.py --roadnet roadnet_3_4.json --flow flow_3_4.json
```

The script prints network details, writes `frame_000.png` and `frame_000.pdf` (or files using the selected output prefix), then opens a Matplotlib animation window. Run headlessly by selecting a Matplotlib non-interactive backend and avoiding the animation display, or adapt the script for batch export.

Options:

```text
--roadnet PATH   Road-network JSON (default: roadnet_3_4.json)
--flow PATH      Traffic-flow JSON (default: flow_3_4.json)
--output PREFIX  Static frame output prefix (default: frame)
--duration SEC   Simulated duration (default: 60)
--dt SEC         Simulation timestep (default: 0.1)
--interval MS    Animation display interval (default: 50)
--save           Save an MP4 animation (requires FFmpeg)
```

For example, to save an MP4 and a static image set:

```bash
python visu.py --roadnet roadnet_3_4.json --flow flow_3_4.json \
  --duration 60 --dt 0.1 --interval 50 --save --output replay
```

The MP4 is written as `traffic_simulation_flow.mp4`; the static outputs use the requested prefix.

## Branch benefits

The branch combines the canonical CityFlow example assets with additional tools for inspection and experimentation. This replay folder makes the map geometry, lane layout, flow routes, and signal visualization easier to review, while the broader repository provides traffic-control features and simulation examples. Together, these materials can help with prototyping, teaching, debugging input files, and comparing alternative traffic-control ideas.

The branch's added utilities are supplementary research code: verify dependencies and scenario assumptions, and compare any policy results against the CityFlow engine and suitable baselines before drawing conclusions.
