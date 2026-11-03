# CityFlow Replay Visualization Tools

This directory contains tools for visualizing CityFlow-style road networks and replay files. The Python visualizer is located in `frontend_replay/python_simu.py`; run the commands below from the repository root unless noted otherwise.

## Python replay visualizer

`frontend_replay/python_simu.py` reads a road-network JSON file and a replay text file. It prints a summary of the network, saves a snapshot of the first replay frame as PNG and PDF, and opens a Matplotlib animation. With `--save`, it also attempts to export the animation as an MP4.

### Requirements

Install Python 3 and the required packages:

```bash
python -m pip install numpy matplotlib
```

`tabulate` is optional and only changes the formatting of the network summary. `tqdm` is imported by the script but is not currently used. MP4 export requires the `ffmpeg` executable to be installed and available on `PATH`; the script passes `writer='ffmpeg'` to Matplotlib.

### Run

The script's default input paths point to a local Windows directory and will not work on most machines. Pass the paths to your own files explicitly:

```bash
python simulation-tools/frontend_replay/python_simu.py \
  --roadnet simulation-tools/frontend_replay/roadnet.json \
  --replay simulation-tools/frontend_replay/replay.txt \
  --output simulation-tools/frontend_replay/frame
```

To save an MP4 as `traffic_simulation.mp4` in the current working directory:

```bash
python simulation-tools/frontend_replay/python_simu.py \
  --roadnet simulation-tools/frontend_replay/roadnet.json \
  --replay simulation-tools/frontend_replay/replay.txt \
  --output simulation-tools/frontend_replay/frame \
  --save
```

Use `--interval 100` to set the animation delay to 100 milliseconds per frame. Run `python simulation-tools/frontend_replay/python_simu.py --help` for all command-line options.

The `--output` argument is a filename prefix for the first-frame exports: for example, `--output results/intersection` creates `results/intersection_000.png` and `results/intersection_000.pdf`. Create the parent output directory before running if it does not already exist. The MP4 filename is currently fixed in the script and is not controlled by `--output`.

### Input expectations and limitations

- `roadnet.json` must contain `static.nodes` and `static.edges` with the fields used by the script, including node `id`, `point`, `virtual`, and `outline`, plus edge `id`, `from`, `to`, `points`, `nLane`, and `laneWidths`.
- The current implementation is specialized to a network containing a node named `intersection_1_1`. It also assumes boundary coordinates at `-300` and `300` when attaching roads. It is not a general-purpose renderer for arbitrary CityFlow networks without adapting the geometry code.
- The replay parser expects one frame per non-empty line, with vehicle and traffic-light sections separated by `;`. Vehicle entries are comma-separated and contain seven whitespace-separated fields: `x y angle name status length width`. Traffic-light entries are comma-separated and contain a road name followed by three color tokens.
- Malformed replay entries are skipped by the current parser. Check the source and your input files if frames or vehicles appear to be missing.
- The script displays only the central intersection traffic lights it can map from the network and does not provide a complete network-wide traffic-signal analysis.
- The animation implementation and its performance have not been benchmarked here. Treat output appearance and runtime as dependent on the input network, replay size, and local Matplotlib backend.

### Outputs

- `<prefix>_000.png` — static visualization of the first parsed replay frame.
- `<prefix>_000.pdf` — PDF export of the same snapshot.
- `traffic_simulation.mp4` — optional animation export when `--save` is supplied and ffmpeg is available.

## Traffic dynamics analysis

A separate analysis script and its usage notes are in [`frontend_replay/readme.md`](frontend_replay/readme.md). That documentation covers `frontend_replay/replay-analysis/current-test/currnet.PY`, which computes replay vehicle-count statistics and optional model fits. Its outputs are descriptive analysis of the supplied replay data, not evidence that a traffic-signal control algorithm has been trained or evaluated.

## Research references and implementation status

These papers provide background for the project. Their citation does not imply that their algorithms or experiments are implemented by the replay visualizer. The Python tool described above is a visualization utility; it does not train a multi-agent reinforcement-learning policy or implement SocialLight's distributed cooperative learning method.

1. *SocialLight: Distributed Cooperation Learning towards Network-Wide Traffic Signal Control.* [ACM Digital Library](https://dl.acm.org/doi/epdf/10.5555/3545946.3598809).
2. *CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario.* [ACM Digital Library](https://dl.acm.org/doi/pdf/10.1145/3308558.3314139).

For CityFlow usage and replay format details, see the [CityFlow replay documentation](https://cityflow.readthedocs.io/en/latest/replay.html). Cite the papers when using their ideas or environment in research, and clearly distinguish reproduced methods from tools that only load or visualize simulation data.
