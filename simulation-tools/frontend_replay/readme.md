# CityFlow Replay Visualization and Traffic Analysis

This directory contains two independent Python tools:

- `python_simu.py` visualizes an existing `roadnet.json` and `replay.txt` as a first-frame PNG/PDF snapshot and an interactive animation.
- `replay-analysis/current-test/currnet.PY` analyzes vehicle counts extracted from a replay, fits simple growth models, and optionally plots results.

These scripts consume simulation data; they do not train or evaluate a traffic-signal control policy.

## Requirements

Use Python 3. Install the packages needed by the tools:

```bash
python -m pip install numpy matplotlib
```

For model comparison and traffic-dynamics analysis, install SciPy as well:

```bash
python -m pip install scipy
```

`tabulate` is optional and only affects the visualizer's console tables. MP4 export from `python_simu.py` requires the `ffmpeg` executable to be installed and available on `PATH`. `tqdm` is imported by the visualizer but is not used.

## Replay visualizer

The visualizer's default input values are machine-specific Windows paths. Pass your own inputs explicitly from the repository root:

```bash
python simulation-tools/frontend_replay/python_simu.py \
  --roadnet simulation-tools/frontend_replay/roadnet.json \
  --replay simulation-tools/frontend_replay/replay.txt \
  --output simulation-tools/frontend_replay/frame
```

This writes `frame_000.png` and `frame_000.pdf` under `simulation-tools/frontend_replay/`, then opens the animation window. To also save an MP4 in the current working directory, add `--save`:

```bash
python simulation-tools/frontend_replay/python_simu.py \
  --roadnet simulation-tools/frontend_replay/roadnet.json \
  --replay simulation-tools/frontend_replay/replay.txt \
  --output simulation-tools/frontend_replay/frame \
  --save
```

Use `--interval 100` to change the animation delay to 100 ms per frame. Run the script with `--help` to see the available options.

The road-network parser expects CityFlow-style JSON with `static.nodes` and `static.edges`. The geometry code is specialized: it expects a node named `intersection_1_1` and boundary nodes at coordinates ±300. It is not a general-purpose renderer for arbitrary network geometry. Replay lines are expected to contain vehicle and traffic-light sections separated by `;`; each vehicle entry has seven fields (`x y angle name status length width`), and each traffic-light entry has a road identifier plus three color tokens. Malformed entries may be skipped by the parser.

## Replay count and model analysis

The analysis script is at `simulation-tools/frontend_replay/replay-analysis/current-test/currnet.PY`. Example commands from the repository root:

```bash
# Basic counts (uses the full-data parser)
python simulation-tools/frontend_replay/replay-analysis/current-test/currnet.PY \
  simulation-tools/frontend_replay/replay.txt --growth

# Compare linear, quadratic, exponential, and logistic fits
python simulation-tools/frontend_replay/replay-analysis/current-test/currnet.PY \
  simulation-tools/frontend_replay/replay.txt --compare

# Analyze count changes, peaks, and rolling trends (requires SciPy)
python simulation-tools/frontend_replay/replay-analysis/current-test/currnet.PY \
  simulation-tools/frontend_replay/replay.txt --dynamics --window 10 --iqr-factor 1.5

# Compare models and open the fit/residual plot
python simulation-tools/frontend_replay/replay-analysis/current-test/currnet.PY \
  simulation-tools/frontend_replay/replay.txt --compare --plot
```

The options `--model`, `--compare`, and `--dynamics` are mutually exclusive. `--growth` prints summary counts and exits, so do not combine it with those analysis modes. The default model is a polynomial fit; select a specific model with `--model linear`, `--model exponential`, or `--model logistic`. `--degree` controls the polynomial degree (default `2`). `--plot` is supported for model fitting/comparison and requires Matplotlib; it does not create a dynamics plot.

For count-only model analysis, the script reads vehicle counts as a stream rather than keeping all vehicle coordinates in memory. The `--growth` path parses full frame data, so it uses more memory. Model comparison reports R², AIC, and BIC; lower AIC/BIC indicate a preferred fit among the candidate models for that data, but do not establish causal explanation or policy effectiveness. SciPy is needed for exponential/logistic fitting and for the dynamics code's peak detection.

## Research background

The following papers are relevant background, not claims that the tools in this directory implement their learning algorithms:

1. *SocialLight: Distributed Cooperation Learning towards Network-Wide Traffic Signal Control.* [ACM Digital Library](https://dl.acm.org/doi/epdf/10.5555/3545946.3598809).
2. *CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario.* [ACM Digital Library](https://dl.acm.org/doi/pdf/10.1145/3308558.3314139).

For the CityFlow replay format, see the [CityFlow replay documentation](https://cityflow.readthedocs.io/en/latest/replay.html).

## Interpretation and reproducibility

Analysis results depend on the replay file, parser assumptions, package versions, and model settings. The scripts do not automatically reproduce the reported values in any paper. Keep the input replay and command-line options with experiment results, and independently validate any conclusions before reporting them.
