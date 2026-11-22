# Grid Traffic Simulation — GridSim

A pure-Python microsimulator for a rectangular grid road network. It
generates a Flow/SUMO-style roadnet, spawns traffic using a logistic
demand model, simulates vehicle movement with car-following and traffic
lights, and renders the result as a GIF, PNG, or interactive matplotlib
window.

The whole pipeline is driven by one Windows batch file, `run_case.cmd`,
so you don't need to remember any Python arguments.

---

## Contents

- [Quick start](#quick-start)
- [Files](#files)
- [Presets](#presets)
- [Options](#options)
- [Pipeline stages](#pipeline-stages)
- [Output files](#output-files)
- [The simulation model](#the-simulation-model)
- [Known issues / troubleshooting](#known-issues--troubleshooting)
- [Examples](#examples)

---

## Quick start

```bat
REM Default demo run (5000 vehicles, 4×4 grid, 300 s)
run_case.cmd

REM Fast smoke test
run_case.cmd quick

REM Heavy 12×12 turn-routed test with 1 M vehicles
run_case.cmd megacrowd

REM Reuse whatever roadnet_*.json + flow_*.json is in runs\live
run_case.cmd case

REM Show all options
run_case.cmd -h
