#!/usr/bin/env python3
"""
scenario_ramps.py
=================
Ramp-region (knee) stress scenarios + diagram analysis for
`traffic_control_algorithm.py`.

Five ramp axes are swept around their known inflection points:

  R1  LATENCY    — sweeps latency through x0 ≈ 101.54 ms  (Eq. 5)
  R2  LOSS       — sweeps Gilbert-Elliott mean loss 0 → 0.45
  R3  CAPACITY   — sweeps weather/incident capacity 1.0 → 0.35
  R4  DEMAND     — sweeps demand multiplier 0.5 → 2.2
  R5  DIRECTION  — sweeps directional bias −0.8 → +0.8

Each ramp is crossed with a representative controller from every class:

      classical  : fixed-time, adaptive
      coordinated: coordinated
      learning   : resilient-marl
      mixed      : resilient-marl+coordinated
      field      : mpc, weather-adaptive

Output:
  * ASCII knee plot per ramp (travel time vs the swept parameter)
  * Numerical inflection table (where the slope changes fastest)
  * CSV  `ramp_<axis>.csv` for external plotting
  * CSV  `ramp_summary.csv` (all ramps, all controllers)

Run:
    python scenario_ramps.py
"""

from __future__ import annotations
import csv
import math
import statistics
from collections import defaultdict

from traffic_control_algorithm import (
    TrafficControlAlgorithm,
    simulate_episode,
)


# ===========================================================================
#  CONTROLLER PROBES  — one representative per class
# ===========================================================================
CLASS_PROBES = {
    'classical':   'adaptive',
    'coordinated': 'coordinated',
    'learning':    'resilient-marl',
    'mixed':       'resilient-coordinated',   # registered by base file
    'field':       'mpc',
    'field-wx':    'weather-adaptive',
}
PROBE_ORDER = ['classical', 'coordinated', 'learning', 'mixed',
               'field', 'field-wx']


# ===========================================================================
#  RAMP DEFINITIONS
# ===========================================================================
# Each ramp sweeps ONE parameter; everything else is fixed to a baseline.
# The extra key 'sweep' contains the ordered grid; 'baseline' is the fixed
# context used when this ramp isn't the one being swept.

RAMPS = {
    # ------------------------------------------------------------------
    # R1 — LATENCY   (sweeps the logistic knee x0 ≈ 101.54 ms)
    # ------------------------------------------------------------------
    'R1-latency': {
        'swept':      'latency_ms',
        'unit':       'ms',
        'grid':       [0, 25, 50, 75, 100, 101.54, 125, 150, 200, 300, 450, 600],
        'knee_hint':  101.54,
        'context':    {'scenario':'S2', 'natural':'N-clear',
                       'base_demand_ns':0.30, 'base_demand_ew':0.30},
    },
    # ------------------------------------------------------------------
    # R2 — LOSS RATE  (0 → 0.45, sweeps through 0.05/0.15/0.30 of S2-S4)
    # ------------------------------------------------------------------
    'R2-loss': {
        'swept':      'loss_rate',
        'unit':       'ρ_loss',
        'grid':       [0.00, 0.02, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        'knee_hint':  0.15,
        'context':    {'scenario':'S2', 'natural':'N-clear',
                       'base_demand_ns':0.30, 'base_demand_ew':0.30},
    },
    # ------------------------------------------------------------------
    # R3 — CAPACITY  (weather / incident severity 1.0 → 0.35)
    # ------------------------------------------------------------------
    'R3-capacity': {
        'swept':      'capacity',
        'unit':       'C',
        'grid':       [1.00, 0.90, 0.85, 0.75, 0.70, 0.65, 0.55, 0.45, 0.35],
        'knee_hint':  0.65,
        'context':    {'scenario':'S2', 'natural':'N-clear',
                       'base_demand_ns':0.30, 'base_demand_ew':0.30},
    },
    # ------------------------------------------------------------------
    # R4 — DEMAND MULTIPLIER  (0.5 → 2.2, sweeps through 1.0)
    # ------------------------------------------------------------------
    'R4-demand': {
        'swept':      'demand_mult',
        'unit':       '×',
        'grid':       [0.50, 0.70, 0.85, 1.00, 1.15, 1.35, 1.50, 1.65, 1.80, 2.00, 2.20],
        'knee_hint':  1.15,
        'context':    {'scenario':'S2', 'natural':'N-clear',
                       'base_demand_ns':0.30, 'base_demand_ew':0.30},
    },
    # ------------------------------------------------------------------
    # R5 — DIRECTIONAL BIAS  (−0.8 → +0.8, sweeps through 0)
    # ------------------------------------------------------------------
    'R5-direction': {
        'swept':      'direction_bias',
        'unit':       'bias',
        'grid':       [-0.80, -0.60, -0.40, -0.20, 0.00, 0.20, 0.40, 0.60, 0.80],
        'knee_hint':  0.40,
        'context':    {'scenario':'S2', 'natural':'N-clear',
                       'base_demand_ns':0.30, 'base_demand_ew':0.30},
    },
}


# ===========================================================================
#  RAMP CELL RUNNER  — one point on a ramp, multi-seed
# ===========================================================================
def _patched_run(controller_name, context, patch, seed, n_steps):
    """
    Run one cell of a ramp.  `patch` overrides:
       * communication latency / loss  → via constructing the comm scenario
       * natural capacity / demand_mult / direction_bias  → via the natural
    We register a *temporary* natural condition per (ramp, point) so the
    base class picks up the override transparently.
    """
    from traffic_control_algorithm import (
        COMMUNICATION_SCENARIOS, NATURAL_SCENARIOS)

    # ------- comm override ------------------------------------------------
    if 'latency_ms' in patch or 'loss_rate' in patch:
        tmp_scen = f"_RAMP_S_{seed}_{patch.get('latency_ms','x')}_"\
                   f"{patch.get('loss_rate','x')}"
        COMMUNICATION_SCENARIOS[tmp_scen] = {
            'latency_ms': float(patch.get('latency_ms', 0.0)),
            'loss_rate':  float(patch.get('loss_rate',  0.0)),
            'label':      'ramp-temp',
        }
        scenario = tmp_scen
    else:
        scenario = context['scenario']

    # ------- natural override --------------------------------------------
    tmp_nat = f"_RAMP_N_{patch.get('capacity', 1.0)}_"\
              f"{patch.get('demand_mult', 1.0)}_"\
              f"{patch.get('direction_bias', 0.0)}"
    base_nat = dict(NATURAL_SCENARIOS[context['natural']])
    base_nat['capacity']          = float(patch.get('capacity',       1.0))
    base_nat['demand_mult']       = float(patch.get('demand_mult',    1.0))
    base_nat['direction_bias']    = float(patch.get('direction_bias', 0.0))
    base_nat['sensor_reliability'] = 1.0
    NATURAL_SCENARIOS[tmp_nat] = base_nat

    c = TrafficControlAlgorithm(controller_name, scenario, tmp_nat,
                                seed=seed, car_following='SAFE')
    simulate_episode(c, n_steps=n_steps,
                     base_demand_ns=context['base_demand_ns'],
                     base_demand_ew=context['base_demand_ew'],
                     seed=seed)
    return c.get_metrics()


def run_ramp_point(controller_name, context, patch,
                   seeds=(1, 2, 3, 4, 5), n_steps=100) -> dict:
    """Mean ± CI over seeds for one point of a ramp."""
    keys = ('travel_time_s', 'queue_veh', 'waiting_time_s',
            'throughput_vph', 'severity_index')
    per_seed = defaultdict(list)
    for s in seeds:
        m = _patched_run(controller_name, context, patch, s, n_steps)
        for k in keys:
            per_seed[k].append(float(m[k]))

    out = {'controller': controller_name}
    out.update(patch)
    for k, xs in per_seed.items():
        mean = statistics.fmean(xs)
        sd   = statistics.stdev(xs) if len(xs) > 1 else 0.0
        out[f'{k}_mean'] = round(mean, 3)
        out[f'{k}_sd']   = round(sd, 3)
        out[f'{k}_ci95'] = round(1.96 * sd / math.sqrt(len(xs)), 3)
    return out


def run_ramp(ramp_name, controllers=None,
             seeds=(1, 2, 3, 4, 5), n_steps=100,
             progress=True) -> list[dict]:
    """Run one ramp for every controller probe."""
    cfg = RAMPS[ramp_name]
    controllers = controllers or [CLASS_PROBES[k] for k in PROBE_ORDER]
    rows = []
    total = len(controllers) * len(cfg['grid'])
    i = 0
    for cname in controllers:
        for v in cfg['grid']:
            i += 1
            if progress and (i % 5 == 0 or i == total):
                print(f"\r    {ramp_name}: {i}/{total} ...",
                      end='', flush=True)
            patch = {cfg['swept']: v}
            try:
                rows.append(run_ramp_point(cname, cfg['context'], patch,
                                           seeds=seeds, n_steps=n_steps))
            except Exception as e:
                if progress:
                    print(f"\n    [warn] {cname}@{v}: {e}")
    if progress:
        print()
    return rows


# ===========================================================================
#  KNEE DETECTION  —  2nd-order difference finds the slope change
# ===========================================================================
def find_knee(grid, ys):
    """
    Return the grid index where the second derivative of the y-series is
    largest in magnitude — the empirical knee.
    """
    if len(ys) < 3:
        return None
    # y is travel_time (or similar); compute discrete 2nd derivative
    d2 = []
    for i in range(1, len(ys) - 1):
        y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
        x0, x1, x2 = grid[i - 1], grid[i], grid[i + 1]
        # uneven grid → use slopes
        s01 = (y1 - y0) / max(x1 - x0, 1e-9)
        s12 = (y2 - y1) / max(x2 - x1, 1e-9)
        d2.append((grid[i], abs(s12 - s01)))
    return max(d2, key=lambda t: t[1])


# ===========================================================================
#  ASCII DIAGRAM  —  travel time vs swept parameter
# ===========================================================================
def ascii_plot(grid, series_dict, ramp_name, unit,
               knee_hint=None, width=64, height=12):
    """Multi-series ASCII line plot: one row per controller."""
    all_y = [y for ys in series_dict.values() for y in ys]
    ymin, ymax = min(all_y), max(all_y)
    if ymax - ymin < 1e-6:
        ymax = ymin + 1.0

    # -- header -----------------------------------------------------------------
    print(f"\n  {ramp_name}  ({unit})")
    bar = "─" * width
    print(f"  {bar}")
    print(f"  {ymin:>6.1f} ┌" + " " * (width - 8) + f"┐ {ymax:>6.1f}")
    # -- body -------------------------------------------------------------------
    # markers for controllers
    markers = dict(zip(series_dict.keys(), "☺☻◐◑○●◇◆▲▼■□"))

    # build grid of rows: y bucket for each point
    canvas = [[' '] * width for _ in range(height)]
    for name, ys in series_dict.items():
        for i, y in enumerate(ys):
            col = int(round(i / (len(grid) - 1) * (width - 1)))
            row = int(round((1 - (y - ymin) / (ymax - ymin)) * (height - 1)))
            row = max(0, min(height - 1, row))
            canvas[row][col] = markers.get(name, '*')

    for r, line in enumerate(canvas):
        y_val = ymax - (r / (height - 1)) * (ymax - ymin)
        print(f"  {y_val:>6.1f} │" + ''.join(line) + "│")
    print(f"  {'':>6} └{bar[:-2]}┘")

    # -- x-axis ticks -----------------------------------------------------------
    ticks = []
    for v in grid:
        col = int(round(grid.index(v) / (len(grid) - 1) * (width - 1)))
        ticks.append((col, f"{v}"))
    # keep only first, last, hint
    shown = [ticks[0], ticks[-1]]
    if knee_hint is not None:
        try:
            idx = min(range(len(grid)), key=lambda i: abs(grid[i] - knee_hint))
            shown.append(ticks[idx])
        except ValueError:
            pass
    shown.sort()
    line = [' '] * (width + 8)
    for col, lbl in shown:
        for i, ch in enumerate(lbl):
            if col + 7 + i < len(line):
                line[col + 7 + i] = ch
    print(f"  {'':>7}" + ''.join(line))
    # vertical marker at hint
    if knee_hint is not None:
        try:
            idx = min(range(len(grid)), key=lambda i: abs(grid[i] - knee_hint))
            col = int(round(idx / (len(grid) - 1) * (width - 1))) + 8
            marker = [' '] * (width + 8)
            marker[col] = '┊'
            print(f"  {'':>7}" + ''.join(marker) +
                  f"  knee ≈ {grid[idx]}{unit}")
        except ValueError:
            pass

    # -- legend -----------------------------------------------------------------
    print(f"\n  legend:")
    for name in series_dict:
        print(f"    {markers.get(name, '*')}  {name}")
    print()


# ===========================================================================
#  ANALYTICAL SUMMARY
# ===========================================================================
def summarise_ramp(ramp_name, rows):
    cfg = RAMPS[ramp_name]
    swept = cfg['swept']
    grid  = cfg['grid']

    print("\n" + "=" * 100)
    print(f"  RAMP {ramp_name}  —  sweeping {swept}  "
          f"({len(grid)} points × {len(CLASS_PROBES)} controllers)")
    print("=" * 100)

    # per-controller series ------------------------------------------------
    series: dict[str, dict[str, list[float]]] = defaultdict(lambda:
                                                            defaultdict(list))
    for r in rows:
        cname = r['controller']
        series[cname]['x'].append(r[swept])
        series[cname]['tt'].append(r['travel_time_s_mean'])
        series[cname]['thr'].append(r['throughput_vph_mean'])
        series[cname]['ci'].append(r['travel_time_s_ci95'])

    # -- table -------------------------------------------------------------
    probe_cols = [CLASS_PROBES[k] for k in PROBE_ORDER]
    hdr = f"{swept:>8}" + ''.join(f"{c[:14]:>16}" for c in probe_cols)
    print(hdr); print('-' * len(hdr))
    for i, v in enumerate(grid):
        line = f"{v:>8}"
        for c in probe_cols:
            line += f"{series[c]['tt'][i]:>16.1f}"
        print(line)
    print()

    # -- knee detection ----------------------------------------------------
    print(f"  Empirical knee per controller "
          f"(peak |Δ²travel| / Δ{swept}²):")
    hdr = f"{'controller':<24}{'knee x':>10}{'knee y (s)':>14}" \
          f"{'Δtravel to hint':>18}"
    print(hdr); print('-' * len(hdr))
    for c in probe_cols:
        xs, ys = series[c]['x'], series[c]['tt']
        k = find_knee(xs, ys)
        if k is None:
            continue
        knee_x, _curv = k
        knee_y = ys[xs.index(knee_x)]
        # change relative to hint
        hint = cfg['knee_hint']
        hint_idx = min(range(len(xs)), key=lambda i: abs(xs[i] - hint))
        delta = ys[-1] - ys[hint_idx]
        print(f"{c:<24}{knee_x:>10.2f}{knee_y:>14.2f}{delta:>18.2f}")
    print()

    # -- ascii diagram -----------------------------------------------------
    ascii_plot(grid,
               {c: series[c]['tt'] for c in probe_cols},
               ramp_name, f" {swept}", cfg['knee_hint'])


def best_controller_per_point(rows, ramp_name):
    cfg = RAMPS[ramp_name]
    swept = cfg['swept']
    grid  = cfg['grid']
    print("\n" + "=" * 100)
    print(f"  BEST CONTROLLER AT EACH POINT OF {ramp_name}")
    print("=" * 100)
    hdr = f"{swept:>8}{'winner':<30}{'travel(s)':>12}" \
          f"{'vs. worst':>12}{'gap':>10}"
    print(hdr); print('-' * len(hdr))
    for v in grid:
        cands = [r for r in rows if abs(r[swept] - v) < 1e-6]
        if not cands:
            continue
        best  = min(cands, key=lambda r: r['travel_time_s_mean'])
        worst = max(cands, key=lambda r: r['travel_time_s_mean'])
        gap   = worst['travel_time_s_mean'] - best['travel_time_s_mean']
        print(f"{v:>8.2f}{best['controller']:<30}"
              f"{best['travel_time_s_mean']:>12.2f}"
              f"{worst['travel_time_s_mean']:>12.2f}"
              f"{gap:>10.2f}")
    print()


# ===========================================================================
#  CSV EXPORT
# ===========================================================================
def export_ramp_csv(ramp_name, rows, path=None):
    path = path or f"ramp_{ramp_name}.csv"
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {len(rows)} rows → {path}")


def export_summary_csv(all_rows, path="ramp_summary.csv"):
    flat = []
    for ramp, rows in all_rows.items():
        swept = RAMPS[ramp]['swept']
        for r in rows:
            r2 = dict(r)
            r2['ramp']  = ramp
            r2['swept'] = swept
            flat.append(r2)
    if not flat:
        return
    fieldnames = sorted({k for r in flat for k in r})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in flat:
            w.writerow(r)
    print(f"  wrote {len(flat)} rows → {path}")


# ===========================================================================
#  HIGH-LEVEL DRIVER
# ===========================================================================
def run_all_ramps(seeds=(1, 2, 3, 4, 5), n_steps=100):
    all_rows: dict[str, list[dict]] = {}
    for ramp_name in RAMPS:
        print(f"\n[ramp] {ramp_name}")
        all_rows[ramp_name] = run_ramp(ramp_name, seeds=seeds,
                                       n_steps=n_steps)
    return all_rows


def main():
    print("#" * 100)
    print("#  RAMP-REGION ANALYSIS  —  knee sweep + diagram synthesis")
    print("#" * 100)
    print(f"\n  Controllers  : {len(CLASS_PROBES)} probes "
          f"({', '.join(PROBE_ORDER)})")
    print(f"  Ramps        : {len(RAMPS)}  "
          f"({', '.join(RAMPS.keys())})")

    SEEDS   = (1, 2, 3, 4, 5)
    N_STEPS = 100

    all_rows = run_all_ramps(seeds=SEEDS, n_steps=N_STEPS)

    # ---- per-ramp reports ------------------------------------------------
    for ramp_name, rows in all_rows.items():
        summarise_ramp(ramp_name, rows)
        best_controller_per_point(rows, ramp_name)
        export_ramp_csv(ramp_name, rows)

    # ---- global summary --------------------------------------------------
    export_summary_csv(all_rows)

    # ---- knee comparison across ramps ------------------------------------
    print("=" * 100)
    print("  GLOBAL KNEE SUMMARY  —  resilient-marl probe")
    print("=" * 100)
    probe = CLASS_PROBES['learning']              # 'resilient-marl'
    hdr = f"{'ramp':<16}{'parameter':<14}{'knee at':>12}" \
          f"{'travel@knee':>14}{'travel@end':>14}"
    print(hdr); print('-' * len(hdr))
    for ramp_name, rows in all_rows.items():
        cfg   = RAMPS[ramp_name]
        swept = cfg['swept']
        sub   = [r for r in rows if r['controller'] == probe]
        if not sub:
            continue
        sub.sort(key=lambda r: r[swept])
        xs = [r[swept] for r in sub]
        ys = [r['travel_time_s_mean'] for r in sub]
        k  = find_knee(xs, ys)
        if k is None:
            continue
        knee_x, _ = k
        y_at = ys[xs.index(knee_x)]
        print(f"{ramp_name:<16}{swept:<14}"
              f"{knee_x:>12.2f}{y_at:>14.2f}{ys[-1]:>14.2f}")
    print()

    print("Done.")


if __name__ == '__main__':
    main()