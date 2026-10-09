#!/usr/bin/env python3
"""
scenario_all.py
===============
Every possible scenario — car-following × controller × communication ×
natural × demand — with 5-seed multi-evaluation (mean ± 95 % CI).

Focus:   SAFE / GIPPS / IDM car-following × non-MARL + adaptive controllers

Axes
----
  CAR      : SAFE, GIPPS, IDM                       (3)
  CTRL     : classical, coordinated, field, non-marl compositions
             (adaptive, fixed-time, actuated, optimized, coordinated,
              max-pressure, scats, scoot, multiband, fuzzy-logic, mpc,
              weather-adaptive, adaptive+coordinated,
              adaptive+weather-adaptive, optimized+scats, mpc+multiband,
              max-pressure+scats, fuzzy-logic+mpc, fuzzy-logic+coordinated,
              scoot+max-pressure, multiband+weather-adaptive)   (20)
  COMM     : S1, S2, S3, S4                         (4)
  NATURAL  : 19 conditions                          (19)
  DEMAND   : D1..D7                                 (7)

Total cells = 3 × 20 × 4 × 19 × 7 = 31 920
× 5 seeds  = 159 600 simulator runs

CLI:
    python scenario_all.py --quick        # 3 car-models × 6 ctrl × S1-S2 × 5 nat × D1  → fast
    python scenario_all.py --n-marl       # non-MARL focused (7 ctrl × all axes)
    python scenario_all.py --cf-full      # full car-following sweep (3 ctrl × all axes)
    python scenario_all.py --full         # everything (slow!)
    python scenario_all.py --export       # write CSV in all modes

Programmatic:
    from scenario_all import run_all_scenarios
    rows = run_all_scenarios()
"""

from __future__ import annotations
import argparse
import csv
import math
import statistics
from collections import defaultdict
from itertools import product

from traffic_control_algorithm import (
    TrafficControlAlgorithm,
    COMMUNICATION_SCENARIOS,
    NATURAL_SCENARIOS,
    simulate_episode,
)

# Pull in the scenario-matrix extras so we have the full natural set
from scenario_matrix import (
    register_mixed_controllers,
    register_extra_naturals,
    NATURAL_GROUPS            as BASE_GROUPS,
    ALL_NATURALS              as BASE_NATURALS,
    DEMAND_PATTERNS           as BASE_DEMANDS,
)


# ===========================================================================
#  AXIS: CAR-FOLLOWING MODELS
# ===========================================================================
CAR_MODELS = ('SAFE', 'GIPPS', 'IDM')


# ===========================================================================
#  AXIS: NON-MARL + ADAPTIVE CONTROLLERS
# ===========================================================================
# All non-MARL classical / coordinated / field, plus the mixed ones whose
# base components are non-MARL.  No learning controllers here — those live
# in scenario_matrix.py.
NON_MARL_CONTROLLERS = [
    # --- classical (non-MARL) ---------------------------------------
    'fixed-time',
    'actuated',
    'adaptive',
    'optimized',
    # --- coordinated ------------------------------------------------
    'coordinated',
    # --- field-inspired ---------------------------------------------
    'max-pressure',
    'scats',
    'scoot',
    'multiband',
    'fuzzy-logic',
    'mpc',
    'weather-adaptive',
    # --- non-MARL compositions (mixed) ------------------------------
    'adaptive+coordinated',
    'adaptive+weather-adaptive',
    'optimized+scats',
    'mpc+multiband',
    'max-pressure+scats',
    'fuzzy-logic+mpc',
    'fuzzy-logic+coordinated',
    'scoot+max-pressure',
    'multiband+weather-adaptive',
]

# The extra pairs we need to register before using them
EXTRA_NON_MARL_PAIRS = [
    ('adaptive',     'coordinated',      0.5),
    ('adaptive',     'weather-adaptive', 0.5),
    ('optimized',    'scats',            0.5),
    ('mpc',          'multiband',        0.5),
    ('max-pressure', 'scats',            0.5),
    ('fuzzy-logic',  'mpc',              0.5),
    ('fuzzy-logic',  'coordinated',      0.5),
    ('scoot',        'max-pressure',     0.5),
    ('multiband',    'weather-adaptive', 0.5),
]


# ===========================================================================
#  REGISTRATION
# ===========================================================================
def register_all_extras() -> list[str]:
    """Register every controller / natural needed by this driver."""
    registered: list[str] = []

    # all scenario_matrix mixed pairs / triplets (superset — harmless)
    registered += register_mixed_controllers()
    register_extra_naturals()

    # the extra non-MARL pairs used here
    for a, b, w in EXTRA_NON_MARL_PAIRS:
        name = f"{a}+{b}"
        if name in TrafficControlAlgorithm.CONTROLLERS:
            continue
        try:
            TrafficControlAlgorithm.compose(a, b, w, new_name=name)
            registered.append(name)
        except KeyError as e:
            print(f"  [skip] {name}: {e}")

    return registered


# ===========================================================================
#  AXES: COMM, NATURAL, DEMAND
# ===========================================================================
COMM_SCENARIOS = list(COMMUNICATION_SCENARIOS)
NATURALS       = BASE_NATURALS + [
    n for n in NATURAL_SCENARIOS if n not in BASE_NATURALS
]
DEMANDS        = list(BASE_DEMANDS)


# ===========================================================================
#  TRACKED METRICS
# ===========================================================================
TRACKED = ('travel_time_s', 'queue_veh', 'waiting_time_s',
           'throughput_vph', 'mean_aoi_ms', 'delivery_ratio')


# ===========================================================================
#  ONE-SEED RUNNER
# ===========================================================================
def run_one_seed(controller: str, car_model: str, scenario: str,
                 natural: str, demand: str,
                 seed: int, n_steps: int) -> dict:
    d = BASE_DEMANDS[demand]
    ctrl = TrafficControlAlgorithm(
        algorithm_type = controller,
        scenario       = scenario,
        natural        = natural,
        seed           = seed,
        car_following  = car_model,
    )
    simulate_episode(
        ctrl,
        n_steps        = n_steps,
        base_demand_ns = d['ns'],
        base_demand_ew = d['ew'],
        seed           = seed,
    )
    return ctrl.get_metrics()


# ===========================================================================
#  MULTI-SEED CELL RUNNER
# ===========================================================================
def run_cell(controller: str, car_model: str, scenario: str,
             natural: str, demand: str,
             seeds=(1, 2, 3, 4, 5), n_steps=100) -> dict:
    per_seed: dict[str, list[float]] = defaultdict(list)
    for s in seeds:
        m = run_one_seed(controller, car_model, scenario, natural,
                         demand, s, n_steps)
        for k in TRACKED:
            per_seed[k].append(float(m[k]))

    out = {'controller': controller, 'car_following': car_model,
           'scenario': scenario, 'natural': natural, 'demand': demand,
           'n_seeds': len(seeds)}
    for k, xs in per_seed.items():
        mean = statistics.fmean(xs)
        if len(xs) > 1:
            sd   = statistics.stdev(xs)
            ci   = 1.96 * sd / math.sqrt(len(xs))
        else:
            sd = ci = 0.0
        out[f'{k}_mean']   = round(mean, 3)
        out[f'{k}_sd']     = round(sd,   3)
        out[f'{k}_ci95']   = round(ci,   3)
        out[f'{k}_values'] = [round(x, 3) for x in xs]
    return out


# ===========================================================================
#  FULL MATRIX DRIVER
# ===========================================================================
def run_all_scenarios(controllers=None, car_models=None,
                      scenarios=None, naturals=None, demands=None,
                      seeds=(1, 2, 3, 4, 5), n_steps=100,
                      progress=True) -> list[dict]:
    controllers = controllers or NON_MARL_CONTROLLERS
    car_models  = car_models  or CAR_MODELS
    scenarios   = scenarios   or COMM_SCENARIOS
    naturals    = naturals    or NATURALS
    demands     = demands     or DEMANDS

    grid = list(product(controllers, car_models, scenarios,
                        naturals, demands))
    total = len(grid)
    rows: list[dict] = []
    for i, (c, cf, sc, nat, dem) in enumerate(grid, 1):
        if progress and (i % 20 == 0 or i == total):
            print(f"\r  cell {i}/{total}  "
                  f"({len(seeds)} seeds each) ...", end='', flush=True)
        try:
            rows.append(run_cell(c, cf, sc, nat, dem,
                                 seeds=seeds, n_steps=n_steps))
        except Exception as e:
            if progress:
                print(f"\n  [warn] {c}/{cf}/{sc}/{nat}/{dem}: {e}")
    if progress:
        print()
    return rows


# ===========================================================================
#  REPORTS
# ===========================================================================
def _short_nat(n: str) -> str:
    return {'N-clear': 'Clear', 'N-rain': 'Rain', 'N-snow': 'Snow',
            'N-fog': 'Fog', 'N-hailstorm': 'Hail', 'N-flooding': 'Flood',
            'N-accident': 'Accid', 'N-workzone': 'Work',
            'N-major-crash': 'Crash',
            'N-rush-am': 'AMpk', 'N-rush-pm': 'PMpk',
            'N-night': 'Night', 'N-event': 'Event', 'N-school': 'Schl',
            'N-holiday': 'Holi', 'N-freight': 'Frt',
            'N-sensor-fail': 'NoDet', 'N-comm-blackout': 'NoV2X',
            'N-power-outage': 'NoPwr'}.get(n, n)


def _short_ctrl(c: str, w: int = 28) -> str:
    return c if len(c) <= w else c[:w - 1] + '…'


def print_cf_x_ctrl(rows, controllers, car_models,
                    scenario='S1', natural='N-clear', demand='D1',
                    metric='travel_time_s'):
    """Car-following × controller matrix (mean)."""
    print("\n" + "=" * 110)
    print(f"  CAR-FOLLOWING × CONTROLLER  —  {metric} (mean)  "
          f"[COMM={scenario}, NAT={natural}, DEM={demand}]")
    print("=" * 110)
    idx = {(r['car_following'], r['controller']): r for r in rows
           if r['scenario'] == scenario and r['natural'] == natural
           and r['demand'] == demand}
    hdr = f"{'Controller':<30}" + ''.join(f"{cf:>14}" for cf in car_models)
    print(hdr); print('-' * len(hdr))
    for c in controllers:
        line = f"{_short_ctrl(c, 30):<30}"
        for cf in car_models:
            r = idx.get((cf, c))
            line += f"{r[f'{metric}_mean']:>14.2f}" if r else f"{'—':>14}"
        print(line)
    print()


def print_cf_x_ctrl_ci(rows, controllers, car_models,
                       scenario='S1', natural='N-clear', demand='D1',
                       metric='travel_time_s'):
    """Car-following × controller — mean ± 95 % CI."""
    print("\n" + "=" * 110)
    print(f"  CAR-FOLLOWING × CONTROLLER  —  {metric} (mean ± 95 % CI)  "
          f"[COMM={scenario}, NAT={natural}, DEM={demand}]")
    print("=" * 110)
    idx = {(r['car_following'], r['controller']): r for r in rows
           if r['scenario'] == scenario and r['natural'] == natural
           and r['demand'] == demand}
    hdr = f"{'Controller':<30}" + ''.join(f"{cf:>22}" for cf in car_models)
    print(hdr); print('-' * len(hdr))
    for c in controllers:
        line = f"{_short_ctrl(c, 30):<30}"
        for cf in car_models:
            r = idx.get((cf, c))
            if r:
                line += (f"{r[f'{metric}_mean']:>12.2f} "
                         f"±{r[f'{metric}_ci95']:>6.2f}")
            else:
                line += f"{'—':>22}"
        print(line)
    print()


def print_comm_ramp(rows, controllers, car_models,
                    natural='N-clear', demand='D1',
                    metric='travel_time_s'):
    """Degradation across S1→S4 for each (controller, car-model)."""
    print("\n" + "=" * 110)
    print(f"  S1 → S4 DEGRADATION  —  {metric}  "
          f"[NAT={natural}, DEM={demand}]")
    print("=" * 110)
    scenarios = COMM_SCENARIOS
    hdr = (f"{'Controller':<28}{'CF':<6}" +
           ''.join(f"{s:>12}" for s in scenarios) + f"{'Δ%':>10}")
    print(hdr); print('-' * len(hdr))
    for c in controllers:
        for cf in car_models:
            line = f"{_short_ctrl(c, 28):<28}{cf:<6}"
            vals = []
            for s in scenarios:
                r = next((r for r in rows
                          if r['controller'] == c
                          and r['car_following'] == cf
                          and r['scenario'] == s
                          and r['natural'] == natural
                          and r['demand'] == demand), None)
                if r:
                    v = r[f'{metric}_mean']
                    vals.append(v)
                    line += f"{v:>12.2f}"
                else:
                    vals.append(None)
                    line += f"{'—':>12}"
            if vals[0] and vals[-1]:
                pct = (vals[-1] - vals[0]) / vals[0] * 100.0
                line += f"{pct:>9.1f}%"
            print(line)
    print()


def print_natural_sweep(rows, controller, car_model,
                        scenario='S2', demand='D1',
                        metric='travel_time_s'):
    print("\n" + "=" * 110)
    print(f"  NATURAL SWEEP  —  {metric}  "
          f"[CTRL={controller}, CF={car_model}, COMM={scenario}, DEM={demand}]")
    print("=" * 110)
    hdr = (f"{'Natural':<18}{'Family':<10}{'Mean':>10}{'±CI95':>9}"
           f"{'Thru(vph)':>12}{'Queue':>9}")
    print(hdr); print('-' * len(hdr))
    for n in NATURALS:
        family = next((f for f, mem in BASE_GROUPS.items() if n in mem),
                      'extra')
        r = next((r for r in rows
                  if r['controller'] == controller
                  and r['car_following'] == car_model
                  and r['scenario'] == scenario
                  and r['natural'] == n
                  and r['demand'] == demand), None)
        if not r:
            continue
        print(f"{_short_nat(n):<18}{family:<10}"
              f"{r[f'{metric}_mean']:>10.2f}"
              f"{r[f'{metric}_ci95']:>9.2f}"
              f"{r['throughput_vph_mean']:>12.1f}"
              f"{r['queue_veh_mean']:>9.2f}")
    print()


def print_best_per_natural(rows, controllers, car_model,
                           scenario='S2', demand='D1',
                           metric='travel_time_s'):
    print("\n" + "=" * 110)
    print(f"  BEST CONTROLLER PER NATURAL CONDITION  —  {metric}  "
          f"[CF={car_model}, COMM={scenario}, DEM={demand}]")
    print("=" * 110)
    hdr = (f"{'Natural':<18}{'Winner':<32}{'Mean':>10}{'±CI95':>9}"
           f"{'Runner-up':<32}{'Mean':>10}")
    print(hdr); print('-' * len(hdr))
    for n in NATURALS:
        cands = [r for r in rows
                 if r['natural'] == n
                 and r['car_following'] == car_model
                 and r['scenario'] == scenario
                 and r['demand'] == demand]
        if not cands:
            continue
        cands.sort(key=lambda r: r[f'{metric}_mean'])
        b, ru = cands[0], cands[1] if len(cands) > 1 else cands[0]
        print(f"{_short_nat(n):<18}"
              f"{_short_ctrl(b['controller'], 32):<32}"
              f"{b[f'{metric}_mean']:>10.2f}"
              f"{b[f'{metric}_ci95']:>9.2f}"
              f"{_short_ctrl(ru['controller'], 32):<32}"
              f"{ru[f'{metric}_mean']:>10.2f}")
    print()


def print_demand_sweep(rows, controllers, car_model,
                       scenario='S2', natural='N-clear',
                       metric='travel_time_s'):
    print("\n" + "=" * 110)
    print(f"  DEMAND SWEEP  —  {metric} (mean)  "
          f"[CF={car_model}, COMM={scenario}, NAT={natural}]")
    print("=" * 110)
    demands = DEMANDS
    hdr = f"{'Controller':<30}" + ''.join(f"{d:>9}" for d in demands)
    print(hdr); print('-' * len(hdr))
    for c in controllers:
        line = f"{_short_ctrl(c, 30):<30}"
        for d in demands:
            r = next((r for r in rows
                      if r['controller'] == c
                      and r['car_following'] == car_model
                      and r['scenario'] == scenario
                      and r['natural'] == natural
                      and r['demand'] == d), None)
            line += f"{r[f'{metric}_mean']:>9.2f}" if r else f"{'—':>9}"
        print(line)
    print()


# ===========================================================================
#  CSV EXPORT
# ===========================================================================
def export_csv(rows, path='scenario_all.csv'):
    if not rows:
        return
    cols = ['controller', 'car_following', 'scenario', 'natural', 'demand',
            'n_seeds'] + [
        f'{k}_{s}'
        for k in TRACKED
        for s in ('mean', 'sd', 'ci95')
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {len(rows)} rows → {path}")


# ===========================================================================
#  CLI
# ===========================================================================
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',   action='store_true',
                   help="small slice: 3 CF × 6 ctrl × S1,S2 × 5 nat × D1")
    p.add_argument('--n-marl',  action='store_true',
                   help="non-MARL focus: 20 ctrl × 3 CF × S1-S4 × 5 nat × D1")
    p.add_argument('--cf-full', action='store_true',
                   help="full car-following sweep: 3 CF × 20 ctrl × "
                        "S1-S4 × 5 nat × D1,D2,D5")
    p.add_argument('--full',    action='store_true',
                   help="everything (slow: 31 920 cells × 5 seeds)")
    p.add_argument('--seeds', type=int, default=5,
                   help="number of seeds (default 5)")
    p.add_argument('--steps', type=int, default=100,
                   help="decision steps per episode (default 100)")
    p.add_argument('--export', action='store_true',
                   help="write scenario_all.csv")
    return p.parse_args()


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    args = _parse_args()

    print("#" * 110)
    print("#  EVERY SCENARIO  —  car-following × controller × comm × natural × demand")
    print("#" * 110)

    # ---- register ------------------------------------------------------
    print("\n[1] Registering controllers + natural conditions ...")
    reg = register_all_extras()
    print(f"    registered {len(reg)} composed controllers")
    print(f"    total controllers        : "
          f"{len(TrafficControlAlgorithm.CONTROLLERS)}")
    print(f"    total natural conditions : {len(NATURAL_SCENARIOS)}")
    print(f"    total demand patterns    : {len(BASE_DEMANDS)}")

    # ---- choose slice --------------------------------------------------
    if args.quick:
        controllers = NON_MARL_CONTROLLERS[:6]
        car_models  = ('SAFE', 'GIPPS', 'IDM')
        scenarios   = ('S1', 'S2')
        naturals    = ['N-clear', 'N-rain', 'N-snow',
                       'N-accident', 'N-rush-am']
        demands     = ['D1']
    elif args.n_marl:
        controllers = NON_MARL_CONTROLLERS
        car_models  = ('SAFE', 'GIPPS', 'IDM')
        scenarios   = COMM_SCENARIOS
        naturals    = ['N-clear', 'N-rain', 'N-snow',
                       'N-accident', 'N-rush-am']
        demands     = ['D1']
    elif args.cf_full:
        controllers = NON_MARL_CONTROLLERS
        car_models  = ('SAFE', 'GIPPS', 'IDM')
        scenarios   = COMM_SCENARIOS
        naturals    = ['N-clear', 'N-rain', 'N-snow',
                       'N-accident', 'N-rush-am']
        demands     = ['D1', 'D2', 'D5']
    elif args.full:
        controllers = NON_MARL_CONTROLLERS
        car_models  = ('SAFE', 'GIPPS', 'IDM')
        scenarios   = COMM_SCENARIOS
        naturals    = NATURALS
        demands     = DEMANDS
    else:
        # default: non-MARL quick slice
        controllers = NON_MARL_CONTROLLERS[:6]
        car_models  = ('SAFE', 'GIPPS', 'IDM')
        scenarios   = ('S1', 'S2')
        naturals    = ['N-clear', 'N-rain', 'N-snow',
                       'N-accident', 'N-rush-am']
        demands     = ['D1']

    seeds = tuple(range(1, args.seeds + 1))

    total = (len(controllers) * len(car_models) * len(scenarios)
             * len(naturals) * len(demands))
    print(f"\n[2] Slice selected:")
    print(f"    car-following  : {car_models}")
    print(f"    controllers    : {len(controllers)}"
          f"  ({controllers[0]} … {controllers[-1]})")
    print(f"    comm scenarios : {scenarios}")
    print(f"    naturals       : {naturals}")
    print(f"    demands        : {demands}")
    print(f"    seeds          : {seeds}  ({len(seeds)} per cell)")
    print(f"    → total cells  : {total}   "
          f"({total * len(seeds)} simulator runs)")

    # ---- run -----------------------------------------------------------
    print("\n[3] Running matrix ...")
    rows = run_all_scenarios(
        controllers = controllers,
        car_models  = car_models,
        scenarios   = scenarios,
        naturals    = naturals,
        demands     = demands,
        seeds       = seeds,
        n_steps     = args.steps,
        progress    = True,
    )

    # ---- reports -------------------------------------------------------
    print("\n[4] Reports ...")

    # 4a. car-following × controller (headline: S1, N-clear, D1)
    print_cf_x_ctrl(rows, controllers, car_models,
                    scenario='S1', natural='N-clear', demand='D1')
    print_cf_x_ctrl_ci(rows, controllers, car_models,
                       scenario='S2', natural='N-clear', demand='D1')

    # 4b. S1 → S4 degradation per (controller, car-model)
    print_comm_ramp(rows, controllers, car_models,
                    natural='N-clear', demand='D1')

    # 4c. natural sweep for adaptive/SAFE and GIPPS/adaptive
    for cf in car_models:
        for ctrl in ('adaptive', 'coordinated', 'mpc', 'weather-adaptive'):
            if ctrl in controllers:
                print_natural_sweep(rows, ctrl, cf,
                                    scenario='S2', demand='D1')
                break  # one representative per car-model in quick mode

    # 4d. best controller per natural (SAFE)
    print_best_per_natural(rows, controllers, 'SAFE',
                           scenario='S2', demand='D1')

    # 4e. demand sweep (adaptive, SAFE)
    print_demand_sweep(rows, controllers, 'SAFE',
                       scenario='S2', natural='N-clear')

    # ---- CSV -----------------------------------------------------------
    if args.export or args.full or args.cf_full:
        print("\n[5] CSV export ...")
        export_csv(rows, 'scenario_all.csv')

    print("\nDone.")


if __name__ == '__main__':
    main()