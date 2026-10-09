#!/usr/bin/env python3
"""
scenario_matrix.py
==================
Scenario catalogue + benchmark driver for `traffic_control_algorithm.py`.

Two axes are enumerated and crossed:
  MIX  — mixed (composed) controllers, incl. pairs + triplets
  NAT  — natural operating conditions, grouped by family

Crossed with:
  COMM — communication-severity scenarios S1–S4
  DEM  — demand patterns D1–D7

Every cell is run with 5 seeds and reported as mean ± 95 % CI
(protocol of §4.5 of the thesis).

CLI:
    python scenario_matrix.py                 # quick run (3 seeds, short)
    python scenario_matrix.py --full          # thesis protocol (5 seeds, 300 steps)
    python scenario_matrix.py --export        # also write the CSV
    python scenario_matrix.py --seeds 5 --steps 300

Programmatic:
    from scenario_matrix import run_scenario_matrix
    rows = run_scenario_matrix(...)
"""

from __future__ import annotations
import argparse
import csv
import math
import statistics
from collections import defaultdict
from typing import Iterable, Optional

from traffic_control_algorithm import (
    TrafficControlAlgorithm,
    COMMUNICATION_SCENARIOS,
    NATURAL_SCENARIOS,
    simulate_episode,
)


# ===========================================================================
#  AXIS 1 — MIXED CONTROLLERS  (pairs + triplets)
# ===========================================================================
# (controller_a, controller_b, weight_of_a)
MIXED_PAIRS = [
    # --- resilient family ---
    ('resilient-marl',     'coordinated',      0.5),
    ('resilient-marl',     'mpc',              0.5),
    ('resilient-marl',     'max-pressure',     0.5),
    ('resilient-marl',     'multiband',        0.5),
    ('resilient-marl',     'multiband',        0.6),
    ('resilient-marl',     'weather-adaptive', 0.5),
    ('resilient-marl',     'scats',            0.5),
    # --- cooperative family ---
    ('cooperative-marl',   'coordinated',      0.5),
    ('cooperative-marl',   'max-pressure',     0.5),
    ('cooperative-marl',   'max-pressure',     0.4),
    ('cooperative-marl',   'multiband',        0.5),
    # --- planning hybrids ---
    ('mpc',                'multiband',        0.5),
    ('mpc',                'coordinated',      0.5),
    ('max-pressure',       'scats',            0.5),
    ('fuzzy-logic',        'mpc',              0.5),
    ('fuzzy-logic',        'coordinated',      0.5),
    ('adaptive',           'coordinated',      0.5),
    ('optimized',          'scats',            0.5),
    # --- field × field ---
    ('scoot',              'max-pressure',     0.5),
    ('multiband',          'weather-adaptive', 0.5),
    ('adaptive',           'weather-adaptive', 0.5),
]

# (a, b, c, (w_a, w_b, w_c))  — weights sum to 1
MIXED_TRIPLETS = [
    ('resilient-marl',   'coordinated',   'mpc',                (0.40, 0.30, 0.30)),
    ('cooperative-marl', 'coordinated',   'multiband',          (0.40, 0.30, 0.30)),
    ('mpc',              'multiband',     'weather-adaptive',   (0.40, 0.30, 0.30)),
    ('resilient-marl',   'max-pressure',  'scats',              (0.40, 0.30, 0.30)),
    ('fuzzy-logic',      'mpc',           'coordinated',        (0.40, 0.30, 0.30)),
    ('resilient-marl',   'multiband',     'weather-adaptive',   (0.40, 0.30, 0.30)),
    ('max-pressure',     'scoot',         'coordinated',        (0.35, 0.30, 0.35)),
    ('mpc',              'multiband',     'max-pressure',       (0.40, 0.30, 0.30)),
]


def register_mixed_controllers() -> list[str]:
    """Register all pairwise + triplet compositions; return their names."""
    names: list[str] = []

    # -------- pairs --------
    for a, b, w in MIXED_PAIRS:
        name = f"{a}+{b}"
        if name in TrafficControlAlgorithm.CONTROLLERS:
            names.append(name); continue
        try:
            TrafficControlAlgorithm.compose(a, b, w, new_name=name)
            names.append(name)
        except KeyError as e:
            print(f"  [skip pair] {name}: {e}")

    # -------- triplets (compose as ((a+b)+c)) --------
    for a, b, c, (wa, wb, wc) in MIXED_TRIPLETS:
        w_ab     = wa + wb
        ab_name  = f"({a}+{b})"
        abc_name = f"{a}+{b}+{c}"
        if abc_name in TrafficControlAlgorithm.CONTROLLERS:
            names.append(abc_name); continue
        try:
            if ab_name not in TrafficControlAlgorithm.CONTROLLERS:
                TrafficControlAlgorithm.compose(a, b, wa / w_ab,
                                                new_name=ab_name)
            TrafficControlAlgorithm.compose(ab_name, c, w_ab,
                                            new_name=abc_name)
            names.append(abc_name)
        except KeyError as e:
            print(f"  [skip triplet] {abc_name}: {e}")

    return names


# ===========================================================================
#  AXIS 2 — NATURAL OPERATING CONDITIONS  (grouped by family)
# ===========================================================================
#  Base conditions live in the base file; extras are declared here and
#  registered into NATURAL_SCENARIOS at import time.
EXTRA_NATURALS = {
    # ---- additional weather -------------------------------------------
    'N-hailstorm': {
        'label': 'Hailstorm', 'capacity': 0.45, 'demand_mult': 0.55,
        'direction_bias': 0.00, 'sensor_reliability': 0.80,
    },
    'N-flooding': {
        'label': 'Localised flooding', 'capacity': 0.35, 'demand_mult': 0.50,
        'direction_bias': 0.05, 'sensor_reliability': 0.75,
    },
    # ---- additional incident ------------------------------------------
    'N-major-crash': {
        'label': 'Multi-lane crash', 'capacity': 0.40, 'demand_mult': 0.85,
        'direction_bias': 0.30, 'sensor_reliability': 1.00,
    },
    # ---- additional demand --------------------------------------------
    'N-holiday': {
        'label': 'Public holiday', 'capacity': 1.00, 'demand_mult': 0.45,
        'direction_bias': 0.00, 'sensor_reliability': 1.00,
    },
    'N-freight': {
        'label': 'Night freight wave', 'capacity': 0.95, 'demand_mult': 1.20,
        'direction_bias': 0.15, 'sensor_reliability': 1.00,
    },
    # ---- additional failure -------------------------------------------
    'N-power-outage': {
        'label': 'Signal power outage', 'capacity': 0.90, 'demand_mult': 0.75,
        'direction_bias': 0.00, 'sensor_reliability': 0.20,
    },
}

NATURAL_GROUPS = {
    'weather':  ['N-clear', 'N-rain', 'N-snow', 'N-fog',
                 'N-hailstorm', 'N-flooding'],
    'incident': ['N-accident', 'N-workzone', 'N-major-crash'],
    'demand':   ['N-rush-am', 'N-rush-pm', 'N-night', 'N-event', 'N-school',
                 'N-holiday', 'N-freight'],
    'failure':  ['N-sensor-fail', 'N-comm-blackout', 'N-power-outage'],
}
ALL_NATURALS = [n for grp in NATURAL_GROUPS.values() for n in grp]


def register_extra_naturals():
    """Add the extra conditions to NATURAL_SCENARIOS at import time."""
    for name, cfg in EXTRA_NATURALS.items():
        NATURAL_SCENARIOS.setdefault(name, cfg)


# ===========================================================================
#  AXES 3 & 4 — COMMUNICATION + DEMAND
# ===========================================================================
COMMUNICATION_GROUPS = {
    'ideal':    ['S1'],
    'moderate': ['S2'],
    'severe':   ['S3'],
    'extreme':  ['S4'],
}

DEMAND_PATTERNS = {
    'D1': {'label': 'Uniform  ~300 vph',           'ns': 0.30, 'ew': 0.30},
    'D2': {'label': 'AM peak  ~450 vph (60 %)',    'ns': 0.40, 'ew': 0.18},
    'D3': {'label': 'PM peak  ~450 vph (60 %)',    'ns': 0.18, 'ew': 0.40},
    'D4': {'label': 'Oversaturated ~600 vph',      'ns': 0.60, 'ew': 0.50},
    'D5': {'label': 'Weekend leisure  ~260 vph',   'ns': 0.22, 'ew': 0.26},
    'D6': {'label': 'Freight corridor ~500 vph',   'ns': 0.42, 'ew': 0.28},
    'D7': {'label': 'School peak  ~380 vph',       'ns': 0.36, 'ew': 0.16},
}


# ===========================================================================
#  ONE-SEED RUNNER
# ===========================================================================
def run_scenario(controller_name: str,
                 scenario: str,
                 natural: str,
                 demand: str,
                 n_steps: int = 100,
                 seed: int = 0) -> dict:
    """Single-seed run of one (controller, scenario, natural, demand) cell."""
    d = DEMAND_PATTERNS[demand]
    ctrl = TrafficControlAlgorithm(controller_name, scenario, natural,
                                   seed, 'SAFE')
    simulate_episode(ctrl, n_steps=n_steps,
                     base_demand_ns=d['ns'], base_demand_ew=d['ew'],
                     seed=seed)
    m = ctrl.get_metrics()
    m['controller'] = controller_name
    m['demand']     = demand
    return m


# ===========================================================================
#  MULTI-SEED RUNNER  (mean ± 95 % CI, thesis protocol)
# ===========================================================================
TRACKED = ('travel_time_s', 'queue_veh', 'waiting_time_s',
           'throughput_vph', 'mean_aoi_ms', 'delivery_ratio')


def run_multi_seed(controller_name: str,
                   scenario: str,
                   natural: str,
                   demand: str,
                   seeds: Iterable[int] = (1, 2, 3, 4, 5),
                   n_steps: int = 100) -> dict:
    """
    Run one cell with n seeds; return mean / stdev / 95 % CI for each metric.

    95 % CI = 1.96 * stdev / sqrt(n)  (normal approximation; standard for n = 5).
    """
    d = DEMAND_PATTERNS[demand]
    per_seed: dict[str, list[float]] = defaultdict(list)
    seeds = tuple(seeds)

    for s in seeds:
        ctrl = TrafficControlAlgorithm(controller_name, scenario, natural,
                                       seed=s, car_following='SAFE')
        simulate_episode(ctrl, n_steps=n_steps,
                         base_demand_ns=d['ns'], base_demand_ew=d['ew'],
                         seed=s)
        m = ctrl.get_metrics()
        for k in TRACKED:
            per_seed[k].append(float(m[k]))

    out = {'controller': controller_name, 'scenario': scenario,
           'natural': natural, 'demand': demand, 'n_seeds': len(seeds)}
    for k, xs in per_seed.items():
        mean = statistics.fmean(xs) if xs else 0.0
        if len(xs) > 1:
            sd  = statistics.stdev(xs)
            sem = sd / math.sqrt(len(xs))
            ci  = 1.96 * sem
        else:
            sd = ci = 0.0
        out[f'{k}_mean']   = round(mean, 3)
        out[f'{k}_sd']     = round(sd,   3)
        out[f'{k}_ci95']   = round(ci,   3)
        out[f'{k}_values'] = [round(x, 3) for x in xs]
    return out


def run_scenario_matrix(controllers: list[str],
                        scenarios: list[str],
                        naturals: list[str],
                        demands: list[str],
                        seeds: Iterable[int] = (1, 2, 3, 4, 5),
                        n_steps: int = 100,
                        progress: bool = True) -> list[dict]:
    """Full cross-product with multi-seed evaluation."""
    rows: list[dict] = []
    total = len(controllers) * len(scenarios) * len(naturals) * len(demands)
    i = 0
    for c in controllers:
        for sc in scenarios:
            for nat in naturals:
                for dem in demands:
                    i += 1
                    if progress and (i % 10 == 0 or i == total):
                        print(f"\r  cell {i}/{total} "
                              f"({len(tuple(seeds))} seeds each) ...",
                              end='', flush=True)
                    try:
                        rows.append(run_multi_seed(
                            c, sc, nat, dem, seeds=seeds, n_steps=n_steps))
                    except Exception as e:
                        if progress:
                            print(f"\n  [warn] {c}/{sc}/{nat}/{dem}: {e}")
    if progress:
        print()
    return rows


# ===========================================================================
#  REPORTS
# ===========================================================================
def _short_natural(n: str) -> str:
    return {'N-clear': 'Clear', 'N-rain': 'Rain', 'N-snow': 'Snow',
            'N-fog': 'Fog', 'N-hailstorm': 'Hail', 'N-flooding': 'Flood',
            'N-accident': 'Accid', 'N-workzone': 'Work',
            'N-major-crash': 'Crash',
            'N-rush-am': 'AMpk', 'N-rush-pm': 'PMpk',
            'N-night': 'Night', 'N-event': 'Event', 'N-school': 'Schl',
            'N-holiday': 'Holi', 'N-freight': 'Frt',
            'N-sensor-fail': 'NoDet', 'N-comm-blackout': 'NoV2X',
            'N-power-outage': 'NoPwr'}.get(n, n)


def _short_controller(c: str, width: int = 32) -> str:
    return c if len(c) <= width else c[:width - 1] + '…'


def print_catalogue(mix_names, naturals, scenarios, demands,
                    seeds=(1, 2, 3, 4, 5), n_steps=100):
    print("\n" + "=" * 100)
    print("  SCENARIO CATALOGUE")
    print("=" * 100)
    print(f"\n  MIXED controllers ({len(mix_names)}):")
    for i, n in enumerate(mix_names, 1):
        print(f"    {i:>2}.  {n}")

    print(f"\n  NATURAL conditions ({len(naturals)}):")
    for grp, members in NATURAL_GROUPS.items():
        labels = ', '.join(_short_natural(m) for m in members)
        print(f"    [{grp:<9}] {labels}")

    print(f"\n  COMMUNICATION scenarios ({len(scenarios)}):")
    for grp, members in COMMUNICATION_GROUPS.items():
        print(f"    [{grp:<9}] {', '.join(members)}")

    print(f"\n  DEMAND patterns ({len(demands)}):")
    for d in demands:
        cfg = DEMAND_PATTERNS[d]
        print(f"    {d}: {cfg['label']}  "
              f"(ns={cfg['ns']}, ew={cfg['ew']})")

    total = len(mix_names) * len(scenarios) * len(naturals) * len(demands)
    print(f"\n  Seeds per cell   : {len(tuple(seeds))} {tuple(seeds)}")
    print(f"  Steps per episode: {n_steps}")
    print(f"  Total scenario cells: {total}  "
          f"({total * len(tuple(seeds))} simulator runs)")


def _matrix(rows, controllers, columns, row_key, col_key,
            natural, scenario, demand, metric):
    idx = {(r['controller'], r[row_key]): r for r in rows
           if r['natural'] == natural and r['scenario'] == scenario
           and r['demand'] == demand}
    return idx


def print_mix_x_nat(rows, mix_names, naturals,
                    scenario='S2', demand='D1',
                    metric='travel_time_s'):
    print("\n" + "=" * 100)
    print(f"  MIXED × NATURAL  —  {metric} (mean)  "
          f"[COMM={scenario}, DEM={demand}]")
    print("=" * 100)
    idx = _matrix(rows, mix_names, naturals, 'natural', 'controller',
                  natural=None, scenario=scenario, demand=demand,
                  metric=metric)
    hdr = f"{'Controller':<34}" + ''.join(f"{_short_natural(n):>8}"
                                          for n in naturals)
    print(hdr); print('-' * len(hdr))
    for m in mix_names:
        line = f"{_short_controller(m, 34):<34}"
        for n in naturals:
            r = idx.get((m, n))
            line += f"{r[f'{metric}_mean']:>8.1f}" if r else f"{'—':>8}"
        print(line)
    print()


def print_mix_x_comm(rows, mix_names, natural='N-clear', demand='D1',
                     metric='travel_time_s'):
    print("\n" + "=" * 100)
    print(f"  MIXED × COMMUNICATION  —  {metric} (mean ± CI)  "
          f"[NAT={natural}, DEM={demand}]")
    print("=" * 100)
    scenarios = list(COMMUNICATION_SCENARIOS)
    idx = _matrix(rows, mix_names, scenarios, 'scenario', 'controller',
                  natural=natural, scenario=None, demand=demand,
                  metric=metric)
    hdr = f"{'Controller':<34}" + ''.join(f"{s:>16}" for s in scenarios)
    print(hdr); print('-' * len(hdr))
    for m in mix_names:
        line = f"{_short_controller(m, 34):<34}"
        for s in scenarios:
            r = idx.get((m, s))
            if r:
                line += f"{r[f'{metric}_mean']:>10.1f}±{r[f'{metric}_ci95']:>4.1f}"
            else:
                line += f"{'—':>16}"
        print(line)
    print()


def print_mix_x_dem(rows, mix_names, natural='N-clear', scenario='S2',
                    metric='travel_time_s'):
    print("\n" + "=" * 100)
    print(f"  MIXED × DEMAND  —  {metric} (mean)  "
          f"[NAT={natural}, COMM={scenario}]")
    print("=" * 100)
    demands = list(DEMAND_PATTERNS)
    idx = _matrix(rows, mix_names, demands, 'demand', 'controller',
                  natural=natural, scenario=scenario, demand=None,
                  metric=metric)
    hdr = f"{'Controller':<34}" + ''.join(f"{d:>8}" for d in demands)
    print(hdr); print('-' * len(hdr))
    for m in mix_names:
        line = f"{_short_controller(m, 34):<34}"
        for d in demands:
            r = idx.get((m, d))
            line += f"{r[f'{metric}_mean']:>8.1f}" if r else f"{'—':>8}"
        print(line)
    print()


def print_nat_x_comm(rows, controller='resilient-marl+coordinated', demand='D1',
                     metric='travel_time_s'):
    print("\n" + "=" * 100)
    print(f"  NATURAL × COMMUNICATION  —  {metric} (mean)  "
          f"[MIX={controller}, DEM={demand}]")
    print("=" * 100)
    scenarios = list(COMMUNICATION_SCENARIOS)
    idx = _matrix(rows, [controller], scenarios, 'scenario', 'natural',
                  natural=None, scenario=None, demand=demand,
                  metric=metric)
    hdr = f"{'Natural':<20}" + ''.join(f"{s:>10}" for s in scenarios)
    print(hdr); print('-' * len(hdr))
    for n in ALL_NATURALS:
        line = f"{_short_natural(n):<20}"
        for s in scenarios:
            r = idx.get((controller, s)) if False else None
            # look up by natural/scenario directly
            cell = next((r for r in rows
                         if r['controller'] == controller
                         and r['scenario'] == s
                         and r['natural'] == n
                         and r['demand'] == demand), None)
            line += f"{cell[f'{metric}_mean']:>10.1f}" if cell else f"{'—':>10}"
        print(line)
    print()


def print_best_per_natural(rows, mix_names, scenario='S2', demand='D1',
                           metric='travel_time_s'):
    print("\n" + "=" * 100)
    print(f"  BEST MIXED CONTROLLER PER NATURAL CONDITION  —  {metric}  "
          f"[COMM={scenario}, DEM={demand}]")
    print("=" * 100)
    hdr = (f"{'Natural':<20}{'Best mixed controller':<38}"
           f"{'Mean':>10}{'±CI':>8}{'Thru(vph)':>12}")
    print(hdr); print('-' * len(hdr))
    for nat in ALL_NATURALS:
        cands = [r for r in rows
                 if r['natural'] == nat
                 and r['scenario'] == scenario
                 and r['demand'] == demand
                 and r['controller'] in mix_names]
        if not cands:
            continue
        best = min(cands, key=lambda r: r[f'{metric}_mean'])
        print(f"{_short_natural(nat):<20}"
              f"{_short_controller(best['controller'], 38):<38}"
              f"{best[f'{metric}_mean']:>10.2f}"
              f"{best[f'{metric}_ci95']:>8.2f}"
              f"{best['throughput_vph_mean']:>12.1f}")
    print()


def print_per_seed_detail(rows, controller, scenario, natural, demand,
                          metric='travel_time_s'):
    print("\n" + "=" * 100)
    print(f"  PER-SEED DETAIL  —  {controller}  •  {scenario}  •  "
          f"{natural}  •  {demand}  —  {metric}")
    print("=" * 100)
    for r in rows:
        if (r['controller'] == controller and r['scenario'] == scenario
                and r['natural'] == natural and r['demand'] == demand):
            print(f"  seeds       : {r['n_seeds']}")
            print(f"  values      : {r[f'{metric}_values']}")
            print(f"  mean        : {r[f'{metric}_mean']}")
            print(f"  stdev       : {r[f'{metric}_sd']}")
            print(f"  95 % CI     : ± {r[f'{metric}_ci95']}")
            return
    print("  (no matching cell)")


# ===========================================================================
#  CSV EXPORT
# ===========================================================================
def export_csv(rows, path='scenario_matrix.csv'):
    cols = ['controller', 'scenario', 'natural', 'demand', 'n_seeds'] + [
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
    p = argparse.ArgumentParser(
        description="Scenario matrix driver for traffic_control_algorithm.")
    p.add_argument('--full', action='store_true',
                   help="thesis protocol: 5 seeds × 300 steps")
    p.add_argument('--seeds', type=int, default=None,
                   help="number of seeds (default 3 quick / 5 full)")
    p.add_argument('--steps', type=int, default=None,
                   help="decision steps per episode "
                        "(default 80 quick / 300 full)")
    p.add_argument('--export', action='store_true',
                   help="write the CSV alongside the reports")
    p.add_argument('--max-controllers', type=int, default=None,
                   help="cap the number of mixed controllers (for quick runs)")
    return p.parse_args()


def main():
    args = _parse_args()

    # -------------------------------------------------------------
    # 1. register extras
    # -------------------------------------------------------------
    print("#" * 100)
    print("#  SCENARIO MATRIX — mixed × natural × comm × demand")
    print("#" * 100)

    print("\n[1] Registering mixed controllers ...")
    mix_names = register_mixed_controllers()
    if args.max_controllers:
        mix_names = mix_names[:args.max_controllers]
    print(f"    registered {len(mix_names)} mixed controllers")

    print("\n[2] Registering extra natural conditions ...")
    register_extra_naturals()
    print(f"    natural conditions now: {len(NATURAL_SCENARIOS)}")

    # -------------------------------------------------------------
    # 2. protocols
    # -------------------------------------------------------------
    if args.full:
        seeds   = tuple(range(1, (args.seeds or 5) + 1))
        n_steps = args.steps or 300
    else:
        seeds   = tuple(range(1, (args.seeds or 3) + 1))
        n_steps = args.steps or 80

    naturals  = ALL_NATURALS
    scenarios = list(COMMUNICATION_SCENARIOS)
    demands   = list(DEMAND_PATTERNS)

    print_catalogue(mix_names, naturals, scenarios, demands,
                    seeds=seeds, n_steps=n_steps)

    # -------------------------------------------------------------
    # 3. run the matrix
    # -------------------------------------------------------------
    print("\n[3] Running scenario matrix ...")
    rows = run_scenario_matrix(mix_names, scenarios, naturals, demands,
                               seeds=seeds, n_steps=n_steps,
                               progress=True)

    # -------------------------------------------------------------
    # 4. reports
    # -------------------------------------------------------------
    print("\n[4] Reports ...")
    print_mix_x_nat (rows, mix_names, naturals, 'S2', 'D1')
    print_mix_x_comm(rows, mix_names, 'N-clear', 'D1')
    print_mix_x_dem (rows, mix_names, 'N-clear', 'S2')
    print_nat_x_comm(rows, mix_names[0], 'D1')
    print_best_per_natural(rows, mix_names, 'S2', 'D1')
    print_per_seed_detail(rows, mix_names[0], 'S3',
                          'N-hailstorm', 'D5')

    # -------------------------------------------------------------
    # 5. CSV
    # -------------------------------------------------------------
    if args.export:
        print("\n[5] CSV export ...")
        export_csv(rows, 'scenario_matrix.csv')

    print("\nDone.")


if __name__ == '__main__':
    main()