#!/usr/bin/env python3
"""
scenario_extensions.py
======================
Four worked examples that extend the scenario matrix:

  (1) NEW MIXED CONTROLLERS   -- append to MIXED_PAIRS / MIXED_TRIPLETS
  (2) NEW NATURAL CONDITIONS  -- register in NATURAL_SCENARIOS + NATURAL_GROUPS
  (3) NEW DEMAND PATTERNS     -- add a key to DEMAND_PATTERNS
  (4) MULTI-SEED EVALUATION   -- 5 seeds + 95 % confidence intervals

Imports the base module and monkey-patches only what's needed.
Run:
    python scenario_extensions.py
"""

from __future__ import annotations
import csv
import math
import statistics
from collections import defaultdict

# ---------------------------------------------------------------------------
#  Base imports  (from traffic_control_algorithm.py)
# ---------------------------------------------------------------------------
from traffic_control_algorithm import (
    TrafficControlAlgorithm,
    COMMUNICATION_SCENARIOS,
    NATURAL_SCENARIOS,
    simulate_episode,
)


# ===========================================================================
#  (1)  NEW MIXED CONTROLLERS  —  append to MIXED_PAIRS / MIXED_TRIPLETS
# ===========================================================================
# Only *new* entries go here.  The driver in scenario_matrix.py already has
# its own MIXED_PAIRS; this file registers the additions on top.

NEW_MIXED_PAIRS = [
    # (controller_a, controller_b, weight_of_a)
    ('scoot',              'max-pressure',     0.5),   # field × field
    ('multiband',          'weather-adaptive', 0.5),   # direction × weather
    ('resilient-marl',     'multiband',        0.6),   # resilience heavy
    ('cooperative-marl',   'max-pressure',     0.4),   # message-light
    ('adaptive',           'weather-adaptive', 0.5),   # local × weather
]

NEW_MIXED_TRIPLETS = [
    # (a, b, c, (w_a, w_b, w_c))  -- weights sum to 1
    ('resilient-marl', 'multiband',        'weather-adaptive',
     (0.40, 0.30, 0.30)),
    ('max-pressure',   'scoot',            'coordinated',
     (0.35, 0.30, 0.35)),
    ('mpc',            'multiband',        'max-pressure',
     (0.40, 0.30, 0.30)),
]


def register_new_mixed_controllers() -> list[str]:
    """Register the new pairs + triplets on the class.  Return their names."""
    names: list[str] = []

    # -------- pairs --------
    for a, b, w in NEW_MIXED_PAIRS:
        name = f"{a}+{b}"
        try:
            TrafficControlAlgorithm.compose(a, b, w, new_name=name)
            names.append(name)
        except KeyError as e:
            print(f"  [skip pair] {name}: {e}")

    # -------- triplets (composed as ((a+b)+c)) --------
    for a, b, c, (wa, wb, wc) in NEW_MIXED_TRIPLETS:
        w_ab     = wa + wb
        ab_name  = f"({a}+{b})"
        abc_name = f"{a}+{b}+{c}"
        try:
            TrafficControlAlgorithm.compose(a, b, wa / w_ab, new_name=ab_name)
            TrafficControlAlgorithm.compose(ab_name, c, w_ab, new_name=abc_name)
            names.append(abc_name)
        except KeyError as e:
            print(f"  [skip triplet] {abc_name}: {e}")

    return names


# ===========================================================================
#  (2)  NEW NATURAL CONDITIONS  —  NATURAL_SCENARIOS + NATURAL_GROUPS
# ===========================================================================
NEW_NATURALS = {
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

# Which family does each new natural condition belong to?
NEW_NATURAL_GROUPS = {
    'weather':  ['N-hailstorm', 'N-flooding'],
    'incident': ['N-major-crash'],
    'demand':   ['N-holiday', 'N-freight'],
    'failure':  ['N-power-outage'],
}


def register_new_naturals():
    """Add the new conditions to the module-level NATURAL_SCENARIOS dict."""
    for name, cfg in NEW_NATURALS.items():
        if name in NATURAL_SCENARIOS:
            print(f"  [skip natural] {name} already registered")
            continue
        NATURAL_SCENARIOS[name] = cfg


# ===========================================================================
#  (3)  NEW DEMAND PATTERNS  —  DEMAND_PATTERNS
# ===========================================================================
NEW_DEMAND_PATTERNS = {
    'D5': {'label': 'Weekend leisure  ~260 vph',
           'ns': 0.22, 'ew': 0.26},
    'D6': {'label': 'Freight corridor  ~500 vph (60 % NS)',
           'ns': 0.42, 'ew': 0.28},
    'D7': {'label': 'School peak  ~380 vph (biased)',
           'ns': 0.36, 'ew': 0.16},
}


def register_new_demands(demand_patterns: dict):
    """Merge the new patterns into the caller's DEMAND_PATTERNS dict."""
    for k, v in NEW_DEMAND_PATTERNS.items():
        if k in demand_patterns:
            print(f"  [skip demand] {k} already registered")
            continue
        demand_patterns[k] = v


# ===========================================================================
#  (4)  MULTI-SEED EVALUATION  —  5 seeds + 95 % confidence intervals
# ===========================================================================
def run_multi_seed(controller_name: str,
                   scenario: str,
                   natural: str,
                   demand: str,
                   demand_patterns: dict,
                   seeds=(1, 2, 3, 4, 5),
                   n_steps: int = 100) -> dict:
    """
    Run one cell with n seeds; return mean and 95 % CI for each metric.

    The 95 % CI is the standard error of the mean × 1.96
    (t-distribution is ≈ normal at n = 5).
    """
    d = demand_patterns[demand]
    per_seed: dict[str, list[float]] = defaultdict(list)

    for s in seeds:
        c = TrafficControlAlgorithm(controller_name, scenario, natural,
                                    seed=s, car_following='SAFE')
        simulate_episode(c, n_steps=n_steps,
                         base_demand_ns=d['ns'],
                         base_demand_ew=d['ew'],
                         seed=s)
        m = c.get_metrics()
        for k in ('travel_time_s', 'queue_veh', 'waiting_time_s',
                  'throughput_vph', 'mean_aoi_ms', 'delivery_ratio'):
            per_seed[k].append(float(m[k]))

    out: dict = {'controller': controller_name, 'scenario': scenario,
                 'natural': natural, 'demand': demand, 'n_seeds': len(seeds)}
    for k, xs in per_seed.items():
        mean = statistics.fmean(xs)
        if len(xs) > 1:
            sd   = statistics.stdev(xs)
            sem  = sd / math.sqrt(len(xs))
            ci95 = 1.96 * sem
        else:
            sd = ci95 = 0.0
        out[f'{k}_mean']   = round(mean, 3)
        out[f'{k}_sd']     = round(sd, 3)
        out[f'{k}_ci95']   = round(ci95, 3)
        out[f'{k}_values'] = [round(x, 3) for x in xs]
    return out


def run_full_matrix(controllers, scenarios, naturals, demands,
                    demand_patterns, seeds=(1, 2, 3, 4, 5),
                    n_steps=100, progress=True) -> list[dict]:
    rows = []
    total = len(controllers) * len(scenarios) * len(naturals) * len(demands)
    i = 0
    for c in controllers:
        for sc in scenarios:
            for nat in naturals:
                for dem in demands:
                    i += 1
                    if progress and (i % 5 == 0 or i == total):
                        print(f"\r  cell {i}/{total}  "
                              f"({len(seeds)} seeds each) ...",
                              end='', flush=True)
                    try:
                        rows.append(run_multi_seed(
                            c, sc, nat, dem, demand_patterns,
                            seeds=seeds, n_steps=n_steps))
                    except Exception as e:
                        if progress:
                            print(f"\n  [warn] {c}/{sc}/{nat}/{dem}: {e}")
    if progress:
        print()
    return rows


# ===========================================================================
#  REPORTS
# ===========================================================================
def print_multi_seed_table(rows, controllers, scenarios,
                           natural='N-clear', demand='D1',
                           metric='travel_time_s'):
    """Show mean ± 95 % CI for each (controller, scenario) pair."""
    print("\n" + "=" * 110)
    print(f"  MULTI-SEED EVALUATION  —  {metric} (mean ± 95 % CI)  "
          f"[NAT={natural}, DEM={demand}]")
    print("=" * 110)
    idx = {(r['controller'], r['scenario']): r for r in rows
           if r['natural'] == natural and r['demand'] == demand}

    hdr = f"{'Controller':<34}" + ''.join(f"{s:>18}" for s in scenarios)
    print(hdr); print('-' * len(hdr))
    for c in controllers:
        line = f"{c:<34}"
        for s in scenarios:
            r = idx.get((c, s))
            if not r:
                line += f"{'—':>18}"
            else:
                mean = r[f'{metric}_mean']
                ci   = r[f'{metric}_ci95']
                line += f"{mean:>10.2f} ±{ci:>5.2f}"
        print(line)
    print()


def print_per_seed_table(rows, controller, scenario,
                        natural='N-clear', demand='D1',
                        metric='travel_time_s'):
    """Show the raw per-seed values for one cell — the thesis protocol."""
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


def export_multi_seed_csv(rows, path='scenario_extensions.csv'):
    cols = ['controller', 'scenario', 'natural', 'demand', 'n_seeds'] + [
        f'{k}_{s}'
        for k in ('travel_time_s', 'queue_veh', 'waiting_time_s',
                  'throughput_vph', 'mean_aoi_ms', 'delivery_ratio')
        for s in ('mean', 'sd', 'ci95')
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {len(rows)} rows to {path}")


# ===========================================================================
#  MAIN  —  demonstration of all four extensions
# ===========================================================================
if __name__ == '__main__':
    print("#" * 100)
    print("#  SCENARIO EXTENSIONS — four worked examples")
    print("#" * 100)

    # -------------------------------------------------------------------
    # (1) register the new mixed controllers
    # -------------------------------------------------------------------
    print("\n[1] Registering new MIXED controllers ...")
    new_mix = register_new_mixed_controllers()
    for n in new_mix:
        print(f"      + {n}")
    print(f"    total new mixed controllers: {len(new_mix)}")

    # -------------------------------------------------------------------
    # (2) register the new natural conditions
    # -------------------------------------------------------------------
    print("\n[2] Registering new NATURAL conditions ...")
    register_new_naturals()
    for name, cfg in NEW_NATURALS.items():
        print(f"      + {name:<16}  ({cfg['label']}, cap={cfg['capacity']:.2f})")
    print(f"    total natural conditions now: {len(NATURAL_SCENARIOS)}")

    # -------------------------------------------------------------------
    # (3) register the new demand patterns
    # -------------------------------------------------------------------
    print("\n[3] Registering new DEMAND patterns ...")
    DEMAND_PATTERNS = {
        'D1': {'label': 'Uniform  ~300 vph',       'ns': 0.30, 'ew': 0.30},
        'D2': {'label': 'AM peak  ~450 vph (60 %)','ns': 0.40, 'ew': 0.18},
        'D3': {'label': 'PM peak  ~450 vph (60 %)','ns': 0.18, 'ew': 0.40},
        'D4': {'label': 'Oversat. ~600 vph',       'ns': 0.60, 'ew': 0.50},
    }
    register_new_demands(DEMAND_PATTERNS)
    for k, v in NEW_DEMAND_PATTERNS.items():
        print(f"      + {k}  {v['label']}")
    print(f"    total demand patterns now: {len(DEMAND_PATTERNS)}")

    # -------------------------------------------------------------------
    # (4) multi-seed evaluation
    # -------------------------------------------------------------------
    print("\n[4] Running multi-seed evaluation (5 seeds, 95 % CI) ...")

    # choose a focused cross-section — the full product is huge
    controllers = new_mix + ['resilient-marl', 'fixed-time']
    scenarios   = ['S1', 'S2', 'S3', 'S4']
    naturals    = ['N-clear', 'N-hailstorm', 'N-major-crash',
                   'N-holiday', 'N-power-outage']
    demands     = ['D1', 'D5', 'D6']

    SEEDS = (1, 2, 3, 4, 5)

    rows = run_full_matrix(
        controllers, scenarios, naturals, demands,
        demand_patterns=DEMAND_PATTERNS,
        seeds=SEEDS, n_steps=80, progress=True)

    # -------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------
    print("\n" + "#" * 100)
    print("#  REPORTS")
    print("#" * 100)

    # 4a. multi-seed mean ± CI table
    print_multi_seed_table(rows, controllers, scenarios,
                           natural='N-clear', demand='D1',
                           metric='travel_time_s')

    # 4b. per-seed detail for one cell (thesis-style)
    print_per_seed_table(rows, controller=new_mix[0], scenario='S3',
                         natural='N-hailstorm', demand='D5',
                         metric='travel_time_s')

    # 4c. throughput mean ± CI table
    print_multi_seed_table(rows, controllers, scenarios,
                           natural='N-major-crash', demand='D6',
                           metric='throughput_vph')

    # -------------------------------------------------------------------
    # CSV export
    # -------------------------------------------------------------------
    print("\n[5] Exporting CSV ...")
    export_multi_seed_csv(rows, 'scenario_extensions.csv')

    print("\nDone.")