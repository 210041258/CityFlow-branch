#!/usr/bin/env python3
"""
analytical_surrogate.py
=======================
Run every scenario WITHOUT stepping the simulator.

Closed-form multiplicative model:

    TT(ctrl, cf, comm, nat, dem)
      = TT_base(cf, regime)
      × K_ctrl(ctrl)
      × K_comm(ctrl_family, latency, loss)
      × K_nat(capacity, demand_mult, bias)
      × K_dem(ns, ew)

Every factor is either:
  * analytical  — logistic curve, M/M/1 approximation, capacity ratio
  * lookup      — a small table fitted from ~50 simulated points

The surrogate replaces `simulate_episode` with `predict(...)`.
The full 5-axis cross product (3 CF × 20 ctrl × 4 comm × 19 nat × 7 dem)
runs in seconds instead of hours.

CLI:
    python analytical_surrogate.py                 # fit + predict grid
    python analytical_surrogate.py --calibrate     # fit from real sim
    python analytical_surrogate.py --invert        # invert the curve
    python analytical_surrogate.py --sensitivity   # Sobol-like analysis
    python analytical_surrogate.py --full --export # 160k cells → CSV

Programmatic:
    from analytical_surrogate import Surrogate
    s = Surrogate()
    s.calibrate()                                 # optional
    row = s.predict('resilient-marl', 'SAFE',
                    'S3', 'N-hailstorm', 'D2')
    grid = s.run_full_grid()
"""

from __future__ import annotations
import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

from traffic_control_algorithm import (
    TrafficControlAlgorithm,
    COMMUNICATION_SCENARIOS,
    NATURAL_SCENARIOS,
    CAR_FOLLOWING_BENCHMARK,
    DEGRADATION_L, DEGRADATION_K, DEGRADATION_X0_MS,
    DECISION_INTERVAL_S, SATURATION_FLOW_VPS,
)


# ===========================================================================
#  CONTROLLER → FAMILY MAP  (which family is it?  affects K_comm)
# ===========================================================================
CONTROLLER_FAMILY = {
    'fixed-time': 'baseline',
    'actuated':   'baseline',
    'adaptive':   'adaptive',
    'optimized':  'adaptive',
    'coordinated': 'coordinated',
    'independent-dqn': 'learning',
    'cooperative-marl': 'learning',
    'resilient-marl':   'resilient',
    'max-pressure': 'field',
    'scats':        'field',
    'scoot':        'field',
    'multiband':    'field',
    'fuzzy-logic':  'field',
    'mpc':          'field',
    'weather-adaptive': 'field',
}

# Family-level K_ctrl (baseline travel-time multiplier) — fitted from
# thesis Table 13 / calibration data
K_CTRL_FAMILY = {
    'baseline':    1.26,
    'adaptive':    1.06,
    'coordinated': 1.00,
    'field':       0.98,
    'learning':    1.02,
    'resilient':   0.99,
}

# Family-level resilience factor: how much the family's policy flattens
# the logistic curve (1.0 = no flattening, higher = flatter)
RESILIENCE_FLATTENING = {
    'baseline':    1.00,
    'adaptive':    0.95,
    'coordinated': 0.90,
    'field':       0.85,
    'learning':    0.75,
    'resilient':   0.35,   # M1-M5 dramatically flatten
}


# ===========================================================================
#  BASE CAR-FOLLOWING TIME
# ===========================================================================
def base_cf_time(cf_model: str, congestion: float) -> float:
    """Blend free-flow and congested in-system time using congestion as weight."""
    b = CAR_FOLLOWING_BENCHMARK[cf_model]
    ff, cg = b['free_flow']['in_system_s'], b['congested']['in_system_s']
    w = max(0.0, min(1.0, congestion))
    return (1 - w) * ff + w * cg


# ===========================================================================
#  FACTOR: COMMUNICATION  (logistic with family flattening)
# ===========================================================================
def K_comm(ctrl_family: str, latency_ms: float, loss_rate: float) -> float:
    """
    Communication degradation factor.
    Logistic in the severity index rho; flattened per controller family.
    """
    # severity index (Eq. 8) — same weights as the base file
    rho = 0.5 * (latency_ms / 600.0) + 0.5 * loss_rate
    # sigmoid: 1 + (L/100) * sigma(k*(x - x0))
    flatten = RESILIENCE_FLATTENING.get(ctrl_family, 1.0)
    # k and x0 scale with flattening: flatter = smaller k, larger x0
    k  = DEGRADATION_K * 100.0 / flatten       # base k ~ 1.43
    x0 = DEGRADATION_X0_MS / 600.0 * (1.0 + 0.5 * (1 - flatten))
    L  = 0.6 * flatten                          # maximum degradation 60 %
    return 1.0 + L / (1.0 + math.exp(-k * (rho - x0)))


# ===========================================================================
#  FACTOR: NATURAL  (capacity + demand_mult + bias)
# ===========================================================================
def K_nat(capacity: float, demand_mult: float, direction_bias: float) -> float:
    """
    Natural-condition degradation factor.
      * capacity 1.0 → no penalty; 0.5 → strong penalty
      * demand_mult 1.0 → no penalty; 1.5 → moderate
      * |bias| penalises controllers that don't handle directionality
    """
    cap_pen  = 1.0 / max(capacity, 0.1)
    dem_pen  = 1.0 + 0.35 * (demand_mult - 1.0)
    bias_pen = 1.0 + 0.15 * abs(direction_bias)
    return cap_pen * dem_pen * bias_pen


# ===========================================================================
#  FACTOR: DEMAND  (M/M/1-like queueing)
# ===========================================================================
def K_dem(ns: float, ew: float, capacity: float = 1.0) -> float:
    """
    Demand factor via M/M/1 utilisation.
    rho = arrival_rate / service_rate
    """
    arrival  = ns + ew
    service  = 2 * SATURATION_FLOW_VPS * capacity * 10.0   # 10 s interval
    rho      = min(0.95, arrival / max(service, 1e-6))
    return 1.0 / (1.0 - rho)


# ===========================================================================
#  FACTOR: CONGESTION STATE (used to pick free-flow vs congested base)
# ===========================================================================
def congestion_state(comm_severity: float, nat_severity: float,
                     dem_severity: float) -> float:
    """Combine the three impairment sources into a congestion weight in [0,1]."""
    total = 0.4 * comm_severity + 0.4 * nat_severity + 0.2 * dem_severity
    return min(0.99, total)


# ===========================================================================
#  THE SURROGATE
# ===========================================================================
class Surrogate:
    """
    Analytical surrogate for `simulate_episode`.
    Predicts travel_time_s, queue_veh, waiting_time_s, throughput_vph
    in O(1) per cell.
    """

    def __init__(self, seed: int = 0):
        self.rng       = random.Random(seed)
        self.calibration_error: dict = {}

    # ---------------------------------------------------------------- predict
    def predict(self, controller: str, car_model: str,
                scenario: str, natural: str, demand: str,
                demand_ns: Optional[float] = None,
                demand_ew: Optional[float] = None) -> dict:
        # ---- resolve the three impairment axes ----------------------
        comm = COMMUNICATION_SCENARIOS[scenario]
        nat  = NATURAL_SCENARIOS[natural]
        # default demand from D1..D7 lookup if not given
        if demand_ns is None or demand_ew is None:
            demand_ns, demand_ew = 0.30, 0.30   # uniform default

        family = CONTROLLER_FAMILY.get(controller, 'adaptive')

        # ---- individual factors -------------------------------------
        latency = comm['latency_ms']
        loss    = comm['loss_rate']
        cap     = nat['capacity']
        dem_m   = nat['demand_mult']
        bias    = nat['direction_bias']

        sev_comm = 0.5 * (latency / 600.0) + 0.5 * loss
        sev_nat  = 1.0 - cap
        sev_dem  = min(0.5, (demand_ns + demand_ew - 0.6) / 1.2)
        cong     = congestion_state(sev_comm, sev_nat, sev_dem)

        # ---- base time ----------------------------------------------
        base_ff = base_cf_time(car_model, 0.0)
        base_cg = base_cf_time(car_model, 0.9)
        base    = base_ff + (base_cg - base_ff) * cong

        # ---- multiplicative factors ---------------------------------
        kc = K_CTRL_FAMILY.get(family, 1.0)
        km = K_comm(family, latency, loss)
        kn = K_nat(cap, dem_m, bias)
        kd = K_dem(demand_ns * dem_m, demand_ew * dem_m, cap)

        travel = base * kc * km * kn * kd

        # ---- derive queue / waiting / throughput --------------------
        # queue grows with total multiplier above 1
        excess = max(0.0, travel / max(base, 1.0) - 1.0)
        queue  = 3.0 + 12.0 * excess
        wait   = 5.0 + 20.0 * excess
        thr    = max(50.0, 950.0 / (1.0 + 0.9 * excess))

        return {
            'controller':    controller,
            'car_following': car_model,
            'scenario':      scenario,
            'natural':       natural,
            'demand':        demand,
            'travel_time_s': round(travel, 2),
            'queue_veh':     round(queue, 2),
            'waiting_time_s': round(wait, 2),
            'throughput_vph': round(thr, 1),
            # diagnostics — useful for analysis
            'K_ctrl':  round(kc, 3),
            'K_comm':  round(km, 3),
            'K_nat':   round(kn, 3),
            'K_dem':   round(kd, 3),
            'congestion': round(cong, 3),
        }

    # -------------------------------------------------------------- calibrate
    def calibrate(self, n_samples: int = 40, n_steps: int = 100,
                  progress: bool = True) -> dict:
        """
        Sample a small random subset of the grid with the real simulator;
        compute residual error per axis; store per-axis correction.
        """
        print(f"  sampling {n_samples} real simulation points ...")
        residuals = []
        for i in range(n_samples):
            ctrl   = self.rng.choice(list(CONTROLLER_FAMILY))
            cf     = self.rng.choice(tuple(CAR_FOLLOWING_BENCHMARK))
            scen   = self.rng.choice(list(COMMUNICATION_SCENARIOS))
            nat    = self.rng.choice(list(NATURAL_SCENARIOS))
            dem_ns = self.rng.uniform(0.10, 0.60)
            dem_ew = self.rng.uniform(0.10, 0.60)
            seed   = i + 1

            # predicted
            pred = self.predict(ctrl, cf, scen, nat, 'D1',
                                demand_ns=dem_ns, demand_ew=dem_ew)
            # simulated
            c = TrafficControlAlgorithm(ctrl, scen, nat,
                                        seed=seed, car_following=cf)
            from traffic_control_algorithm import simulate_episode
            simulate_episode(c, n_steps=n_steps,
                             base_demand_ns=dem_ns,
                             base_demand_ew=dem_ew,
                             seed=seed)
            m = c.get_metrics()

            ratio = m['travel_time_s'] / max(pred['travel_time_s'], 1e-6)
            residuals.append({
                'controller': ctrl, 'car_following': cf,
                'scenario': scen, 'natural': nat,
                'sim': m['travel_time_s'], 'pred': pred['travel_time_s'],
                'ratio': ratio,
            })
            if progress and (i + 1) % 10 == 0:
                print(f"    {i+1}/{n_samples}")

        # aggregate residuals
        by_ctrl = defaultdict(list)
        by_cf   = defaultdict(list)
        for r in residuals:
            by_ctrl[r['controller']].append(r['ratio'])
            by_cf[r['car_following']].append(r['ratio'])

        global_ratio = statistics.fmean(r['ratio'] for r in residuals)
        self.calibration_error = {
            'global_ratio': round(global_ratio, 4),
            'global_rmse':  round(math.sqrt(statistics.fmean(
                (r['ratio'] - global_ratio) ** 2 for r in residuals)), 4),
            'by_controller': {k: round(statistics.fmean(v), 3)
                              for k, v in by_ctrl.items()},
            'by_cf':         {k: round(statistics.fmean(v), 3)
                              for k, v in by_cf.items()},
            'n_samples':     len(residuals),
        }

        # apply global correction to the family map
        for c, r in self.calibration_error['by_controller'].items():
            fam = CONTROLLER_FAMILY.get(c)
            if fam:
                K_CTRL_FAMILY[fam] *= r
        return self.calibration_error

    # -------------------------------------------------------------- grid
    def run_full_grid(self, controllers=None, car_models=None,
                      scenarios=None, naturals=None, demands=None,
                      demand_patterns=None, progress=True) -> list[dict]:
        """
        Predict the full cross product.  ~160k cells in seconds.
        """
        from scenario_matrix import DEMAND_PATTERNS as DP
        demand_patterns = demand_patterns or DP
        controllers = controllers or list(CONTROLLER_FAMILY)
        car_models  = car_models  or list(CAR_FOLLOWING_BENCHMARK)
        scenarios   = scenarios   or list(COMMUNICATION_SCENARIOS)
        naturals    = naturals    or list(NATURAL_SCENARIOS)
        demands     = demands     or list(demand_patterns)

        grid = list(product(controllers, car_models, scenarios,
                            naturals, demands))
        total = len(grid)
        rows: list[dict] = []
        for i, (c, cf, sc, nat, dem) in enumerate(grid, 1):
            d = demand_patterns[dem]
            r = self.predict(c, cf, sc, nat, dem,
                             demand_ns=d['ns'], demand_ew=d['ew'])
            rows.append(r)
            if progress and (i % 5000 == 0 or i == total):
                print(f"\r  {i}/{total} cells predicted ...",
                      end='', flush=True)
        if progress:
            print()
        return rows

    # -------------------------------------------------------------- invert
    def invert_latency(self, controller: str, car_model: str = 'SAFE',
                       natural: str = 'N-clear', demand: str = 'D1',
                       target_pct: float = 10.0,
                       demand_patterns=None) -> float:
        """
        Find the latency (ms) at which travel time rises by target_pct
        relative to the ideal case (S1).  Uses bisection on the analytic
        model — no simulation.
        """
        from scenario_matrix import DEMAND_PATTERNS as DP
        demand_patterns = demand_patterns or DP
        d = demand_patterns[demand]

        base = self.predict(controller, car_model, 'S1', natural, demand,
                            demand_ns=d['ns'], demand_ew=d['ew'])['travel_time_s']
        target = base * (1 + target_pct / 100.0)

        def f(latency):
            comm = COMMUNICATION_SCENARIOS['S3']       # use S3 loss
            rho = 0.5 * (latency / 600.0) + 0.5 * comm['loss_rate']
            family = CONTROLLER_FAMILY.get(controller, 'adaptive')
            km = K_comm(family, latency, comm['loss_rate'])
            kn = K_nat(NATURAL_SCENARIOS[natural]['capacity'], 1.0, 0.0)
            kd = K_dem(d['ns'], d['ew'])
            return base * K_CTRL_FAMILY.get(family, 1.0) * km * kn * kd - target

        lo, hi = 0.0, 3000.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0: lo = mid
            else:          hi = mid
        return round(0.5 * (lo + hi), 2)


# ===========================================================================
#  REPORTS
# ===========================================================================
def print_calibration(report: dict):
    print("\n" + "=" * 90)
    print("  CALIBRATION REPORT")
    print("=" * 90)
    print(f"  samples         : {report['n_samples']}")
    print(f"  global ratio    : {report['global_ratio']}   "
          f"(1.0 = perfect)")
    print(f"  global RMSE     : {report['global_rmse']}")
    print(f"\n  per-controller residual ratio:")
    for c, r in sorted(report['by_controller'].items()):
        print(f"    {c:<26}  {r:.3f}")
    print(f"\n  per-car-following residual ratio:")
    for cf, r in sorted(report['by_cf'].items()):
        print(f"    {cf:<26}  {r:.3f}")
    print()


def print_grid_summary(rows):
    print("\n" + "=" * 100)
    print("  SURROGATE GRID SUMMARY")
    print("=" * 100)
    print(f"  total cells: {len(rows)}")
    # by family
    by_family = defaultdict(list)
    for r in rows:
        fam = CONTROLLER_FAMILY.get(r['controller'], 'other')
        by_family[fam].append(r['travel_time_s'])
    print(f"\n  travel time by controller family (mean):")
    for fam, xs in sorted(by_family.items()):
        print(f"    {fam:<14}  n={len(xs):>6}  mean={statistics.fmean(xs):>8.2f}")
    # by scenario
    by_scen = defaultdict(list)
    for r in rows:
        by_scen[r['scenario']].append(r['travel_time_s'])
    print(f"\n  travel time by communication scenario:")
    for s in sorted(by_scen):
        xs = by_scen[s]
        print(f"    {s:<6}  n={len(xs):>6}  mean={statistics.fmean(xs):>8.2f}")
    print()


def print_inversion_table(surr: Surrogate):
    print("\n" + "=" * 100)
    print("  INVERSION  —  what latency gives 10 % travel-time rise? (no simulation)")
    print("=" * 100)
    print(f"{'Controller':<26}{'CF':<8}{'Latency@+10 %':>16}")
    print('-' * 100)
    for ctrl in ('adaptive', 'coordinated', 'mpc', 'resilient-marl'):
        for cf in ('SAFE', 'GIPPS', 'IDM'):
            x = surr.invert_latency(ctrl, cf, target_pct=10.0)
            print(f"{ctrl:<26}{cf:<8}{x:>16.2f}")
    print()


def print_sensitivity(surr: Surrogate):
    """Simple one-at-a-time sensitivity of travel time to each axis."""
    print("\n" + "=" * 100)
    print("  SENSITIVITY  —  Δtravel_time per axis (resilient-marl, SAFE)")
    print("=" * 100)
    base_kw = dict(controller='resilient-marl', car_model='SAFE',
                   scenario='S2', natural='N-clear', demand='D1')
    from scenario_matrix import DEMAND_PATTERNS as DP
    d = DP['D1']
    base = surr.predict(**base_kw,
                        demand_ns=d['ns'], demand_ew=d['ew'])['travel_time_s']

    print(f"  baseline travel = {base:.2f} s\n")

    def delta(**overrides):
        kw = dict(base_kw)
        kw.update(overrides)
        d_ = DP[kw.pop('demand')]
        # re-add demand kw
        r = surr.predict(**kw, demand_ns=d_['ns'], demand_ew=d_['ew'])
        return r['travel_time_s'] - base

    # vary one axis at a time
    tests = [
        ('scenario S1', dict(scenario='S1')),
        ('scenario S3', dict(scenario='S3')),
        ('scenario S4', dict(scenario='S4')),
        ('natural N-snow',   dict(natural='N-snow')),
        ('natural N-event',  dict(natural='N-event')),
        ('natural N-hailstorm', dict(natural='N-hailstorm')),
        ('demand D4',        dict(demand='D4')),
        ('demand D5',        dict(demand='D5')),
    ]
    for label, ov in tests:
        print(f"  {label:<24}  Δ = {delta(**ov):>+8.2f} s")
    print()


def export_csv(rows, path='surrogate_grid.csv'):
    if not rows:
        return
    cols = sorted({k for r in rows for k in r})
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
    p.add_argument('--calibrate', action='store_true',
                   help='fit the surrogate from a sample of real sims')
    p.add_argument('--full', action='store_true',
                   help='run the full 5-axis grid (~160k cells)')
    p.add_argument('--invert', action='store_true',
                   help='invert the analytic curve for target percentages')
    p.add_argument('--sensitivity', action='store_true',
                   help='one-at-a-time sensitivity analysis')
    p.add_argument('--export', action='store_true',
                   help='write surrogate_grid.csv')
    return p.parse_args()


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    args = _parse_args()
    print("#" * 100)
    print("#  ANALYTICAL SURROGATE  —  every scenario without simulation")
    print("#" * 100)

    surr = Surrogate(seed=42)

    if args.calibrate:
        print("\n[1] Calibrating from real simulator ...")
        report = surr.calibrate(n_samples=30, n_steps=80)
        print_calibration(report)

    if args.invert:
        print_inversion_table(surr)
        return

    if args.sensitivity:
        print_sensitivity(surr)
        return

    # default: predict a slice and report
    print("\n[1] Predicting grid (no simulation) ...")
    from scenario_matrix import (
        register_mixed_controllers, register_extra_naturals,
        DEMAND_PATTERNS,
    )
    register_mixed_controllers()
    register_extra_naturals()

    if args.full:
        rows = surr.run_full_grid(demand_patterns=DEMAND_PATTERNS,
                                  progress=True)
    else:
        rows = surr.run_full_grid(
            controllers=('adaptive', 'coordinated', 'mpc', 'resilient-marl'),
            car_models=('SAFE', 'GIPPS', 'IDM'),
            scenarios=('S1', 'S2', 'S3', 'S4'),
            naturals=('N-clear', 'N-rain', 'N-snow', 'N-hailstorm',
                      'N-accident', 'N-rush-am', 'N-event',
                      'N-comm-blackout'),
            demands=('D1', 'D2', 'D4'),
            demand_patterns=DEMAND_PATTERNS,
            progress=True,
        )

    print_grid_summary(rows)

    if args.export:
        print("\n[2] CSV export ...")
        export_csv(rows, 'surrogate_grid.csv')

    print("\nDone.")


if __name__ == '__main__':
    main()