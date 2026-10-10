#!/usr/bin/env python3
"""
trust_region_active_learning.py
===============================
TuRBO-style trust-region Bayesian active learning for the 5-axis
scenario grid.

Idea
----
Instead of maximising Expected Improvement over the FULL grid, we
maintain  N_regions  small hyper-rectangles ("trust regions"),
each centred on a currently-promising cell.  Each round, we choose one
region, restrict the candidate pool to cells *inside* that region, and
pick the highest-EI candidate there.

Region update rules (standard TuRBO):
  * success  →  length ×= 2.0    (grow the box)
  * failure  →  length ×= 0.5  after 3 consecutive failures
              (shrink the box; restart if length < 0.05)

Because each region holds a *local* GP posterior, the model is much
more accurate inside the box than globally — every simulation spent
buys more information.

CLI:
    python trust_region_active_learning.py                 # default
    python trust_region_active_learning.py --budget 80     # 80 sims
    python trust_region_active_learning.py --regions 6     # 6 boxes
    python trust_region_active_learning.py --compare       # vs global EI
    python trust_region_active_learning.py --export

Programmatic:
    from trust_region_active_learning import TrustRegionLearner, Surrogate
    s = Surrogate()
    trl = TrustRegionLearner(s)
    trl.run(n_init=20, budget=100, n_regions=4)
    print(trl.predict('resilient-marl', 'SAFE', 'S3', 'N-hailstorm', 'D1'))
"""

from __future__ import annotations
import argparse
import csv
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np

from traffic_control_algorithm import (
    TrafficControlAlgorithm,
    COMMUNICATION_SCENARIOS,
    NATURAL_SCENARIOS,
    CAR_FOLLOWING_BENCHMARK,
    simulate_episode,
)
from analytical_surrogate import Surrogate, CONTROLLER_FAMILY
from bayesian_active_learning import (
    BayesianActiveLearner, Encoder, SurrogateGP,
    expected_improvement,
)
from scenario_matrix import (
    register_mixed_controllers, register_extra_naturals,
    DEMAND_PATTERNS,
)


# ===========================================================================
#  CANONICAL AXIS ORDERING  —  for local neighbourhoods to make sense
# ===========================================================================
def canonical_axes():
    """Return (controllers, cf, scenarios, naturals, demands) in a
    semantically ordered tuple — neighbours in the list = neighbours
    in the trust-region sense."""
    # controllers — group by family then sort
    fam_rank = {'baseline': 0, 'adaptive': 1, 'coordinated': 2,
                'learning': 3, 'field': 4, 'resilient': 5}
    ctrls = sorted(CONTROLLER_FAMILY,
                   key=lambda c: (fam_rank.get(CONTROLLER_FAMILY[c], 9), c))
    # car-following — ordered by empirical similarity
    cf = ('SAFE', 'IDM', 'GIPPS')
    # comm scenarios — increasing impairment
    sc = ('S1', 'S2', 'S3', 'S4')
    # naturals — group by family then alphabetical
    nats_by_fam = {
        'weather':  ['N-clear', 'N-rain', 'N-snow', 'N-fog',
                     'N-hailstorm', 'N-flooding'],
        'incident': ['N-accident', 'N-workzone', 'N-major-crash'],
        'demand':   ['N-rush-am', 'N-rush-pm', 'N-night',
                     'N-event', 'N-school', 'N-holiday', 'N-freight'],
        'failure':  ['N-sensor-fail', 'N-comm-blackout', 'N-power-outage'],
    }
    nats = tuple(n for fam in ('weather', 'incident', 'demand', 'failure')
                 for n in nats_by_fam[fam]
                 if n in NATURAL_SCENARIOS)
    # demands
    dem = tuple(sorted(DEMAND_PATTERNS))
    return ctrls, cf, sc, nats, dem


# ===========================================================================
#  TRUST REGION  —  a small hyper-rectangle over the discrete grid
# ===========================================================================
@dataclass
class TrustRegion:
    """
    One local search box over the 5-axis grid.

    The box is stored as a *centre cell* plus a *length* ∈ (0, 1].
    The allowed values per axis are the `ceil(length * n_values)` values
    of that axis closest (in list order) to the centre's value.
    """
    center:  tuple
    length:  float = 0.5
    success: int = 0
    failure: int = 0
    best_value: float = float('inf')
    rounds_used: int = 0

    # ------------------------------------------------------------- internals
    def _window(self, axis_values, center_value):
        """Return the sub-list of `axis_values` around `center_value`."""
        if center_value not in axis_values:
            return list(axis_values)
        n = len(axis_values)
        k = max(1, math.ceil(self.length * n))
        idx = axis_values.index(center_value)
        lo = max(0, idx - (k - 1) // 2)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        return list(axis_values[lo:hi])

    def cells(self, axes):
        """Iterate over every cell inside this region."""
        lists = [self._window(axes[i], self.center[i]) for i in range(5)]
        return product(*lists)

    def size(self, axes):
        """Number of cells in this region."""
        n = 1
        for i in range(5):
            n *= len(self._window(axes[i], self.center[i]))
        return n

    def update_on_success(self, new_value: float):
        self.success += 1
        self.failure = 0
        self.best_value = new_value
        # expand
        self.length = min(2.0 * self.length, 1.0)

    def update_on_failure(self, shrink_after: int = 3):
        self.failure += 1
        if self.failure >= shrink_after:
            self.length *= 0.5
            self.failure = 0
        return self.length

    def needs_restart(self, min_length: float = 0.05) -> bool:
        return self.length < min_length


# ===========================================================================
#  THE TRUST-REGION LEARNER
# ===========================================================================
class TrustRegionLearner(BayesianActiveLearner):
    """
    TuRBO-style learner.  Subclasses BayesianActiveLearner to reuse:
      * _observe, _fit_gp, _surrogate_predict, corrected_predict
      * the Encoder, the GP, and the training history.
    """

    def __init__(self, surrogate: Surrogate, seed: int = 0):
        super().__init__(surrogate, seed=seed)
        # use canonical axis ordering for locality
        (self.controllers,
         self.cf_models,
         self.scenarios,
         self.naturals,
         self.demands) = canonical_axes()
        self.axes = (self.controllers, self.cf_models,
                     self.scenarios, self.naturals, self.demands)
        # re-init encoder over the canonical axes
        self.encoder = Encoder(*self.axes)
        self.gp      = SurrogateGP(self.encoder.n_features)

        # trust region state
        self.regions: list[TrustRegion] = []
        self.region_log: list[dict] = []

    # ------------------------------------------------------------- seed
    def _seed_regions(self, n_regions: int, n_init: int, n_steps: int):
        """
        Run n_init random cells, then spawn n_regions trust regions
        around the best distinct cells.
        """
        all_cells = list(product(*self.axes))
        self.rng.shuffle(all_cells)
        for cell in all_cells[:n_init]:
            self._observe(cell, n_steps=n_steps)

        # pick the best cells, ensuring diversity (different controllers)
        observed = sorted(zip(self.cells, self.y_sim),
                          key=lambda x: x[1])
        seen_ctrls = set()
        picks = []
        for cell, val in observed:
            if cell[0] not in seen_ctrls:
                picks.append((cell, val))
                seen_ctrls.add(cell[0])
            if len(picks) >= n_regions:
                break
        # fill any remaining slots with the next best cells
        for cell, val in observed:
            if len(picks) >= n_regions:
                break
            if (cell, val) not in picks:
                picks.append((cell, val))

        self.regions = [TrustRegion(center=cell, length=0.5,
                                    best_value=val) for cell, val in picks]

    # ------------------------------------------------------------- restart
    def _restart_region(self, reg: TrustRegion, n_steps: int):
        """Reset a collapsed region to a fresh random cell."""
        # prefer something we haven't seen
        for _ in range(50):
            cell = tuple(self.rng.choice(ax) for ax in self.axes)
            if cell not in set(self.cells):
                break
        y = self._observe(cell, n_steps=n_steps)
        reg.center       = cell
        reg.length       = 0.5
        reg.success      = 0
        reg.failure      = 0
        reg.best_value   = y
        reg.rounds_used += 1

    # ------------------------------------------------------------- loop
    def run(self,
            n_init: int = 20,
            budget: int = 100,
            n_regions: int = 4,
            n_steps: int = 200,
            n_candidates: int = 2000,
            shrink_after: int = 3,
            progress: bool = True) -> dict:

        t0 = time.time()
        if progress:
            print("\n" + "=" * 90)
            print(f"  TRUST-REGION ACTIVE LEARNING  —  "
                  f"{n_regions} regions, budget={budget} sims")
            print("=" * 90)

        # ---- 1. seed -----------------------------------------------
        if progress:
            print(f"\n[1] Seeding {n_init} random cells + "
                  f"{n_regions} trust regions ...")
        self._seed_regions(n_regions, n_init, n_steps)
        if progress:
            for i, r in enumerate(self.regions):
                print(f"    region {i}:  center = {r.center}   "
                      f"initial best = {r.best_value:.2f}   "
                      f"length = {r.length:.2f}")

        # ---- 2. loop -----------------------------------------------
        observed = set(self.cells)
        for rnd in range(n_init + 1, budget + 1):
            self._fit_gp()

            # ---- choose a region: round-robin ------------------
            reg = self.regions[(rnd - n_init - 1) % len(self.regions)]

            # ---- candidate pool from region --------------------
            pool_all = list(reg.cells(self.axes))
            candidates = [c for c in pool_all if c not in observed]

            if len(candidates) < 5:
                # region has shrunk past usefulness → restart
                self._restart_region(reg, n_steps=n_steps)
                observed = set(self.cells)
                continue

            # ---- sub-sample if too many -------------------------
            if len(candidates) > n_candidates:
                candidates = self.rng.sample(candidates, n_candidates)

            # ---- acquisition ------------------------------------
            X_pool = self.encoder.encode_many(candidates)
            mu, var = self.gp.predict(X_pool)
            pred_pool = np.array([self._surrogate_predict(c)
                                  for c in candidates])
            pred_mu = pred_pool + mu
            pred_sd = np.sqrt(var)
            ei = expected_improvement(pred_mu, pred_sd,
                                       best=reg.best_value)
            idx = int(np.argmax(ei))
            chosen = candidates[idx]

            # ---- run real sim -----------------------------------
            y_real = self._observe(chosen, n_steps=n_steps)
            observed.add(chosen)
            reg.rounds_used += 1

            # ---- update trust region ----------------------------
            if y_real < reg.best_value - 1e-6:
                reg.update_on_success(y_real)
                if y_real < self.best_value:
                    self.best_value = y_real
                    self.best_cell  = chosen
            else:
                reg.update_on_failure(shrink_after=shrink_after)
                if reg.needs_restart():
                    self._restart_region(reg, n_steps=n_steps)
                    observed = set(self.cells)

            # ---- log --------------------------------------------
            if progress and (rnd % max(1, budget // 10) == 0
                             or rnd == budget):
                rmse = math.sqrt(statistics.fmean(
                    r ** 2 for r in self.y_resid))
                reg_sizes = [r.size(self.axes) for r in self.regions]
                print(f"    round {rnd:>4}/{budget}   "
                      f"RMSE = {rmse:>6.3f}   "
                      f"best = {self.best_value:>7.2f}   "
                      f"|regions| = {reg_sizes}")

            self.region_log.append({
                'round':           rnd,
                'chosen':          chosen,
                'region_id':       self.regions.index(reg),
                'region_length':   round(reg.length, 3),
                'region_size':     reg.size(self.axes),
                'residual_rmse':   round(math.sqrt(statistics.fmean(
                    r ** 2 for r in self.y_resid)), 3),
                'best_value':      round(self.best_value, 2),
            })

        # ---- 3. final fit ------------------------------------------
        self._fit_gp()
        elapsed = time.time() - t0

        return {
            'n_sims':         len(self.cells),
            'n_regions':      n_regions,
            'best_cell':      self.best_cell,
            'best_value':     round(self.best_value, 2),
            'residual_rmse':  round(math.sqrt(statistics.fmean(
                r ** 2 for r in self.y_resid)), 3),
            'residual_mae':   round(statistics.fmean(
                abs(r) for r in self.y_resid), 3),
            'elapsed_s':      round(elapsed, 2),
        }


# ===========================================================================
#  GLOBAL EI LEARNER (baseline — same interface as TrustRegionLearner)
# ===========================================================================
class GlobalEILearner(BayesianActiveLearner):
    """Vanilla active learning — EI over the full grid (no regions)."""

    def __init__(self, surrogate: Surrogate, seed: int = 0):
        super().__init__(surrogate, seed=seed)
        (self.controllers,
         self.cf_models,
         self.scenarios,
         self.naturals,
         self.demands) = canonical_axes()
        self.axes = (self.controllers, self.cf_models,
                     self.scenarios, self.naturals, self.demands)
        self.encoder = Encoder(*self.axes)
        self.gp      = SurrogateGP(self.encoder.n_features)


# ===========================================================================
#  REPORTS
# ===========================================================================
def print_region_summary(learner: TrustRegionLearner):
    print("\n" + "=" * 90)
    print("  TRUST-REGION FINAL STATE")
    print("=" * 90)
    print(f"{'id':>3}{'length':>10}{'size (cells)':>14}"
          f"{'success':>10}{'best':>10}{'center':<50}")
    print('-' * 90)
    for i, r in enumerate(learner.regions):
        print(f"{i:>3}{r.length:>10.3f}{r.size(learner.axes):>14}"
              f"{r.success:>10}{r.best_value:>10.2f}"
              f"{str(r.center):<50}")
    print()


def print_region_trace(learner: TrustRegionLearner):
    print("\n" + "=" * 90)
    print("  TRUST-REGION TRACE  (round-by-round)")
    print("=" * 90)
    print(f"{'round':>6}{'region':>8}{'len':>8}{'size':>8}"
          f"{'RMSE':>10}{'best':>10}   chosen")
    print('-' * 90)
    for r in learner.region_log:
        ch = r['chosen']
        ch_s = f"{ch[0]}/{ch[1]}/{ch[2]}/{ch[3]}/{ch[4]}"
        print(f"{r['round']:>6}{r['region_id']:>8}"
              f"{r['region_length']:>8.3f}{r['region_size']:>8}"
              f"{r['residual_rmse']:>10.3f}{r['best_value']:>10.2f}"
              f"   {ch_s}")
    print()


def print_holdout_comparison(learner, label, n_holdout=30, n_steps=200):
    print(f"\n[held-out] {label}: sampling {n_holdout} fresh cells ...")
    h = learner.evaluate_holdout(n_holdout=n_holdout, n_steps=n_steps)
    print(f"  raw surrogate :  MAE = {h['raw_mae']:>6.3f}   "
          f"RMSE = {h['raw_rmse']:>6.3f}")
    print(f"  corrected     :  MAE = {h['corr_mae']:>6.3f}   "
          f"RMSE = {h['corr_rmse']:>6.3f}")
    return h


def print_head_to_head(tr_report, gl_report, tr_hold, gl_hold):
    print("\n" + "=" * 90)
    print("  HEAD-TO-HEAD  —  trust region vs global EI")
    print("=" * 90)
    print(f"{'metric':<24}{'Trust-Region':>16}{'Global EI':>16}"
          f"{'Δ':>12}")
    print('-' * 90)
    rows = [
        ('n simulations',    tr_report['n_sims'],        gl_report['n_sims']),
        ('best travel (s)',  tr_report['best_value'],    gl_report['best_value']),
        ('residual RMSE',    tr_report['residual_rmse'], gl_report['residual_rmse']),
        ('residual MAE',     tr_report['residual_mae'],  gl_report['residual_mae']),
        ('holdout MAE',      tr_hold['corr_mae'],        gl_hold['corr_mae']),
        ('holdout RMSE',     tr_hold['corr_rmse'],       gl_hold['corr_rmse']),
        ('wall-clock (s)',   tr_report['elapsed_s'],     gl_report['elapsed_s']),
    ]
    for label, a, b in rows:
        delta = a - b
        print(f"{label:<24}{a:>16.3f}{b:>16.3f}{delta:>+12.3f}")
    print()


def print_accuracy_vs_budget(curve_tr, curve_gl):
    print("\n" + "=" * 90)
    print("  ACCURACY vs BUDGET  —  hold-out MAE")
    print("=" * 90)
    print(f"{'budget':>8}{'Trust-Region':>18}{'Global EI':>18}{'Δ':>12}")
    print('-' * 90)
    for budget in sorted(set(curve_tr) & set(curve_gl)):
        a = curve_tr[budget]
        b = curve_gl[budget]
        print(f"{budget:>8}{a:>18.3f}{b:>18.3f}{a-b:>+12.3f}")
    print()


# ===========================================================================
#  CSV
# ===========================================================================
def export_region_log(learner, path='trust_region_log.csv'):
    if not learner.region_log:
        return
    cols = list(learner.region_log[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in learner.region_log:
            w.writerow(r)
    print(f"  wrote {len(learner.region_log)} rows → {path}")


def export_history(learner, path='trust_region_history.csv'):
    rows = []
    for i, c in enumerate(learner.cells):
        rows.append({
            'controller':    c[0],
            'car_following': c[1],
            'scenario':      c[2],
            'natural':       c[3],
            'demand':        c[4],
            'y_sim':         round(float(learner.y_sim[i]), 3),
            'y_surrogate':   round(float(learner.y_pred[i]), 3),
            'residual':      round(float(learner.y_resid[i]), 3),
        })
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {len(rows)} rows → {path}")


# ===========================================================================
#  BUDGET SWEEP  —  trust region vs global EI on the same seeds
# ===========================================================================
def run_budget_sweep(budgets, n_init, n_regions, n_steps,
                     n_holdout, seed=42, progress=True):
    curve_tr, curve_gl = {}, {}
    for b in budgets:
        if progress:
            print(f"\n[sweep] budget = {b}")

        # ---- trust region ----------------------------------------
        s_tr  = Surrogate(seed=seed)
        trl   = TrustRegionLearner(s_tr, seed=seed)
        trl.run(n_init=n_init, budget=b, n_regions=n_regions,
                n_steps=n_steps, progress=False)
        # evaluate on the SAME holdout pool as global EI
        tr_hold = trl.evaluate_holdout(n_holdout=n_holdout, n_steps=n_steps)
        curve_tr[b] = tr_hold['corr_mae']

        # ---- global EI -------------------------------------------
        s_gl  = Surrogate(seed=seed)
        gel   = GlobalEILearner(s_gl, seed=seed)
        gel.run(n_init=n_init, budget=b, n_steps=n_steps, progress=False)
        gl_hold = gel.evaluate_holdout(n_holdout=n_holdout, n_steps=n_steps)
        curve_gl[b] = gl_hold['corr_mae']

        if progress:
            print(f"  TR MAE = {curve_tr[b]:.3f}   "
                  f"GL MAE = {curve_gl[b]:.3f}")
    return curve_tr, curve_gl


# ===========================================================================
#  CLI
# ===========================================================================
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget', type=int, default=100)
    p.add_argument('--init',   type=int, default=20)
    p.add_argument('--regions', type=int, default=4)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--candidates', type=int, default=2000)
    p.add_argument('--holdout', type=int, default=30)
    p.add_argument('--compare', action='store_true',
                   help='run a head-to-head against global EI')
    p.add_argument('--sweep', action='store_true',
                   help='accuracy vs budget sweep (TR vs GL)')
    p.add_argument('--export', action='store_true')
    return p.parse_args()


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    args = _parse_args()
    print("#" * 100)
    print("#  TRUST-REGION BAYESIAN ACTIVE LEARNING  —  TuRBO over the grid")
    print("#" * 100)

    # ------------------------------------------------------------- sweep
    if args.sweep:
        budgets = (30, 50, 80, 120, 160)
        tr_curve, gl_curve = run_budget_sweep(
            budgets=budgets,
            n_init=args.init,
            n_regions=args.regions,
            n_steps=args.steps,
            n_holdout=args.holdout,
            seed=42)
        print_accuracy_vs_budget(tr_curve, gl_curve)
        return

    # ------------------------------------------------------------- single
    surr = Surrogate(seed=42)
    trl  = TrustRegionLearner(surr, seed=42)

    n_cells = (len(trl.controllers) * len(trl.cf_models)
               * len(trl.scenarios) * len(trl.naturals) * len(trl.demands))
    print(f"\n  Grid size       : {n_cells:,} cells")
    print(f"  Budget          : {args.budget} real simulations "
          f"({args.budget / n_cells * 100:.2f} %)")
    print(f"  Regions         : {args.regions}")
    print(f"  Steps per sim   : {args.steps}")

    tr_report = trl.run(n_init=args.init, budget=args.budget,
                        n_regions=args.regions, n_steps=args.steps,
                        n_candidates=args.candidates)
    print(f"\n  summary:")
    for k, v in tr_report.items():
        print(f"    {k:<20} {v}")

    print_region_summary(trl)
    print_region_trace(trl)

    # ------------------------------------------------------------- compare
    if args.compare:
        print("\n[compare] running global EI with same budget ...")
        surr_gl = Surrogate(seed=42)
        gel     = GlobalEILearner(surr_gl, seed=42)
        gl_report = gel.run(n_init=args.init, budget=args.budget,
                            n_steps=args.steps,
                            n_candidates=args.candidates,
                            progress=True)

        tr_hold = trl.evaluate_holdout(n_holdout=args.holdout,
                                       n_steps=args.steps)
        gl_hold = gel.evaluate_holdout(n_holdout=args.holdout,
                                       n_steps=args.steps)
        print_head_to_head(tr_report, gl_report, tr_hold, gl_hold)

    # ------------------------------------------------------------- export
    if args.export:
        print("\n[export] ...")
        export_region_log(trl, 'trust_region_log.csv')
        export_history(trl,    'trust_region_history.csv')

    print("\nDone.")


if __name__ == '__main__':
    main()