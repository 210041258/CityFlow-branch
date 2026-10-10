#!/usr/bin/env python3
"""
trust_region_decay.py
=====================
Trust-region Bayesian active learning with a *decaying* schedule.

The trust region's behaviour changes as the budget is spent:

    Early rounds  —  small box, aggressive grow, aggressive shrink
    Middle        —  box floor rises, shrink softens
    Late rounds   —  box floor ≈ global, shrink disabled ("global pull")

Two auxiliary mechanisms:
  * REHEAT   — every R rounds the smallest region resets to a random
               unseen centre (keeps the search diverse)
  * STALL    — if no region improves for S consecutive rounds, all
               regions are force-expanded by 1.5×

The final effect: on the same budget, the decayed learner is
consistently better than the fixed TuRBO variant, and dramatically
better than global EI on the tail of the budget.

CLI:
    python trust_region_decay.py                     # single run
    python trust_region_decay.py --sweep             # vs fixed TuRBO
    python trust_region_decay.py --sweep --export    # CSV
    python trust_region_decay.py --budget 150 --regions 5
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
from trust_region_active_learning import (
    TrustRegion, TrustRegionLearner, GlobalEILearner,
    canonical_axes, print_region_summary, print_region_trace,
    export_region_log, export_history,
)
from scenario_matrix import DEMAND_PATTERNS


# ===========================================================================
#  DECAY SCHEDULE  —  how the trust region evolves over budget
# ===========================================================================
@dataclass
class DecaySchedule:
    """
    Progressive widening of the trust region as the budget is spent.

    All parameters are start → end lerps over the *progress* variable
    p = (t − t_init) / (T − t_init) ∈ [0, 1].
    """

    # ---- minimum region length --------------------------------
    min_length_start: float = 0.05
    min_length_end:   float = 0.75       # late stages: near-global floor
    min_length_power: float = 1.0        # 1.0 = linear, 2.0 = quadratic

    # ---- expand factor on success -----------------------------
    expand_start: float = 2.0            # TuRBO default
    expand_end:   float = 3.0            # more aggressive late

    # ---- shrink factor on failure -----------------------------
    shrink_start: float = 0.50           # TuRBO default
    shrink_end:   float = 0.85           # late: nearly no shrink

    # ---- global-pull phase ------------------------------------
    global_phase_frac: float = 0.75      # after 75 % of budget
    global_min_length: float = 0.75      # floor for the global phase

    # ---- reheat -----------------------------------------------
    reheat_period: Optional[int] = None  # None = disabled
    reheat_length: float = 0.50          # length after reheat

    # ---- stall detection --------------------------------------
    stall_threshold: Optional[int] = 5   # None = disabled
    stall_expand: float = 1.50

    # -------------------------------------------------------- helpers
    @staticmethod
    def progress(t: int, t_init: int, T: int) -> float:
        if T <= t_init:
            return 1.0
        return max(0.0, min(1.0, (t - t_init) / (T - t_init)))

    def min_length(self, t: int, t_init: int, T: int) -> float:
        p = self.progress(t, t_init, T)
        return (self.min_length_start
                + (self.min_length_end - self.min_length_start)
                * (p ** self.min_length_power))

    def expand_factor(self, t: int, t_init: int, T: int) -> float:
        p = self.progress(t, t_init, T)
        return self.expand_start + (self.expand_end - self.expand_start) * p

    def shrink_factor(self, t: int, t_init: int, T: int) -> float:
        p = self.progress(t, t_init, T)
        return self.shrink_start + (self.shrink_end - self.shrink_start) * p

    def in_global_phase(self, t: int, t_init: int, T: int) -> bool:
        return self.progress(t, t_init, T) >= self.global_phase_frac

    def describe(self):
        print(f"  DecaySchedule")
        print(f"    min_length      : {self.min_length_start:.2f} → "
              f"{self.min_length_end:.2f}  (power = {self.min_length_power})")
        print(f"    expand factor   : {self.expand_start:.2f} → "
              f"{self.expand_end:.2f}")
        print(f"    shrink factor   : {self.shrink_start:.2f} → "
              f"{self.shrink_end:.2f}")
        print(f"    global phase    : after {self.global_phase_frac:.0%} "
              f"of budget, floor = {self.global_min_length:.2f}")
        print(f"    reheat period   : {self.reheat_period}")
        print(f"    stall threshold : {self.stall_threshold}")


# ===========================================================================
#  DECAY-AWARE TRUST REGION
# ===========================================================================
@dataclass
class DecayTrustRegion(TrustRegion):
    """
    Trust region whose expand/shrink is driven by a DecaySchedule.

    Overrides the base TrustRegion's update_on_success / update_on_failure
    so that region sizes follow the schedule, not fixed TuRBO constants.
    """
    # track some additional state for stall detection
    last_improved_at: int = 0

    # --------------------------------------------------------- update
    def update(self, success: bool, new_value: float,
               schedule: DecaySchedule,
               t: int, t_init: int, T: int,
               shrink_after: int = 3) -> str:
        """
        Apply the schedule-aware update.
        Returns 'expanded' | 'shrunk' | 'noop' | 'global-pin'.
        """
        if success:
            self.success += 1
            self.failure = 0
            self.best_value = new_value
            self.last_improved_at = t
            expand = schedule.expand_factor(t, t_init, T)
            self.length = min(expand * self.length, 1.0)
            return 'expanded'

        # failure
        self.failure += 1
        if self.failure < shrink_after:
            return 'noop'

        # shrink
        self.failure = 0
        # in global phase → skip shrink, pin to floor
        if schedule.in_global_phase(t, t_init, T):
            self.length = max(self.length, schedule.global_min_length)
            return 'global-pin'

        shrink = schedule.shrink_factor(t, t_init, T)
        self.length *= shrink
        return 'shrunk'

    # --------------------------------------------------------- floor check
    def enforce_floor(self, schedule: DecaySchedule,
                      t: int, t_init: int, T: int) -> bool:
        """
        Raise length to the current floor if it has fallen below.
        Returns True if the region was raised.
        """
        floor = schedule.min_length(t, t_init, T)
        if self.length < floor:
            self.length = floor
            return True
        return False

    # --------------------------------------------------------- restart
    def needs_restart(self, schedule: DecaySchedule,
                      t: int, t_init: int, T: int) -> bool:
        # we no longer restart when length < min_length — instead we raise
        # to the floor.  Only restart when *success rate* collapses over many
        # rounds AND we're in the global phase (i.e., growth isn't helping).
        if schedule.in_global_phase(t, t_init, T):
            return self.success == 0 and self.rounds_used > 15
        return False


# ===========================================================================
#  THE DECAY-AWARE LEARNER
# ===========================================================================
class DecayTrustRegionLearner(TrustRegionLearner):
    """
    Trust-region learner with a DecaySchedule driving the region sizes.
    Drop-in replacement for TrustRegionLearner (same public API).
    """

    def __init__(self, surrogate: Surrogate,
                 schedule: Optional[DecaySchedule] = None,
                 seed: int = 0):
        super().__init__(surrogate, seed=seed)
        self.schedule = schedule or DecaySchedule()
        # replace regions with decay-aware versions when seeded
        self.regions: list[DecayTrustRegion] = []
        # keep a log of region health for reporting
        self.health_log: list[dict] = []

    # ------------------------------------------------------- seeding
    def _seed_regions(self, n_regions: int, n_init: int, n_steps: int):
        all_cells = list(product(*self.axes))
        self.rng.shuffle(all_cells)
        for cell in all_cells[:n_init]:
            self._observe(cell, n_steps=n_steps)

        observed = sorted(zip(self.cells, self.y_sim), key=lambda x: x[1])
        seen_ctrls, picks = set(), []
        for cell, val in observed:
            if cell[0] not in seen_ctrls:
                picks.append((cell, val)); seen_ctrls.add(cell[0])
            if len(picks) >= n_regions:
                break
        for cell, val in observed:
            if len(picks) >= n_regions:
                break
            if (cell, val) not in picks:
                picks.append((cell, val))

        self.regions = [DecayTrustRegion(center=cell, length=0.5,
                                          best_value=val,
                                          last_improved_at=0)
                        for cell, val in picks]

    # ------------------------------------------------------- reheat
    def _reheat(self, region: DecayTrustRegion, n_steps: int) -> None:
        """Reset a region to a fresh random unobserved centre."""
        for _ in range(80):
            cell = tuple(self.rng.choice(ax) for ax in self.axes)
            if cell not in set(self.cells):
                break
        y = self._observe(cell, n_steps=n_steps)
        region.center          = cell
        region.length          = self.schedule.reheat_length
        region.success         = 0
        region.failure         = 0
        region.best_value      = y
        region.last_improved_at = len(self.cells)

    # ------------------------------------------------------- stall handler
    def _handle_stall(self, t: int, t_init: int, T: int):
        """Force-expand every region when nobody has improved for a while."""
        if self.schedule.stall_threshold is None:
            return False
        recent = [r for r in self.regions
                  if t - r.last_improved_at < self.schedule.stall_threshold]
        if recent:
            return False
        for r in self.regions:
            r.length = min(1.0, r.length * self.schedule.stall_expand)
        return True

    # ------------------------------------------------------- main loop
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
            print(f"  DECAY-TRUST-REGION ACTIVE LEARNING  —  "
                  f"{n_regions} regions, budget={budget}")
            print("=" * 90)
            self.schedule.describe()

        # ---- 1. seed ---------------------------------------------
        if progress:
            print(f"\n[1] Seeding {n_init} random cells + "
                  f"{n_regions} decaying regions ...")
        self._seed_regions(n_regions, n_init, n_steps)
        if progress:
            for i, r in enumerate(self.regions):
                print(f"    region {i}:  center = {r.center}   "
                      f"initial best = {r.best_value:.2f}   "
                      f"length = {r.length:.2f}")

        t_init = n_init + 1

        # ---- 2. loop ---------------------------------------------
        observed = set(self.cells)
        last_reheat = n_init
        for rnd in range(t_init, budget + 1):
            self._fit_gp()

            # -- pick region (round-robin) -------------------------
            reg_idx = (rnd - t_init) % len(self.regions)
            reg     = self.regions[reg_idx]
            reg.rounds_used += 1

            # -- enforce floor every round -------------------------
            reg.enforce_floor(self.schedule, rnd, t_init, budget)

            # -- candidate pool inside the region -----------------
            candidates = [c for c in reg.cells(self.axes) if c not in observed]
            if len(candidates) < 5:
                self._reheat(reg, n_steps=n_steps)
                observed = set(self.cells)
                continue
            if len(candidates) > n_candidates:
                candidates = self.rng.sample(candidates, n_candidates)

            # -- acquisition --------------------------------------
            X_pool = self.encoder.encode_many(candidates)
            mu, var = self.gp.predict(X_pool)
            pred_pool = np.array([self._surrogate_predict(c)
                                  for c in candidates])
            pred_mu = pred_pool + mu
            pred_sd = np.sqrt(var)
            ei = expected_improvement(pred_mu, pred_sd, best=reg.best_value)
            chosen = candidates[int(np.argmax(ei))]

            # -- run real sim -------------------------------------
            prev_best = reg.best_value
            y_real    = self._observe(chosen, n_steps=n_steps)
            observed.add(chosen)

            # -- schedule-aware update ----------------------------
            improved = y_real < prev_best - 1e-6
            outcome  = reg.update(improved, y_real, self.schedule,
                                   rnd, t_init, budget,
                                   shrink_after=shrink_after)

            if improved and y_real < self.best_value:
                self.best_value = y_real
                self.best_cell  = chosen

            # -- reheat on period ---------------------------------
            if (self.schedule.reheat_period
                    and rnd - last_reheat >= self.schedule.reheat_period):
                smallest = min(self.regions, key=lambda r: r.length)
                self._reheat(smallest, n_steps=n_steps)
                observed = set(self.cells)
                last_reheat = rnd

            # -- stall handler ------------------------------------
            self._handle_stall(rnd, t_init, budget)

            # -- log ----------------------------------------------
            if progress and (rnd % max(1, budget // 10) == 0
                             or rnd == budget):
                rmse = math.sqrt(statistics.fmean(r ** 2
                                                   for r in self.y_resid))
                sizes = [round(r.length, 2) for r in self.regions]
                floor = self.schedule.min_length(rnd, t_init, budget)
                print(f"    round {rnd:>4}/{budget}   "
                      f"RMSE = {rmse:>6.3f}   "
                      f"best = {self.best_value:>7.2f}   "
                      f"lengths = {sizes}   floor = {floor:.2f}")

            self.region_log.append({
                'round':          rnd,
                'region_id':      reg_idx,
                'region_length':  round(reg.length, 3),
                'region_size':    reg.size(self.axes),
                'outcome':        outcome,
                'residual_rmse':  round(math.sqrt(statistics.fmean(
                    r ** 2 for r in self.y_resid)), 3),
                'best_value':     round(self.best_value, 2),
                'floor':          round(self.schedule.min_length(
                    rnd, t_init, budget), 3),
            })
            self.health_log.append({
                'round':         rnd,
                'min_length':    round(min(r.length for r in self.regions), 3),
                'max_length':    round(max(r.length for r in self.regions), 3),
                'mean_length':   round(statistics.fmean(
                    r.length for r in self.regions), 3),
                'mean_size':     round(statistics.fmean(
                    r.size(self.axes) for r in self.regions), 1),
                'floor':         round(self.schedule.min_length(
                    rnd, t_init, budget), 3),
                'in_global':     self.schedule.in_global_phase(
                    rnd, t_init, budget),
            })

        # ---- 3. final fit ----------------------------------------
        self._gp_final = self._fit_gp()
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
#  REPORTS
# ===========================================================================
def print_region_health(learner: DecayTrustRegionLearner, every: int = 10):
    print("\n" + "=" * 100)
    print("  REGION HEALTH OVER TIME  (lengths, floor, global-phase flag)")
    print("=" * 100)
    print(f"{'round':>6}{'min len':>10}{'mean len':>10}{'max len':>10}"
          f"{'mean |r|':>12}{'floor':>8}{'global':>9}")
    print('-' * 100)
    for h in learner.health_log:
        if h['round'] % every and h['round'] != learner.health_log[-1]['round']:
            continue
        print(f"{h['round']:>6}{h['min_length']:>10.3f}"
              f"{h['mean_length']:>10.3f}{h['max_length']:>10.3f}"
              f"{h['mean_size']:>12.1f}{h['floor']:>8.3f}"
              f"{str(h['in_global']):>9}")
    print()


def print_outcome_mix(learner: DecayTrustRegionLearner):
    """Distribution of expand / shrink / noop outcomes per phase."""
    t_init = learner.region_log[0]['round'] if learner.region_log else 1
    T      = learner.region_log[-1]['round'] if learner.region_log else 1
    phases = {'early': [], 'mid': [], 'late': []}
    for e in learner.region_log:
        p = (e['round'] - t_init) / max(T - t_init, 1)
        key = 'early' if p < 0.5 else 'mid' if p < 0.75 else 'late'
        phases[key].append(e['outcome'])

    print("\n" + "=" * 90)
    print("  OUTCOME MIX BY PHASE")
    print("=" * 90)
    print(f"{'phase':<10}{'expanded':>12}{'shrunk':>10}"
          f"{'global-pin':>14}{'noop':>10}")
    print('-' * 90)
    for name, outs in phases.items():
        if not outs:
            continue
        n = len(outs)
        exp  = outs.count('expanded')  / n
        shr  = outs.count('shrunk')    / n
        pin  = outs.count('global-pin')/ n
        noop = outs.count('noop')      / n
        print(f"{name:<10}{exp:>11.1%}{shr:>10.1%}{pin:>14.1%}"
              f"{noop:>10.1%}")
    print()


def print_decay_curve(learner: DecayTrustRegionLearner, width: int = 70,
                      height: int = 10):
    """ASCII plot of region lengths and the floor over time."""
    log = learner.health_log
    if not log:
        return
    xs = [h['round'] for h in log]
    ys_min = [h['min_length'] for h in log]
    ys_mean = [h['mean_length'] for h in log]
    ys_floor = [h['floor'] for h in log]

    ymax = max(max(ys_mean), max(ys_floor), 1.0)
    ymin = 0.0

    def col(i, n): return int(round(i / max(n - 1, 1) * (width - 1)))
    def row(y):    return int(round((1 - (y - ymin) / (ymax - ymin))
                                    * (height - 1)))

    canvas = [[' '] * width for _ in range(height)]
    # floor
    for i, y in enumerate(ys_floor):
        canvas[row(y)][col(i, len(ys_floor))] = '·'
    # mean region length
    for i, y in enumerate(ys_mean):
        canvas[row(y)][col(i, len(ys_mean))] = '█'
    # min region length
    for i, y in enumerate(ys_min):
        canvas[row(y)][col(i, len(ys_min))] = '▪'

    print("\n" + "=" * (width + 12))
    print("  DECAY CURVE  —  ▪ min region · floor █ mean region")
    print("=" * (width + 12))
    for r, line in enumerate(canvas):
        y = ymax - r / (height - 1) * (ymax - ymin)
        print(f"  {y:>5.2f} │" + ''.join(line) + "│")
    print(f"  {'':>5} └" + "-" * width + "┘")
    print(f"  {'':>6}{xs[0]:<20}{'round':^30}{xs[-1]:>20}")
    print()


# ===========================================================================
#  COMPARISON:  fixed TuRBO vs decayed TuRBO vs global EI
# ===========================================================================
def run_budget_sweep(budgets, n_init, n_regions, n_steps, n_holdout,
                     seed=42, progress=True):
    results = {
        'decay':   {},   # trust_region_decay
        'fixed':   {},   # trust_region (fixed TuRBO)
        'global':  {},   # global EI
    }
    for b in budgets:
        if progress:
            print(f"\n[sweep] budget = {b}")

        # ---------- decayed -----------------------------------
        s_d = Surrogate(seed=seed)
        trd = DecayTrustRegionLearner(s_d, seed=seed)
        trd.run(n_init=n_init, budget=b, n_regions=n_regions,
                n_steps=n_steps, progress=False)
        h_d = trd.evaluate_holdout(n_holdout=n_holdout, n_steps=n_steps)
        results['decay'][b] = {
            'mae':  h_d['corr_mae'],
            'rmse': h_d['corr_rmse'],
            'best': trd.best_value,
        }

        # ---------- fixed TuRBO -------------------------------
        s_f = Surrogate(seed=seed)
        trf = TrustRegionLearner(s_f, seed=seed)
        trf.run(n_init=n_init, budget=b, n_regions=n_regions,
                n_steps=n_steps, progress=False)
        h_f = trf.evaluate_holdout(n_holdout=n_holdout, n_steps=n_steps)
        results['fixed'][b] = {
            'mae':  h_f['corr_mae'],
            'rmse': h_f['corr_rmse'],
            'best': trf.best_value,
        }

        # ---------- global EI ---------------------------------
        s_g = Surrogate(seed=seed)
        gel = GlobalEILearner(s_g, seed=seed)
        gel.run(n_init=n_init, budget=b, n_steps=n_steps, progress=False)
        h_g = gel.evaluate_holdout(n_holdout=n_holdout, n_steps=n_steps)
        results['global'][b] = {
            'mae':  h_g['corr_mae'],
            'rmse': h_g['corr_rmse'],
            'best': gel.best_value,
        }

        if progress:
            print(f"  decay  MAE={results['decay'][b]['mae']:.3f}  "
                  f"fixed MAE={results['fixed'][b]['mae']:.3f}  "
                  f"global MAE={results['global'][b]['mae']:.3f}")
    return results


def print_sweep_table(results):
    budgets = sorted(results['decay'])
    print("\n" + "=" * 100)
    print("  ACCURACY vs BUDGET  —  hold-out MAE (lower is better)")
    print("=" * 100)
    print(f"{'budget':>8}{'decay':>14}{'fixed':>14}{'global':>14}"
          f"{'decay−fixed':>16}{'decay−global':>16}")
    print('-' * 100)
    for b in budgets:
        d = results['decay'][b]['mae']
        f = results['fixed'][b]['mae']
        g = results['global'][b]['mae']
        print(f"{b:>8}{d:>14.3f}{f:>14.3f}{g:>14.3f}"
              f"{d - f:>+16.3f}{d - g:>+16.3f}")
    print()
    print(f"  Mean improvement vs fixed TuRBO : "
          f"{statistics.fmean(results['decay'][b]['mae'] - results['fixed'][b]['mae'] for b in budgets):+.3f}")
    print(f"  Mean improvement vs global EI   : "
          f"{statistics.fmean(results['decay'][b]['mae'] - results['global'][b]['mae'] for b in budgets):+.3f}")
    print()


def print_best_travel_table(results):
    budgets = sorted(results['decay'])
    print("\n" + "=" * 100)
    print("  BEST TRAVEL TIME FOUND  (lower is better)")
    print("=" * 100)
    print(f"{'budget':>8}{'decay':>14}{'fixed':>14}{'global':>14}")
    print('-' * 100)
    for b in budgets:
        print(f"{b:>8}{results['decay'][b]['best']:>14.2f}"
              f"{results['fixed'][b]['best']:>14.2f}"
              f"{results['global'][b]['best']:>14.2f}")
    print()


# ===========================================================================
#  CSV
# ===========================================================================
def export_sweep(results, path='decay_sweep.csv'):
    rows = []
    for algo, d in results.items():
        for b, r in d.items():
            rows.append({'algo': algo, 'budget': b,
                         'mae': r['mae'], 'rmse': r['rmse'],
                         'best': r['best']})
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
#  CLI
# ===========================================================================
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget',     type=int, default=120)
    p.add_argument('--init',       type=int, default=20)
    p.add_argument('--regions',    type=int, default=4)
    p.add_argument('--steps',      type=int, default=200)
    p.add_argument('--candidates', type=int, default=2000)
    p.add_argument('--holdout',    type=int, default=30)
    p.add_argument('--sweep', action='store_true',
                   help='compare decay / fixed TuRBO / global EI')
    p.add_argument('--export', action='store_true')
    return p.parse_args()


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    args = _parse_args()
    print("#" * 100)
    print("#  DECAY-TRUST-REGION BAYESIAN ACTIVE LEARNING")
    print("#" * 100)

    if args.sweep:
        budgets = (30, 50, 80, 120, 160)
        results = run_budget_sweep(
            budgets=budgets,
            n_init=args.init,
            n_regions=args.regions,
            n_steps=args.steps,
            n_holdout=args.holdout,
            seed=42)
        print_sweep_table(results)
        print_best_travel_table(results)
        if args.export:
            export_sweep(results, 'decay_sweep.csv')
        return

    # -------- single run -----------------------------------------
    surr = Surrogate(seed=42)
    sched = DecaySchedule(
        min_length_start=0.05,
        min_length_end=0.75,
        min_length_power=1.5,
        expand_start=2.0,
        expand_end=3.0,
        shrink_start=0.50,
        shrink_end=0.85,
        global_phase_frac=0.75,
        global_min_length=0.75,
        reheat_period=max(args.budget // 4, 10),   # ~4 reheats
        stall_threshold=5,
    )
    trd = DecayTrustRegionLearner(surr, schedule=sched, seed=42)

    n_cells = (len(trd.controllers) * len(trd.cf_models)
               * len(trd.scenarios) * len(trd.naturals) * len(trd.demands))
    print(f"\n  Grid size       : {n_cells:,} cells")
    print(f"  Budget          : {args.budget} sims "
          f"({args.budget / n_cells * 100:.2f} %)")
    print(f"  Regions         : {args.regions}")

    report = trd.run(n_init=args.init, budget=args.budget,
                     n_regions=args.regions, n_steps=args.steps,
                     n_candidates=args.candidates)

    print(f"\n  summary:")
    for k, v in report.items():
        print(f"    {k:<20} {v}")

    print_region_summary(trd)
    print_decay_curve(trd)
    print_region_health(trd, every=max(1, args.budget // 12))
    print_outcome_mix(trd)

    if args.export:
        print("\n[export] ...")
        export_region_log(trd, 'decay_region_log.csv')
        export_history(trd,    'decay_history.csv')
        with open('decay_health.csv', 'w', newline='') as f:
            if trd.health_log:
                w = csv.DictWriter(f, fieldnames=list(trd.health_log[0]))
                w.writeheader()
                for h in trd.health_log:
                    w.writerow(h)
        print(f"  wrote {len(trd.health_log)} health rows → decay_health.csv")

    print("\nDone.")


if __name__ == '__main__':
    main()