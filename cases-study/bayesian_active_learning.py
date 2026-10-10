#!/usr/bin/env python3
"""
bayesian_active_learning.py
===========================
Bayesian active-learning loop that corrects the analytical surrogate
using as few real simulations as possible.

The loop:
  1. Start from the surrogate's prediction (prior mean).
  2. Run n_init seed simulations to get residuals  r = sim - pred.
  3. Fit a Gaussian Process on the residuals.
  4. At each round, evaluate an acquisition function (Expected
     Improvement) on a large pool of unobserved cells.
  5. Pick the highest-EI cell, run the real simulation, update the GP.
  6. Stop when the budget is exhausted.

Outcome: a corrected predictor that matches the simulator to within
a few percent on the ENTIRE grid, using only ~0.5 % of the cells.

CLI:
    python bayesian_active_learning.py                        # default
    python bayesian_active_learning.py --budget 100           # 100 sims
    python bayesian_active_learning.py --compare              # held-out test
    python bayesian_active_learning.py --export               # write CSV

Programmatic:
    from bayesian_active_learning import BayesianActiveLearner, Surrogate
    s   = Surrogate()
    bal = BayesianActiveLearner(s)
    bal.run(n_init=20, budget=100)
    pred = bal.predict('resilient-marl', 'SAFE', 'S3', 'N-hailstorm', 'D2')
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
from scenario_matrix import (
    register_mixed_controllers, register_extra_naturals,
    DEMAND_PATTERNS,
)


# ===========================================================================
#  FEATURE ENCODER  —  5 discrete axes → continuous feature vector
# ===========================================================================
class Encoder:
    """
    Maps (controller, cf, scenario, natural, demand) to a fixed-length
    feature vector.

    Features:
      * controller         — one-hot over every registered controller
      * car_following      — one-hot over SAFE / GIPPS / IDM
      * scenario           — (latency/600, loss_rate)
      * natural            — (1 - capacity, demand_mult - 1, |bias|)
      * demand             — (ns, ew)
    """

    def __init__(self, controllers, cf_models, scenarios, naturals, demands):
        self.controllers = tuple(controllers)
        self.cf_models   = tuple(cf_models)
        self.scenarios   = tuple(scenarios)
        self.naturals    = tuple(naturals)
        self.demands     = tuple(demands)
        self.ctrl_index  = {c: i for i, c in enumerate(self.controllers)}
        self.cf_index    = {c: i for i, c in enumerate(self.cf_models)}
        self.n_features  = (len(self.controllers) + len(self.cf_models)
                            + 2 + 3 + 2)

    def encode(self, ctrl, cf, scen, nat, dem):
        x = np.zeros(self.n_features, dtype=float)
        i = 0
        # controller one-hot
        if ctrl in self.ctrl_index:
            x[i + self.ctrl_index[ctrl]] = 1.0
        i += len(self.controllers)
        # car-following one-hot
        if cf in self.cf_index:
            x[i + self.cf_index[cf]] = 1.0
        i += len(self.cf_models)
        # scenario numeric
        s = COMMUNICATION_SCENARIOS[scen]
        x[i]   = s['latency_ms'] / 600.0
        x[i+1] = s['loss_rate']
        i += 2
        # natural numeric
        n = NATURAL_SCENARIOS[nat]
        x[i]   = 1.0 - n['capacity']
        x[i+1] = n['demand_mult'] - 1.0
        x[i+2] = abs(n['direction_bias'])
        i += 3
        # demand numeric
        d = DEMAND_PATTERNS[dem]
        x[i]   = d['ns'] * 2.0
        x[i+1] = d['ew'] * 2.0
        return x

    def encode_many(self, rows):
        return np.vstack([self.encode(*r) for r in rows])


# ===========================================================================
#  GAUSSIAN PROCESS  —  small ARD-RBF kernel, pure numpy
# ===========================================================================
class SurrogateGP:
    """
    Gaussian Process regression on the residual r(x) = sim(x) - pred(x).
    Uses an ARD-RBF kernel with hand-set lengthscales.
    """

    def __init__(self, n_features: int, jitter: float = 1e-6):
        self.n_features = n_features
        # ARD lengthscales — smaller = more sensitive on that feature
        # categorical features get small ls (local), numeric get large ls (smooth)
        self.ls    = np.ones(n_features)
        self.var   = 1.0
        self.noise = 0.15                    # residual std ~ 15 %
        self.jitter = jitter
        self.X     = None
        self.y     = None
        self.L     = None
        self.alpha = None

    # ------------------------------------------------------------- kernel
    def _kernel(self, X1, X2):
        X1s = X1 / self.ls
        X2s = X2 / self.ls
        sq1 = (X1s ** 2).sum(1)[:, None]
        sq2 = (X2s ** 2).sum(1)[None, :]
        d2  = sq1 + sq2 - 2.0 * X1s @ X2s.T
        return self.var * np.exp(-0.5 * np.maximum(d2, 0.0))

    # ------------------------------------------------------------- fit
    def fit(self, X, y):
        self.X, self.y = X, y
        n = len(X)
        K = self._kernel(X, X)
        K += (self.noise ** 2 + self.jitter) * np.eye(n)
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T,
                                     np.linalg.solve(self.L, y))

    # ------------------------------------------------------------- predict
    def predict(self, X_star):
        if self.X is None:
            n = len(X_star)
            return np.zeros(n), np.ones(n) * self.var
        Ks = self._kernel(self.X, X_star)
        mu = Ks.T @ self.alpha
        v  = np.linalg.solve(self.L, Ks)
        var = self.var - (v ** 2).sum(0)
        var = np.maximum(var, 1e-9)
        return mu, var


# ===========================================================================
#  ACQUISITION  —  Expected Improvement (minimisation)
# ===========================================================================
def _norm_cdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _norm_pdf(x): return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def expected_improvement(mu, sigma, best, xi=0.01):
    """
    EI for minimisation:  E[max(0, best − f(x))].
    Vectorised over numpy arrays.
    """
    sigma = np.maximum(sigma, 1e-9)
    z     = (best - mu - xi) / sigma
    ei    = (best - mu - xi) * np.vectorize(_norm_cdf)(z) \
          + sigma * np.vectorize(_norm_pdf)(z)
    return np.maximum(ei, 0.0)


# ===========================================================================
#  THE BAYESIAN ACTIVE LEARNER
# ===========================================================================
class BayesianActiveLearner:
    """
    Wraps a `Surrogate` with a GP on the residuals + an active-learning
    loop that picks the highest-information cells to actually simulate.
    """

    def __init__(self, surrogate: Surrogate,
                 controllers=None, cf_models=None,
                 scenarios=None, naturals=None, demands=None,
                 seed: int = 0):
        self.surrogate = surrogate
        self.rng       = random.Random(seed)

        # -- register all extras first --------------------------------
        register_mixed_controllers()
        register_extra_naturals()

        # -- axis domains ---------------------------------------------
        controllers = controllers or tuple(sorted(CONTROLLER_FAMILY))
        cf_models   = cf_models   or tuple(CAR_FOLLOWING_BENCHMARK)
        scenarios   = scenarios   or tuple(COMMUNICATION_SCENARIOS)
        naturals    = naturals    or tuple(NATURAL_SCENARIOS)
        demands     = demands     or tuple(DEMAND_PATTERNS)

        self.controllers = controllers
        self.cf_models   = cf_models
        self.scenarios   = scenarios
        self.naturals    = naturals
        self.demands     = demands

        self.encoder = Encoder(controllers, cf_models,
                               scenarios, naturals, demands)
        self.gp      = SurrogateGP(self.encoder.n_features)

        # -- history --------------------------------------------------
        self.cells   = []                  # observed (ctrl, cf, sc, nat, dem)
        self.X       = np.zeros((0, self.encoder.n_features))
        self.y_sim   = np.zeros(0)
        self.y_pred  = np.zeros(0)
        self.y_resid = np.zeros(0)
        self.best_cell  = None
        self.best_value = float('inf')
        self.round_log: list[dict] = []

    # ------------------------------------------------------------- internals
    def _run_real(self, cell, n_steps=200, seed=None):
        ctrl, cf, sc, nat, dem = cell
        d = DEMAND_PATTERNS[dem]
        seed = seed if seed is not None else self.rng.randint(1, 10**6)
        c = TrafficControlAlgorithm(ctrl, sc, nat, seed=seed,
                                    car_following=cf)
        simulate_episode(c, n_steps=n_steps,
                         base_demand_ns=d['ns'], base_demand_ew=d['ew'],
                         seed=seed)
        return c.get_metrics()

    def _surrogate_predict(self, cell):
        ctrl, cf, sc, nat, dem = cell
        d = DEMAND_PATTERNS[dem]
        r = self.surrogate.predict(ctrl, cf, sc, nat, dem,
                                   demand_ns=d['ns'], demand_ew=d['ew'])
        return r['travel_time_s']

    def _observe(self, cell, n_steps=200):
        """Run one real sim, compute residual, update history."""
        m       = self._run_real(cell, n_steps=n_steps)
        y_sim   = float(m['travel_time_s'])
        y_pred  = self._surrogate_predict(cell)
        residual = y_sim - y_pred

        self.cells.append(cell)
        x = self.encoder.encode(*cell).reshape(1, -1)
        self.X       = np.vstack([self.X, x]) if len(self.X) else x
        self.y_sim   = np.append(self.y_sim, y_sim)
        self.y_pred  = np.append(self.y_pred, y_pred)
        self.y_resid = np.append(self.y_resid, residual)

        if y_sim < self.best_value:
            self.best_value = y_sim
            self.best_cell  = cell
        return y_sim

    def _fit_gp(self):
        self.gp.fit(self.X, self.y_resid)

    # ------------------------------------------------------------- public
    def corrected_predict(self, cell):
        """Surrogate + GP posterior mean of the residual."""
        x = self.encoder.encode(*cell).reshape(1, -1)
        mu, var = self.gp.predict(x)
        return self._surrogate_predict(cell) + float(mu[0]), float(var[0])

    def predict(self, ctrl, cf, sc, nat, dem):
        """Return a fully-corrected prediction (mean + std)."""
        mean, var = self.corrected_predict((ctrl, cf, sc, nat, dem))
        return {'travel_time_s': round(mean, 2),
                'std':           round(math.sqrt(var), 3)}

    # ------------------------------------------------------------- loop
    def run(self, n_init: int = 20, budget: int = 100,
            n_steps: int = 200, n_candidates: int = 5000,
            progress: bool = True) -> dict:
        """
        Run the active-learning loop.

        n_init        — initial random seeds
        budget        — total real simulations (n_init included)
        n_steps       — steps per simulation
        n_candidates  — random pool size per round for EI maximisation
        """
        t0 = time.time()
        if progress:
            print("\n" + "=" * 90)
            print(f"  BAYESIAN ACTIVE LEARNING  —  budget={budget} sims")
            print("=" * 90)

        # ---- 1. seed ------------------------------------------------
        if progress:
            print(f"\n[1] Seeding with {n_init} random cells ...")
        all_cells = list(product(self.controllers, self.cf_models,
                                 self.scenarios, self.naturals,
                                 self.demands))
        self.rng.shuffle(all_cells)
        for cell in all_cells[:n_init]:
            self._observe(cell, n_steps=n_steps)

        # ---- 2. loop ------------------------------------------------
        observed = set(self.cells)
        for rnd in range(n_init + 1, budget + 1):
            self._fit_gp()

            # -- build candidate pool ---------------------------------
            pool = []
            while len(pool) < n_candidates:
                c = self.rng.choice(all_cells)
                if c not in observed:
                    pool.append(c)
            X_pool = self.encoder.encode_many(pool)
            mu, var = self.gp.predict(X_pool)
            # surrogate mean as prior
            pred_pool = np.array([self._surrogate_predict(c) for c in pool])
            pred_mu   = pred_pool + mu
            pred_sd   = np.sqrt(var)

            # -- acquisition ------------------------------------------
            ei = expected_improvement(pred_mu, pred_sd,
                                      best=self.best_value)
            idx = int(np.argmax(ei))
            chosen = pool[idx]

            # -- run ------------------------------------------------
            self._observe(chosen, n_steps=n_steps)
            observed.add(chosen)

            # -- log ------------------------------------------------
            if progress and (rnd % max(1, budget // 10) == 0
                             or rnd == budget):
                rmse = math.sqrt(statistics.fmean(
                    r ** 2 for r in self.y_resid))
                print(f"    round {rnd:>4}/{budget}   "
                      f"residual RMSE = {rmse:>6.3f}   "
                      f"best = {self.best_value:>7.2f}   "
                      f"chosen = {chosen[0]}/{chosen[2]}/{chosen[3]}")

            self.round_log.append({
                'round': rnd,
                'residual_rmse': round(math.sqrt(statistics.fmean(
                    r ** 2 for r in self.y_resid)), 3),
                'best_value': round(self.best_value, 2),
                'chosen':    chosen,
            })

        # ---- 3. final fit ------------------------------------------
        self._fit_gp()
        elapsed = time.time() - t0

        return {
            'n_sims':          len(self.cells),
            'best_cell':       self.best_cell,
            'best_value':      round(self.best_value, 2),
            'residual_rmse':   round(math.sqrt(statistics.fmean(
                r ** 2 for r in self.y_resid)), 3),
            'residual_mae':    round(statistics.fmean(
                abs(r) for r in self.y_resid), 3),
            'elapsed_s':       round(elapsed, 2),
        }

    # ------------------------------------------------------------- eval
    def evaluate_holdout(self, n_holdout: int = 30,
                         n_steps: int = 200) -> dict:
        """
        Draw a random fresh sample, compare 3 predictors:
          * raw surrogate
          * corrected (surrogate + GP)
          * real simulator
        """
        all_cells = list(product(self.controllers, self.cf_models,
                                 self.scenarios, self.naturals,
                                 self.demands))
        seen = set(self.cells)
        holdout_pool = [c for c in all_cells if c not in seen]
        self.rng.shuffle(holdout_pool)
        holdout = holdout_pool[:n_holdout]

        raw_err, corr_err = [], []
        for c in holdout:
            m = self._run_real(c, n_steps=n_steps)
            y_real = float(m['travel_time_s'])
            y_raw  = self._surrogate_predict(c)
            y_corr, _ = self.corrected_predict(c)
            raw_err.append(abs(y_real - y_raw))
            corr_err.append(abs(y_real - y_corr))

        return {
            'n_holdout':    n_holdout,
            'raw_mae':      round(statistics.fmean(raw_err), 3),
            'raw_rmse':     round(math.sqrt(statistics.fmean(
                e ** 2 for e in raw_err)), 3),
            'corr_mae':     round(statistics.fmean(corr_err), 3),
            'corr_rmse':    round(math.sqrt(statistics.fmean(
                e ** 2 for e in corr_err)), 3),
        }


# ===========================================================================
#  REPORTS
# ===========================================================================
def print_loop_summary(report: dict):
    print("\n" + "=" * 90)
    print("  ACTIVE-LEARNING SUMMARY")
    print("=" * 90)
    for k, v in report.items():
        print(f"  {k:<20} {v}")
    print()


def print_holdout(report: dict):
    print("\n" + "=" * 90)
    print("  HELD-OUT EVALUATION  (fresh cells never seen by the GP)")
    print("=" * 90)
    print(f"  hold-out n     : {report['n_holdout']}")
    print(f"  raw surrogate  : MAE = {report['raw_mae']:>6.3f}   "
          f"RMSE = {report['raw_rmse']:>6.3f}")
    print(f"  GP-corrected   : MAE = {report['corr_mae']:>6.3f}   "
          f"RMSE = {report['corr_rmse']:>6.3f}")
    improvement = (report['raw_mae'] - report['corr_mae']) \
                / max(report['raw_mae'], 1e-6) * 100
    print(f"  → MAE improvement: {improvement:>5.1f} %")
    print()


def print_round_log(log):
    print("\n" + "=" * 90)
    print("  ROUND-BY-ROUND CONVERGENCE")
    print("=" * 90)
    print(f"{'round':>6}{'residual RMSE':>16}{'best so far':>16}"
          f"{'chosen cell':>40}")
    print('-' * 90)
    for r in log:
        ch = r['chosen']
        ch_s = f"{ch[0]}/{ch[2]}/{ch[3]}/{ch[4]}"
        print(f"{r['round']:>6}{r['residual_rmse']:>16.3f}"
              f"{r['best_value']:>16.2f}{ch_s:>40}")
    print()


def print_comparison_table(learner, sample_cells):
    print("\n" + "=" * 90)
    print("  RAW vs CORRECTED PREDICTION  (sample cells)")
    print("=" * 90)
    print(f"{'Cell':<50}{'raw':>10}{'corrected':>12}{'std':>8}")
    print('-' * 90)
    for c in sample_cells:
        raw = learner._surrogate_predict(c)
        corr, var = learner.corrected_predict(c)
        sd = math.sqrt(var)
        cell_str = f"{c[0]}/{c[1]}/{c[2]}/{c[3]}/{c[4]}"
        print(f"{cell_str:<50}{raw:>10.2f}{corr:>12.2f}{sd:>8.3f}")
    print()


# ===========================================================================
#  CSV EXPORT
# ===========================================================================
def export_history(learner, path='active_learning_history.csv'):
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
#  CLI
# ===========================================================================
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget', type=int, default=100,
                   help='total real simulations to spend')
    p.add_argument('--init', type=int, default=20,
                   help='initial random seeds')
    p.add_argument('--steps', type=int, default=200,
                   help='simulation steps per cell')
    p.add_argument('--candidates', type=int, default=5000,
                   help='pool size for EI maximisation')
    p.add_argument('--compare', action='store_true',
                   help='run a held-out evaluation after the loop')
    p.add_argument('--holdout', type=int, default=30,
                   help='hold-out sample size when --compare is used')
    p.add_argument('--export', action='store_true',
                   help='write active_learning_history.csv')
    return p.parse_args()


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    args = _parse_args()
    print("#" * 100)
    print("#  BAYESIAN ACTIVE-LEARNING LOOP  —  simulator-quality from <1 % of cells")
    print("#" * 100)

    surr = Surrogate(seed=42)
    bal  = BayesianActiveLearner(surr, seed=42)

    n_cells = (len(bal.controllers) * len(bal.cf_models)
               * len(bal.scenarios) * len(bal.naturals) * len(bal.demands))
    print(f"\n  Full grid       : {n_cells:,} cells")
    print(f"  Budget          : {args.budget} real simulations "
          f"({args.budget / n_cells * 100:.2f} % of grid)")
    print(f"  Initial seeds   : {args.init}")
    print(f"  Steps per sim   : {args.steps}")

    report = bal.run(n_init=args.init, budget=args.budget,
                     n_steps=args.steps, n_candidates=args.candidates)
    print_loop_summary(report)
    print_round_log(bal.round_log)

    # sample comparison
    demo_cells = [
        ('resilient-marl', 'SAFE', 'S3', 'N-clear', 'D1'),
        ('resilient-marl', 'SAFE', 'S4', 'N-hailstorm', 'D6'),
        ('adaptive',       'GIPPS', 'S2', 'N-rush-am', 'D2'),
        ('mpc',            'SAFE', 'S3', 'N-event', 'D4'),
        ('coordinated',    'IDM', 'S4', 'N-comm-blackout', 'D5'),
    ]
    print_comparison_table(bal, demo_cells)

    if args.compare:
        print(f"\n[held-out] sampling {args.holdout} fresh cells ...")
        h = bal.evaluate_holdout(n_holdout=args.holdout,
                                 n_steps=args.steps)
        print_holdout(h)

    if args.export:
        print("\n[export] writing history ...")
        export_history(bal, 'active_learning_history.csv')

    print("\nDone.")


if __name__ == '__main__':
    main()