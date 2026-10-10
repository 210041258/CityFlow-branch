#!/usr/bin/env python3
"""
scenario_advanced.py
====================
Advanced impairment scenarios for `traffic_control_algorithm.py`.

Each of the 8 new scenarios has its OWN logistic degradation signature:

    f(x) = L / (1 + exp(-k (x - x0)))

The signature's (L, k, x0) encode the scenario's characteristic:
  * L      — saturation level (max travel time under total failure)
  * k      — steepness (how abruptly performance falls off)
  * x0     — inflection point (the scenario's knee)

The 8 scenarios
---------------
  A1  Correlated Latency       AR(1) latency, temporal clustering
  A2  Burst Packet Loss        Poisson bursts of high loss
  A3  Regional Blackout        2×2 subgrid loses V2X for a window
  A4  Incident Surge           demand multiplier spiking 3× for 2–10 min
  A5  Heterogeneous Fleet      per-vehicle CF mix + 20% heavy vehicles
  A6  Mixed Comm Stack         per-agent protocol (5G / DSRC / LTE)
  A7  Adversarial Messages     % corrupted messages + false AoI
  A8  Real Trace Replay        load a CSV trace of (t, latency, loss)

Public API
----------
    AdvancedScenario(name, seed, **kw)        — one scenario instance
        .generate(t, node_id) -> dict         — per-step impairment
        .logistic(x) -> float                 — scenario's knee curve
        .describe() -> str                    — human-readable summary

    compare_signatures()                      — ASCII overlay of 8 curves
    run_scenario(name, controller, ...)       — one cell
    run_all_advanced(...)                     — every scenario × controller

Run:
    python scenario_advanced.py
"""

from __future__ import annotations
import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional

# ---- imports from the base simulator --------------------------------------
from traffic_control_algorithm import (
    TrafficControlAlgorithm,
    COMMUNICATION_SCENARIOS,
    NATURAL_SCENARIOS,
    simulate_episode,
    degrade_logistic,
    severity_index,
    DECISION_INTERVAL_S,
    GRID_ROWS, GRID_COLS,
)


# ===========================================================================
#  LOGISTIC SIGNATURE  —  the "expression in the logistics"
# ===========================================================================
@dataclass(frozen=True)
class LogisticSignature:
    """
    One scenario's logistic degradation curve.

    f(x) = L / (1 + exp(-k (x - x0)))

    x is the scenario's severity variable (latency_ms, loss_rate,
    demand_mult, etc.) — see AdvancedScenario.severity_of().
    """
    L:     float          # saturation level
    k:     float          # steepness
    x0:    float          # inflection point  (the knee)
    label: str = ''
    x_unit: str = ''
    x_max:  float = 1.0   # for plotting

    def __call__(self, x: float) -> float:
        return self.L / (1.0 + math.exp(-self.k * (x - self.x0)))

    def derivative_at(self, x: float) -> float:
        """Analytic slope — useful for knee-finding."""
        e = math.exp(-self.k * (x - self.x0))
        return self.L * self.k * e / (1.0 + e) ** 2

    def summary(self) -> str:
        return (f"{self.label:<20}  L={self.L:>6.1f}  "
                f"k={self.k:>7.5f}  x0={self.x0:>7.2f}{self.x_unit}")


# ===========================================================================
#  THE 8 SCENARIOS  —  each with its own logistic signature
# ===========================================================================
#  Curves hand-fitted to reflect the qualitative behaviour of each
#  impairment type (steep for blackout, shallow for correlated latency, etc.)

ADVANCED_SIGNATURES: dict[str, LogisticSignature] = {
    # A1 — Correlated Latency: shallow knee, high L (recovers between spikes)
    'A1-correlated-latency': LogisticSignature(
        L=245.0, k=0.0120, x0=118.0,
        label='Correlated latency',
        x_unit=' ms', x_max=600.0),

    # A2 — Burst Loss: sharp knee, moderate L (bursts are short but severe)
    'A2-burst-loss': LogisticSignature(
        L=280.0, k=0.0220, x0=0.28,
        label='Burst packet loss',
        x_unit=' ρ_loss', x_max=0.60),

    # A3 — Regional Blackout: sharpest knee, near-saturated above x0
    'A3-regional-blackout': LogisticSignature(
        L=320.0, k=0.0500, x0=0.35,
        label='Regional blackout',
        x_unit=' fraction', x_max=1.00),

    # A4 — Incident Surge: moderate knee on demand multiplier
    'A4-incident-surge': LogisticSignature(
        L=295.0, k=1.2000, x0=1.65,
        label='Incident demand surge',
        x_unit=' ×demand', x_max=3.00),

    # A5 — Heterogeneous Fleet: shallow knee, low L (diversity helps)
    'A5-heterogeneous-fleet': LogisticSignature(
        L=220.0, k=0.0300, x0=0.55,
        label='Heterogeneous fleet',
        x_unit=' heavy-frac', x_max=1.00),

    # A6 — Mixed Comm Stack: knee below the S2 inflection
    'A6-mixed-comm-stack': LogisticSignature(
        L=270.0, k=0.0150, x0=88.0,
        label='Mixed comm stack',
        x_unit=' ms', x_max=600.0),

    # A7 — Adversarial Messages: very sharp knee on adversary rate
    'A7-adversarial': LogisticSignature(
        L=310.0, k=12.000, x0=0.30,
        label='Adversarial messages',
        x_unit=' attack-rate', x_max=0.80),

    # A8 — Real Trace Replay: lowest L, closest to the thesis baseline
    'A8-trace-replay': LogisticSignature(
        L=250.0, k=0.0143, x0=101.54,
        label='Real trace replay',
        x_unit=' ms', x_max=600.0),
}


# ===========================================================================
#  IMPAIRMENT GENERATORS  (one per scenario)
# ===========================================================================
class ImpairmentGenerator:
    """Base class — produce a per-step impairment dict for one node."""
    name = 'base'

    def __init__(self, seed: int = 0, **kw):
        self.rng  = random.Random(seed)
        self.kw   = kw
        self.state = {}

    def step(self, t: float, node_id: str) -> dict:
        """Return {'latency_ms':..., 'loss_rate':..., 'blackout':bool, ...}."""
        return {}

    def severity_of(self, impairment: dict) -> float:
        """Map the impairment to the scenario's logistic x-variable."""
        return impairment.get('latency_ms', 0.0)


# --- A1  Correlated latency (AR(1)) ----------------------------------------
class CorrelatedLatency(ImpairmentGenerator):
    name = 'A1-correlated-latency'

    def __init__(self, seed=0, mean_ms=100.0, autocorr=0.85,
                 sigma=0.4, base_loss=0.05):
        super().__init__(seed)
        self.mean_ms   = mean_ms
        self.autocorr  = autocorr
        self.sigma     = sigma
        self.base_loss = base_loss
        self._prev     = math.log(mean_ms)

    def step(self, t, node_id):
        # AR(1) in log-space -> multiplicative, right-skewed
        eps  = self.rng.gauss(0.0, self.sigma)
        x    = self.autocorr * self._prev + (1 - self.autocorr) * math.log(self.mean_ms) + eps
        self._prev = x
        latency = math.exp(x)
        latency = max(1.0, min(3000.0, latency))
        return {'latency_ms': latency, 'loss_rate': self.base_loss,
                'autocorr': self.autocorr}

    def severity_of(self, imp): return imp['latency_ms']


# --- A2  Burst packet loss --------------------------------------------------
class BurstPacketLoss(ImpairmentGenerator):
    name = 'A2-burst-loss'

    def __init__(self, seed=0, base_loss=0.02, burst_rate=0.15,
                 burst_loss=0.65, burst_len_s=8.0):
        super().__init__(seed)
        self.base_loss   = base_loss
        self.burst_rate  = burst_rate          # prob/s
        self.burst_loss  = burst_loss
        self.burst_len_s = burst_len_s
        self._burst_until = -1.0

    def step(self, t, node_id):
        in_burst = t < self._burst_until
        if not in_burst and self.rng.random() < self.burst_rate * DECISION_INTERVAL_S / 60.0:
            self._burst_until = t + self.rng.uniform(2.0, self.burst_len_s)
            in_burst = True
        loss = self.burst_loss if in_burst else self.base_loss
        return {'latency_ms': 100.0, 'loss_rate': loss,
                'in_burst': in_burst}

    def severity_of(self, imp): return imp['loss_rate']


# --- A3  Regional blackout --------------------------------------------------
class RegionalBlackout(ImpairmentGenerator):
    name = 'A3-regional-blackout'

    def __init__(self, seed=0, block=((0, 0), (1, 1)),
                 start_s=200.0, duration_s=300.0):
        super().__init__(seed)
        (r0, c0), (r1, c1) = block
        self.block_nodes = {
            f"J{r+1}{c+1}" for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
        }
        self.start    = start_s
        self.duration = duration_s
        self.outside_loss = 0.05

    def step(self, t, node_id):
        active = (self.start <= t < self.start + self.duration)
        inside = node_id in self.block_nodes
        if active and inside:
            return {'latency_ms': 0.0, 'loss_rate': 1.0, 'blackout': True}
        return {'latency_ms': 100.0, 'loss_rate': self.outside_loss,
                'blackout': False}

    def severity_of(self, imp):
        return 1.0 if imp.get('blackout') else 0.0


# --- A4  Incident surge -----------------------------------------------------
class IncidentSurge(ImpairmentGenerator):
    name = 'A4-incident-surge'

    def __init__(self, seed=0, start_s=150.0, duration_s=240.0,
                 surge_mult=2.8, base_loss=0.05):
        super().__init__(seed)
        self.start    = start_s
        self.duration = duration_s
        self.mult     = surge_mult
        self.base_loss = base_loss

    def step(self, t, node_id):
        active = (self.start <= t < self.start + self.duration)
        return {'demand_mult': self.mult if active else 1.0,
                'latency_ms': 100.0, 'loss_rate': self.base_loss,
                'surge': active}

    def severity_of(self, imp): return imp['demand_mult']


# --- A5  Heterogeneous fleet ------------------------------------------------
class HeterogeneousFleet(ImpairmentGenerator):
    name = 'A5-heterogeneous-fleet'

    def __init__(self, seed=0, safe=0.6, idm=0.3, gipps=0.1,
                 heavy_frac=0.2):
        super().__init__(seed)
        # normalise
        s = safe + idm + gipps
        self.probs = {'SAFE': safe / s, 'IDM': idm / s, 'GIPPS': gipps / s}
        self.heavy_frac = heavy_frac

    def step(self, t, node_id):
        r = self.rng.random()
        if r < self.probs['SAFE']: cf = 'SAFE'
        elif r < self.probs['SAFE'] + self.probs['IDM']: cf = 'IDM'
        else: cf = 'GIPPS'
        heavy = self.rng.random() < self.heavy_frac
        return {'latency_ms': 100.0, 'loss_rate': 0.05,
                'car_model': cf, 'heavy': heavy,
                'heavy_frac': self.heavy_frac}

    def severity_of(self, imp): return imp['heavy_frac']


# --- A6  Mixed comm stack ---------------------------------------------------
STACK_PROFILES = {
    '5G':   {'latency_ms':  30.0, 'loss_rate': 0.02},
    'DSRC': {'latency_ms':  60.0, 'loss_rate': 0.08},
    'LTE':  {'latency_ms': 150.0, 'loss_rate': 0.15},
}

class MixedCommStack(ImpairmentGenerator):
    name = 'A6-mixed-comm-stack'

    def __init__(self, seed=0, fractions=None):
        super().__init__(seed)
        self.fractions = fractions or {'5G': 0.5, 'DSRC': 0.3, 'LTE': 0.2}
        self.assignment = {}

    def _proto(self, node_id):
        if node_id not in self.assignment:
            r, cum = self.rng.random(), 0.0
            for p, f in self.fractions.items():
                cum += f
                if r < cum:
                    self.assignment[node_id] = p
                    break
            else:
                self.assignment[node_id] = 'LTE'
        return self.assignment[node_id]

    def step(self, t, node_id):
        proto = self._proto(node_id)
        prof  = STACK_PROFILES[proto]
        # handover jitter
        jit = self.rng.gauss(0.0, 5.0)
        return {'latency_ms': max(0.0, prof['latency_ms'] + jit),
                'loss_rate': prof['loss_rate'],
                'protocol': proto}

    def severity_of(self, imp): return imp['latency_ms']


# --- A7  Adversarial messages -----------------------------------------------
class AdversarialMessages(ImpairmentGenerator):
    name = 'A7-adversarial'

    ATTACKS = ('false_queue', 'stale_aoi', 'replay')

    def __init__(self, seed=0, attack_rate=0.20, base_loss=0.05,
                 latency_ms=100.0):
        super().__init__(seed)
        self.attack_rate = attack_rate
        self.base_loss   = base_loss
        self.latency_ms  = latency_ms

    def step(self, t, node_id):
        attacked = self.rng.random() < self.attack_rate
        attack   = self.rng.choice(self.ATTACKS) if attacked else None
        return {'latency_ms': self.latency_ms,
                'loss_rate': self.base_loss,
                'attacked': attacked, 'attack': attack}

    def severity_of(self, imp): return self.attack_rate


# --- A8  Real trace replay --------------------------------------------------
class RealTraceReplay(ImpairmentGenerator):
    name = 'A8-trace-replay'

    def __init__(self, seed=0, trace=None):
        super().__init__(seed)
        # default trace = synthetic AR(1) fallback
        if trace is None:
            rng = random.Random(seed)
            trace = []
            x = math.log(100.0)
            for _ in range(120):
                x = 0.9 * x + 0.1 * math.log(100.0) + rng.gauss(0, 0.3)
                trace.append((math.exp(x), rng.uniform(0.0, 0.15)))
        self.trace = trace

    def step(self, t, node_id):
        idx = int(t / DECISION_INTERVAL_S) % len(self.trace)
        lat, loss = self.trace[idx]
        return {'latency_ms': lat, 'loss_rate': loss, 'trace_idx': idx}

    def severity_of(self, imp): return imp['latency_ms']


# Factory map
GENERATOR_CLASSES = {
    'A1-correlated-latency':  CorrelatedLatency,
    'A2-burst-loss':          BurstPacketLoss,
    'A3-regional-blackout':   RegionalBlackout,
    'A4-incident-surge':      IncidentSurge,
    'A5-heterogeneous-fleet': HeterogeneousFleet,
    'A6-mixed-comm-stack':    MixedCommStack,
    'A7-adversarial':         AdversarialMessages,
    'A8-trace-replay':        RealTraceReplay,
}


# ===========================================================================
#  THE ADVANCED SCENARIO CLASS
# ===========================================================================
class AdvancedScenario:
    """
    One advanced scenario:  signature + impairment generator + wrapper
    around the base simulator.  Exposes:

        .signature          — the logistic expression
        .logistic(x)        — evaluate the curve
        .impairment_at(t,n) — per-step impairment for one node
        .severity(t)        — aggregate severity at time t
        .describe()         — human-readable summary
    """
    def __init__(self, name: str, seed: int = 0, **generator_kw):
        if name not in ADVANCED_SIGNATURES:
            raise ValueError(f"Unknown advanced scenario '{name}'.")
        if name not in GENERATOR_CLASSES:
            raise ValueError(f"No generator for '{name}'.")
        self.name      = name
        self.signature = ADVANCED_SIGNATURES[name]
        self.generator = GENERATOR_CLASSES[name](seed=seed, **generator_kw)

    def logistic(self, x: float) -> float:
        return self.signature(x)

    def impairment_at(self, t: float, node_id: str) -> dict:
        return self.generator.step(t, node_id)

    def severity(self, t: float) -> float:
        """Mean severity across the grid at time t."""
        xs = [self.signature.x_max * 0.5]  # fallback
        out = [self.generator.step(t, f"J{r+1}{c+1}")
               for r in range(GRID_ROWS) for c in range(GRID_COLS)]
        return statistics.fmean(self.generator.severity_of(o) for o in out)

    def describe(self) -> str:
        return self.signature.summary()

    # ---------------------------------------------------------------- run
    def simulate(self, controller_name: str,
                 n_steps: int = 200,
                 base_demand_ns: float = 0.30,
                 base_demand_ew: float = 0.30,
                 car_following: str = 'SAFE',
                 seed: int = 0) -> dict:
        """
        Run the base simulator, then apply a *logistic-adjusted* correction
        that reflects this scenario's impairment signature.
        """
        ctrl = TrafficControlAlgorithm(controller_name, 'S2', 'N-clear',
                                       seed, car_following)
        simulate_episode(ctrl, n_steps=n_steps,
                         base_demand_ns=base_demand_ns,
                         base_demand_ew=base_demand_ew,
                         seed=seed)
        m = ctrl.get_metrics()

        # sample the mean severity across the episode
        sev = statistics.fmean(self.severity(t * DECISION_INTERVAL_S)
                               for t in range(0, n_steps, 5))
        # apply the scenario's logistic correction to the base metrics
        correction = self.logistic(sev) / self.logistic(self.signature.x0)
        m['travel_time_s']  = round(m['travel_time_s'] * correction, 2)
        m['queue_veh']      = round(m['queue_veh'] * correction, 2)
        m['waiting_time_s'] = round(m['waiting_time_s'] * correction, 2)
        m['throughput_vph'] = round(m['throughput_vph'] / correction, 1)

        m['advanced_scenario'] = self.name
        m['severity_sampled']  = round(sev, 4)
        m['logistic_correction'] = round(correction, 3)
        return m


# ===========================================================================
#  DRIVERS  —  one cell, one scenario, all scenarios
# ===========================================================================
def run_cell(scenario_name: str, controller_name: str,
             seeds=(1, 2, 3, 4, 5), n_steps: int = 200) -> dict:
    """Multi-seed run of one (advanced scenario, controller) cell."""
    keys = ('travel_time_s', 'queue_veh', 'waiting_time_s',
            'throughput_vph', 'severity_sampled', 'logistic_correction')
    per_seed = defaultdict(list)
    for s in seeds:
        sc = AdvancedScenario(scenario_name, seed=s)
        m = sc.simulate(controller_name, n_steps=n_steps, seed=s)
        for k in keys:
            per_seed[k].append(float(m[k]))

    out = {'advanced_scenario': scenario_name, 'controller': controller_name,
           'n_seeds': len(seeds)}
    for k, xs in per_seed.items():
        mean = statistics.fmean(xs)
        sd   = statistics.stdev(xs) if len(xs) > 1 else 0.0
        ci   = 1.96 * sd / math.sqrt(len(xs))
        out[f'{k}_mean'] = round(mean, 3)
        out[f'{k}_sd']   = round(sd, 3)
        out[f'{k}_ci95'] = round(ci, 3)
    return out


def run_all_advanced(controllers=None, scenarios=None,
                     seeds=(1, 2, 3, 4, 5), n_steps: int = 200,
                     progress: bool = True) -> list[dict]:
    controllers = controllers or ('adaptive', 'coordinated',
                                  'mpc', 'resilient-marl')
    scenarios   = scenarios   or list(ADVANCED_SIGNATURES)
    rows = []
    total = len(controllers) * len(scenarios)
    i = 0
    for c in controllers:
        for s in scenarios:
            i += 1
            if progress:
                print(f"\r  cell {i}/{total} ...", end='', flush=True)
            rows.append(run_cell(s, c, seeds=seeds, n_steps=n_steps))
    if progress:
        print()
    return rows


# ===========================================================================
#  REPORTS  —  ASCII overlay + tables
# ===========================================================================
def compare_signatures(width: int = 78, height: int = 18):
    """Plot all 8 logistic signatures on one ASCII canvas."""
    print("\n" + "=" * 100)
    print("  LOGISTIC SIGNATURES  —  f(x) = L / (1 + exp(-k (x - x0)))")
    print("=" * 100)
    for name, sig in ADVANCED_SIGNATURES.items():
        print(f"  {sig.summary()}")

    # normalise each curve to [0, 1] over its own x_max, then plot all
    markers = "12345678"
    canvas  = [[' '] * width for _ in range(height)]

    for i, (name, sig) in enumerate(ADVANCED_SIGNATURES.items()):
        for col in range(width):
            x    = (col / (width - 1)) * sig.x_max
            y    = sig(x) / max(sig.L, 1e-6)
            row  = int(round((1 - y) * (height - 1)))
            row  = max(0, min(height - 1, row))
            canvas[row][col] = markers[i]

    print()
    print("  1.0 ┌" + "-" * (width - 2) + "┐")
    for r, line in enumerate(canvas):
        y = 1.0 - r / (height - 1)
        print(f"  {y:>4.2f} │" + ''.join(line) + "│")
    print("  0.0 └" + "-" * (width - 2) + "┘")
    print("       " + "normalised x (each scenario's own axis)")
    print()
    print("  legend:")
    for i, (name, sig) in enumerate(ADVANCED_SIGNATURES.items()):
        print(f"    {markers[i]}  {name:<26}  knee at x0={sig.x0:>7.2f}{sig.x_unit}")
    print()


def print_knee_table():
    print("\n" + "=" * 100)
    print("  KNEE / SATURATION COMPARISON  —  advanced vs thesis baseline")
    print("=" * 100)
    print(f"{'Scenario':<26}{'L':>8}{'k':>10}{'x0':>10}"
          f"{'unit':>14}{'L/L_thesis':>12}{'k/k_thesis':>12}")
    print('-' * 100)
    L0, k0, x00 = 302.23, 0.0143, 101.54
    for name, sig in ADVANCED_SIGNATURES.items():
        print(f"{name:<26}{sig.L:>8.1f}{sig.k:>10.5f}{sig.x0:>10.2f}"
              f"{sig.x_unit:>14}{sig.L/L0:>12.2f}{sig.k/k0:>12.2f}")
    print(f"\n  thesis baseline          {L0:>8.2f}{k0:>10.5f}{x00:>10.2f}"
          f"{' ms':>14}{1.0:>12.2f}{1.0:>12.2f}")
    print()


def print_matrix(rows, controllers, scenarios,
                 metric='travel_time_s'):
    print("\n" + "=" * 110)
    print(f"  ADVANCED SCENARIO × CONTROLLER  —  {metric} (mean ± 95 % CI)")
    print("=" * 110)
    idx = {(r['advanced_scenario'], r['controller']): r for r in rows}
    hdr = f"{'Scenario':<26}" + ''.join(f"{c[:16]:>18}" for c in controllers)
    print(hdr); print('-' * len(hdr))
    for s in scenarios:
        line = f"{s:<26}"
        for c in controllers:
            r = idx.get((s, c))
            if r:
                mean = r[f'{metric}_mean']
                ci   = r[f'{metric}_ci95']
                line += f"{mean:>10.2f}±{ci:>6.2f}"
            else:
                line += f"{'—':>18}"
        print(line)
    print()


def print_best_per_scenario(rows, controllers,
                            metric='travel_time_s'):
    print("\n" + "=" * 100)
    print(f"  BEST CONTROLLER PER ADVANCED SCENARIO  —  {metric}")
    print("=" * 100)
    print(f"{'Scenario':<26}{'Winner':<22}{'Mean':>10}"
          f"{'±CI':>8}{'vs 2nd':>10}{'2nd':<22}")
    print('-' * 100)
    for s in ADVANCED_SIGNATURES:
        cands = [r for r in rows if r['advanced_scenario'] == s]
        if not cands:
            continue
        cands.sort(key=lambda r: r[f'{metric}_mean'])
        b  = cands[0]
        ru = cands[1] if len(cands) > 1 else b
        gap = ru[f'{metric}_mean'] - b[f'{metric}_mean']
        print(f"{s:<26}{b['controller']:<22}"
              f"{b[f'{metric}_mean']:>10.2f}"
              f"{b[f'{metric}_ci95']:>8.2f}"
              f"{gap:>10.2f}"
              f"{ru['controller']:<22}")
    print()


def export_csv(rows, path='scenario_advanced.csv'):
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
    p.add_argument('--scenario', default=None,
                   help='single advanced scenario to run')
    p.add_argument('--controller', default='adaptive')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--full', action='store_true',
                   help='run every scenario × 4 controllers')
    p.add_argument('--export', action='store_true')
    return p.parse_args()


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    args = _parse_args()
    print("#" * 100)
    print("#  ADVANCED SCENARIOS  —  logistic signatures for 8 new impairments")
    print("#" * 100)

    # always print the signature comparison + knee table
    compare_signatures()
    print_knee_table()

    if args.scenario:
        # single cell ------------------------------------------------
        sc = AdvancedScenario(args.scenario, seed=1)
        print(f"  Scenario : {sc.name}")
        print(f"  Curve    : {sc.describe()}")
        m = sc.simulate(args.controller, n_steps=args.steps)
        print(f"  Metrics  : {m}")
        return

    if not args.full:
        # default: run 8 scenarios × 2 controllers
        controllers = ('adaptive', 'resilient-marl')
    else:
        controllers = ('adaptive', 'coordinated', 'mpc', 'resilient-marl')

    seeds = tuple(range(1, args.seeds + 1))
    scenarios = list(ADVANCED_SIGNATURES)

    print(f"\n[1] Running {len(scenarios)} scenarios × "
          f"{len(controllers)} controllers × {len(seeds)} seeds ...")
    rows = run_all_advanced(controllers=controllers,
                            scenarios=scenarios,
                            seeds=seeds,
                            n_steps=args.steps)

    print("\n[2] Reports ...")
    print_matrix(rows, controllers, scenarios)
    print_best_per_scenario(rows, controllers)

    if args.export:
        print("\n[3] CSV export ...")
        export_csv(rows, 'scenario_advanced.csv')

    print("\nDone.")


if __name__ == '__main__':
    main()