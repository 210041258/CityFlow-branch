#!/usr/bin/env python3
"""
adaptive_multi_execution.py
===========================
Distributed adaptive signal execution.

  * One AdaptiveSignalController instance PER intersection
  * Real signal state machine (green → yellow → all-red → green)
  * Online parameter adaptation (min_green, switch_threshold, cycle_target)
  * Decentralised SignalBus for neighbour queries
  * run_parallel(n) → n concurrent episodes via ProcessPoolExecutor

Imports only the network + car-following from traffic_control_algorithm.
Run:
    python adaptive_multi_execution.py
    python adaptive_multi_execution.py --parallel 8 --steps 200
"""

from __future__ import annotations
import argparse
import math
import random
import statistics
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from traffic_control_algorithm import (
    DECISION_INTERVAL_S, GREEN_MIN_S, GREEN_MAX_S, YELLOW_S,
    GRID_ROWS, GRID_COLS, SATURATION_FLOW_VPS,
    CarFollowingModel, Intersection,
)


# ===========================================================================
#  SIGNAL STATE MACHINE
# ===========================================================================
class SignalPhase(IntEnum):
    NS_GREEN   = 0
    NS_YELLOW  = 1
    ALL_RED_1  = 2
    EW_GREEN   = 3
    EW_YELLOW  = 4
    ALL_RED_2  = 5


# Which approach is served during which phase?
SERVED_BY_PHASE = {
    SignalPhase.NS_GREEN:  ('N', 'S'),
    SignalPhase.EW_GREEN:  ('E', 'W'),
    SignalPhase.NS_YELLOW: (),
    SignalPhase.EW_YELLOW: (),
    SignalPhase.ALL_RED_1: (),
    SignalPhase.ALL_RED_2: (),
}

ALL_RED_S = 1.5     # clearance interval


@dataclass
class SignalStateMachine:
    """
    One intersection's real signal state machine.

    The controller does NOT specify green duration directly — it *requests*
    a phase change, and the machine enforces min-green, max-green, yellow,
    and all-red before the change takes effect.
    """
    node_id: str
    current_phase: SignalPhase = SignalPhase.NS_GREEN
    time_in_phase: float = 0.0
    pending_switch: bool = False
    log: list = field(default_factory=list)

    # --- timing (mutated online by the controller) -----------------------
    min_green: float = GREEN_MIN_S
    max_green: float = GREEN_MAX_S

    def request_switch(self):
        """Controller asks for a phase change; machine will honour it when allowed."""
        self.pending_switch = True

    def step(self, dt: float, demand_ns: float, demand_ew: float):
        """Advance the signal machine by dt.  Return the phase served."""
        self.time_in_phase += dt

        # --- enforce max-green: auto-switch if green held too long --------
        if self.current_phase in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN):
            if self.time_in_phase >= self.max_green:
                self.pending_switch = True

        # --- process pending switch if min-green satisfied -----------------
        if self.pending_switch and self.time_in_phase >= self.min_green:
            if self.current_phase == SignalPhase.NS_GREEN:
                self.current_phase = SignalPhase.NS_YELLOW
            elif self.current_phase == SignalPhase.NS_YELLOW:
                self.current_phase = SignalPhase.ALL_RED_1
            elif self.current_phase == SignalPhase.ALL_RED_1:
                self.current_phase = SignalPhase.EW_GREEN
            elif self.current_phase == SignalPhase.EW_GREEN:
                self.current_phase = SignalPhase.EW_YELLOW
            elif self.current_phase == SignalPhase.EW_YELLOW:
                self.current_phase = SignalPhase.ALL_RED_2
            elif self.current_phase == SignalPhase.ALL_RED_2:
                self.current_phase = SignalPhase.NS_GREEN
            self.time_in_phase = 0.0
            self.pending_switch = False

            # --- schedule automatic yellow / all-red transitions -----------
        else:
            if (self.current_phase == SignalPhase.NS_YELLOW
                    and self.time_in_phase >= YELLOW_S):
                self.current_phase = SignalPhase.ALL_RED_1
                self.time_in_phase = 0.0
            elif (self.current_phase == SignalPhase.ALL_RED_1
                  and self.time_in_phase >= ALL_RED_S):
                self.current_phase = SignalPhase.EW_GREEN
                self.time_in_phase = 0.0
            elif (self.current_phase == SignalPhase.EW_YELLOW
                  and self.time_in_phase >= YELLOW_S):
                self.current_phase = SignalPhase.ALL_RED_2
                self.time_in_phase = 0.0
            elif (self.current_phase == SignalPhase.ALL_RED_2
                  and self.time_in_phase >= ALL_RED_S):
                self.current_phase = SignalPhase.NS_GREEN
                self.time_in_phase = 0.0

        return self.current_phase

    def served_approaches(self):
        return SERVED_BY_PHASE[self.current_phase]

    def is_green(self):
        return self.current_phase in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN)

    def remaining_min_green(self):
        return max(0.0, self.min_green - self.time_in_phase)

    def snapshot(self):
        return {
            'id': self.node_id,
            'phase': int(self.current_phase),
            'time_in_phase': self.time_in_phase,
            'is_green': self.is_green(),
            'min_green': self.min_green,
        }


# ===========================================================================
#  DECENTRALISED SIGNAL BUS
# ===========================================================================
class SignalBus:
    """
    Decentralised message bus — each controller can ask neighbours
    for their current phase and remaining min-green.  No central
    coordinator, no shared mutable state visible to all.
    """

    def __init__(self):
        self._snapshots: dict[str, dict] = {}

    def publish(self, snapshot: dict):
        self._snapshots[snapshot['id']] = snapshot

    def query(self, neighbour_ids) -> list[dict]:
        return [self._snapshots[n] for n in neighbour_ids
                if n in self._snapshots]


# ===========================================================================
#  ADAPTIVE SIGNAL CONTROLLER — one per intersection
# ===========================================================================
@dataclass
class AdaptiveSignalController:
    """
    One per intersection.  Observes local queues + neighbour phases,
    requests signal changes, and *adapts its own parameters online*.
    """
    node_id: str
    seed: int = 0
    # adaptive knobs — start from thesis defaults, mutate during episode
    min_green:       float = GREEN_MIN_S
    max_green:       float = GREEN_MAX_S
    switch_threshold: float = 1.5     # queue-ratio gap that justifies switching
    cycle_target:    float = 60.0     # target cycle length

    # bookkeeping
    rng:            random.Random = field(init=False)
    last_switch_t:  float = 0.0
    switches:       int   = 0
    adaptations:    dict  = field(default_factory=dict)
    queue_history:  deque = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.adaptations = {'min_green': 0, 'switch_threshold': 0, 'cycle_target': 0}

    # ------------------------------------------------------------- decide
    def decide(self,
               sm: SignalStateMachine,
               q_ns: float, q_ew: float,
               neighbour_phases: list[dict]) -> None:
        """
        Decide whether to request a phase switch.  The signal machine
        enforces the actual transition.
        """
        if not sm.is_green():
            return                                    # wait for yellow/all-red

        self.queue_history.append((q_ns, q_ew))
        total = max(q_ns + q_ew, 1e-6)
        ns_ratio = q_ns / total

        # --- adaptation 1: min_green responds to sustained queue ---------
        if len(self.queue_history) >= 5:
            recent_total = statistics.fmean(q[0] + q[1]
                                            for q in self.queue_history)
            if recent_total > 20 and self.min_green < 20.0:
                self.min_green = min(20.0, self.min_green + 1.0)
                self.adaptations['min_green'] += 1
            elif recent_total < 5 and self.min_green > 8.0:
                self.min_green = max(8.0, self.min_green - 1.0)
                self.adaptations['min_green'] += 1

        # --- adaptation 2: switch_threshold responds to neighbour sync ---
        same_as_us = sum(1 for n in neighbour_phases
                         if n['phase'] == int(sm.current_phase))
        if same_as_us >= 2 and self.switch_threshold > 1.0:
            # too many neighbours on the same phase -> break symmetry sooner
            self.switch_threshold = max(1.0, self.switch_threshold - 0.1)
            self.adaptations['switch_threshold'] += 1

        # --- adaptation 3: cycle_target shrinks under high congestion ----
        if recent_total := (self.queue_history[-1][0] + self.queue_history[-1][1]):
            if recent_total > 25 and self.cycle_target > 45.0:
                self.cycle_target = max(45.0, self.cycle_target - 0.5)
                self.adaptations['cycle_target'] += 1
            elif recent_total < 8 and self.cycle_target < 90.0:
                self.cycle_target = min(90.0, self.cycle_target + 0.5)
                self.adaptations['cycle_target'] += 1

        # --- decision: queue-ratio gap exceeds adaptive threshold --------
        gap = ns_ratio - 0.5
        serving_ns = (sm.current_phase == SignalPhase.NS_GREEN)

        # served direction has no queue, other does -> switch
        if serving_ns and q_ew > q_ns + self.switch_threshold * 3:
            sm.request_switch()
            self.switches += 1
        elif (not serving_ns) and q_ns > q_ew + self.switch_threshold * 3:
            sm.request_switch()
            self.switches += 1
        # or we've served long enough and the gap is modest
        elif abs(gap) > 0.15 and sm.time_in_phase > sm.min_green + 5.0:
            sm.request_switch()
            self.switches += 1

        # propagate adaptive limits into the machine
        sm.min_green = self.min_green
        sm.max_green = self.max_green

    def snapshot(self):
        return {
            'id': self.node_id,
            'min_green': round(self.min_green, 2),
            'switch_threshold': round(self.switch_threshold, 2),
            'cycle_target': round(self.cycle_target, 2),
            'switches': self.switches,
            'adaptations': dict(self.adaptations),
        }


# ===========================================================================
#  DISTRIBUTED ADAPTIVE EXECUTOR — one episode
# ===========================================================================
class DistributedAdaptiveExecutor:
    """
    Runs one episode with one AdaptiveSignalController + one
    SignalStateMachine per intersection.
    """

    def __init__(self, seed: int = 0, car_following: str = 'SAFE'):
        self.seed = seed
        self.rng = random.Random(seed)
        self.car_model = CarFollowingModel(car_following, self.rng)

        # --- build the network (reuse the base Intersection) ------------
        self.nodes: dict[str, Intersection] = {}
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                nid = f"J{r+1}{c+1}"
                node = Intersection(nid, self.rng)
                node.row, node.col = r, c
                self.nodes[nid] = node

        # --- one controller + state machine per intersection ------------
        self.controllers: dict[str, AdaptiveSignalController] = {}
        self.machines:    dict[str, SignalStateMachine]       = {}
        for nid in self.nodes:
            self.controllers[nid] = AdaptiveSignalController(
                node_id=nid, seed=seed + hash(nid) % 10_000)
            self.machines[nid] = SignalStateMachine(node_id=nid)

        self.bus = SignalBus()

        # --- metric accumulators ----------------------------------------
        self.metrics = {
            'travel_time_sum_s': 0.0,
            'queue_sum':         0.0,
            'waiting_sum_s':     0.0,
            'vehicles_completed': 0.0,
            'samples':           0,
        }
        self.timeline: list[dict] = []

    # ------------------------------------------------------------- helpers
    def _neighbours(self, nid: str) -> list[str]:
        r, c = self.nodes[nid].row, self.nodes[nid].col
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                out.append(f"J{nr+1}{nc+1}")
        return out

    # ------------------------------------------------------------- episode
    def run(self, n_steps: int = 200,
            base_demand_ns: float = 0.30,
            base_demand_ew: float = 0.30,
            congestion_start: float = 0.4,
            congestion_end:   float = 0.9) -> dict:
        dt = DECISION_INTERVAL_S

        for t in range(n_steps):
            c = congestion_start + (congestion_end - congestion_start) * t / max(n_steps - 1, 1)
            cur_time = t * dt

            # ---- 1. every controller decides, using the shared bus ----
            for nid, ctrl in self.controllers.items():
                node = self.nodes[nid]
                nbrs = self.bus.query(self._neighbours(nid))
                ctrl.decide(self.machines[nid],
                            node.queue_ns(), node.queue_ew(), nbrs)

            # ---- 2. every signal machine advances (yellow/all-red) ----
            for nid, sm in self.machines.items():
                sm.step(dt, self.nodes[nid].queue_ns(),
                        self.nodes[nid].queue_ew())

            # ---- 3. publish new snapshots to the bus ------------------
            for nid, sm in self.machines.items():
                self.bus.publish(sm.snapshot())

            # ---- 4. advance each intersection's queue dynamics --------
            for nid, node in self.nodes.items():
                served = self.machines[nid].served_approaches()
                # arrivals
                for a in ('N', 'S'):
                    node.q[a] += 0.5 * base_demand_ns * (0.8 + 0.6 * c) * dt
                for a in ('E', 'W'):
                    node.q[a] += 0.5 * base_demand_ew * (0.8 + 0.6 * c) * dt
                # discharge during green
                for a in served:
                    d = min(node.q[a], SATURATION_FLOW_VPS * dt)
                    node.q[a]      -= d
                    node.departures += d
                    node.completed  += d
                # waiting accumulation
                for a in node.q:
                    node.W[a] += node.q[a] * dt

            # ---- 5. metrics ------------------------------------------
            for node in self.nodes.values():
                self.metrics['waiting_sum_s']      += sum(node.W.values())
                self.metrics['queue_sum']          += node.total_queue()
                self.metrics['samples']            += 1
                total_q = node.total_queue()
                self.metrics['vehicles_completed'] += min(2.5, 0.025 * total_q)
                cong = min(0.99, c)
                cf = self.car_model.simulate_step(
                    'free_flow' if cong < 0.5 else 'congested',
                    int(self.machines[node.id].current_phase), 0, 0, cong)
                self.metrics['travel_time_sum_s'] += (
                    0.6 * cf['in_system_s'] + 0.4 * 100.0)

            # ---- 6. per-step timeline (for dashboards / plots) -------
            self.timeline.append({
                't': cur_time,
                'phases': {nid: int(sm.current_phase)
                           for nid, sm in self.machines.items()},
                'queues': {nid: round(n.total_queue(), 2)
                           for nid, n in self.nodes.items()},
            })

        return self.metrics

    # ------------------------------------------------------------- metrics
    def get_metrics(self) -> dict:
        s = max(self.metrics['samples'], 1)
        sim_s = s * DECISION_INTERVAL_S
        agg_adapt = defaultdict(int)
        total_switches = 0
        for ctrl in self.controllers.values():
            total_switches += ctrl.switches
            for k, v in ctrl.adaptations.items():
                agg_adapt[k] += v
        return {
            'travel_time_s':  round(self.metrics['travel_time_sum_s'] / s, 2),
            'queue_veh':      round(self.metrics['queue_sum'] / s, 2),
            'waiting_time_s': round(self.metrics['waiting_sum_s'] / s, 2),
            'throughput_vph': round(self.metrics['vehicles_completed']
                                     / max(sim_s, 1.0) * 3600.0, 1),
            'total_switches': total_switches,
            'adaptations':    dict(agg_adapt),
        }


# ===========================================================================
#  PARALLEL MULTI-EPISODE EXECUTION
# ===========================================================================
def _worker(args):
    seed, n_steps = args
    ex = DistributedAdaptiveExecutor(seed=seed)
    ex.run(n_steps=n_steps)
    m = ex.get_metrics()
    m['seed'] = seed
    return m


def run_parallel(n_episodes: int = 5, n_steps: int = 200) -> list[dict]:
    """
    Run n_episodes concurrently using ProcessPoolExecutor.
    Returns per-episode metrics (mean ± 95 % CI computed by the caller).
    """
    seeds = list(range(1, n_episodes + 1))
    with ProcessPoolExecutor(max_workers=n_episodes) as pool:
        futures = [pool.submit(_worker, (s, n_steps)) for s in seeds]
        return [f.result() for f in as_completed(futures)]


# ===========================================================================
#  DASHBOARD
# ===========================================================================
PHASE_GLYPH = {
    int(SignalPhase.NS_GREEN):  '↑',   # NS green
    int(SignalPhase.NS_YELLOW): '⚠',
    int(SignalPhase.ALL_RED_1): '·',
    int(SignalPhase.EW_GREEN):  '→',   # EW green
    int(SignalPhase.EW_YELLOW): '⚠',
    int(SignalPhase.ALL_RED_2): '·',
}


def print_dashboard(ex: DistributedAdaptiveExecutor, every: int = 10):
    """Print a per-timestep grid of intersection phases."""
    print("\n  t (s)   " + "  ".join(f"J{r+1}{c+1}"
                                       for r in range(GRID_ROWS)
                                       for c in range(GRID_COLS)))
    for step, row in enumerate(ex.timeline):
        if step % every: continue
        cells = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                nid = f"J{r+1}{c+1}"
                ph = row['phases'][nid]
                cells.append(f"{PHASE_GLYPH[ph]}{row['queues'][nid]:>3.0f}")
        print(f"  {row['t']:>5.0f}   " + "  ".join(cells))
    print("\n  legend: ↑ NS green | → EW green | ⚠ yellow | · all-red")
    print("  number = total queue at that intersection")


def print_controller_report(ex: DistributedAdaptiveExecutor):
    print("\n" + "=" * 100)
    print("  PER-CONTROLLER ADAPTATION REPORT")
    print("=" * 100)
    print(f"{'Node':<6}{'min_green':>12}{'switch_thr':>14}"
          f"{'cycle_tgt':>12}{'switches':>12}{'adapt (mg/st/ct)':>22}")
    print('-' * 100)
    for nid, ctrl in ex.controllers.items():
        s = ctrl.snapshot()
        a = s['adaptations']
        print(f"{nid:<6}{s['min_green']:>12.2f}{s['switch_threshold']:>14.2f}"
              f"{s['cycle_target']:>12.2f}{s['switches']:>12d}"
              f"{a['min_green']:>8d}/{a['switch_threshold']:<6d}"
              f"{a['cycle_target']:<6d}")
    print()


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--parallel', type=int, default=None,
                    help='run N concurrent episodes instead of one')
    args = ap.parse_args()

    print("#" * 100)
    print("#  DISTRIBUTED ADAPTIVE SIGNAL EXECUTION")
    print("#" * 100)

    if args.parallel:
        print(f"\n[1] Running {args.parallel} episodes concurrently "
              f"({args.steps} steps each) ...")
        rows = run_parallel(args.parallel, args.steps)
        print("\n[2] Per-episode results:")
        print(f"{'seed':>6}{'travel':>10}{'queue':>9}"
              f"{'wait':>9}{'throughput':>14}{'switches':>12}")
        print('-' * 60)
        for r in sorted(rows, key=lambda x: x['seed']):
            print(f"{r['seed']:>6}{r['travel_time_s']:>10.2f}"
                  f"{r['queue_veh']:>9.2f}{r['waiting_time_s']:>9.2f}"
                  f"{r['throughput_vph']:>14.1f}"
                  f"{r['total_switches']:>12d}")

        # aggregate
        print("\n[3] Aggregate over episodes:")
        for k in ('travel_time_s', 'queue_veh', 'waiting_time_s',
                  'throughput_vph', 'total_switches'):
            xs = [r[k] for r in rows]
            mean = statistics.fmean(xs)
            sd   = statistics.stdev(xs) if len(xs) > 1 else 0.0
            ci   = 1.96 * sd / math.sqrt(len(xs))
            print(f"    {k:<20} {mean:>10.2f}  ± {ci:>5.2f}")
    else:
        print("\n[1] Running a single episode ...")
        ex = DistributedAdaptiveExecutor(seed=1)
        ex.run(n_steps=args.steps)
        m = ex.get_metrics()
        print_dashboard(ex, every=10)
        print_controller_report(ex)
        print("\n[2] Episode metrics:")
        for k, v in m.items():
            print(f"    {k:<18} {v}")

    print("\nDone.")


if __name__ == '__main__':
    main()