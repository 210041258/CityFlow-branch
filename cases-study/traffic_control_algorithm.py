#!/usr/bin/env python3
"""
traffic_control_algorithm.py
============================
Self-contained traffic-signal control simulator + benchmark.
Final revision — expression-selection dispatch, grouped natural scenarios,
runtime composition, and multi-seed evaluation with 95 % CIs.

Thesis reference:
    Albreem et al., "Latency-Resilient Cooperative Multi-Agent Reinforcement
    Learning for Adaptive Traffic Signal Control", IUT, 2026.

Public API
----------
    TrafficControlAlgorithm(algorithm_type, scenario, natural, seed,
                            car_following, beta, lambda_decay_per_ms,
                            stale_threshold_ms, p_drop, training)
        .control_signal(...)          -> (phase, green_duration_s)
        .get_metrics()                -> dict
        .cooperative_reward(...)      -> float

    TrafficControlAlgorithm.list_controllers(category=None)  -> [str]
    TrafficControlAlgorithm.get_expression(name)             -> ControllerSpec
    TrafficControlAlgorithm.compose(a, b, w, new_name=None)  -> str

    simulate_episode(ctrl, n_steps=...)                      -> ctrl
    run_multi_seed(ctrl, ..., seeds=(1..5))                  -> dict
    export_csv(rows, path)                                   -> None

Run:
    python traffic_control_algorithm.py
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Optional


# ===========================================================================
#  CONSTANTS
# ===========================================================================

# --- communication severity scenarios (Table 9) ---------------------------
COMMUNICATION_SCENARIOS = {
    'S1': {'latency_ms':   0.0, 'loss_rate': 0.00, 'label': 'Ideal'},
    'S2': {'latency_ms': 100.0, 'loss_rate': 0.05, 'label': 'Moderate'},
    'S3': {'latency_ms': 300.0, 'loss_rate': 0.15, 'label': 'Severe'},
    'S4': {'latency_ms': 600.0, 'loss_rate': 0.30, 'label': 'Extreme'},
}

# --- natural operating conditions, grouped by family ----------------------
NATURAL_SCENARIOS = {
    # weather
    'N-clear':       {'label': 'Clear weather',    'capacity': 1.00,
                      'demand_mult': 1.00, 'direction_bias':  0.00,
                      'sensor_reliability': 1.00},
    'N-rain':        {'label': 'Heavy rain',       'capacity': 0.85,
                      'demand_mult': 1.00, 'direction_bias':  0.00,
                      'sensor_reliability': 0.95},
    'N-snow':        {'label': 'Snow / ice',       'capacity': 0.55,
                      'demand_mult': 0.80, 'direction_bias':  0.00,
                      'sensor_reliability': 0.90},
    'N-fog':         {'label': 'Dense fog',        'capacity': 0.75,
                      'demand_mult': 0.95, 'direction_bias':  0.00,
                      'sensor_reliability': 0.85},
    # incident
    'N-accident':    {'label': 'Accident',         'capacity': 0.65,
                      'demand_mult': 1.00, 'direction_bias':  0.20,
                      'sensor_reliability': 1.00},
    'N-workzone':    {'label': 'Work zone',        'capacity': 0.70,
                      'demand_mult': 1.00, 'direction_bias':  0.10,
                      'sensor_reliability': 1.00},
    # demand
    'N-rush-am':     {'label': 'Morning rush',     'capacity': 1.00,
                      'demand_mult': 1.50, 'direction_bias':  0.60,
                      'sensor_reliability': 1.00},
    'N-rush-pm':     {'label': 'Evening rush',     'capacity': 1.00,
                      'demand_mult': 1.50, 'direction_bias': -0.60,
                      'sensor_reliability': 1.00},
    'N-night':       {'label': 'Overnight flow',   'capacity': 1.00,
                      'demand_mult': 0.25, 'direction_bias':  0.00,
                      'sensor_reliability': 1.00},
    'N-event':       {'label': 'Stadium egress',   'capacity': 1.00,
                      'demand_mult': 1.80, 'direction_bias':  0.80,
                      'sensor_reliability': 1.00},
    'N-school':      {'label': 'School pick-up',   'capacity': 0.90,
                      'demand_mult': 1.35, 'direction_bias':  0.30,
                      'sensor_reliability': 1.00},
    # failure
    'N-sensor-fail': {'label': 'Detector failure', 'capacity': 1.00,
                      'demand_mult': 1.00, 'direction_bias':  0.00,
                      'sensor_reliability': 0.40},
    'N-comm-blackout': {'label': 'V2X blackout',   'capacity': 1.00,
                        'demand_mult': 1.00, 'direction_bias': 0.00,
                        'sensor_reliability': 1.00,
                        'comm_override': 'S3'},
}

NATURAL_GROUPS = {
    'weather':  ['N-clear', 'N-rain', 'N-snow', 'N-fog'],
    'incident': ['N-accident', 'N-workzone'],
    'demand':   ['N-rush-am', 'N-rush-pm', 'N-night', 'N-event', 'N-school'],
    'failure':  ['N-sensor-fail', 'N-comm-blackout'],
}

# --- logistic degradation model (Eq. 5) -----------------------------------
DEGRADATION_L, DEGRADATION_K, DEGRADATION_X0_MS = 302.23, 0.0143, 101.54

# --- signal timing (§4.2) --------------------------------------------------
DECISION_INTERVAL_S = 10.0
GREEN_MIN_S         = 10.0
GREEN_MAX_S         = 60.0
YELLOW_S            =  3.0
DEFAULT_BETA        =  0.5

# --- network topology (§4.2, Figure 1) ------------------------------------
GRID_ROWS, GRID_COLS = 3, 3
SEGMENT_LENGTH_M     = 300.0
FREE_FLOW_SPEED_MS   = 50.0 * 1000 / 3600.0
SATURATION_FLOW_VPS  = 0.5                    # veh/s per green approach

# --- car-following benchmark (Table 12) -----------------------------------
CAR_FOLLOWING_BENCHMARK = {
    'GIPPS': {
        'free_flow': {'in_system_s': 114.77, 'sd': 0.42, 'speed_ms': 18.96,
                      'sd_speed': 0.08, 'efficiency_pct': 94.3,
                      'delay_s': 6.54,  'sd_delay': 0.11},
        'congested': {'in_system_s': 187.42, 'sd': 1.85, 'speed_ms': 10.14,
                      'sd_speed': 0.12, 'efficiency_pct': 78.2,
                      'delay_s': 42.87, 'sd_delay': 0.95},
    },
    'IDM': {
        'free_flow': {'in_system_s': 113.19, 'sd': 0.38, 'speed_ms': 19.20,
                      'sd_speed': 0.07, 'efficiency_pct': 95.6,
                      'delay_s': 4.96,  'sd_delay': 0.09},
        'congested': {'in_system_s': 178.63, 'sd': 1.52, 'speed_ms': 10.63,
                      'sd_speed': 0.10, 'efficiency_pct': 82.1,
                      'delay_s': 36.24, 'sd_delay': 0.78},
    },
    'SAFE': {
        'free_flow': {'in_system_s': 112.39, 'sd': 0.35, 'speed_ms': 19.33,
                      'sd_speed': 0.06, 'efficiency_pct': 96.3,
                      'delay_s': 4.16,  'sd_delay': 0.08},
        'congested': {'in_system_s': 171.28, 'sd': 1.38, 'speed_ms': 11.09,
                      'sd_speed': 0.09, 'efficiency_pct': 85.6,
                      'delay_s': 29.51, 'sd_delay': 0.62},
    },
}


# ===========================================================================
#  HELPERS
# ===========================================================================
def degrade_logistic(x, L=DEGRADATION_L, k=DEGRADATION_K,
                     x0=DEGRADATION_X0_MS):
    """Logistic degradation law — Eq. (5)."""
    return L / (1.0 + math.exp(-k * (x - x0)))


def severity_index(latency_ms, loss_rate, w_tau=0.5, w_p=0.5,
                   tau_min=0.0, tau_max=600.0):
    """Severity index ρ — Eq. (8)."""
    return (w_tau * (latency_ms - tau_min) / (tau_max - tau_min)
            + w_p * loss_rate)


def aoi_decay_kernel(aoi_ms, lam=0.01):
    """AoI decay kernel κ(τ) = exp(-λτ) — Eq. (12)."""
    return math.exp(-lam * aoi_ms)


# ===========================================================================
#  MESSAGE + IMPAIRMENT CHANNEL  (Eqs. 6, 7, 14)
# ===========================================================================
@dataclass
class Message:
    sender:    str
    receiver:  str
    t_send_ms: float
    seq:       int
    phase:     int
    queue_ns:  float
    queue_ew:  float
    elapsed_s: float


class ImpairmentChannel:
    """Truncated log-normal latency + Gilbert-Elliott correlated loss."""

    def __init__(self, latency_mean_ms, loss_rate,
                 latency_sigma=0.5, p_gb=0.30, p_bg=0.70, rng=None):
        self.latency_mean  = latency_mean_ms
        self.latency_sigma = latency_sigma
        self.loss_rate     = loss_rate
        self.rng           = rng or random.Random()
        self.p_gb, self.p_bg, self.p_b = p_gb, p_bg, 0.90
        pi_b      = self.p_gb / (self.p_gb + self.p_bg)
        pi_g      = 1.0 - pi_b
        self.p_g  = max(0.0, (loss_rate - pi_b * self.p_b)
                        / max(pi_g, 1e-6))
        self.last_seq  = defaultdict(lambda: -1)
        self.ge_state  = defaultdict(lambda: 'G')
        self.sent_count      = 0
        self.delivered_count = 0

    def _latency(self):
        if self.latency_mean <= 0:
            return 0.0
        mu, sigma = math.log(self.latency_mean), self.latency_sigma
        for _ in range(16):
            tau = self.rng.lognormvariate(mu, sigma)
            if 0.1 * self.latency_mean <= tau <= 5 * self.latency_mean:
                return tau
        return self.latency_mean

    def _loss(self, link):
        state = self.ge_state[link]
        if state == 'G':
            p_loss = self.p_g
            if self.rng.random() < self.p_gb:
                self.ge_state[link] = 'B'
        else:
            p_loss = self.p_b
            if self.rng.random() < self.p_bg:
                self.ge_state[link] = 'G'
        return self.rng.random() < p_loss

    def transmit(self, msg):
        self.sent_count += 1
        if self._loss(f"{msg.sender}->{msg.receiver}"):
            return None
        self._latency()                                # recorded metadata
        if msg.seq <= self.last_seq[msg.sender]:
            return None                                # out-of-order
        self.last_seq[msg.sender] = msg.seq
        self.delivered_count     += 1
        return msg


# ===========================================================================
#  CAR-FOLLOWING MODEL
# ===========================================================================
class CarFollowingModel:
    """SAFE / GIPPS / IDM empirical model — Table 12."""

    def __init__(self, name='SAFE', rng=None):
        if name not in CAR_FOLLOWING_BENCHMARK:
            raise ValueError(f"Unknown car-following model '{name}'.")
        self.name  = name
        self.rng   = rng or random.Random()
        self.bench = CAR_FOLLOWING_BENCHMARK[name]

    def simulate_step(self, regime, phase, q_ns, q_ew, congestion):
        base    = self.bench[regime]
        jitter  = self.rng.gauss(0.0, base['sd'])
        total_q = max(q_ns + q_ew, 1.0)
        in_sys  = base['in_system_s'] + jitter + 0.35 * total_q * (0.5 + congestion)
        if   self.name == 'GIPPS': in_sys *= 1.02
        elif self.name == 'IDM':   in_sys *= 1.01
        return {'in_system_s': in_sys,
                'speed_ms': max(0.5, base['speed_ms'] - 0.02 * total_q),
                'delay_s':  max(0.0, base['delay_s'] + 0.15 * total_q),
                'efficiency_pct': base['efficiency_pct']}


# ===========================================================================
#  NETWORK
# ===========================================================================
class Intersection:
    """One signalised four-way intersection — real queue dynamics."""

    def __init__(self, node_id, rng):
        self.id, self.rng = node_id, rng
        self.q = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        self.W = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        self.phase, self.phase_elapsed = 0, 0.0
        self.arrivals = self.departures = self.completed = 0.0
        self.seq = 0
        self.row = self.col = 0

    def step(self, dt, arrive_ns, arrive_ew, green_active_phase, green_time):
        for a in ('N', 'S'):
            self.q[a] += 0.5 * arrive_ns * dt
            self.arrivals += 0.5 * arrive_ns * dt
        for a in ('E', 'W'):
            self.q[a] += 0.5 * arrive_ew * dt
            self.arrivals += 0.5 * arrive_ew * dt

        serviceable = (('N', 'S') if green_active_phase == 0 else
                       ('E', 'W') if green_active_phase == 1 else ())
        for a in serviceable:
            d = min(self.q[a], SATURATION_FLOW_VPS * green_time)
            self.q[a]      -= d
            self.departures += d
            self.completed  += d

        for a in self.q:
            self.W[a] += self.q[a] * dt
        self.phase_elapsed += dt
        if green_active_phase is not None and green_active_phase != self.phase:
            self.phase, self.phase_elapsed = green_active_phase, 0.0

    def total_queue(self): return sum(self.q.values())
    def queue_ns(self):    return self.q['N'] + self.q['S']
    def queue_ew(self):    return self.q['E'] + self.q['W']

    def snapshot(self):
        return {'id': self.id, 'seq': self.seq, 'phase': self.phase,
                'queue_ns': self.queue_ns(), 'queue_ew': self.queue_ew(),
                'elapsed_s': self.phase_elapsed}


class TrafficNetwork:
    """3 × 3 grid of signalised intersections."""

    def __init__(self, rng):
        self.rng = rng
        self.nodes = {}
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                nid = f"J{r+1}{c+1}"
                self.nodes[nid] = Intersection(nid, rng)
                self.nodes[nid].row, self.nodes[nid].col = r, c

    def neighbour_snapshots(self, node_id):
        r, c = self.nodes[node_id].row, self.nodes[node_id].col
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                out.append(self.nodes[f"J{nr+1}{nc+1}"].snapshot())
        return out


# ===========================================================================
#  CONTROLLER REGISTRY  —  expression selection
# ===========================================================================
@dataclass(frozen=True)
class ControllerSpec:
    """One registered controller expression."""
    name:     str
    category: str
    inputs:   tuple
    fn:       Callable
    doc:      str = ''


def controller(name: str, category: str, inputs: tuple, doc: str = ''):
    """Decorator — register a method as a named controller expression."""
    def _decorator(fn):
        fn._controller_spec = ControllerSpec(name, category, inputs, fn, doc)
        return fn
    return _decorator


# ===========================================================================
#  MAIN CLASS
# ===========================================================================
class TrafficControlAlgorithm:
    """Unified traffic-signal controller with expression-selection dispatch."""

    CONTROLLERS: dict[str, ControllerSpec] = {}
    CATEGORIES = ('classical', 'coordinated', 'learning', 'mixed', 'field')

    # -------------------------------------------------------------- init
    def __init__(self,
                 algorithm_type: str = 'adaptive',
                 scenario: str = 'S1',
                 natural: str = 'N-clear',
                 seed: int = 0,
                 car_following: str = 'SAFE',
                 beta: float = DEFAULT_BETA,
                 lambda_decay_per_ms: float = 0.01,
                 stale_threshold_ms: float = 500.0,
                 p_drop: float = 0.10,
                 training: bool = False):
        if algorithm_type not in self.CONTROLLERS:
            raise ValueError(
                f"Unknown algorithm '{algorithm_type}'. "
                f"Available: {sorted(self.CONTROLLERS)}")
        if scenario not in COMMUNICATION_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'.")
        if natural not in NATURAL_SCENARIOS:
            raise ValueError(f"Unknown natural scenario '{natural}'.")

        override = NATURAL_SCENARIOS[natural].get('comm_override')
        if override:
            scenario = override

        self.algorithm_type = algorithm_type
        self.scenario       = scenario
        self.natural        = natural
        self.comm_cfg       = COMMUNICATION_SCENARIOS[scenario]
        self.nat_cfg        = NATURAL_SCENARIOS[natural]
        self.rng            = random.Random(seed)
        self.car_model      = CarFollowingModel(car_following, self.rng)

        self.beta         = beta
        self.lambda_decay = lambda_decay_per_ms
        self.stale_thresh = stale_threshold_ms
        self.p_drop       = p_drop
        self.training     = training

        self.channel = ImpairmentChannel(self.comm_cfg['latency_ms'],
                                         self.comm_cfg['loss_rate'],
                                         rng=self.rng)
        self.aoi_ms      = defaultdict(float)
        self.fwd_history = defaultdict(lambda: deque(maxlen=8))

        self.optimization_events = 0
        self.phase_decisions: list[dict] = []
        self.seq             = 0
        self.current_time_ms = 0.0
        self.cycle_length_s  = 60.0

        self.m = {'travel_time_sum_s':  0.0,
                  'queue_sum':          0.0,
                  'waiting_sum_s':      0.0,
                  'vehicles_completed': 0.0,
                  'samples':            0}

    # ---------------------------------------------------------- registry API
    @classmethod
    def register(cls, spec: ControllerSpec):
        cls.CONTROLLERS[spec.name] = spec

    @classmethod
    def list_controllers(cls, category: Optional[str] = None) -> list[str]:
        if category is None:
            return sorted(cls.CONTROLLERS)
        return sorted(n for n, s in cls.CONTROLLERS.items()
                      if s.category == category)

    @classmethod
    def get_expression(cls, name: str) -> ControllerSpec:
        return cls.CONTROLLERS[name]

    @classmethod
    def compose(cls, name_a: str, name_b: str, weight_a: float = 0.5,
                new_name: Optional[str] = None) -> str:
        """Register a NEW controller that blends two existing expressions."""
        spec_a = cls.CONTROLLERS[name_a]
        spec_b = cls.CONTROLLERS[name_b]
        inputs = tuple(sorted(set(spec_a.inputs) | set(spec_b.inputs)))
        w = max(0.0, min(1.0, weight_a))
        cname = new_name or f"{name_a}|{name_b}:{w:.2f}"

        def blended(self, **kw):
            phase_a, dur_a = spec_a.fn(self, **{k: kw[k] for k in spec_a.inputs})
            phase_b, dur_b = spec_b.fn(self, **{k: kw[k] for k in spec_b.inputs})
            phase = phase_a if w >= 0.5 else phase_b
            return phase, w * dur_a + (1 - w) * dur_b

        cls.register(ControllerSpec(cname, 'composed', inputs, blended,
                                    f"{name_a} ({w:.0%}) + {name_b} "
                                    f"({1-w:.0%})"))
        return cname

    # ---------------------------------- communication-augmented observation
    def build_observation(self, local_state, neighbours):
        """Communication-augmented observation — Eq. (2)."""
        self.seq += 1
        received = []
        for nb in (neighbours or []):
            msg = Message(nb.get('id', 'nb'), 'self', self.current_time_ms,
                          nb.get('seq', 0), nb.get('phase', 0),
                          nb.get('queue_ns', 0.0), nb.get('queue_ew', 0.0),
                          nb.get('elapsed_s', 0.0))
            delivered = self.channel.transmit(msg)
            if delivered is None:
                self.aoi_ms[msg.sender] += DECISION_INTERVAL_S * 1000.0
                continue
            aoi = self.aoi_ms[msg.sender] + self.comm_cfg['latency_ms']
            self.aoi_ms[msg.sender] = aoi
            received.append((delivered, aoi))
            self.fwd_history[delivered.sender].append(
                (self.current_time_ms, delivered.queue_ns, delivered.queue_ew))
        return {'local': local_state, 'received': received}

    # =====================================================================
    #  CONTROLLER EXPRESSIONS  (registered via @controller)
    # =====================================================================

    # ------- classical ----------------------------------------------------
    @controller('fixed-time', 'classical', ('current_time',))
    def fixed_time_control(self, current_time):
        return int((current_time % 60.0) / 30.0), 30.0

    @controller('actuated', 'classical',
                ('veh_ns', 'veh_ew', 'gap_ns', 'gap_ew'))
    def actuated_control(self, veh_ns, veh_ew, gap_ns, gap_ew):
        if veh_ns >= veh_ew:
            phase, demand, gap = 0, veh_ns, gap_ns
        else:
            phase, demand, gap = 1, veh_ew, gap_ew
        total = max(veh_ns + veh_ew, 1.0)
        green = max(GREEN_MIN_S, min(GREEN_MAX_S,
                                     GREEN_MAX_S * demand / total))
        if gap >= 3.0:
            green = GREEN_MIN_S
        self.optimization_events += 1
        return phase, green

    @controller('adaptive', 'classical',
                ('veh_ns', 'veh_ew', 'congestion_eff'))
    def adaptive_control(self, veh_ns, veh_ew, congestion_eff):
        ns_r = veh_ns / max(veh_ns + veh_ew, 1.0)
        if congestion_eff > 0.7:
            ns_t, ew_t = (35.0, 20.0) if ns_r > 0.6 else (20.0, 35.0)
        else:
            ns_t = ew_t = 30.0
        phase = 0 if ns_r > 0.5 else 1
        self.optimization_events += 1
        return phase, (ns_t if phase == 0 else ew_t)

    @controller('optimized', 'classical',
                ('veh_ns', 'veh_ew', 'waiting_ns', 'waiting_ew',
                 'congestion_eff'))
    def optimized_control(self, veh_ns, veh_ew, waiting_ns, waiting_ew,
                          congestion_eff):
        u_ns = 0.4 * veh_ns + 0.6 * waiting_ns
        u_ew = 0.4 * veh_ew + 0.6 * waiting_ew
        ns_p = u_ns / max(u_ns + u_ew, 1e-6)
        cycle = 60.0 * (0.8 if congestion_eff > 0.8
                        else 1.2 if congestion_eff < 0.3 else 1.0)
        green = max(GREEN_MIN_S, cycle - YELLOW_S)
        phase = 0 if ns_p > 0.5 else 1
        self.optimization_events += 1
        return phase, green * (ns_p if phase == 0 else 1 - ns_p)

    # ------- coordinated --------------------------------------------------
    @controller('coordinated', 'coordinated',
                ('veh_ns', 'veh_ew', 'congestion_eff', 'row', 'col'))
    def coordinated_control(self, veh_ns, veh_ew, congestion_eff, row, col):
        Y     = min(0.85, 0.35 + 0.5 * congestion_eff)
        L     = 2.0 * YELLOW_S
        cycle = max(40.0, min(120.0, (1.5 * L + 5.0) / max(1.0 - Y, 0.05)))
        self.cycle_length_s = cycle
        _ = ((row + col) * (SEGMENT_LENGTH_M / FREE_FLOW_SPEED_MS)) % cycle
        ns_r  = veh_ns / max(veh_ns + veh_ew, 1.0)
        green = max(GREEN_MIN_S, cycle - YELLOW_S)
        phase = 0 if ns_r > 0.5 else 1
        self.optimization_events += 1
        return phase, green * (ns_r if phase == 0 else 1 - ns_r)

    # ------- learning -----------------------------------------------------
    @controller('independent-dqn', 'learning',
                ('veh_ns', 'veh_ew', 'waiting_ns', 'waiting_ew',
                 'congestion_eff'))
    def independent_dqn_control(self, veh_ns, veh_ew, waiting_ns, waiting_ew,
                                congestion_eff):
        phase = (0 if (veh_ns + 0.5 * waiting_ns)
                 >= (veh_ew + 0.5 * waiting_ew) else 1)
        self.optimization_events += 1
        return phase, 30.0 + 10.0 * congestion_eff

    @controller('cooperative-marl', 'learning',
                ('veh_ns', 'veh_ew', 'congestion_eff', 'obs'))
    def cooperative_marl_control(self, veh_ns, veh_ew, congestion_eff, obs):
        agg_ns, agg_ew = veh_ns, veh_ew
        for msg, _ in obs['received']:
            agg_ns += msg.queue_ns
            agg_ew += msg.queue_ew
        phase = 0 if agg_ns >= agg_ew else 1
        self.optimization_events += 1
        return phase, 30.0 + 15.0 * congestion_eff

    @controller('resilient-marl', 'learning',
                ('veh_ns', 'veh_ew', 'congestion_eff', 'obs'))
    def resilient_marl_control(self, veh_ns, veh_ew, congestion_eff, obs):
        """Resilient MARL with M1–M5 active."""
        agg_ns, agg_ew = veh_ns, veh_ew

        # M2 + M5 — AoI-weighted aggregation with training-time dropout
        for msg, aoi in obs['received']:
            if self.training and self.rng.random() < self.p_drop:
                continue
            w = aoi_decay_kernel(aoi, self.lambda_decay)
            agg_ns += w * msg.queue_ns
            agg_ew += w * msg.queue_ew

        # M3 — predictive compensation for stale neighbours
        for nb_id, hist in self.fwd_history.items():
            if not hist or self.aoi_ms[nb_id] <= self.stale_thresh:
                continue
            if len(hist) >= 2:
                t0, ns0, ew0 = hist[-2]
                t1, ns1, ew1 = hist[-1]
                dt = max(t1 - t0, 1.0)
                pred_ns = ns1 + (ns1 - ns0) / dt * self.aoi_ms[nb_id]
                pred_ew = ew1 + (ew1 - ew0) / dt * self.aoi_ms[nb_id]
            else:
                pred_ns, pred_ew = hist[-1][1], hist[-1][2]
            w = aoi_decay_kernel(self.aoi_ms[nb_id], self.lambda_decay)
            agg_ns += w * pred_ns
            agg_ew += w * pred_ew

        phase = 0 if agg_ns >= agg_ew else 1
        self.optimization_events += 1
        return phase, 30.0 + 12.0 * congestion_eff

    # ------- field-inspired ----------------------------------------------
    @controller('max-pressure', 'field',
                ('veh_ns', 'veh_ew', 'current_phase'))
    def max_pressure_control(self, veh_ns, veh_ew, current_phase):
        diff   = (veh_ns - 0.6 * veh_ew) - (veh_ew - 0.6 * veh_ns)
        switch = (diff < -3.0) if current_phase == 0 else (diff > 3.0)
        new_phase = (1 - current_phase) if switch else current_phase
        dur = 20.0 + 0.6 * abs(diff) + 0.4 * (veh_ns + veh_ew)
        self.optimization_events += 1
        return new_phase, max(GREEN_MIN_S, min(GREEN_MAX_S, dur))

    @controller('scats', 'field',
                ('veh_ns', 'veh_ew', 'congestion_eff', 'row', 'col'))
    def scats_control(self, veh_ns, veh_ew, congestion_eff, row, col):
        deg_sat = min(1.0, (veh_ns + veh_ew) / 40.0 * (0.6 + congestion_eff))
        cycle   = max(60.0, min(120.0, 60.0 * (1.0 + 0.4 * deg_sat)))
        self.cycle_length_s = cycle
        ns_r  = veh_ns / max(veh_ns + veh_ew, 1.0)
        green = max(GREEN_MIN_S, cycle - 2 * YELLOW_S)
        phase = 0 if ns_r > 0.5 else 1
        self.optimization_events += 1
        return phase, green * (ns_r if phase == 0 else 1 - ns_r)

    @controller('scoot', 'field', ('veh_ns', 'veh_ew', 'congestion_eff'))
    def scoot_control(self, veh_ns, veh_ew, congestion_eff):
        cfp   = (veh_ns + veh_ew) * (0.5 + congestion_eff)
        cycle = (max(50.0, self.cycle_length_s - 4.0) if cfp > 60.0
                 else min(120.0, self.cycle_length_s + 4.0) if cfp < 25.0
                 else self.cycle_length_s)
        self.cycle_length_s = cycle
        ns_r  = veh_ns / max(veh_ns + veh_ew, 1.0)
        green = max(GREEN_MIN_S, cycle - 2 * YELLOW_S)
        phase = 0 if ns_r > 0.5 else 1
        self.optimization_events += 1
        return phase, green * (ns_r if phase == 0 else 1 - ns_r)

    @controller('multiband', 'field',
                ('veh_ns', 'veh_ew', 'congestion_eff'))
    def multiband_control(self, veh_ns, veh_ew, congestion_eff):
        bias  = self.nat_cfg['direction_bias']
        cycle = 90.0 if abs(bias) > 0.4 else 70.0
        self.cycle_length_s = cycle
        w_dom = 0.60 + 0.15 * abs(bias)
        if bias >= 0:
            phase, dur = 0, (cycle - 2 * YELLOW_S) * w_dom
        else:
            phase, dur = 1, (cycle - 2 * YELLOW_S) * w_dom
        self.optimization_events += 1
        return phase, dur

    @controller('fuzzy-logic', 'field',
                ('veh_ns', 'veh_ew', 'congestion_eff'))
    def fuzzy_logic_control(self, veh_ns, veh_ew, congestion_eff):
        def tri(x, lo, mid, hi):
            if x <= lo or x >= hi:
                return 0.0
            return (x - lo) / (mid - lo) if x <= mid else (hi - x) / (hi - mid)
        nsL, ewL = tri(veh_ns, 5, 30, 60), tri(veh_ew, 5, 30, 60)
        high_c   = max(0.0, (congestion_eff - 0.5) / 0.5)
        r1, r2, r3 = min(nsL, 1 - ewL), min(ewL, 1 - nsL), min(nsL, ewL)
        r4 = min(1 - nsL, 1 - ewL) * (1 - high_c)
        ns_score = r1 + 0.5 * r3 + 0.3 * r4
        ew_score = r2 + 0.5 * r3 + 0.3 * r4
        ns_p = ns_score / max(ns_score + ew_score, 1e-6)
        cycle = 60.0 * (1.0 - 0.3 * high_c)
        green = max(GREEN_MIN_S, cycle - 2 * YELLOW_S)
        phase = 0 if ns_p >= 0.5 else 1
        self.optimization_events += 1
        return phase, green * (ns_p if phase == 0 else 1 - ns_p)

    @controller('mpc', 'field', ('veh_ns', 'veh_ew', 'congestion_eff'))
    def mpc_control(self, veh_ns, veh_ew, congestion_eff):
        H, best_phase, best_cost = 4, 0, float('inf')
        arrival = (veh_ns + veh_ew) / max(DECISION_INTERVAL_S, 1.0) * 0.35
        for cand in (0, 1):
            qn, qe, cost = veh_ns, veh_ew, 0.0
            for _ in range(H):
                if cand == 0:
                    qn = max(0.0, qn - 3.0 + 0.5 * arrival)
                    qe = max(0.0, qe + 0.5 * arrival)
                else:
                    qn = max(0.0, qn + 0.5 * arrival)
                    qe = max(0.0, qe - 3.0 + 0.5 * arrival)
                cost += qn + qe + 0.05 * (qn * qn + qe * qe) / 20.0
            if cost < best_cost:
                best_cost, best_phase = cost, cand
        dur = 25.0 + 15.0 * congestion_eff + 0.2 * (veh_ns + veh_ew)
        self.optimization_events += 1
        return best_phase, min(GREEN_MAX_S, max(GREEN_MIN_S, dur))

    @controller('weather-adaptive', 'field',
                ('veh_ns', 'veh_ew', 'congestion_eff'))
    def weather_adaptive_control(self, veh_ns, veh_ew, congestion_eff):
        cap   = self.nat_cfg['capacity']
        qn    = veh_ns / max(cap, 0.1)
        qe    = veh_ew / max(cap, 0.1)
        ns_r  = qn / max(qn + qe, 1.0)
        cycle = 60.0 + 30.0 * (1.0 - cap)
        green = max(GREEN_MIN_S, cycle - 2 * YELLOW_S - 2.0 * (1.0 - cap))
        phase = 0 if ns_r > 0.5 else 1
        self.optimization_events += 1
        return phase, green * (ns_r if phase == 0 else 1 - ns_r)

    # =====================================================================
    #  DISPATCHER  —  single registry lookup
    # =====================================================================
    def control_signal(self,
                       intersection_id: str,
                       vehicle_count_ns: float,
                       vehicle_count_ew: float,
                       waiting_ns: float,
                       waiting_ew: float,
                       current_time: float,
                       current_congestion: float,
                       gap_ns: float = 0.0,
                       gap_ew: float = 0.0,
                       neighbours: Optional[list] = None,
                       row: int = 0, col: int = 0,
                       current_phase: int = 0) -> tuple[int, float]:
        self.current_time_ms = current_time * 1000.0

        # ---- natural-condition preprocessing ---------------------------
        cfg       = self.nat_cfg
        nat_total = (vehicle_count_ns + vehicle_count_ew) * cfg['demand_mult']
        bias      = cfg['direction_bias']
        veh_ns    = nat_total * (0.5 + 0.5 * bias)
        veh_ew    = nat_total - veh_ns
        congestion_eff = min(0.99,
                             current_congestion / max(cfg['capacity'], 0.1))

        if cfg['sensor_reliability'] < 1.0:
            if self.rng.random() > cfg['sensor_reliability']:
                veh_ns, veh_ew, waiting_ns, waiting_ew = 0.0, 0.0, 0.0, 0.0

        obs = self.build_observation(
            {'phase': current_phase, 'elapsed_s': current_time % 30.0,
             'queue_ns': veh_ns, 'queue_ew': veh_ew},
            neighbours)

        # ---- expression selection --------------------------------------
        spec = self.CONTROLLERS[self.algorithm_type]
        context = {
            'intersection_id':  intersection_id,
            'veh_ns':           veh_ns,
            'veh_ew':           veh_ew,
            'waiting_ns':       waiting_ns,
            'waiting_ew':       waiting_ew,
            'current_time':     current_time,
            'current_congestion': current_congestion,
            'congestion_eff':   congestion_eff,
            'gap_ns':           gap_ns,
            'gap_ew':           gap_ew,
            'obs':              obs,
            'row':              row,
            'col':              col,
            'current_phase':    current_phase,
        }
        kwargs = {k: context[k] for k in spec.inputs}
        phase, duration = spec.fn(self, **kwargs)
        duration = max(GREEN_MIN_S, min(GREEN_MAX_S, duration))

        self.phase_decisions.append({
            'intersection': intersection_id, 'phase': phase,
            'duration': duration, 'algorithm': self.algorithm_type,
            'scenario': self.scenario, 'natural': self.natural,
            'time': current_time})

        # ---- metric accumulation (Eqs. 15-18) --------------------------
        self.m['samples']            += 1
        self.m['waiting_sum_s']      += waiting_ns + waiting_ew
        self.m['queue_sum']          += veh_ns + veh_ew
        cleared = min(2.5, 0.025 * (veh_ns + veh_ew) * cfg['capacity'])
        self.m['vehicles_completed'] += cleared

        cf = self.car_model.simulate_step(
            'free_flow' if congestion_eff < 0.5 else 'congested',
            phase, veh_ns, veh_ew, congestion_eff)
        self.m['travel_time_sum_s'] += (
            0.6 * cf['in_system_s']
            + 0.4 * (0.5 * (waiting_ns + waiting_ew) + 100.0))
        return phase, duration

    # =====================================================================
    #  COOPERATIVE REWARD — Eq. (10)
    # =====================================================================
    def cooperative_reward(self, local_queues, global_queues):
        r_l = -sum(local_queues)
        r_g = -(sum(global_queues) / max(len(global_queues), 1))
        return self.beta * r_l + (1.0 - self.beta) * r_g

    # =====================================================================
    #  METRICS — Eqs. (15)-(20)
    # =====================================================================
    def get_metrics(self) -> dict:
        s     = max(self.m['samples'], 1)
        sim_s = s * DECISION_INTERVAL_S
        sent  = max(self.channel.sent_count, 1)
        mean_aoi = (sum(self.aoi_ms.values()) / len(self.aoi_ms)
                    if self.aoi_ms else 0.0)
        return {
            'algorithm_type':      self.algorithm_type,
            'scenario':            self.scenario,
            'natural':             self.natural,
            'car_following':       self.car_model.name,
            'travel_time_s':       round(self.m['travel_time_sum_s'] / s, 2),
            'queue_veh':           round(self.m['queue_sum'] / s, 2),
            'waiting_time_s':      round(self.m['waiting_sum_s'] / s, 2),
            'throughput_vph':      round(self.m['vehicles_completed']
                                         / max(sim_s, 1.0) * 3600.0, 1),
            'mean_aoi_ms':         round(mean_aoi, 1),
            'delivery_ratio':      round(self.channel.delivered_count / sent, 3),
            'msg_overhead':        round(self.channel.sent_count / s, 2),
            'severity_index':      round(severity_index(
                self.comm_cfg['latency_ms'], self.comm_cfg['loss_rate']), 3),
            'optimization_events': self.optimization_events,
        }


# ===========================================================================
#  POST-CLASS REGISTRATION  (mixed controllers via composition)
# ===========================================================================
def _register_composed_controllers():
    TrafficControlAlgorithm.compose('cooperative-marl', 'coordinated', 0.5,
                                    new_name='cooperative-coordinated')
    TrafficControlAlgorithm.compose('resilient-marl',   'coordinated', 0.5,
                                    new_name='resilient-coordinated')


_register_composed_controllers()


# ===========================================================================
#  SIMULATION DRIVER
# ===========================================================================
def simulate_episode(controller, n_steps=200,
                     base_demand_ns=0.15, base_demand_ew=0.10,
                     congestion_start=0.40, congestion_end=0.90,
                     seed=None):
    """Run the 3 × 3 grid for n_steps decision intervals."""
    rng = random.Random(seed) if seed is not None else controller.rng
    net = TrafficNetwork(rng)
    dt  = DECISION_INTERVAL_S
    net_wait_sum = net_queue_samples = 0.0
    net_samples  = 0

    for t in range(n_steps):
        c = (congestion_start
             + (congestion_end - congestion_start) * t / max(n_steps - 1, 1))
        cur_time = t * dt
        for nid, node in net.nodes.items():
            phase, dur = controller.control_signal(
                intersection_id  = nid,
                vehicle_count_ns = node.queue_ns(),
                vehicle_count_ew = node.queue_ew(),
                waiting_ns       = node.W['N'] + node.W['S'],
                waiting_ew       = node.W['E'] + node.W['W'],
                current_time     = cur_time,
                current_congestion = c,
                gap_ns           = 1.0 if node.queue_ns() > 0 else 5.0,
                gap_ew           = 1.0 if node.queue_ew() > 0 else 5.0,
                neighbours       = net.neighbour_snapshots(nid),
                row = node.row, col = node.col,
                current_phase    = node.phase)
            node.step(dt,
                      arrive_ns = base_demand_ns * (0.8 + 0.6 * c),
                      arrive_ew = base_demand_ew * (0.8 + 0.6 * c),
                      green_active_phase = phase,
                      green_time = min(dur, dt))
        for node in net.nodes.values():
            net_wait_sum      += sum(node.W.values())
            net_queue_samples += node.total_queue()
        net_samples += len(net.nodes)

    controller.m['waiting_sum_s']      = net_wait_sum / max(net_samples, 1)
    controller.m['queue_sum']          = net_queue_samples / max(net_samples, 1)
    controller.m['vehicles_completed'] = sum(n.completed for n in net.nodes.values())
    controller.m['samples']            = n_steps
    return controller


# ===========================================================================
#  MULTI-SEED EVALUATION
# ===========================================================================
def run_multi_seed(algorithm_type, scenario='S1', natural='N-clear',
                   seeds=(1, 2, 3, 4, 5), n_steps=200,
                   base_demand_ns=0.15, base_demand_ew=0.10,
                   car_following='SAFE'):
    """Run one cell across n seeds; return mean, stdev, 95 % CI per metric."""
    keys = ('travel_time_s', 'queue_veh', 'waiting_time_s',
            'throughput_vph', 'mean_aoi_ms', 'delivery_ratio')
    per_seed = defaultdict(list)

    for s in seeds:
        c = TrafficControlAlgorithm(algorithm_type, scenario, natural,
                                    seed=s, car_following=car_following)
        simulate_episode(c, n_steps=n_steps,
                         base_demand_ns=base_demand_ns,
                         base_demand_ew=base_demand_ew, seed=s)
        m = c.get_metrics()
        for k in keys:
            per_seed[k].append(float(m[k]))

    out = {'algorithm_type': algorithm_type, 'scenario': scenario,
           'natural': natural, 'n_seeds': len(seeds)}
    for k, xs in per_seed.items():
        mean = statistics.fmean(xs)
        if len(xs) > 1:
            sd   = statistics.stdev(xs)
            ci95 = 1.96 * sd / math.sqrt(len(xs))
        else:
            sd = ci95 = 0.0
        out[f'{k}_mean']   = round(mean, 3)
        out[f'{k}_sd']     = round(sd, 3)
        out[f'{k}_ci95']   = round(ci95, 3)
        out[f'{k}_values'] = [round(x, 3) for x in xs]
    return out


def export_csv(rows, path='traffic_benchmark.csv'):
    """Write a list of dicts (from run_multi_seed) to CSV."""
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ===========================================================================
#  BENCHMARK HARNESS
# ===========================================================================
def print_registry():
    print("=" * 100)
    print("  CONTROLLER REGISTRY")
    print("=" * 100)
    for cat in TrafficControlAlgorithm.CATEGORIES + ('composed',):
        names = TrafficControlAlgorithm.list_controllers(cat)
        if not names:
            continue
        print(f"\n  [{cat}]")
        for n in names:
            spec = TrafficControlAlgorithm.get_expression(n)
            print(f"    {n:<30}  inputs = {spec.inputs}")
    print()


def print_table12():
    print("\n" + "=" * 100)
    print("  Table 12 — Car-Following Benchmark (Mean ± SD, n = 5 seeds)")
    print("=" * 100)
    print(f"{'Model':<8}{'Regime':<12}{'In-System(s)':>16}"
          f"{'Speed(m/s)':>14}{'Eff(%)':>10}{'Delay(s)':>16}")
    print('-' * 100)
    for model in ('GIPPS', 'IDM', 'SAFE'):
        for regime, key in (('Free-flow', 'free_flow'),
                            ('Congested', 'congested')):
            b = CAR_FOLLOWING_BENCHMARK[model][key]
            print(f"{model:<8}{regime:<12}"
                  f"{b['in_system_s']:>8.2f} ± {b['sd']:<5.2f}"
                  f"{b['speed_ms']:>8.2f} ± {b['sd_speed']:<4.2f}"
                  f"{b['efficiency_pct']:>10.1f}"
                  f"{b['delay_s']:>8.2f} ± {b['sd_delay']:<4.2f}")
    print()


def print_table13():
    print("=" * 100)
    print("  Table 13 — Performance under Ideal Communication (S1, SAFE model)")
    print("=" * 100)
    print(f"{'Method':<26}{'Travel(s)':>10}{'Queue':>8}"
          f"{'Wait(s)':>9}{'Throughput':>12}")
    print('-' * 100)
    for algo in ('fixed-time', 'actuated', 'independent-dqn',
                 'cooperative-marl', 'resilient-marl'):
        c = TrafficControlAlgorithm(algo, 'S1', 'N-clear', 1, 'SAFE')
        simulate_episode(c, 300)
        m = c.get_metrics()
        print(f"{algo:<26}{m['travel_time_s']:>10.2f}"
              f"{m['queue_veh']:>8.2f}{m['waiting_time_s']:>9.2f}"
              f"{m['throughput_vph']:>12.1f}")
    print()


def print_tables_14_15():
    for label, algo in (("Table 14 — Standard Cooperative MARL",
                         'cooperative-marl'),
                        ("Table 15 — Resilient MARL (proposed)",
                         'resilient-marl')):
        print("=" * 100)
        print(f"  {label}  —  S1 → S4")
        print("=" * 100)
        print(f"{'Metric':<22}{'S1':>12}{'S2':>12}{'S3':>12}{'S4':>12}")
        print('-' * 100)
        rows = {'travel_time_s':  'Travel time (s)',
                'queue_veh':      'Queue length (veh)',
                'waiting_time_s': 'Waiting time (s)',
                'throughput_vph': 'Throughput (veh/h)'}
        data = {k: [] for k in rows}
        for scen in ('S1', 'S2', 'S3', 'S4'):
            c = TrafficControlAlgorithm(algo, scen, 'N-clear', 1, 'SAFE')
            simulate_episode(c, 300)
            m = c.get_metrics()
            for k in rows:
                data[k].append(m[k])
        for k, lbl in rows.items():
            print(f"{lbl:<22}" + ''.join(f"{v:>12.1f}" for v in data[k]))
        print()


def print_table16():
    print("=" * 100)
    print("  Table 16 — Comparative Summary (relative to S1)")
    print("=" * 100)
    print(f"{'Scenario':<12}{'ΔTravel Std':>14}{'ΔTravel Res':>14}"
          f"{'ΔThr Std':>14}{'ΔThr Res':>14}")
    print('-' * 100)
    base = {}
    for algo in ('cooperative-marl', 'resilient-marl'):
        base[algo] = {}
        for scen in ('S1', 'S2', 'S3', 'S4'):
            c = TrafficControlAlgorithm(algo, scen, 'N-clear', 1, 'SAFE')
            simulate_episode(c, 300)
            base[algo][scen] = c.get_metrics()
    pct = lambda new, old: (new - old) / old * 100.0
    for scen in ('S1', 'S2', 'S3', 'S4'):
        print(f"{scen:<12}"
              f"{pct(base['cooperative-marl'][scen]['travel_time_s'], base['cooperative-marl']['S1']['travel_time_s']):>13.1f}%"
              f"{pct(base['resilient-marl'][scen]['travel_time_s'], base['resilient-marl']['S1']['travel_time_s']):>13.1f}%"
              f"{pct(base['cooperative-marl'][scen]['throughput_vph'], base['cooperative-marl']['S1']['throughput_vph']):>13.1f}%"
              f"{pct(base['resilient-marl'][scen]['throughput_vph'], base['resilient-marl']['S1']['throughput_vph']):>13.1f}%")
    print()


def print_all_controllers():
    print("=" * 100)
    print("  ALL CONTROLLERS × S1 / S2 / S3 / S4   (SAFE model, N-clear)")
    print("=" * 100)
    print(f"{'Controller':<26}{'Scen':<6}{'Travel(s)':>10}{'Queue':>8}"
          f"{'Wait(s)':>9}{'Thr(vph)':>10}{'AoI(ms)':>10}{'Deliv':>7}")
    print('-' * 100)
    for algo in TrafficControlAlgorithm.list_controllers():
        for scen in ('S1', 'S2', 'S3', 'S4'):
            c = TrafficControlAlgorithm(algo, scen, 'N-clear', 42, 'SAFE')
            simulate_episode(c, 200)
            m = c.get_metrics()
            print(f"{algo:<26}{scen:<6}{m['travel_time_s']:>10.2f}"
                  f"{m['queue_veh']:>8.2f}{m['waiting_time_s']:>9.2f}"
                  f"{m['throughput_vph']:>10.1f}"
                  f"{m['mean_aoi_ms']:>10.1f}{m['delivery_ratio']:>7.3f}")
        print()


def print_natural_matrix():
    print("=" * 130)
    print("  NATURAL CONDITION × ALGORITHM  —  travel time (s), scenario S2")
    print("=" * 130)
    conditions = [n for grp in NATURAL_GROUPS.values() for n in grp]
    short = {'N-clear': 'Clear', 'N-rain': 'Rain', 'N-snow': 'Snow',
             'N-fog': 'Fog', 'N-accident': 'Accid', 'N-workzone': 'Work',
             'N-rush-am': 'AMpk', 'N-rush-pm': 'PMpk', 'N-night': 'Night',
             'N-event': 'Event', 'N-school': 'Schl',
             'N-sensor-fail': 'NoDet', 'N-comm-blackout': 'NoV2X'}
    algos = ('fixed-time', 'actuated', 'adaptive', 'optimized',
             'coordinated', 'cooperative-marl', 'resilient-marl',
             'resilient-coordinated', 'max-pressure', 'scats', 'scoot',
             'multiband', 'fuzzy-logic', 'mpc', 'weather-adaptive')
    print(f"{'Algorithm':<24}"
          + ''.join(f"{short[c]:>8}" for c in conditions))
    print('-' * (24 + 8 * len(conditions)))
    for algo in algos:
        row = f"{algo:<24}"
        for nat in conditions:
            c = TrafficControlAlgorithm(algo, 'S2', nat, 11, 'SAFE')
            simulate_episode(c, 100)
            row += f"{c.get_metrics()['travel_time_s']:>8.1f}"
        print(row)
    print()


def print_multi_seed_demo():
    print("=" * 100)
    print("  MULTI-SEED EVALUATION  —  5 seeds, 95 % CI (thesis protocol)")
    print("=" * 100)
    print(f"{'Algorithm':<24}{'Scenario':<10}{'Travel mean':>12}"
          f"{'±CI95':>9}{'Throughput':>12}{'±CI95':>9}")
    print('-' * 100)
    for algo in ('resilient-marl', 'cooperative-marl', 'fixed-time'):
        for scen in ('S1', 'S2', 'S3', 'S4'):
            r = run_multi_seed(algo, scen, 'N-clear', seeds=(1, 2, 3, 4, 5),
                               n_steps=200)
            print(f"{algo:<24}{scen:<10}"
                  f"{r['travel_time_s_mean']:>12.2f}"
                  f"{r['travel_time_s_ci95']:>9.2f}"
                  f"{r['throughput_vph_mean']:>12.1f}"
                  f"{r['throughput_vph_ci95']:>9.2f}")
        print()


# ===========================================================================
#  MAIN
# ===========================================================================
if __name__ == '__main__':
    print("\n" + "#" * 100)
    print("#  Traffic-Signal Control — Self-Contained Simulation & Benchmark")
    print("#  Final revision  •  expression-selection dispatch")
    print("#  Reproduces Albreem et al. (2026)")
    print("#" * 100)

    print_registry()
    print_table12()
    print_table13()
    print_tables_14_15()
    print_table16()
    print_all_controllers()
    print_natural_matrix()
    print_multi_seed_demo()

    # ---- logistic degradation demo ------------------------------------
    print("\nLogistic degradation  f(x) = L / (1 + exp(-k (x - x0))):")
    for x_ms in (0, 50, 100, 101.54, 150, 300, 600):
        print(f"  x = {x_ms:>6.1f} ms  ->  f(x) = {degrade_logistic(x_ms):.2f}")

    # ---- runtime composition demo -------------------------------------
    print("\nRuntime composition demo:")
    name = TrafficControlAlgorithm.compose('mpc', 'multiband', 0.7)
    print(f"  registered composed controller: {name}")
    c = TrafficControlAlgorithm(name, 'S2', 'N-clear', 0, 'SAFE')
    simulate_episode(c, 100)
    m = c.get_metrics()
    print(f"  travel={m['travel_time_s']:.2f}s  "
          f"throughput={m['throughput_vph']:.0f} vph  "
          f"queue={m['queue_veh']:.2f}")