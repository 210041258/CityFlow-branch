#!/usr/bin/env python3
"""
Reward Calculator Module
========================
Second stage of the traffic-signal RL pipeline.

Pipeline
--------
    SignalCharacterization  ->  observation space  (what the agent sees)
    RewardCalculator        ->  scalar reward       (how the agent is scored)

Used by: run.py --training-env (paired with signal_characterization.py)

Reward shape
------------
The scalar reward is a weighted sum of four interpretable components so
that each contribution can be inspected during debugging / tensorboard
logging:

    R = w_queue    * (-queue_pressure)
      + w_wait     * (-wait_penalty)
      + w_through  * (+throughput_gain)
      + w_fair     * (-fairness_penalty)

Every component is normalized to roughly [0, 1] *before* weighting, so
weight tuning stays interpretable regardless of intersection count or
simulation step size:

    queue_pressure     in [0, 1]   (0 = empty, 1 = saturated & imbalanced)
    wait_penalty       in [0, 1]   (saturates at `wait_penalty_horizon`)
    throughput_gain    in [0, 1]   (soft-capped at 10 veh/step)
    fairness_penalty   in [0, 1]   (0 when served side is not behind)

Phase convention
----------------
    0 / 2 -> NS green    1 / 3 -> EW green
"""

from collections import defaultdict
from typing import Any, Dict, Hashable, Mapping, Optional


class RewardCalculator:
    """Multi-objective reward for traffic signal control RL training."""

    #: Default weights. They need not sum to 1 -- they are independent levers.
    DEFAULT_WEIGHTS: Dict[str, float] = {
        'queue':      0.4,   # discourage long queues building up
        'wait':       0.3,   # discourage long cumulative wait times
        'throughput': 0.2,   # reward vehicles clearing the intersection
        'fairness':   0.1,   # discourage one approach starving another
    }

    #: Names of the four breakdown components, in a stable order.
    COMPONENTS = (
        'queue_pressure',
        'wait_penalty',
        'throughput_gain',
        'fairness_penalty',
    )

    def __init__(
        self,
        num_intersections: int = 10,
        weights: Optional[Mapping[str, float]] = None,
        max_queue_per_lane: int = 50,
        wait_penalty_horizon: float = 300.0,
    ) -> None:
        """
        Args:
            num_intersections: number of intersections being controlled.
            weights: optional dict overriding any subset of ``DEFAULT_WEIGHTS``.
            max_queue_per_lane: queue length considered "saturated"; used to
                normalize ``queue_pressure`` into ~[0, 1].
            wait_penalty_horizon: simulation seconds over which a stopped
                vehicle becomes fully penalized; used to normalize
                ``wait_penalty``.
        """
        self.num_intersections = num_intersections
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.max_queue_per_lane = max_queue_per_lane
        self.wait_penalty_horizon = wait_penalty_horizon

        # Per-intersection accumulators
        self._step_reward: Dict[Hashable, float] = defaultdict(float)
        self._episode_return: Dict[Hashable, float] = defaultdict(float)
        self._component_log: Dict[Hashable, Dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._component_sum: Dict[Hashable, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._component_count: Dict[Hashable, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Throughput bookkeeping: throughput is a *delta* of the cumulative
        # cleared-vehicle counter, so we track the last value we saw.
        self._last_cleared_seen: Dict[Hashable, int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute(
        self,
        intersection_id: Hashable,
        observation: Mapping[str, Any],
        cleared_vehicles: int = 0,
        stopped_vehicles: int = 0,
        sim_dt: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compute the scalar reward for one intersection at one step.

        Args:
            intersection_id: hashable id of the intersection.
            observation: dict from ``SignalCharacterization.get_observation_space``.
                Must contain ``vehicles_ns``, ``vehicles_ew``, ``queue_ratio``,
                ``phase`` (``duration`` is accepted but unused here).
            cleared_vehicles: *cumulative* vehicles cleared since simulation
                start; the reward uses the per-step delta.
            stopped_vehicles: vehicles currently stopped on any approach.
            sim_dt: simulation step length in seconds.

        Returns:
            ``{'reward': float, 'components': {<name>: <value>, ...}}``
            so callers can log / visualise individual contributions.
        """
        v_ns = observation.get('vehicles_ns', 0)
        v_ew = observation.get('vehicles_ew', 0)
        queue_ratio = observation.get('queue_ratio', 0.0)
        phase = observation.get('phase', 0)

        queue_pressure = self._queue_pressure(v_ns, v_ew, queue_ratio)
        wait_penalty = self._wait_penalty(stopped_vehicles, sim_dt)
        throughput_gain = self._throughput_gain(intersection_id, cleared_vehicles)
        fairness_penalty = self._fairness_penalty(v_ns, v_ew, phase)

        w = self.weights
        reward = (
            w['queue']       * (-queue_pressure)
            + w['wait']      * (-wait_penalty)
            + w['throughput'] * throughput_gain
            + w['fairness']  * (-fairness_penalty)
        )

        # Bookkeeping
        self._step_reward[intersection_id] = reward
        self._episode_return[intersection_id] += reward

        breakdown = {
            'queue_pressure':   queue_pressure,
            'wait_penalty':     wait_penalty,
            'throughput_gain':  throughput_gain,
            'fairness_penalty': fairness_penalty,
        }
        for k, v in breakdown.items():
            self._component_log[intersection_id][k].append(v)
            self._component_sum[intersection_id][k] += v
            self._component_count[intersection_id][k] += 1

        return {'reward': reward, 'components': breakdown}

    def reset_episode(self, intersection_id: Optional[Hashable] = None) -> None:
        """Reset episodic accumulators. If ``intersection_id`` is None, reset all."""
        if intersection_id is not None:
            targets = [intersection_id]
        else:
            targets = list(self._episode_return.keys())
            if not targets:
                targets = [f'int_{i}' for i in range(self.num_intersections)]

        for iid in targets:
            self._episode_return[iid] = 0.0
            self._last_cleared_seen[iid] = 0
            self._component_log[iid].clear()
            self._component_sum[iid].clear()
            self._component_count[iid].clear()

    def get_episode_return(self, intersection_id: Hashable) -> float:
        """Total accumulated reward for this intersection in the current episode."""
        return self._episode_return.get(intersection_id, 0.0)

    def get_metrics(self) -> Dict[str, float]:
        """Aggregate metrics for logging / tensorboard."""
        returns = list(self._episode_return.values())
        n = max(len(returns), 1)
        avg_return = sum(returns) / n if returns else 0.0

        # O(1) per component: uses the running sums/counts rather than
        # re-walking the full component log on every call.
        component_avgs: Dict[str, float] = {}
        for component in self.COMPONENTS:
            total = sum(
                self._component_sum[iid].get(component, 0.0)
                for iid in self._component_sum
            )
            count = sum(
                self._component_count[iid].get(component, 0)
                for iid in self._component_count
            )
            component_avgs[component] = total / count if count else 0.0

        return {
            'reward_intersections_tracked': len(self._episode_return),
            'reward_avg_episode_return':   avg_return,
            'reward_max_episode_return':   max(returns) if returns else 0.0,
            'reward_min_episode_return':   min(returns) if returns else 0.0,
            **{f'reward_avg_{k}': v for k, v in component_avgs.items()},
        }

    # ------------------------------------------------------------------ #
    # Internal component functions (each normalized to ~[0, 1])
    # ------------------------------------------------------------------ #
    def _queue_pressure(self, v_ns: float, v_ew: float, queue_ratio: float) -> float:
        """
        Combine total load and directional imbalance.

        Returns ~[0, 1]: 0 when empty, ~1 when saturated *and* imbalanced.
        """
        total = v_ns + v_ew
        load = min(total / (2 * self.max_queue_per_lane), 1.0)
        imbalance = abs(queue_ratio - 0.5) * 2 if total > 0 else 0.0
        return 0.6 * load + 0.4 * imbalance

    def _wait_penalty(self, stopped_vehicles: float, sim_dt: float) -> float:
        """Normalized wait penalty in [0, 1]; saturates at the configured horizon."""
        effective_wait = stopped_vehicles * sim_dt
        return min(effective_wait / self.wait_penalty_horizon, 1.0)

    def _throughput_gain(self, intersection_id: Hashable, cleared_vehicles: int) -> float:
        """
        Delta of cleared vehicles this step, normalized to [0, 1] by a soft cap
        so a single very large platoon cannot dominate the reward.
        """
        delta = cleared_vehicles - self._last_cleared_seen.get(intersection_id, 0)
        self._last_cleared_seen[intersection_id] = cleared_vehicles
        # Soft cap at 10 veh/step -- typical for a single intersection phase.
        return min(max(delta, 0) / 10.0, 1.0)

    def _fairness_penalty(self, v_ns: float, v_ew: float, phase: int) -> float:
        """
        Penalize serving an empty approach while the other approach is loaded.

        Returns [0, 1]. Phase convention: 0/2 = NS green, 1/3 = EW green.
        """
        total = v_ns + v_ew
        if total == 0:
            return 0.0

        if phase % 2 == 0:          # NS green
            served_side, starved_side = v_ns, v_ew
        else:                        # EW green
            served_side, starved_side = v_ew, v_ns

        if served_side >= starved_side:
            return 0.0

        return (starved_side - served_side) / total


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    from signal_characterization import SignalCharacterization

    INT = 'int_0'
    sig = SignalCharacterization(num_intersections=10)
    rew = RewardCalculator(num_intersections=10)

    # (label, phase, duration, v_ns, v_ew, cleared_cumulative, stopped, dt)
    #
    # `cleared_cumulative` is the total number of vehicles cleared since
    # simulation start, so it MUST be non-decreasing across scenarios.
    # Scenarios are grouped by the property they exercise.
    scenarios = [
        # --- baseline & smooth demand ------------------------------------
        ("empty",              0, 30,   0,   0,   0,   0, 1.0),
        ("light-balanced",     0, 30,  10,  12,   3,   1, 1.0),
        ("moderate-balanced",  1, 25,  30,  30,  10,   5, 1.0),

        # --- directional load & imbalance --------------------------------
        ("heavy-ns-skew",      0, 30,  80,  10,  12,   2, 1.0),
        ("heavy-ew-skew",      1, 30,   5,  70,  15,   3, 1.0),

        # --- fairness stress: serve an empty approach while other loads ---
        ("fairness-starved",   0, 30,   0,  40,  20,  10, 1.0),

        # --- wait stress: 350 stopped * 1 s = 350 s > 300 s horizon -----
        ("wait-saturated",     1, 20,  20,  20,  25, 350, 1.0),

        # --- throughput stress: +25 clears in one step (>10 soft cap) ---
        ("throughput-capped",  0, 30,  15,  15,  50,   5, 1.0),

        # --- zero-throughput step: cleared_cumulative unchanged ----------
        ("throughput-zero",    0, 20,  10,  10,  50,   3, 1.0),

        # --- sub-second step: 60 stopped * 0.5 s / 300 = 0.10 -----------
        ("dt-half-second",     1, 15,  20,  20,  50,  60, 0.5),

        # --- phase 3 == EW green; EW is empty while NS is loaded --------
        ("phase-3-ew-green",   3, 30,  30,   0,  50,   5, 1.0),
    ]

    # ---------------------------------------------------------------- #
    # Run every scenario and print a single aligned table
    # ---------------------------------------------------------------- #
    header = (
        f"{'scenario':<20} {'ph':>3} {'dt':>5} {'v_ns':>5} {'v_ew':>5} "
        f"{'queue':>7} {'wait':>7} {'thru':>7} {'fair':>7} {'reward':>9}"
    )
    print(header)
    print("-" * len(header))

    results: Dict[str, Dict[str, Any]] = {}
    prev_cleared = 0
    for label, ph, dur, v_ns, v_ew, cleared, stopped, dt in scenarios:
        assert cleared >= prev_cleared, (
            f"'{label}': cleared_cumulative must be non-decreasing "
            f"({cleared} < {prev_cleared})"
        )
        prev_cleared = cleared

        sig.record_signal_state(INT, ph, dur, v_ns, v_ew)
        obs = sig.get_observation_space(INT)
        r = rew.compute(
            INT, obs,
            cleared_vehicles=cleared,
            stopped_vehicles=stopped,
            sim_dt=dt,
        )
        c = r['components']

        # Structural invariant: every component stays inside [0, 1].
        for name in RewardCalculator.COMPONENTS:
            val = c[name]
            assert 0.0 <= val <= 1.0, f"{label}: {name}={val} outside [0, 1]"

        results[label] = r
        print(
            f"{label:<20} {ph:>3} {dt:>5.1f} {v_ns:>5} {v_ew:>5} "
            f"{c['queue_pressure']:>7.3f} {c['wait_penalty']:>7.3f} "
            f"{c['throughput_gain']:>7.3f} {c['fairness_penalty']:>7.3f} "
            f"{r['reward']:>9.4f}"
        )

    # ---------------------------------------------------------------- #
    # Scenario-specific correctness assertions
    # ---------------------------------------------------------------- #
    def comp(label: str, key: str) -> float:
        return results[label]['components'][key]

    # 1. Empty intersection -> every component exactly zero.
    assert all(v == 0.0 for v in results['empty']['components'].values()), \
        "empty scenario should yield zero components"

    # 2. Fairness fires when the green phase serves an empty approach
    #    while the opposing approach is loaded (NS green, NS empty).
    assert comp('fairness-starved', 'fairness_penalty') >= 0.99, (
        f"expected near-max fairness penalty, "
        f"got {comp('fairness-starved', 'fairness_penalty')}"
    )

    # 3. When the served approach has MORE vehicles than the starved one,
    #    fairness penalty must be exactly zero (no false positives).
    assert comp('heavy-ns-skew', 'fairness_penalty') == 0.0
    assert comp('heavy-ew-skew', 'fairness_penalty') == 0.0

    # 4. Wait penalty saturates at the configured horizon (300 s).
    assert comp('wait-saturated', 'wait_penalty') == 1.0

    # 5. Throughput gain is clamped to [0, 1]:
    #    - a large delta (+25) hits the soft cap,
    #    - zero delta floors at 0.
    assert comp('throughput-capped', 'throughput_gain') == 1.0
    assert comp('throughput-zero',   'throughput_gain') == 0.0

    # 6. Wait penalty scales linearly with sim_dt:
    #    60 stopped * 0.5 s / 300 = 0.10.
    assert abs(comp('dt-half-second', 'wait_penalty') - 0.10) < 1e-9

    # 7. Phase 3 must be interpreted as EW green -> fairness fires when EW
    #    is empty and NS is loaded.
    assert comp('phase-3-ew-green', 'fairness_penalty') >= 0.99, (
        f"phase 3 not treated as EW green: "
        f"fairness={comp('phase-3-ew-green', 'fairness_penalty')}"
    )

    print("\nAll scenario assertions passed.")

    # ---------------------------------------------------------------- #
    # Aggregate metrics (captured BEFORE reset so they are non-trivial)
    # ---------------------------------------------------------------- #
    print("\n--- Signal metrics ---")
    print(sig.get_metrics())
    print("\n--- Reward metrics ---")
    print(rew.get_metrics())
    print(f"\nEpisode return ({INT}): {rew.get_episode_return(INT):.4f}")

    # ---------------------------------------------------------------- #
    # reset_episode sanity check
    # ---------------------------------------------------------------- #
    rew.reset_episode(INT)
    assert rew.get_episode_return(INT) == 0.0, \
        "reset_episode did not zero the episode return"
    print(f"reset_episode({INT}) -> episode return zeroed.")