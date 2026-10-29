#!/usr/bin/env python3
"""
Reward Calculator Module
========================
Next step after signal_characterization.py in the traffic-signal RL pipeline.

Pipeline so far:
    SignalCharacterization  ->  observation space  (what the agent sees)
    RewardCalculator        ->  scalar reward       (how the agent is scored)

Used by: run.py --training-env (paired with signal_characterization.py)

The reward is a weighted sum of four interpretable components so that
individual contributions can be inspected during debugging / tensorboard
logging:

    R = w_queue   * (-queue_pressure)
      + w_wait    * (-avg_wait_penalty)
      + w_through * ( +throughput_gain)
      + w_fair    * (-fairness_penalty)

All components are normalized to roughly [-1, 1] before weighting so that
weight tuning stays interpretable regardless of intersection count or
simulation step size.
"""

from collections import defaultdict


class RewardCalculator:
    """Multi-objective reward for traffic signal control RL training."""

    # Default component weights (sum need not be 1; they are independent levers)
    DEFAULT_WEIGHTS = {
        'queue': 0.4,    # discourage long queues building up
        'wait': 0.3,     # discourage long cumulative wait times
        'throughput': 0.2,  # reward vehicles clearing the intersection
        'fairness': 0.1, # discourage one approach starving another
    }

    def __init__(self, num_intersections=10, weights=None,
                 max_queue_per_lane=50, wait_penalty_horizon=300):
        """
        Args:
            num_intersections: number of intersections being controlled.
            weights: optional dict overriding DEFAULT_WEIGHTS for any subset
                     of the four components ('queue', 'wait', 'throughput',
                     'fairness').
            max_queue_per_lane: queue length considered "saturated"; used to
                     normalize the queue component into ~[-1, 0].
            wait_penalty_horizon: number of simulation seconds over which a
                     stopped vehicle is considered "fully penalized"; used
                     to normalize the wait component.
        """
        self.num_intersections = num_intersections
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.max_queue_per_lane = max_queue_per_lane
        self.wait_penalty_horizon = wait_penalty_horizon

        # Per-intersection accumulators
        self._step_reward = defaultdict(float)
        self._episode_return = defaultdict(float)
        self._component_log = defaultdict(lambda: defaultdict(list))

        # Throughput bookkeeping: track which vehicles we have already counted
        # so throughput is a *delta*, not a cumulative sum.
        self._cleared_vehicles_total = defaultdict(int)
        self._last_cleared_seen = defaultdict(int)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute(self, intersection_id, observation, cleared_vehicles=0,
                stopped_vehicles=0, sim_dt=1.0):
        """
        Compute the scalar reward for one intersection at one step.

        Args:
            intersection_id: hashable id of the intersection.
            observation: dict from SignalCharacterization.get_observation_space
                         (must contain 'vehicles_ns', 'vehicles_ew',
                         'queue_ratio', 'phase', 'duration').
            cleared_vehicles: total vehicles that have cleared this
                              intersection since simulation start (cumulative).
            stopped_vehicles: vehicles currently stopped on any approach.
            sim_dt: simulation step length in seconds (used for wait scaling).

        Returns:
            dict with 'reward' (float) and per-component breakdown so callers
            can log/visualize contributions.
        """
        v_ns = observation.get('vehicles_ns', 0)
        v_ew = observation.get('vehicles_ew', 0)
        queue_ratio = observation.get('queue_ratio', 0.0)

        # --- queue pressure: high when both approaches are loaded & imbalanced
        queue_pressure = self._queue_pressure(v_ns, v_ew, queue_ratio)

        # --- wait penalty: grows with stopped vehicles, normalized by horizon
        wait_penalty = self._wait_penalty(stopped_vehicles, sim_dt)

        # --- throughput gain: delta of cleared vehicles this step
        throughput_gain = self._throughput_gain(intersection_id, cleared_vehicles)

        # --- fairness penalty: penalize one approach starving the other
        fairness_penalty = self._fairness_penalty(v_ns, v_ew,
                                                  observation.get('phase', 0))

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
            'queue_pressure': queue_pressure,
            'wait_penalty': wait_penalty,
            'throughput_gain': throughput_gain,
            'fairness_penalty': fairness_penalty,
        }
        for k, v in breakdown.items():
            self._component_log[intersection_id][k].append(v)

        return {'reward': reward, 'components': breakdown}

    def reset_episode(self, intersection_id=None):
        """Reset episodic accumulators. If intersection_id is None, reset all."""
        targets = ([intersection_id] if intersection_id is not None
                   else list(self._episode_return.keys()) or
                   [f'int_{i}' for i in range(self.num_intersections)])
        for iid in targets:
            self._episode_return[iid] = 0.0
            self._cleared_vehicles_total[iid] = 0
            self._last_cleared_seen[iid] = 0
            self._component_log[iid].clear()

    def get_episode_return(self, intersection_id):
        """Total accumulated reward for this intersection in the current episode."""
        return self._episode_return.get(intersection_id, 0.0)

    def get_metrics(self):
        """Aggregate metrics for logging / tensorboard."""
        n = max(len(self._episode_return), 1)
        returns = list(self._episode_return.values())
        avg_return = sum(returns) / n if returns else 0.0

        # Average per-component magnitudes across all intersections & steps
        component_avgs = {}
        for component in ('queue_pressure', 'wait_penalty',
                          'throughput_gain', 'fairness_penalty'):
            all_vals = []
            for logs in self._component_log.values():
                all_vals.extend(logs.get(component, []))
            component_avgs[component] = (
                sum(all_vals) / len(all_vals) if all_vals else 0.0
            )

        return {
            'reward_intersections_tracked': len(self._episode_return),
            'reward_avg_episode_return': avg_return,
            'reward_max_episode_return': max(returns) if returns else 0.0,
            'reward_min_episode_return': min(returns) if returns else 0.0,
            **{f'reward_avg_{k}': v for k, v in component_avgs.items()},
        }

    # ------------------------------------------------------------------ #
    # Internal component functions (each normalized to ~[0, 1] or [-1, 1])
    # ------------------------------------------------------------------ #
    def _queue_pressure(self, v_ns, v_ew, queue_ratio):
        """
        Combines total load and directional imbalance.
        Returns value in ~[0, 1]: 0 when empty, ~1 when saturated & imbalanced.
        """
        total = v_ns + v_ew
        load = min(total / (2 * self.max_queue_per_lane), 1.0)
        # |queue_ratio - 0.5| * 2  -> 0 when balanced, 1 when one side empty
        imbalance = abs(queue_ratio - 0.5) * 2 if total > 0 else 0.0
        return 0.6 * load + 0.4 * imbalance

    def _wait_penalty(self, stopped_vehicles, sim_dt):
        """
        Normalized wait penalty in [0, 1]. Saturates when stopped vehicles
        accumulate enough seconds to fill the configured horizon.
        """
        effective_wait = stopped_vehicles * sim_dt
        return min(effective_wait / self.wait_penalty_horizon, 1.0)

    def _throughput_gain(self, intersection_id, cleared_vehicles):
        """
        Returns the delta in cleared vehicles this step, normalized to [0, 1]
        by a soft cap so a single very-large platoon doesn't dominate.
        """
        delta = cleared_vehicles - self._last_cleared_seen.get(intersection_id, 0)
        self._last_cleared_seen[intersection_id] = cleared_vehicles
        self._cleared_vehicles_total[intersection_id] += max(delta, 0)
        # Soft cap at 10 vehicles/step -> typical for an intersection phase
        return min(max(delta, 0) / 10.0, 1.0)

    def _fairness_penalty(self, v_ns, v_ew, phase):
        """
        Penalize situations where the green phase serves an empty approach
        while the other approach has waiting vehicles. Returns [0, 1].
        phase convention: 0/2 = NS green, 1/3 = EW green.
        """
        total = v_ns + v_ew
        if total == 0:
            return 0.0
        ns_green = (phase % 2 == 0)
        if ns_green:
            # Penalize if NS is being served but has fewer vehicles
            starved_side = v_ew
            served_side = v_ns
        else:
            starved_side = v_ns
            served_side = v_ew
        if served_side >= starved_side:
            return 0.0
        # Scale by how badly the other side is starved relative to total
        return (starved_side - served_side) / total


# ---------------------------------------------------------------------- #
# Smoke test — mirrors the style of signal_characterization.py's __main__
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    from signal_characterization import SignalCharacterization

    sig = SignalCharacterization(num_intersections=10)
    rew = RewardCalculator(num_intersections=10)

    # Simulate a few steps for intersection 'int_0'
    sim_steps = [
        # (phase, duration, v_ns, v_ew, cleared, stopped, dt)
        (0, 30, 50, 30, 5, 10, 1.0),
        (0, 30, 35, 28, 12, 7, 1.0),
        (1, 25, 20, 45, 18, 9, 1.0),   # EW green but NS still loaded -> fairness hit
        (0, 30, 60, 5, 22, 4, 1.0),    # NS green serving empty EW -> also fairness hit
    ]

    for phase, dur, v_ns, v_ew, cleared, stopped, dt in sim_steps:
        sig.record_signal_state('int_0', phase, dur, v_ns, v_ew)
        obs = sig.get_observation_space('int_0')
        result = rew.compute('int_0', obs,
                             cleared_vehicles=cleared,
                             stopped_vehicles=stopped,
                             sim_dt=dt)
        print(f"phase={phase} dur={dur} v_ns={v_ns} v_ew={v_ew} "
              f"cleared={cleared} stopped={stopped}")
        print(f"  reward={result['reward']:.4f}  components={result['components']}")
        print()

    print("--- Signal metrics ---")
    print(sig.get_metrics())
    print("\n--- Reward metrics ---")
    print(rew.get_metrics())
    print(f"\nEpisode return (int_0): {rew.get_episode_return('int_0'):.4f}")
