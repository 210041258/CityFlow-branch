#!/usr/bin/env python3
"""
Training Environment Module
===========================
Third step in the traffic-signal RL pipeline:

    signal_characterization.py  ->  observation space
    reward_calculator.py        ->  scalar reward
    training_env.py             ->  gym-style env wiring them together

Used by: run.py --training-env

Design notes
------------
* Minimal dependencies: only stdlib + the two project modules above.
  No hard requirement on `gymnasium` or `sumo` -- both are abstracted:
    - `TrafficSimulator`  : interface the env talks to.
    - `MockTrafficSimulator`: a self-contained stochastic sim used for
      tests / bootstrapping. Swap for a SUMO-backed implementation later.
* Follows the Gymnasium step API:
      obs, reward, terminated, truncated, info = env.step(action)
      obs, info = env.reset()
* Single agent controls ALL intersections simultaneously.
  Action  : list[int] of length num_intersections, each in [0, num_phases)
  Obs     : dict mapping intersection_id -> observation dict
            (same shape as SignalCharacterization.get_observation_space)
* Phase-change bookkeeping that signal_characterization.py currently
  doesn't do is tracked HERE (env detects action != previous action
  and increments `info['phase_changes']`).
"""

import random
from collections import defaultdict

from signal_characterization import SignalCharacterization
from reward_calculator import RewardCalculator


# ---------------------------------------------------------------------- #
# Simulator interface + a mock implementation
# ---------------------------------------------------------------------- #
class TrafficSimulator:
    """Abstract interface. Override `step` and `get_intersection_state`."""

    def reset(self):
        raise NotImplementedError

    def step(self, actions, dt=1.0):
        """Advance the simulation by dt seconds given per-intersection phases."""
        raise NotImplementedError

    def get_intersection_state(self, intersection_id):
        """
        Return per-intersection state needed by env/reward:
            {
              'vehicles_ns': int,
              'vehicles_ew': int,
              'cleared_total': int,    # cumulative vehicles cleared since reset
              'stopped': int,          # currently stopped vehicles on approaches
              'phase_index': int,      # the phase actually applied (may differ
                                       # from requested if min-green enforced)
            }
        """
        raise NotImplementedError


class MockTrafficSimulator(TrafficSimulator):
    """
    A lightweight stochastic traffic simulator for testing.

    Each intersection has two approaches (NS / EW). Vehicles arrive
    poisson-style and depart at a rate proportional to green-time on the
    served approach. Phase convention:
        0, 2 -> NS green
        1, 3 -> EW green
    """

    def __init__(self, num_intersections=10, num_phases=4,
                 arrival_rate=0.4, departure_rate=0.9,
                 seed=None):
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.arrival_rate = arrival_rate
        self.departure_rate = departure_rate
        self._rng = random.Random(seed)

        # Per-intersection dynamic state
        self._v_ns = [0] * num_intersections
        self._v_ew = [0] * num_intersections
        self._cleared = [0] * num_intersections
        self._stopped = [0] * num_intersections
        self._phase = [0] * num_intersections

    def reset(self):
        self._v_ns = [self._rng.randint(0, 30) for _ in range(self.num_intersections)]
        self._v_ew = [self._rng.randint(0, 30) for _ in range(self.num_intersections)]
        self._cleared = [0] * self.num_intersections
        self._stopped = [0] * self.num_intersections
        self._phase = [0] * self.num_intersections

    def step(self, actions, dt=1.0):
        for i in range(self.num_intersections):
            phase = int(actions[i]) % self.num_phases
            self._phase[i] = phase

            # Arrivals (poisson-ish via binomial approximation for stability)
            arrivals_ns = self._rng.binomial if hasattr(self._rng, 'binomial') \
                else None
            # Fallback: use simple bernoulli sum
            new_ns = sum(1 for _ in range(int(self.arrival_rate * 30))
                         if self._rng.random() < (self.arrival_rate * dt))
            new_ew = sum(1 for _ in range(int(self.arrival_rate * 30))
                         if self._rng.random() < (self.arrival_rate * dt))

            # Departures: only on the green approach
            ns_green = (phase % 2 == 0)
            depart_cap = int(self.departure_rate * 30 * dt)
            if ns_green and self._v_ns[i] > 0:
                departed = min(self._v_ns[i], depart_cap)
                self._v_ns[i] -= departed
                self._cleared[i] += departed
            elif (not ns_green) and self._v_ew[i] > 0:
                departed = min(self._v_ew[i], depart_cap)
                self._v_ew[i] -= departed
                self._cleared[i] += departed

            # Apply arrivals
            self._v_ns[i] += new_ns
            self._v_ew[i] += new_ew

            # Stopped vehicles = vehicles on the red approach (rough proxy)
            if ns_green:
                self._stopped[i] = self._v_ew[i]
            else:
                self._stopped[i] = self._v_ns[i]

    def get_intersection_state(self, intersection_id):
        i = int(intersection_id.split('_')[-1]) if isinstance(intersection_id, str) \
            else int(intersection_id)
        return {
            'vehicles_ns': self._v_ns[i],
            'vehicles_ew': self._v_ew[i],
            'cleared_total': self._cleared[i],
            'stopped': self._stopped[i],
            'phase_index': self._phase[i],
        }


# ---------------------------------------------------------------------- #
# Environment
# ---------------------------------------------------------------------- #
class TrafficSignalEnv:
    """
    Gym-style environment controlling N intersections.

    Args:
        num_intersections: number of intersections.
        num_phases: discrete phases per intersection (default 4).
        simulator: a TrafficSimulator instance (defaults to MockTrafficSimulator).
        reward_weights: optional dict forwarded to RewardCalculator.
        max_steps: episode length in simulation steps.
        sim_dt: seconds per step.
        min_green_steps: minimum number of steps a phase must be held before
                the env will accept a phase change. Enforced inside the env,
                not the simulator. Prevents flickering.
        seed: RNG seed for reproducibility (passed to MockTrafficSimulator
                if no simulator is supplied).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_intersections=10, num_phases=4,
                 simulator=None, reward_weights=None,
                 max_steps=3600, sim_dt=1.0,
                 min_green_steps=3, seed=None):
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.max_steps = max_steps
        self.sim_dt = sim_dt
        self.min_green_steps = min_green_steps

        self.simulator = simulator if simulator is not None \
            else MockTrafficSimulator(num_intersections, num_phases, seed=seed)

        self.signal_char = SignalCharacterization(num_intersections=num_intersections)
        self.reward_calc = RewardCalculator(num_intersections=num_intersections,
                                            weights=reward_weights)

        # Per-intersection phase tracking
        self._current_phase = [0] * num_intersections
        self._phase_hold_steps = [0] * num_intersections
        self._step_count = 0
        self._episode_phase_changes = 0

        self._intersection_ids = [f'int_{i}' for i in range(num_intersections)]

    # ------------------------------------------------------------------ #
    # Spaces (returned as plain dicts so no gym dependency is needed)
    # ------------------------------------------------------------------ #
    def get_action_space(self):
        """Returns action space spec. Action = list of N ints in [0, num_phases)."""
        return {
            'type': 'MultiDiscrete',
            'nvec': [self.num_phases] * self.num_intersections,
            'shape': (self.num_intersections,),
        }

    def get_observation_space(self):
        """Returns observation space spec (per-intersection dict)."""
        return {
            'type': 'Dict',
            'per_intersection': {
                'phase': {'type': 'Discrete', 'n': self.num_phases},
                'duration': {'type': 'Box', 'low': 0, 'high': self.max_steps},
                'queue_ratio': {'type': 'Box', 'low': 0.0, 'high': 1.0},
                'avg_duration': {'type': 'Box', 'low': 0.0, 'high': self.max_steps},
                'vehicles_ns': {'type': 'Box', 'low': 0, 'high': 10_000},
                'vehicles_ew': {'type': 'Box', 'low': 0, 'high': 10_000},
            },
        }

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        """Reset episode. Returns (observation, info)."""
        if seed is not None and hasattr(self.simulator, '_rng'):
            self.simulator._rng = random.Random(seed)

        self.simulator.reset()
        self.signal_char = SignalCharacterization(num_intersections=self.num_intersections)
        self.reward_calc.reset_episode()
        self._current_phase = [0] * self.num_intersections
        self._phase_hold_steps = [0] * self.num_intersections
        self._step_count = 0
        self._episode_phase_changes = 0

        # Apply an initial no-op step so we have real observations to return
        self.simulator.step(self._current_phase, dt=self.sim_dt)
        for iid in self._intersection_ids:
            i = int(iid.split('_')[-1])
            state = self.simulator.get_intersection_state(iid)
            self.signal_char.record_signal_state(
                iid,
                phase_index=state['phase_index'],
                duration=self._phase_hold_steps[i] * self.sim_dt,
                vehicle_count_ns=state['vehicles_ns'],
                vehicle_count_ew=state['vehicles_ew'],
            )

        obs = self._build_observation()
        info = {'step': 0, 'phase_changes': 0}
        return obs, info

    def step(self, action):
        """
        Args:
            action: iterable of N phase indices, each in [0, num_phases).

        Returns:
            observation, reward, terminated, truncated, info
        """
        if len(action) != self.num_intersections:
            raise ValueError(
                f"action length {len(action)} != num_intersections "
                f"{self.num_intersections}"
            )

        # Enforce min-green: ignore requested changes that come too early
        applied = []
        for i, req in enumerate(action):
            req = int(req) % self.num_phases
            if req != self._current_phase[i] and \
                    self._phase_hold_steps[i] < self.min_green_steps:
                # Hold current phase
                applied.append(self._current_phase[i])
            else:
                if req != self._current_phase[i]:
                    self._episode_phase_changes += 1
                applied.append(req)
                self._phase_hold_steps[i] = 0

        # Advance simulator with the (possibly min-green-corrected) phases
        self.simulator.step(applied, dt=self.sim_dt)
        self._step_count += 1

        # Update per-intersection phase bookkeeping & record observations
        total_reward = 0.0
        per_intersection_info = {}
        for i, iid in enumerate(self._intersection_ids):
            self._current_phase[i] = applied[i]
            self._phase_hold_steps[i] += 1

            state = self.simulator.get_intersection_state(iid)
            self.signal_char.record_signal_state(
                iid,
                phase_index=state['phase_index'],
                duration=self._phase_hold_steps[i] * self.sim_dt,
                vehicle_count_ns=state['vehicles_ns'],
                vehicle_count_ew=state['vehicles_ew'],
            )

            obs_i = self.signal_char.get_observation_space(iid)
            result = self.reward_calc.compute(
                iid, obs_i,
                cleared_vehicles=state['cleared_total'],
                stopped_vehicles=state['stopped'],
                sim_dt=self.sim_dt,
            )
            total_reward += result['reward']
            per_intersection_info[iid] = {
                'reward': result['reward'],
                'components': result['components'],
                'cleared': state['cleared_total'],
                'stopped': state['stopped'],
                'phase': state['phase_index'],
            }

        obs = self._build_observation()
        terminated = False  # episodes end by truncation (time limit), not failure
        truncated = self._step_count >= self.max_steps
        info = {
            'step': self._step_count,
            'phase_changes': self._episode_phase_changes,
            'per_intersection': per_intersection_info,
            'signal_metrics': self.signal_char.get_metrics(),
            'reward_metrics': self.reward_calc.get_metrics(),
        }
        return obs, total_reward, terminated, truncated, info

    def render(self, mode='human'):
        """Print a compact one-line summary of the current episode state."""
        sig_m = self.signal_char.get_metrics()
        rew_m = self.reward_calc.get_metrics()
        print(f"[step {self._step_count:>4}] "
              f"phase_changes={self._episode_phase_changes} "
              f"avg_return={rew_m['reward_avg_episode_return']:.3f} "
              f"avg_queue={rew_m['reward_avg_queue_pressure']:.3f} "
              f"avg_throughput={rew_m['reward_avg_throughput_gain']:.3f} "
              f"intersections_tracked={sig_m['signal_intersections_tracked']}")

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_observation(self):
        """Build the full per-intersection observation dict."""
        return {iid: self.signal_char.get_observation_space(iid)
                for iid in self._intersection_ids}


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    import random as _r

    env = TrafficSignalEnv(
        num_intersections=5,
        num_phases=4,
        max_steps=20,
        sim_dt=1.0,
        min_green_steps=2,
        seed=42,
    )

    print("Action space:      ", env.get_action_space())
    print("Observation space: ", env.get_observation_space())
    print()

    obs, info = env.reset()
    print(f"Reset -> {len(obs)} intersections, initial int_0 obs: {obs['int_0']}")
    print()

    rng = _r.Random(123)
    episode_return = 0.0
    step_count = 0
    done = False
    while not done:
        action = [rng.randint(0, env.num_phases - 1)
                  for _ in range(env.num_intersections)]
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        step_count += 1
        if step_count % 5 == 0 or truncated:
            env.render()
        done = terminated or truncated

    print()
    print(f"Episode finished after {step_count} steps, total return: {episode_return:.3f}")
    print(f"Total phase changes during episode: {info['phase_changes']}")
    print()
    print("--- Final signal metrics ---")
    print(info['signal_metrics'])
    print()
    print("--- Final reward metrics ---")
    print(info['reward_metrics'])
    print()
    print("--- Final per-intersection snapshot (int_0) ---")
    print(info['per_intersection']['int_0'])
