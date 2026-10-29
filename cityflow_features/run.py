#!/usr/bin/env python3
"""
Run Module
==========
Entry point for the traffic-signal RL pipeline. Wires together:

    signal_characterization.py  ->  observation space
    reward_calculator.py        ->  scalar reward
    action_space.py             ->  action space
    training_env.py             ->  gym-style env

Usage:
    python run.py --training-env [--steps N] [--seed S] [--mode MODE]
    python run.py --smoke-test

The `--training-env` mode runs a self-contained training loop using the
MockTrafficSimulator (or a CityFlowTrafficSimulator if configured).
"""

import argparse
import random
import sys

from action_space import ActionSpace
from training_env import TrafficSignalEnv, MockTrafficSimulator
from signal_characterization import SignalCharacterization
from reward_calculator import RewardCalculator


# ---------------------------------------------------------------------- #
# CityFlow Adapter
# ---------------------------------------------------------------------- #
class CityFlowTrafficSimulator:
    """
    Adapter that makes a CityFlow engine conform to the TrafficSimulator
    interface expected by TrafficSignalEnv.

    This is a *thin* wrapper. It assumes:
        - You have a CityFlow config file (e.g., `config.json`).
        - The roadnet defines intersections with controllable phases.
        - You have mapped intersection IDs (from CityFlow) to the
          `int_0`, `int_1`, ... convention used by the env.

    Args:
        config_path: Path to the CityFlow JSON config.
        intersection_ids: List of CityFlow intersection IDs, in the same
            order as the env's internal `int_0` ... `int_{N-1}`.
        threads: Number of simulation threads.
        seed: Random seed for CityFlow (if supported).
    """

    def __init__(self, config_path, intersection_ids, threads=1, seed=None):
        try:
            import cityflow
        except ImportError as exc:
            raise ImportError(
                "CityFlow is not installed. Install it via `pip install cityflow` "
                "or follow the official build instructions."
            ) from exc

        self._cf = cityflow
        self._engine = cityflow.Engine(config_path, thread_num=threads)
        if seed is not None and hasattr(self._engine, 'set_random_seed'):
            self._engine.set_random_seed(seed)

        self.intersection_ids = list(intersection_ids)
        self.num_intersections = len(intersection_ids)
        self._phase_cache = [0] * self.num_intersections

    def reset(self):
        self._engine.reset()
        self._phase_cache = [0] * self.num_intersections

    def step(self, actions, dt=1.0):
        # CityFlow's step advances one simulation tick. The env's `sim_dt`
        # controls how many ticks we advance per decision.
        for phase in actions:
            self._engine.set_tl_phase(self.intersection_ids, int(phase))
        self._phase_cache = [int(a) for a in actions]
        # Advance one tick (the env calls this once per decision step).
        self._engine.next_step()

    def get_intersection_state(self, intersection_id):
        idx = self.intersection_ids.index(intersection_id)
        # CityFlow exposes lane vehicle counts; we map them to NS/EW.
        # This is a simplified mapping -- adjust based on your roadnet.
        lanes = self._engine.get_lane_vehicle_count()
        waiting = self._engine.get_lane_waiting_vehicle_count()

        # Example heuristic: split lanes by direction.
        v_ns = 0
        v_ew = 0
        stopped = 0
        for lane_id, count in lanes.items():
            if lane_id.startswith('n_') or lane_id.startswith('s_'):
                v_ns += count
            else:
                v_ew += count
        for lane_id, w in waiting.items():
            stopped += w

        return {
            'vehicles_ns': v_ns,
            'vehicles_ew': v_ew,
            'cleared_total': self._engine.get_vehicle_count(),  # proxy
            'stopped': stopped,
            'phase_index': self._phase_cache[idx],
        }


# ---------------------------------------------------------------------- #
# Training Loop
# ---------------------------------------------------------------------- #
def run_training(args):
    """Run a single-episode training loop for demonstration / smoke testing."""
    print("=" * 70)
    print("Traffic Signal RL Pipeline -- Training Environment")
    print("=" * 70)
    print(f"Mode:          {args.mode}")
    print(f"Intersections: {args.num_intersections}")
    print(f"Phases:        {args.num_phases}")
    print(f"Steps:         {args.steps}")
    print(f"Seed:          {args.seed}")
    print()

    # --- Build the action space ---
    action_space = ActionSpace(
        num_intersections=args.num_intersections,
        num_phases=args.num_phases,
        mode=args.mode,
        seed=args.seed,
    )
    print(f"Action space: {action_space}")

    # --- Build the environment ---
    env = TrafficSignalEnv(
        num_intersections=args.num_intersections,
        num_phases=args.num_phases,
        max_steps=args.steps,
        sim_dt=args.sim_dt,
        min_green_steps=args.min_green_steps,
        seed=args.seed,
    )
    print(f"Env action spec: {env.get_action_space()}")
    print()

    # --- Reset ---
    obs, info = env.reset(seed=args.seed)
    print(f"Reset -> {len(obs)} intersections, step={info['step']}")
    print()

    # --- Training loop (random policy for demonstration) ---
    rng = random.Random(args.seed + 1)
    episode_return = 0.0
    step_count = 0
    done = False

    while not done:
        # Use the action space to sample a valid action, optionally with masks.
        # For 'switch_hold' mode, pass masks to respect min-green.
        if action_space.mode == 'switch_hold':
            min_green_ok = [env._phase_hold_steps[i] >= env.min_green_steps
                            for i in range(env.num_intersections)]
            masks = action_space.get_action_mask(
                env._current_phase, min_green_ok)
            action = action_space.sample(masks)
        else:
            action = action_space.sample()

        action_space.validate(action)
        applied = action_space.resolve(action, env._current_phase)

        obs, reward, terminated, truncated, info = env.step(applied)
        episode_return += reward
        step_count += 1

        if step_count % 10 == 0 or truncated:
            env.render()

        done = terminated or truncated

    # --- Final report ---
    print()
    print("=" * 70)
    print(f"Episode finished after {step_count} steps")
    print(f"Total return: {episode_return:.3f}")
    print(f"Phase changes: {info['phase_changes']}")
    print()
    print("--- Final Signal Metrics ---")
    for k, v in info['signal_metrics'].items():
        print(f"  {k}: {v}")
    print()
    print("--- Final Reward Metrics ---")
    for k, v in info['reward_metrics'].items():
        print(f"  {k}: {v}")
    print()
    print("--- Action Space Metrics ---")
    for k, v in action_space.get_metrics().items():
        print(f"  {k}: {v}")
    print("=" * 70)

    env.close()
    return episode_return


# ---------------------------------------------------------------------- #
# Smoke Test
# ---------------------------------------------------------------------- #
def run_smoke_test():
    """Quick sanity check across all three action modes."""
    print("Running pipeline smoke test...\n")

    for mode in ('discrete_phase', 'switch_hold', 'phase_duration'):
        print(f"--- mode: {mode} ---")
        action_space = ActionSpace(
            num_intersections=3,
            num_phases=4,
            mode=mode,
            seed=0,
        )
        env = TrafficSignalEnv(
            num_intersections=3,
            num_phases=4,
            max_steps=5,
            sim_dt=1.0,
            min_green_steps=2,
            seed=0,
        )
        obs, _ = env.reset()
        for _ in range(3):
            if mode == 'switch_hold':
                masks = action_space.get_action_mask(
                    env._current_phase, [True] * 3)
                action = action_space.sample(masks)
            else:
                action = action_space.sample()
            action_space.validate(action)
            applied = action_space.resolve(action, env._current_phase)
            obs, reward, term, trunc, info = env.step(applied)
        print(f"  OK -- return so far: {info['reward_metrics']['reward_avg_episode_return']:.3f}")
        env.close()

    print("\nSmoke test passed.")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Traffic Signal RL Pipeline runner")
    parser.add_argument(
        '--training-env', action='store_true',
        help="Run the training environment loop.")
    parser.add_argument(
        '--smoke-test', action='store_true',
        help="Run a quick smoke test across all action modes.")
    parser.add_argument(
        '--num-intersections', type=int, default=5,
        help="Number of intersections (default: 5).")
    parser.add_argument(
        '--num-phases', type=int, default=4,
        help="Number of phases per intersection (default: 4).")
    parser.add_argument(
        '--steps', type=int, default=50,
        help="Episode length in steps (default: 50).")
    parser.add_argument(
        '--sim-dt', type=float, default=1.0,
        help="Simulation seconds per step (default: 1.0).")
    parser.add_argument(
        '--min-green-steps', type=int, default=3,
        help="Minimum green steps before a phase change is allowed "
             "(default: 3).")
    parser.add_argument(
        '--mode', choices=ActionSpace.VALID_MODES,
        default='switch_hold',
        help="Action mode (default: switch_hold).")
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed (default: 42).")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
    elif args.training_env:
        run_training(args)
    else:
        parser.print_help()
        print("\nHint: use `--training-env` or `--smoke-test`.")
        sys.exit(0)


if __name__ == '__main__':
    main()