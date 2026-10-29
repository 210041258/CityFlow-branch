#!/usr/bin/env python3
"""
Multi Simulator Module
======================
Orchestrates multiple simulations in parallel or hierarchically. Sits on
top of training_env.py's TrafficSimulator interface, so the env itself
does not need to know how many simulators exist underneath.

Two complementary utilities:

1. MultiSimulator
   ---------------
   A TrafficSimulator that wraps N child simulators and routes per-
   intersection calls to the correct child. Useful for:
     - mixing engine types (some CityFlow, some Mock) in one env
     - partitioning a large network across multiple engine instances
     - attaching a "shadow" simulator (e.g., a replay) alongside a live one

2. ParallelEvaluator
   ------------------
   Runs N independent episodes concurrently and aggregates their
   MetricsPanel results. Useful for:
     - policy evaluation across seeds
     - hyperparameter sweeps across action modes
     - data collection for offline RL / imitation learning

Pipeline:
    signal_characterization.py  ->  observation space
    reward_calculator.py        ->  scalar reward
    action_space.py             ->  action space
    training_env.py             ->  gym-style env
    run.py                      ->  entry point
    panel.py                    ->  metrics panel
    multi_sim.py                ->  multi-simulator orchestration (this file)

Design notes
------------
* stdlib only (concurrent.futures, threading, argparse).
* MultiSimulator preserves the child's `int_{i}` ID convention: each
  child handles its own intersections as if it were a standalone env.
  The MultiSimulator maintains the global<->local ID bijection.
* ParallelEvaluator uses ThreadPoolExecutor. The mock simulator is pure
  Python, so the GIL limits speedup there; the interface is intended for
  external backends (CityFlow, SUMO) that release the GIL during
  stepping. Swap to ProcessPoolExecutor if your env becomes picklable.
"""

import argparse
import concurrent.futures
import threading

from action_space import ActionSpace
from training_env import TrafficSignalEnv, MockTrafficSimulator, TrafficSimulator
from panel import MetricsPanel


# ====================================================================== #
# Part 1: MultiSimulator -- hierarchical routing
# ====================================================================== #
class MultiSimulator(TrafficSimulator):
    """
    A TrafficSimulator that routes per-intersection calls to N child
    simulators.

    Each child is expected to handle its own subset using the standard
    `int_0`, `int_1`, ... convention (local IDs). The MultiSimulator
    maintains the bijection:

        global 'int_k'  ->  (child, local 'int_j')

    Constraints
    -----------
    * All children must share the same `num_phases` (phase indices are
      passed through unchanged).
    * The union of all children's global IDs must equal the full set the
      env will pass to `step`.

    Args:
        num_intersections: total N the env will pass in.
        children: list of (child_simulator, [global_ids_it_handles]).
            Global IDs may be any strings; the child will see them
            remapped to its local `int_0..int_{k-1}` convention.
        intersection_ids: optional explicit global ordering; defaults to
            [f'int_{i}' for i in range(num_intersections)].
    """

    def __init__(self, num_intersections, children, intersection_ids=None):
        self.num_intersections = int(num_intersections)
        self.intersection_ids = intersection_ids or [
            f'int_{i}' for i in range(self.num_intersections)
        ]
        if len(self.intersection_ids) != self.num_intersections:
            raise ValueError(
                f"intersection_ids has {len(self.intersection_ids)} entries, "
                f"expected {self.num_intersections}"
            )

        self._children = []
        self._global_index = {gid: i for i, gid in enumerate(self.intersection_ids)}
        self._id_route = {}  # global_id -> (child, local_id)

        claimed = set()
        for child_idx, (child, global_ids) in enumerate(children):
            if not global_ids:
                raise ValueError(f"child[{child_idx}] has no intersection_ids")
            local_ids = [f'int_{j}' for j in range(len(global_ids))]
            for global_id, local_id in zip(global_ids, local_ids):
                if global_id in self._id_route:
                    raise ValueError(
                        f"global intersection {global_id!r} claimed by "
                        f"multiple children"
                    )
                if global_id not in self._global_index:
                    raise ValueError(
                        f"child claims unknown global intersection {global_id!r}"
                    )
                self._id_route[global_id] = (child, local_id)
                claimed.add(global_id)
            self._children.append((child, list(global_ids)))

        missing = set(self.intersection_ids) - claimed
        if missing:
            raise ValueError(
                f"no child handles intersections: {sorted(missing)}"
            )

    # -------------------------------------------------------------- #
    # TrafficSimulator interface
    # -------------------------------------------------------------- #
    def reset(self):
        for child, _ in self._children:
            child.reset()

    def step(self, actions, dt=1.0):
        if len(actions) != self.num_intersections:
            raise ValueError(
                f"MultiSimulator.step got {len(actions)} actions, "
                f"expected {self.num_intersections}"
            )
        for child, global_ids in self._children:
            local_actions = [actions[self._global_index[gid]]
                             for gid in global_ids]
            child.step(local_actions, dt=dt)

    def get_intersection_state(self, intersection_id):
        if intersection_id not in self._id_route:
            raise KeyError(
                f"unknown intersection {intersection_id!r} "
                f"(known: {sorted(self._id_route)})"
            )
        child, local_id = self._id_route[intersection_id]
        return child.get_intersection_state(local_id)

    # -------------------------------------------------------------- #
    # Introspection
    # -------------------------------------------------------------- #
    def get_metrics(self):
        return {
            'multi_sim_num_children': len(self._children),
            'multi_sim_num_intersections': self.num_intersections,
            'multi_sim_children': [
                {'intersections': list(global_ids),
                 'num_intersections': len(global_ids)}
                for _, global_ids in self._children
            ],
        }

    def __repr__(self):
        return (f"MultiSimulator(N={self.num_intersections}, "
                f"children={len(self._children)})")


# ====================================================================== #
# Part 2: ParallelEvaluator -- independent episodes
# ====================================================================== #
class ParallelEvaluator:
    """
    Runs N independent episodes concurrently (threaded).

    Each worker builds its own env + action_space (via the supplied
    builders), runs one episode, and returns a MetricsPanel. The
    aggregated result includes per-episode returns plus a combined panel
    with one record per episode -- suitable for downstream comparison
    with `MetricsPanel.compare()`.

    Args:
        env_builder: callable(seed) -> TrafficSignalEnv.
        action_space_builder: callable(seed) -> ActionSpace.
        num_workers: max concurrent workers (default 4).
        mode: action mode passed to action_space_builder; used to drive
            masking in the episode loop.
    """

    def __init__(self, env_builder, action_space_builder,
                 num_workers=4, mode='switch_hold'):
        self.env_builder = env_builder
        self.action_space_builder = action_space_builder
        self.num_workers = max(1, int(num_workers))
        self.mode = mode
        self._lock = threading.Lock()

    # -------------------------------------------------------------- #
    # Single episode
    # -------------------------------------------------------------- #
    def _run_episode(self, episode_idx, seed):
        env = self.env_builder(seed)
        act = self.action_space_builder(seed)

        panel = MetricsPanel(history_size=100)
        obs, info = env.reset(seed=seed)
        panel.record(env=env, action_space=act, step=0)

        total_return = 0.0
        step_count = 0
        done = False
        last_info = info

        while not done:
            if act.mode == 'switch_hold':
                min_green_ok = [
                    env._phase_hold_steps[i] >= env.min_green_steps
                    for i in range(env.num_intersections)
                ]
                masks = act.get_action_mask(env._current_phase, min_green_ok)
                action = act.sample(masks)
            else:
                action = act.sample()

            act.validate(action)
            applied = act.resolve(action, env._current_phase)
            obs, reward, term, trunc, info = env.step(applied)
            total_return += reward
            step_count += 1
            last_info = info

            if step_count % 5 == 0 or term or trunc:
                panel.record(env=env, action_space=act,
                             extra={'last_reward': float(reward),
                                    'episode_idx': episode_idx,
                                    'seed': seed})

            done = term or trunc

        env.close()
        return {
            'episode_idx': episode_idx,
            'seed': seed,
            'return': float(total_return),
            'step_count': int(step_count),
            'final_info': last_info,
            'panel': panel,
        }

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #
    def evaluate(self, num_episodes, seeds=None, on_progress=None):
        """Run `num_episodes` concurrently and return aggregate stats.

        Args:
            num_episodes: number of episodes to run.
            seeds: optional explicit seeds (must have len == num_episodes).
            on_progress: optional callable(result_dict) invoked as each
                episode completes.

        Returns:
            dict with per-episode results and aggregate statistics.
        """
        if num_episodes < 1:
            raise ValueError("num_episodes must be >= 1")
        if seeds is None:
            seeds = list(range(num_episodes))
        if len(seeds) != num_episodes:
            raise ValueError(
                f"seeds has {len(seeds)} entries, expected {num_episodes}"
            )

        results = []
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(self._run_episode, i, seed): i
                for i, seed in enumerate(seeds)
            }
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                results.append(result)
                if on_progress is not None:
                    on_progress(result)

        results.sort(key=lambda r: r['episode_idx'])
        return self._aggregate(results)

    # -------------------------------------------------------------- #
    # Aggregation
    # -------------------------------------------------------------- #
    def _aggregate(self, results):
        returns = [r['return'] for r in results]
        n = len(returns)
        avg = sum(returns) / n if n else 0.0
        std = ((sum((x - avg) ** 2 for x in returns) / n) ** 0.5
               if n > 1 else 0.0)

        combined_panel = MetricsPanel(history_size=n + 1)
        for r in results:
            latest = r['panel'].latest() or {}
            combined_panel.record(
                step=r['step_count'],
                extra={
                    'episode_idx': r['episode_idx'],
                    'seed': r['seed'],
                    'episode_return': r['return'],
                    'episode_steps': r['step_count'],
                    'final_avg_return':
                        latest.get('reward', {}).get(
                            'reward_avg_episode_return', 0.0),
                },
            )

        return {
            'num_episodes': n,
            'avg_return': avg,
            'std_return': std,
            'min_return': min(returns) if returns else 0.0,
            'max_return': max(returns) if returns else 0.0,
            'total_return': sum(returns),
            'per_episode': [
                {'episode_idx': r['episode_idx'],
                 'seed': r['seed'],
                 'return': r['return'],
                 'step_count': r['step_count']}
                for r in results
            ],
            'combined_panel': combined_panel,
        }


# ====================================================================== #
# High-level driver: run multiple modes and compare
# ====================================================================== #
def run_multi(modes=('discrete_phase', 'switch_hold'),
              num_episodes=4, num_workers=2,
              num_intersections=3, num_phases=4,
              max_steps=30, seed=0, verbose=True):
    """Run the same pipeline across action modes and compare returns."""
    all_results = {}
    for mode in modes:
        if verbose:
            print(f"\n=== mode={mode!r}  episodes={num_episodes} ===")
        evaluator = ParallelEvaluator(
            env_builder=lambda s: TrafficSignalEnv(
                num_intersections=num_intersections,
                num_phases=num_phases,
                max_steps=max_steps,
                seed=s,
            ),
            action_space_builder=lambda s, _m=mode: ActionSpace(
                num_intersections=num_intersections,
                num_phases=num_phases,
                mode=_m,
                seed=s,
            ),
            num_workers=num_workers,
            mode=mode,
        )
        result = evaluator.evaluate(num_episodes, seeds=list(range(seed, seed + num_episodes)))
        all_results[mode] = result
        if verbose:
            print(f"  avg_return={result['avg_return']:+.3f}  "
                  f"std={result['std_return']:.3f}  "
                  f"min={result['min_return']:+.3f}  "
                  f"max={result['max_return']:+.3f}")

    if verbose:
        print("\n=== ranking (avg_return, higher is better) ===")
        for mode, res in sorted(all_results.items(),
                                key=lambda kv: -kv[1]['avg_return']):
            print(f"  {mode:<20s} {res['avg_return']:+.3f}  "
                  f"(std {res['std_return']:.3f}, n={res['num_episodes']})")
    return all_results


# ====================================================================== #
# CLI
# ====================================================================== #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Multi-simulator orchestration for the RL pipeline.")
    p.add_argument(
        '--modes', default='discrete_phase,switch_hold',
        help="Comma-separated action modes to compare "
             "(choices: discrete_phase,switch_hold,phase_duration).")
    p.add_argument('--episodes', type=int, default=4,
                   help="Episodes per mode (default 4).")
    p.add_argument('--workers', type=int, default=2,
                   help="Concurrent workers (default 2).")
    p.add_argument('--num-intersections', type=int, default=3)
    p.add_argument('--num-phases', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--quiet', action='store_true',
                   help="Suppress progress output.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    modes = tuple(m.strip() for m in args.modes.split(',') if m.strip())
    for m in modes:
        if m not in ActionSpace.VALID_MODES:
            raise SystemExit(f"invalid mode {m!r}; "
                             f"expected one of {ActionSpace.VALID_MODES}")
    run_multi(
        modes=modes,
        num_episodes=args.episodes,
        num_workers=args.workers,
        num_intersections=args.num_intersections,
        num_phases=args.num_phases,
        max_steps=args.max_steps,
        seed=args.seed,
        verbose=not args.quiet,
    )


# ====================================================================== #
# Smoke test -- exercises both utilities, mirrors the pipeline style
# ====================================================================== #
if __name__ == '__main__':
    print("=" * 74)
    print(" 1. MultiSimulator: two child mocks -> one 4-intersection env")
    print("=" * 74)

    child_a = MockTrafficSimulator(num_intersections=2, num_phases=4, seed=1)
    child_b = MockTrafficSimulator(num_intersections=2, num_phases=4, seed=2)

    multi = MultiSimulator(
        num_intersections=4,
        children=[
            (child_a, ['int_0', 'int_1']),
            (child_b, ['int_2', 'int_3']),
        ],
    )
    print(multi)
    print("routing:", {gid: (id(c), lid)
                       for gid, (c, lid) in multi._id_route.items()})

    env = TrafficSignalEnv(
        num_intersections=4, num_phases=4, simulator=multi,
        max_steps=5, sim_dt=1.0, min_green_steps=2, seed=0,
    )
    obs, _ = env.reset()
    for _ in range(3):
        action = [0, 1, 2, 3]
        obs, reward, term, trunc, info = env.step(action)
    print(f"after 3 steps -> phase_changes={info['phase_changes']} "
          f"avg_return={info['reward_metrics']['reward_avg_episode_return']:+.3f}")
    env.close()

    print()
    print("=" * 74)
    print(" 2. ParallelEvaluator: run 3 episodes across 2 modes")
    print("=" * 74)

    results = run_multi(
        modes=('discrete_phase', 'switch_hold'),
        num_episodes=3,
        num_workers=2,
        num_intersections=3,
        num_phases=4,
        max_steps=15,
        seed=42,
    )

    print()
    print("=== combined panel pitch per mode ===")
    for mode, res in results.items():
        print(f"  [{mode}]")
        print(f"    {res['combined_panel'].pitch()}")

    print()
    print("=== compare discrete_phase vs switch_hold ===")
    if 'discrete_phase' in results and 'switch_hold' in results:
        a = results['discrete_phase']['combined_panel']
        b = results['switch_hold']['combined_panel']
        deltas = a.compare(b)
        for k, v in sorted(deltas.items()):
            if abs(v) > 1e-9:
                print(f"  {k:<40s} {v:+.4f}")

    print("\nSmoke test passed.")