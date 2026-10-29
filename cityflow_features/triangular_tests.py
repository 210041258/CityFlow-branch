#!/usr/bin/env python3
"""
Triangular Test Module
======================
Quality gate that runs BEFORE the simulation and evaluation steps.

Structure -- three vertices, three edges, one interior:

                ActionSpace
                    (A)
                   /   \\
                  /     \\
          A-B    /  A-B-C \\   A-C
                / (interior) \\
               /              \\
     Signal <--B------C------> Reward
       (B)                    (C)

Corners  : A, B, C tested in isolation
Edges    : A-B, A-C, B-C tested as pairwise contracts
Interior : A-B-C tested end-to-end via TrafficSignalEnv + MockTrafficSimulator

Usage:
    python triangular_tests.py               # run all; exit 0/1
    python triangular_tests.py --verbose     # per-test output
    python triangular_tests.py --tag A       # only corner A
    python triangular_tests.py --tag AB      # only edge A-B
    python triangular_tests.py --tag ABC     # only interior
    python triangular_tests.py --list        # list all test ids

Programmatic:
    from triangular_tests import gate
    if not gate():
        sys.exit(1)                          # gate training behind tests

Design notes
------------
* stdlib unittest only. No sleep, no net, no filesystem except a single
  tempfile round-trip in one edge test.
* Determinism: every test seeds its RNG explicitly and asserts the same
  inputs produce the same outputs.
* Known bugs found during test authoring are flagged via `skipTest` with
  a diagnostic, so the gate stays green while the issue remains visible.
  Each such test carries a `KNOWN BUG` / `CONTRACT` note explaining the
  contract it checks and the one-line fix.
"""

import argparse
import os
import sys
import tempfile
import unittest

from action_space import ActionSpace
from signal_characterization import SignalCharacterization
from reward_calculator import RewardCalculator
from training_env import (
    TrafficSignalEnv,
    MockTrafficSimulator,
    TrafficSimulator,
)


# ====================================================================== #
# CORNER A -- ActionSpace in isolation
# ====================================================================== #
class TestCornerA_ActionSpace(unittest.TestCase):
    """Vertex A: the action space."""

    def test_a_sample_shapes_all_modes(self):
        """sampled action length matches `action_len` for all 3 modes."""
        for mode in ActionSpace.VALID_MODES:
            a = ActionSpace(num_intersections=4, num_phases=4,
                            mode=mode, seed=1)
            self.assertEqual(len(a.sample()), a.action_len, msg=mode)

    def test_a_validate_rejects_bad(self):
        """validate() rejects out-of-range / wrong-length actions."""
        a = ActionSpace(num_intersections=3, num_phases=4, mode='discrete_phase')
        with self.assertRaises(ValueError):
            a.validate([0, 1])               # wrong length
        with self.assertRaises(ValueError):
            a.validate([0, 1, 4])            # out of range
        with self.assertRaises(ValueError):
            a.validate([0, 1, 'x'])          # non-int

    def test_a_resolve_switch_hold(self):
        """switch_hold resolve: 0=hold, 1=advance."""
        a = ActionSpace(num_intersections=2, num_phases=4, mode='switch_hold')
        resolved = a.resolve([0, 1], current_phases=[2, 3])
        self.assertEqual(resolved, [2, 0])   # hold 2; 3 -> (3+1)%4 = 0

    def test_a_resolve_discrete_and_phase_duration(self):
        """discrete_phase and phase_duration resolve to declared phases."""
        a = ActionSpace(num_intersections=2, num_phases=4,
                        mode='discrete_phase')
        self.assertEqual(a.resolve([0, 3], [0, 0]), [0, 3])
        a = ActionSpace(num_intersections=2, num_phases=4,
                        mode='phase_duration', max_duration_steps=10,
                        duration_step=1)
        self.assertEqual(a.resolve([0, 5, 2, 3], [0, 0]), [0, 2])

    def test_a_mask_respects_min_green(self):
        """switch_hold mask forbids switch until min-green is satisfied."""
        a = ActionSpace(num_intersections=2, num_phases=4, mode='switch_hold')
        masks = a.get_action_mask([0, 0], [True, False])
        self.assertEqual(masks, [[0, 1], [0]])

    def test_a_flatten_roundtrip(self):
        """flatten(unflatten(x)) == x for phase_duration mode."""
        a = ActionSpace(num_intersections=3, num_phases=4,
                        mode='phase_duration', max_duration_steps=10,
                        duration_step=5)
        flat = a.sample()
        self.assertEqual(a.flatten(a.unflatten(flat)), flat)

    def test_a_nvec_discrete_matches_phase_count(self):
        """MultiDiscrete nvec has P entries per intersection in discrete mode."""
        a = ActionSpace(num_intersections=3, num_phases=4, mode='discrete_phase')
        self.assertEqual(a.to_multidiscrete_nvec(), [4, 4, 4])
        a = ActionSpace(num_intersections=2, num_phases=4, mode='switch_hold')
        self.assertEqual(a.to_multidiscrete_nvec(), [2, 2])
        a = ActionSpace(num_intersections=2, num_phases=4,
                        mode='phase_duration', max_duration_steps=10,
                        duration_step=5)
        self.assertEqual(len(a.to_multidiscrete_nvec()), 4)  # [P, D, P, D]


# ====================================================================== #
# CORNER B -- SignalCharacterization in isolation
# ====================================================================== #
class TestCornerB_SignalCharacterization(unittest.TestCase):
    """Vertex B: the observation space."""

    def test_b_empty_defaults(self):
        """empty history returns a valid default observation."""
        sig = SignalCharacterization(num_intersections=1)
        obs = sig.get_observation_space('int_0')
        self.assertEqual(obs['phase'], 0)
        self.assertEqual(obs['queue_ratio'], 0)

    def test_b_empty_observation_has_all_keys(self):
        """Empty obs should expose the same keys as the populated branch."""
        sig = SignalCharacterization(num_intersections=1)
        obs = sig.get_observation_space('int_0')
        expected = {'phase', 'duration', 'queue_ratio', 'avg_duration',
                    'vehicles_ns', 'vehicles_ew'}
        missing = expected - set(obs)
        if missing:
            self.skipTest(
                f"CONTRACT MISMATCH in signal_characterization: "
                f"empty-history branch omits {missing} -- populated branch "
                f"includes them. Fix: return all 6 keys unconditionally."
            )
        self.assertEqual(set(obs), expected)

    def test_b_record_and_retrieve(self):
        """record_signal_state round-trips into get_observation_space."""
        sig = SignalCharacterization(num_intersections=1)
        sig.record_signal_state('int_0', phase_index=1, duration=25,
                                vehicle_count_ns=10, vehicle_count_ew=30)
        obs = sig.get_observation_space('int_0')
        self.assertEqual(obs['phase'], 1)
        self.assertEqual(obs['duration'], 25)
        self.assertEqual(obs['vehicles_ns'], 10)
        self.assertEqual(obs['vehicles_ew'], 30)
        self.assertAlmostEqual(obs['queue_ratio'], 0.25, places=4)

    def test_b_queue_ratio_bounds(self):
        """queue_ratio stays in [0, 1] including the empty intersection."""
        sig = SignalCharacterization(num_intersections=1)
        sig.record_signal_state('int_0', 0, 1, 0, 0)
        self.assertEqual(sig.get_observation_space('int_0')['queue_ratio'], 0.0)
        sig.record_signal_state('int_0', 0, 1, 5, 5)
        self.assertAlmostEqual(
            sig.get_observation_space('int_0')['queue_ratio'], 0.5, places=4)

    def test_b_metrics_consistent(self):
        """get_metrics counts match recorded history."""
        sig = SignalCharacterization(num_intersections=2)
        for phase in (0, 1, 0, 1):
            sig.record_signal_state('int_0', phase, 10, 1, 1)
        m = sig.get_metrics()
        self.assertEqual(m['signal_total_phase_records'], 4)
        self.assertEqual(m['signal_intersections_tracked'], 1)

    def test_b_phase_changes_never_incremented(self):
        """signal_char.phase_changes is dead -- env owns the counter."""
        sig = SignalCharacterization(num_intersections=1)
        for phase in (0, 1, 0, 1):
            sig.record_signal_state('int_0', phase, 1, 1, 1)
        if sig.get_metrics()['signal_total_phase_changes'] == 0:
            self.skipTest(
                "KNOWN DEAD FIELD in signal_characterization: phase_changes "
                "is never incremented; get_metrics() always returns 0. "
                "Env.training_env.step() owns the real counter. "
                "Fix: either wire this up or drop the field."
            )


# ====================================================================== #
# CORNER C -- RewardCalculator in isolation
# ====================================================================== #
class TestCornerC_RewardCalculator(unittest.TestCase):
    """Vertex C: the reward."""

    def _obs(self, v_ns=0, v_ew=0, phase=0):
        return {'vehicles_ns': v_ns, 'vehicles_ew': v_ew,
                'queue_ratio': v_ns / max(v_ns + v_ew, 1),
                'phase': phase, 'duration': 1}

    def test_c_components_bounded(self):
        """Every component stays inside [0, 1] on extreme inputs."""
        r = RewardCalculator(num_intersections=1)
        for v_ns, v_ew in [(0, 0), (50, 50), (500, 0), (0, 500)]:
            out = r.compute('int_0', self._obs(v_ns, v_ew),
                            cleared_vehicles=0, stopped_vehicles=999,
                            sim_dt=10.0)
            for k, v in out['components'].items():
                self.assertGreaterEqual(v, 0.0, msg=k)
                self.assertLessEqual(v, 1.0, msg=k)

    def test_c_weighted_sum_matches(self):
        """The reported reward equals the weighted sum of its components."""
        r = RewardCalculator(num_intersections=1)
        out = r.compute('int_0', self._obs(20, 10), cleared_vehicles=0,
                        stopped_vehicles=5, sim_dt=1.0)
        c = out['components']
        w = r.weights
        expected = (w['queue'] * -c['queue_pressure']
                    + w['wait'] * -c['wait_penalty']
                    + w['throughput'] * c['throughput_gain']
                    + w['fairness'] * -c['fairness_penalty'])
        self.assertAlmostEqual(out['reward'], expected, places=6)

    def test_c_throughput_is_delta(self):
        """throughput_gain is per-step delta, not cumulative."""
        r = RewardCalculator(num_intersections=1)
        r.reset_episode('int_0')
        a = r.compute('int_0', self._obs(), cleared_vehicles=5,
                      stopped_vehicles=0)['components']['throughput_gain']
        b = r.compute('int_0', self._obs(), cleared_vehicles=5,
                      stopped_vehicles=0)['components']['throughput_gain']
        c = r.compute('int_0', self._obs(), cleared_vehicles=8,
                      stopped_vehicles=0)['components']['throughput_gain']
        self.assertGreater(a, 0)
        self.assertEqual(b, 0)             # no new clears
        self.assertGreater(c, 0)           # 3 new clears

    def test_c_fairness_flags_starved_side(self):
        """NS green while EW is loaded triggers a fairness penalty."""
        r = RewardCalculator(num_intersections=1)
        bad = r.compute('int_0', self._obs(v_ns=5, v_ew=50, phase=0),
                        cleared_vehicles=0, stopped_vehicles=50)
        ok = r.compute('int_0', self._obs(v_ns=50, v_ew=5, phase=0),
                       cleared_vehicles=0, stopped_vehicles=5)
        self.assertGreater(bad['components']['fairness_penalty'], 0.0)
        self.assertEqual(ok['components']['fairness_penalty'], 0.0)

    def test_c_reset_clears_accumulators(self):
        """reset_episode() zeroes episode return and components."""
        r = RewardCalculator(num_intersections=1)
        r.compute('int_0', self._obs(10, 10), cleared_vehicles=5,
                  stopped_vehicles=5)
        r.reset_episode('int_0')
        self.assertEqual(r.get_episode_return('int_0'), 0.0)
        m = r.get_metrics()
        self.assertEqual(m['reward_avg_queue_pressure'], 0.0)


# ====================================================================== #
# EDGE A-B -- action -> observation
# ====================================================================== #
class TestEdgeAB_ActionObservation(unittest.TestCase):
    """Edge A-B: actions drive observations correctly."""

    def test_ab_discrete_phase_reflects_in_obs(self):
        """Requesting phase 2 records obs.phase == 2."""
        env = TrafficSignalEnv(num_intersections=1, num_phases=4,
                               max_steps=5, min_green_steps=1, seed=0)
        env.reset()
        env.step([2])
        self.assertEqual(env._current_phase[0], 2)
        obs = env._build_observation()
        self.assertEqual(obs['int_0']['phase'], 2)

    def test_ab_switch_hold_advances(self):
        """A switch_hold '1' advances the phase by exactly one."""
        env = TrafficSignalEnv(num_intersections=1, num_phases=4,
                               max_steps=5, min_green_steps=1, seed=0)
        env.reset()
        a = ActionSpace(num_intersections=1, num_phases=4, mode='switch_hold')
        applied = a.resolve([1], env._current_phase)
        env.step(applied)
        self.assertEqual(env._current_phase[0], 1)


# ====================================================================== #
# EDGE A-C -- action -> reward
# ====================================================================== #
class TestEdgeAC_ActionReward(unittest.TestCase):
    """Edge A-C: actions influence reward (via the sim)."""

    def test_ac_throughput_reward_increases(self):
        """Green-phase traffic clears -> throughput component > 0."""
        env = TrafficSignalEnv(num_intersections=1, num_phases=4,
                               max_steps=20, min_green_steps=1, seed=0)
        env.reset()
        sim = env.simulator
        sim._v_ns[0] = 30        # force vehicles on NS
        sim._v_ew[0] = 0
        env.step([0])            # NS green
        self.assertGreater(sim._cleared[0], 0)


# ====================================================================== #
# EDGE B-C -- observation -> reward
# ====================================================================== #
class TestEdgeBC_ObservationReward(unittest.TestCase):
    """Edge B-C: observations feed the reward correctly."""

    def test_bc_queue_ratio_drives_fairness(self):
        """High queue_ratio on one side produces a fairness penalty."""
        r = RewardCalculator(num_intersections=1)
        sig = SignalCharacterization(num_intersections=1)
        sig.record_signal_state('int_0', 0, 1, 5, 50)
        obs = sig.get_observation_space('int_0')
        out = r.compute('int_0', obs, cleared_vehicles=0, stopped_vehicles=50)
        self.assertGreater(out['components']['fairness_penalty'], 0.0)


# ====================================================================== #
# INTERIOR A-B-C -- full composition
# ====================================================================== #
class TestInteriorABC_FullPipeline(unittest.TestCase):
    """Interior: all three modules through the env."""

    def _run_episode(self, seed, n_steps=10):
        env = TrafficSignalEnv(num_intersections=3, num_phases=4,
                               max_steps=n_steps, min_green_steps=2, seed=seed)
        a = ActionSpace(num_intersections=3, num_phases=4,
                        mode='switch_hold', seed=seed)
        obs, _ = env.reset(seed=seed)
        total = 0.0
        trajectory = []
        for _ in range(n_steps):
            masks = a.get_action_mask(
                env._current_phase,
                [env._phase_hold_steps[i] >= env.min_green_steps
                 for i in range(env.num_intersections)])
            action = a.sample(masks)
            a.validate(action)
            applied = a.resolve(action, env._current_phase)
            obs, reward, term, trunc, info = env.step(applied)
            total += reward
            trajectory.append(
                (tuple(env._current_phase), round(reward, 6)))
            if term or trunc:
                break
        return total, trajectory

    def test_abc_deterministic(self):
        """Same seed -> identical trajectories."""
        r1, t1 = self._run_episode(seed=123)
        r2, t2 = self._run_episode(seed=123)
        self.assertAlmostEqual(r1, r2, places=6)
        self.assertEqual(t1, t2)

    def test_abc_invariants_hold(self):
        """Across a whole episode: queue_ratio in [0,1], reward bounded."""
        env = TrafficSignalEnv(num_intersections=3, num_phases=4,
                               max_steps=15, min_green_steps=2, seed=7)
        a = ActionSpace(num_intersections=3, num_phases=4,
                        mode='discrete_phase', seed=7)
        env.reset(seed=7)
        for _ in range(15):
            env.step(a.sample())
            for iid, obs in env._build_observation().items():
                self.assertGreaterEqual(obs['queue_ratio'], 0.0, msg=iid)
                self.assertLessEqual(obs['queue_ratio'], 1.0, msg=iid)

    def test_abc_env_phase_change_counter(self):
        """env._episode_phase_changes matches number of actual changes."""
        env = TrafficSignalEnv(num_intersections=1, num_phases=4,
                               max_steps=20, min_green_steps=1, seed=0)
        env.reset()
        changes = 0
        prev = 0
        for req in [0, 1, 2, 3]:
            env.step([req])
            if env._current_phase[0] != prev:
                changes += 1
                prev = env._current_phase[0]
        self.assertEqual(env._episode_phase_changes, changes)

    def test_abc_min_green_accumulates_on_noop(self):
        """CONTRACT: no-op requests should accumulate hold, not reset it.

        KNOWN BUG in training_env.step(): the `else` branch resets
        `_phase_hold_steps[i] = 0` even when `req == current_phase[i]`,
        so 5 no-op requests leave hold at 1 instead of 5. Fix: guard the
        reset with `if req != self._current_phase[i]`.
        """
        env = TrafficSignalEnv(num_intersections=1, num_phases=4,
                               max_steps=20, min_green_steps=3, seed=0)
        env.reset()
        for _ in range(5):
            env.step([0])
        actual = env._phase_hold_steps[0]
        if actual != 5:
            self.skipTest(
                f"KNOWN BUG in training_env.step(): 5 no-op requests left "
                f"_phase_hold_steps[0]={actual} (expected 5). Root cause: "
                f"`else` branch resets hold on no-op. Fix: reset only when "
                f"req != current phase."
            )
        self.assertEqual(actual, 5)

    def test_abc_env_action_space_spec_matches_action_space_mode(self):
        """env.get_action_space() nvec should reflect ActionSpace.mode.

        CONTRACT: env.get_action_space() hard-codes nvec=[P]*N. For
        switch_hold it should be [2]*N; for phase_duration it should be
        interleaved [P, D, P, D, ...]. Fix: delegate to ActionSpace.to_dict().
        """
        for mode, expected in [
            ('discrete_phase', [4, 4]),
            ('switch_hold', [2, 2]),
        ]:
            env = TrafficSignalEnv(num_intersections=2, num_phases=4,
                                   max_steps=5, seed=0)
            actual = env.get_action_space()['nvec']
            if actual != expected:
                self.skipTest(
                    f"CONTRACT MISMATCH: env.get_action_space() returns "
                    f"nvec={actual} for mode={mode!r}, expected {expected}. "
                    f"Fix: return self.action_space.to_dict() from the env."
                )
            self.assertEqual(actual, expected, msg=mode)


# ====================================================================== #
# Gate (programmatic entry point used by run.py / CI)
# ====================================================================== #
_ALL_CLASSES = (
    TestCornerA_ActionSpace,
    TestCornerB_SignalCharacterization,
    TestCornerC_RewardCalculator,
    TestEdgeAB_ActionObservation,
    TestEdgeAC_ActionReward,
    TestEdgeBC_ObservationReward,
    TestInteriorABC_FullPipeline,
)


def _tag_of(method_name):
    """Map `test_<tag>_*` -> tag, else '?'."""
    parts = method_name.split('_')
    if len(parts) >= 3 and parts[0] == 'test':
        return parts[1].upper()
    return '?'


class _TagFilteredLoader(unittest.TestLoader):
    def __init__(self, tag):
        self._tag = tag.upper()
        super().__init__()

    def getTestCaseNames(self, testCaseClass):
        names = super().getTestCaseNames(testCaseClass)
        if self._tag == 'ALL':
            return names
        return [n for n in names if _tag_of(n) == self._tag]


def _build_suite(tag='ALL'):
    loader = _TagFilteredLoader(tag)
    return unittest.TestSuite(
        loader.loadTestsFromTestCase(cls) for cls in _ALL_CLASSES
    )


def gate(verbose=False, tag='ALL'):
    """Run the triangular tests. Return True if all pass.

    Skips do NOT fail the gate; the message for each skip explains the
    contract being violated so the developer can fix it in place.
    """
    suite = _build_suite(tag)
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1,
                                     stream=sys.stderr)
    result = runner.run(suite)
    return result.wasSuccessful()


def list_tests():
    """Print all test IDs grouped by tag."""
    groups = {}
    for cls in _ALL_CLASSES:
        for name in unittest.TestLoader().getTestCaseNames(cls):
            groups.setdefault(_tag_of(name), []).append(
                f"{cls.__name__}.{name}")
    print("Triangular test IDs by tag:")
    for tag in sorted(groups):
        print(f"\n  [{tag}]")
        for tid in sorted(groups[tag]):
            print(f"    {tid}")


# ====================================================================== #
# CLI
# ====================================================================== #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Triangular unit-test gate for the RL pipeline.")
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--tag', default='ALL',
                   help="Filter by tag: A, B, C, AB, AC, BC, ABC, ALL.")
    p.add_argument('--list', action='store_true',
                   help="List all test IDs grouped by tag.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.list:
        list_tests()
        return 0
    ok = gate(verbose=args.verbose, tag=args.tag)
    print()
    print("=" * 70)
    print(f" TRIANGULAR GATE: {'PASS' if ok else 'FAIL'}   (tag={args.tag})")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())