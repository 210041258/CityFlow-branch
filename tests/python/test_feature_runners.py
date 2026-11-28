import contextlib
import io
from pathlib import Path
import sys
import unittest

from cityflow_features.run import build_parser, run_training
from cityflow_features.training_env import TrafficSignalEnv


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CASE_STUDY_DIRECTORY = REPOSITORY_ROOT / "cases-study"
sys.path.insert(0, str(CASE_STUDY_DIRECTORY))

import scenario_matrix


class TrafficSignalEnvTests(unittest.TestCase):
    def test_minimum_green_delays_phase_change(self):
        env = TrafficSignalEnv(
            num_intersections=1,
            max_steps=5,
            min_green_steps=2,
            seed=1,
        )
        env.reset()

        for _ in range(2):
            _, _, _, _, info = env.step([1])
            self.assertEqual(info["per_intersection"]["int_0"]["phase"], 0)
            self.assertEqual(info["phase_changes"], 0)

        _, _, _, _, info = env.step([1])
        self.assertEqual(info["per_intersection"]["int_0"]["phase"], 1)
        self.assertEqual(info["phase_changes"], 1)

    def test_step_rejects_wrong_action_count(self):
        env = TrafficSignalEnv(num_intersections=2, seed=1)
        env.reset()

        with self.assertRaisesRegex(ValueError, "action length 1"):
            env.step([0])

    def test_episode_truncates_at_step_limit(self):
        env = TrafficSignalEnv(
            num_intersections=1,
            max_steps=2,
            min_green_steps=0,
            seed=1,
        )
        env.reset()

        _, _, terminated, truncated, _ = env.step([0])
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        _, _, terminated, truncated, info = env.step([0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["step"], 2)


class RunnerTests(unittest.TestCase):
    def test_training_runner_returns_episode_summary(self):
        args = build_parser().parse_args([
            "--training-env",
            "--simulator", "mock",
            "--num-intersections", "2",
            "--steps", "2",
            "--episodes", "1",
            "--policy", "hold",
            "--seed", "7",
        ])

        with contextlib.redirect_stdout(io.StringIO()):
            results = run_training(args)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["steps"], 2)
        self.assertEqual(results[0]["seed"], 7)
        self.assertEqual(results[0]["tracked_intersections"], 2)

    def test_scenario_matrix_runs_and_aggregates_small_case(self):
        rows = scenario_matrix.run_scenario_matrix(
            controllers=["fixed-time"],
            scenarios=["S1"],
            naturals=["N-clear"],
            demands=["D1"],
            seeds=(7,),
            n_steps=1,
            progress=False,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["controller"], "fixed-time")
        self.assertEqual(row["n_seeds"], 1)
        self.assertEqual(len(row["travel_time_s_values"]), 1)
        self.assertEqual(row["travel_time_s_sd"], 0.0)
        self.assertEqual(row["travel_time_s_ci95"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
