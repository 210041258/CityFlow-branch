import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "cityflow_features"))

from reward_calculator import RewardCalculator


class RewardCalculatorTests(unittest.TestCase):
    def test_defaults(self):
        rc = RewardCalculator(num_intersections=2)
        self.assertEqual(rc.num_intersections, 2)
        self.assertEqual(rc.weights, RewardCalculator.DEFAULT_WEIGHTS)

    def test_weights_override(self):
        rc = RewardCalculator(weights={"queue": 0.9})
        self.assertEqual(rc.weights["queue"], 0.9)
        self.assertEqual(rc.weights["wait"], 0.3)

    def test_compute_returns_structure(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 10, "vehicles_ew": 5, "queue_ratio": 0.5, "phase": 0}
        result = rc.compute("i0", obs, cleared_vehicles=3, stopped_vehicles=2, sim_dt=1.0)
        self.assertIn("reward", result)
        self.assertIn("components", result)
        self.assertEqual(set(result["components"].keys()),
                         {"queue_pressure", "wait_penalty", "throughput_gain", "fairness_penalty"})

    def test_throughput_delta(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        r1 = rc.compute("i0", obs, cleared_vehicles=5)
        r2 = rc.compute("i0", obs, cleared_vehicles=5)
        self.assertEqual(r1["components"]["throughput_gain"], 0.5)
        self.assertEqual(r2["components"]["throughput_gain"], 0.0)

    def test_throughput_negative_delta_clamped(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        rc.compute("i0", obs, cleared_vehicles=10)
        r = rc.compute("i0", obs, cleared_vehicles=3)
        self.assertEqual(r["components"]["throughput_gain"], 0.0)

    def test_reset_episode(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 1, "vehicles_ew": 1, "queue_ratio": 0.5, "phase": 0}
        rc.compute("i0", obs, cleared_vehicles=4)
        self.assertNotEqual(rc.get_episode_return("i0"), 0.0)
        rc.reset_episode("i0")
        self.assertEqual(rc.get_episode_return("i0"), 0.0)

    def test_reset_all(self):
        rc = RewardCalculator(num_intersections=2)
        obs = {"vehicles_ns": 1, "vehicles_ew": 1, "queue_ratio": 0.5, "phase": 0}
        rc.compute("int_0", obs, cleared_vehicles=1)
        rc.compute("int_1", obs, cleared_vehicles=1)
        rc.reset_episode()
        self.assertEqual(rc.get_episode_return("int_0"), 0.0)
        self.assertEqual(rc.get_episode_return("int_1"), 0.0)

    def test_get_metrics_empty(self):
        rc = RewardCalculator()
        m = rc.get_metrics()
        self.assertEqual(m["reward_intersections_tracked"], 0)
        self.assertEqual(m["reward_avg_episode_return"], 0.0)

    def test_fairness_no_penalty_balanced(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 10, "vehicles_ew": 10, "queue_ratio": 0.5, "phase": 0}
        r = rc.compute("i0", obs)
        self.assertEqual(r["components"]["fairness_penalty"], 0.0)

    def test_fairness_penalty_starved(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 1, "vehicles_ew": 10, "queue_ratio": 0.5, "phase": 0}
        r = rc.compute("i0", obs)
        self.assertGreater(r["components"]["fairness_penalty"], 0.0)

    def test_queue_pressure_empty(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        r = rc.compute("i0", obs)
        self.assertEqual(r["components"]["queue_pressure"], 0.0)

    def test_wait_penalty_saturation(self):
        rc = RewardCalculator(wait_penalty_horizon=10)
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        r = rc.compute("i0", obs, stopped_vehicles=100, sim_dt=1.0)
        self.assertEqual(r["components"]["wait_penalty"], 1.0)

    def test_reward_formula_matches_weighted_components(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 50, "vehicles_ew": 50, "queue_ratio": 0.5, "phase": 0}
        r = rc.compute("i0", obs, cleared_vehicles=10, stopped_vehicles=0, sim_dt=1.0)
        # queue_pressure=0.6, wait=0, throughput=1.0, fairness=0
        # reward = 0.4*(-0.6) + 0.3*(-0) + 0.2*(1.0) + 0.1*(-0) = -0.04
        self.assertAlmostEqual(r["reward"], -0.04, places=6)
        self.assertAlmostEqual(r["components"]["queue_pressure"], 0.6, places=6)
        self.assertAlmostEqual(r["components"]["wait_penalty"], 0.0, places=6)
        self.assertAlmostEqual(r["components"]["throughput_gain"], 1.0, places=6)
        self.assertAlmostEqual(r["components"]["fairness_penalty"], 0.0, places=6)

    def test_queue_pressure_saturates_at_one(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 100, "vehicles_ew": 0, "queue_ratio": 1.0, "phase": 0}
        r = rc.compute("i0", obs)
        self.assertAlmostEqual(r["components"]["queue_pressure"], 1.0, places=6)

    def test_queue_pressure_imbalance_only(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 10, "vehicles_ew": 0, "queue_ratio": 1.0, "phase": 0}
        r = rc.compute("i0", obs)
        # load=0.1, imbalance=1.0 -> 0.6*0.1 + 0.4*1.0 = 0.46
        self.assertAlmostEqual(r["components"]["queue_pressure"], 0.46, places=6)

    def test_wait_penalty_linear_before_saturation(self):
        rc = RewardCalculator(wait_penalty_horizon=100)
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        r = rc.compute("i0", obs, stopped_vehicles=25, sim_dt=1.0)
        self.assertAlmostEqual(r["components"]["wait_penalty"], 0.25, places=6)

    def test_wait_penalty_scales_with_sim_dt(self):
        rc = RewardCalculator(wait_penalty_horizon=300)
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        r1 = rc.compute("i0", obs, stopped_vehicles=100, sim_dt=1.0)
        r2 = rc.compute("i0", obs, stopped_vehicles=100, sim_dt=3.0)
        self.assertAlmostEqual(r1["components"]["wait_penalty"], 100 / 300, places=6)
        self.assertAlmostEqual(r2["components"]["wait_penalty"], 1.0, places=6)

    def test_throughput_gain_soft_cap(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        r = rc.compute("i0", obs, cleared_vehicles=25)
        self.assertAlmostEqual(r["components"]["throughput_gain"], 1.0, places=6)

    def test_throughput_independent_across_intersections(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        r0 = rc.compute("int_0", obs, cleared_vehicles=5)
        r1 = rc.compute("int_1", obs, cleared_vehicles=5)
        self.assertAlmostEqual(r0["components"]["throughput_gain"], 0.5, places=6)
        self.assertAlmostEqual(r1["components"]["throughput_gain"], 0.5, places=6)

    def test_fairness_penalty_ns_green_starves_ew(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 0, "vehicles_ew": 50, "queue_ratio": 0.0, "phase": 0}
        r = rc.compute("i0", obs)
        self.assertAlmostEqual(r["components"]["fairness_penalty"], 1.0, places=6)

    def test_fairness_penalty_ew_green_starves_ns(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 50, "vehicles_ew": 0, "queue_ratio": 1.0, "phase": 1}
        r = rc.compute("i0", obs)
        self.assertAlmostEqual(r["components"]["fairness_penalty"], 1.0, places=6)

    def test_fairness_no_penalty_when_served_side_loaded(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 50, "vehicles_ew": 0, "queue_ratio": 1.0, "phase": 0}
        r = rc.compute("i0", obs)
        self.assertEqual(r["components"]["fairness_penalty"], 0.0)

    def test_episode_return_accumulates(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        rc.compute("i0", obs, cleared_vehicles=10)
        rc.compute("i0", obs, cleared_vehicles=20)
        # each step: throughput=1.0 -> reward=0.2; two steps -> 0.4
        self.assertAlmostEqual(rc.get_episode_return("i0"), 0.4, places=6)

    def test_get_metrics_with_data(self):
        rc = RewardCalculator(num_intersections=2)
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        rc.compute("int_0", obs, cleared_vehicles=10)
        rc.compute("int_1", obs, cleared_vehicles=0)
        m = rc.get_metrics()
        self.assertEqual(m["reward_intersections_tracked"], 2)
        self.assertIn("reward_avg_episode_return", m)
        self.assertIn("reward_max_episode_return", m)
        self.assertIn("reward_min_episode_return", m)
        for comp in ("queue_pressure", "wait_penalty", "throughput_gain", "fairness_penalty"):
            self.assertIn(f"reward_avg_{comp}", m)

    def test_default_num_intersections(self):
        rc = RewardCalculator()
        self.assertEqual(rc.num_intersections, 10)

    def test_weights_override_partial(self):
        rc = RewardCalculator(weights={"queue": 0.0})
        self.assertEqual(rc.weights["queue"], 0.0)
        self.assertEqual(rc.weights["wait"], 0.3)
        self.assertEqual(rc.weights["throughput"], 0.2)
        self.assertEqual(rc.weights["fairness"], 0.1)

    def test_observation_missing_keys_use_defaults(self):
        rc = RewardCalculator()
        r = rc.compute("i0", {})
        self.assertEqual(r["components"]["queue_pressure"], 0.0)
        self.assertEqual(r["components"]["fairness_penalty"], 0.0)

    def test_reset_episode_clears_throughput_bookkeeping(self):
        rc = RewardCalculator()
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        rc.compute("i0", obs, cleared_vehicles=10)
        rc.compute("i0", obs, cleared_vehicles=10)
        self.assertEqual(rc.compute("i0", obs, cleared_vehicles=10)["components"]["throughput_gain"], 0.0)
        rc.reset_episode("i0")
        self.assertEqual(rc.compute("i0", obs, cleared_vehicles=10)["components"]["throughput_gain"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
