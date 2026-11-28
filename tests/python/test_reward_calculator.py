"""Tests for RewardCalculator.

Covers initialization, component calculations, episode tracking, metrics,
and edge cases. Uses pytest with parameterized tests to reduce duplication.
"""

import pytest

from cityflow_features.reward_calculator import RewardCalculator


class TestRewardCalculatorInitialization:
    """Tests for RewardCalculator construction and default values."""

    def test_defaults(self):
        """Default num_intersections=10 and weights=DEFAULT_WEIGHTS."""
        rc = RewardCalculator(num_intersections=2)
        assert rc.num_intersections == 2
        assert rc.weights == RewardCalculator.DEFAULT_WEIGHTS

    def test_default_num_intersections(self):
        """Default num_intersections is 10."""
        rc = RewardCalculator()
        assert rc.num_intersections == 10

    def test_weights_override_full(self):
        """All weights can be overridden."""
        rc = RewardCalculator(weights={"queue": 0.9, "wait": 0.1})
        assert rc.weights["queue"] == 0.9
        assert rc.weights["wait"] == 0.1
        assert rc.weights["throughput"] == RewardCalculator.DEFAULT_WEIGHTS["throughput"]
        assert rc.weights["fairness"] == RewardCalculator.DEFAULT_WEIGHTS["fairness"]

    def test_weights_override_partial(self):
        """Partial weight override preserves other defaults."""
        rc = RewardCalculator(weights={"queue": 0.0})
        assert rc.weights["queue"] == 0.0
        assert rc.weights["wait"] == RewardCalculator.DEFAULT_WEIGHTS["wait"]
        assert rc.weights["throughput"] == RewardCalculator.DEFAULT_WEIGHTS["throughput"]
        assert rc.weights["fairness"] == RewardCalculator.DEFAULT_WEIGHTS["fairness"]

    def test_custom_max_queue_and_wait_horizon(self):
        """Custom normalization constants are stored."""
        rc = RewardCalculator(max_queue_per_lane=100, wait_penalty_horizon=600)
        assert rc.max_queue_per_lane == 100
        assert rc.wait_penalty_horizon == 600

    def test_components_constant_is_stable_order(self):
        """COMPONENTS lists the four breakdown names in a fixed order."""
        assert RewardCalculator.COMPONENTS == (
            "queue_pressure",
            "wait_penalty",
            "throughput_gain",
            "fairness_penalty",
        )

    def test_components_constant_matches_breakdown_keys(self):
        """COMPONENTS exactly matches the keys returned by compute()."""
        rc = RewardCalculator()
        obs = {"vehicles_ns": 10, "vehicles_ew": 5, "queue_ratio": 0.5, "phase": 0}
        result = rc.compute("i0", obs)
        assert set(result["components"].keys()) == set(RewardCalculator.COMPONENTS)


class TestRewardCalculatorComputeStructure:
    """Tests for compute() return structure and basic behavior."""

    def setup_method(self):
        """Create a fresh calculator and common observation."""
        self.rc = RewardCalculator()
        self.obs = {"vehicles_ns": 10, "vehicles_ew": 5, "queue_ratio": 0.5, "phase": 0}

    def test_compute_returns_structure(self):
        """Result dict contains reward and all four components."""
        result = self.rc.compute("i0", self.obs, cleared_vehicles=3, stopped_vehicles=2, sim_dt=1.0)
        assert "reward" in result
        assert "components" in result
        assert set(result["components"].keys()) == {
            "queue_pressure", "wait_penalty", "throughput_gain", "fairness_penalty"
        }

    def test_observation_missing_keys_use_defaults(self):
        """Missing observation keys default to zero without error."""
        result = self.rc.compute("i0", {})
        assert result["components"]["queue_pressure"] == 0.0
        assert result["components"]["fairness_penalty"] == 0.0


class TestQueuePressureComponent:
    """Tests for the _queue_pressure internal component."""

    def setup_method(self):
        self.rc = RewardCalculator()

    def test_empty_intersection(self):
        """Zero vehicles gives zero queue pressure."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        result = self.rc.compute("i0", obs)
        assert result["components"]["queue_pressure"] == 0.0

    def test_saturates_at_one(self):
        """High load with imbalance saturates at 1.0."""
        obs = {"vehicles_ns": 100, "vehicles_ew": 0, "queue_ratio": 1.0, "phase": 0}
        result = self.rc.compute("i0", obs)
        assert result["components"]["queue_pressure"] == pytest.approx(1.0, abs=1e-6)

    def test_imbalance_only(self):
        """Load=0.1, imbalance=1.0 -> 0.6*0.1 + 0.4*1.0 = 0.46."""
        obs = {"vehicles_ns": 10, "vehicles_ew": 0, "queue_ratio": 1.0, "phase": 0}
        result = self.rc.compute("i0", obs)
        expected = 0.6 * (10 / 100) + 0.4 * 1.0  # 0.46
        assert result["components"]["queue_pressure"] == pytest.approx(expected, abs=1e-6)

    def test_balanced_load(self):
        """Balanced queue_ratio=0.5 gives zero imbalance contribution."""
        obs = {"vehicles_ns": 50, "vehicles_ew": 50, "queue_ratio": 0.5, "phase": 0}
        result = self.rc.compute("i0", obs)
        # load = 100/100 = 1.0, imbalance = 0 -> 0.6*1.0 = 0.6
        assert result["components"]["queue_pressure"] == pytest.approx(0.6, abs=1e-6)


class TestWaitPenaltyComponent:
    """Tests for the _wait_penalty internal component."""

    def setup_method(self):
        self.rc = RewardCalculator(wait_penalty_horizon=300)

    def test_linear_before_saturation(self):
        """Wait penalty grows linearly with stopped vehicles."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        result = self.rc.compute("i0", obs, stopped_vehicles=25, sim_dt=1.0)
        assert result["components"]["wait_penalty"] == pytest.approx(25 / 300, abs=1e-6)

    def test_saturation(self):
        """Wait penalty clamps at 1.0."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        result = self.rc.compute("i0", obs, stopped_vehicles=100, sim_dt=1.0)
        # 100 * 1.0 / 300 = 0.333..., not saturated
        assert result["components"]["wait_penalty"] == pytest.approx(100 / 300, abs=1e-6)

    def test_saturation_with_large_stopped(self):
        """Wait penalty clamps at 1.0 with enough stopped vehicles."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        result = self.rc.compute("i0", obs, stopped_vehicles=300, sim_dt=1.0)
        assert result["components"]["wait_penalty"] == pytest.approx(1.0, abs=1e-6)

    def test_scales_with_sim_dt(self):
        """Wait penalty scales proportionally with sim_dt."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        r1 = self.rc.compute("i0", obs, stopped_vehicles=100, sim_dt=1.0)
        r2 = self.rc.compute("i0", obs, stopped_vehicles=100, sim_dt=3.0)
        # horizon=300: 100*1/300=0.333, 100*3/300=1.0 (saturated)
        assert r1["components"]["wait_penalty"] == pytest.approx(100 / 300, abs=1e-6)
        assert r2["components"]["wait_penalty"] == pytest.approx(1.0, abs=1e-6)


class TestThroughputGainComponent:
    """Tests for the _throughput_gain internal component."""

    def setup_method(self):
        self.rc = RewardCalculator()

    def test_delta_positive(self):
        """First call with cleared=5 gives 0.5 (soft cap at 10)."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        result = self.rc.compute("i0", obs, cleared_vehicles=5)
        assert result["components"]["throughput_gain"] == pytest.approx(0.5, abs=1e-6)

    def test_delta_zero_on_repeat(self):
        """Repeated same cleared count gives zero gain."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        self.rc.compute("i0", obs, cleared_vehicles=5)
        result = self.rc.compute("i0", obs, cleared_vehicles=5)
        assert result["components"]["throughput_gain"] == pytest.approx(0.0, abs=1e-6)

    def test_negative_delta_clamped(self):
        """Decrease in cleared vehicles clamps to zero gain."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        self.rc.compute("i0", obs, cleared_vehicles=10)
        result = self.rc.compute("i0", obs, cleared_vehicles=3)
        assert result["components"]["throughput_gain"] == pytest.approx(0.0, abs=1e-6)

    def test_soft_cap(self):
        """Gain caps at 1.0 for large deltas (soft cap at 10 vehicles)."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        result = self.rc.compute("i0", obs, cleared_vehicles=25)
        assert result["components"]["throughput_gain"] == pytest.approx(1.0, abs=1e-6)

    def test_independent_across_intersections(self):
        """Throughput bookkeeping is per-intersection."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        r0 = self.rc.compute("int_0", obs, cleared_vehicles=5)
        r1 = self.rc.compute("int_1", obs, cleared_vehicles=5)
        assert r0["components"]["throughput_gain"] == pytest.approx(0.5, abs=1e-6)
        assert r1["components"]["throughput_gain"] == pytest.approx(0.5, abs=1e-6)


class TestFairnessPenaltyComponent:
    """Tests for the _fairness_penalty internal component."""

    def setup_method(self):
        self.rc = RewardCalculator()

    @pytest.mark.parametrize("v_ns,v_ew,phase,expected", [
        (10, 10, 0, 0.0),           # balanced, NS green
        (10, 10, 1, 0.0),           # balanced, EW green
        (1, 10, 0, 0.818181),       # NS green starves EW: (10-1)/11
        (10, 1, 1, 0.818181),       # EW green starves NS: (10-1)/11
        (0, 50, 0, 1.0),            # NS green, EW fully starved
        (50, 0, 1, 1.0),            # EW green, NS fully starved
        (50, 0, 0, 0.0),            # NS green serves loaded NS -> no penalty
        (0, 50, 1, 0.0),            # EW green serves loaded EW -> no penalty
    ])
    def test_fairness_scenarios(self, v_ns, v_ew, phase, expected):
        """Fairness penalty matches expected for various scenarios."""
        obs = {"vehicles_ns": v_ns, "vehicles_ew": v_ew, "queue_ratio": 0.5, "phase": phase}
        result = self.rc.compute("i0", obs)
        assert result["components"]["fairness_penalty"] == pytest.approx(expected, abs=1e-6)

    def test_empty_intersection_no_penalty(self):
        """Zero vehicles on both approaches gives zero fairness penalty."""
        obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}
        result = self.rc.compute("i0", obs)
        assert result["components"]["fairness_penalty"] == 0.0


class TestRewardFormula:
    """Tests that the total reward matches the weighted component formula."""

    def setup_method(self):
        self.rc = RewardCalculator()
        self.obs = {"vehicles_ns": 50, "vehicles_ew": 50, "queue_ratio": 0.5, "phase": 0}

    def test_reward_matches_weighted_components(self):
        """Reward = sum(weight * signed_component) per the documented formula."""
        result = self.rc.compute("i0", self.obs, cleared_vehicles=10, stopped_vehicles=0, sim_dt=1.0)
        comp = result["components"]
        w = self.rc.weights
        expected = (
            w["queue"] * (-comp["queue_pressure"])
            + w["wait"] * (-comp["wait_penalty"])
            + w["throughput"] * comp["throughput_gain"]
            + w["fairness"] * (-comp["fairness_penalty"])
        )
        assert result["reward"] == pytest.approx(expected, abs=1e-6)

    def test_explicit_component_values(self):
        """Known input yields expected component values (regression guard)."""
        result = self.rc.compute("i0", self.obs, cleared_vehicles=10, stopped_vehicles=0, sim_dt=1.0)
        comp = result["components"]
        assert comp["queue_pressure"] == pytest.approx(0.6, abs=1e-6)
        assert comp["wait_penalty"] == pytest.approx(0.0, abs=1e-6)
        assert comp["throughput_gain"] == pytest.approx(1.0, abs=1e-6)
        assert comp["fairness_penalty"] == pytest.approx(0.0, abs=1e-6)


class TestEpisodeTracking:
    """Tests for episode return accumulation and reset."""

    def setup_method(self):
        self.rc = RewardCalculator()
        self.obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}

    def test_episode_return_accumulates(self):
        """Episode return sums rewards across steps."""
        self.rc.compute("i0", self.obs, cleared_vehicles=10)
        self.rc.compute("i0", self.obs, cleared_vehicles=20)
        # Each step: throughput=1.0 -> reward = 0.2 * 1.0 = 0.2; two steps = 0.4
        assert self.rc.get_episode_return("i0") == pytest.approx(0.4, abs=1e-6)

    def test_reset_episode_single(self):
        """reset_episode(intersection_id) clears return and throughput bookkeeping."""
        self.rc.compute("i0", self.obs, cleared_vehicles=4)
        assert self.rc.get_episode_return("i0") != 0.0
        self.rc.reset_episode("i0")
        assert self.rc.get_episode_return("i0") == 0.0
        # Throughput bookkeeping also cleared
        result = self.rc.compute("i0", self.obs, cleared_vehicles=10)
        assert result["components"]["throughput_gain"] == pytest.approx(1.0, abs=1e-6)

    def test_reset_all(self):
        """reset_episode() with no args clears all tracked intersections."""
        self.rc = RewardCalculator(num_intersections=2)
        self.rc.compute("int_0", self.obs, cleared_vehicles=1)
        self.rc.compute("int_1", self.obs, cleared_vehicles=1)
        self.rc.reset_episode()
        assert self.rc.get_episode_return("int_0") == 0.0
        assert self.rc.get_episode_return("int_1") == 0.0

    def test_reset_clears_throughput_bookkeeping(self):
        """reset_episode clears _last_cleared_seen so next call gives full gain."""
        self.rc.compute("i0", self.obs, cleared_vehicles=10)
        self.rc.compute("i0", self.obs, cleared_vehicles=10)
        assert self.rc.compute("i0", self.obs, cleared_vehicles=10)["components"]["throughput_gain"] == 0.0
        self.rc.reset_episode("i0")
        assert self.rc.compute("i0", self.obs, cleared_vehicles=10)["components"]["throughput_gain"] == 1.0


class TestMetrics:
    """Tests for get_metrics() aggregate statistics."""

    def setup_method(self):
        self.rc = RewardCalculator(num_intersections=2)
        self.obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}

    def test_empty_metrics(self):
        """Empty calculator returns zeros for all metrics."""
        m = self.rc.get_metrics()
        assert m["reward_intersections_tracked"] == 0
        assert m["reward_avg_episode_return"] == 0.0

    def test_metrics_with_data(self):
        """Metrics reflect tracked intersections and component averages."""
        self.rc.compute("int_0", self.obs, cleared_vehicles=10)
        self.rc.compute("int_1", self.obs, cleared_vehicles=0)
        m = self.rc.get_metrics()
        assert m["reward_intersections_tracked"] == 2
        assert "reward_avg_episode_return" in m
        assert "reward_max_episode_return" in m
        assert "reward_min_episode_return" in m
        for comp in ("queue_pressure", "wait_penalty", "throughput_gain", "fairness_penalty"):
            assert f"reward_avg_{comp}" in m


class TestEdgeCases:
    """Tests for invalid/edge case inputs."""

    def setup_method(self):
        self.rc = RewardCalculator()
        self.obs = {"vehicles_ns": 0, "vehicles_ew": 0, "queue_ratio": 0.5, "phase": 0}

    def test_negative_cleared_vehicles(self):
        """Negative cleared_vehicles treated as zero delta (no crash)."""
        self.rc.compute("i0", self.obs, cleared_vehicles=10)
        result = self.rc.compute("i0", self.obs, cleared_vehicles=-5)
        assert result["components"]["throughput_gain"] == 0.0

    def test_negative_stopped_vehicles(self):
        """Negative stopped_vehicles produces negative wait penalty (no clamping in impl)."""
        result = self.rc.compute("i0", self.obs, stopped_vehicles=-10, sim_dt=1.0)
        # effective_wait = -10 * 1.0 = -10, penalty = -10/300 = -0.0333...
        assert result["components"]["wait_penalty"] == pytest.approx(-10 / 300, abs=1e-6)

    def test_negative_sim_dt(self):
        """Negative sim_dt produces negative wait penalty (no clamping in impl)."""
        result = self.rc.compute("i0", self.obs, stopped_vehicles=10, sim_dt=-1.0)
        # effective_wait = 10 * -1.0 = -10, penalty = -10/300 = -0.0333...
        assert result["components"]["wait_penalty"] == pytest.approx(-10 / 300, abs=1e-6)

    def test_zero_wait_horizon_raises(self):
        """wait_penalty_horizon=0 raises ZeroDivisionError on compute."""
        rc = RewardCalculator(wait_penalty_horizon=0)
        with pytest.raises(ZeroDivisionError):
            rc.compute("i0", self.obs, stopped_vehicles=10, sim_dt=1.0)

    def test_zero_max_queue_raises(self):
        """max_queue_per_lane=0 raises ZeroDivisionError on compute."""
        rc = RewardCalculator(max_queue_per_lane=0)
        with pytest.raises(ZeroDivisionError):
            rc.compute("i0", {"vehicles_ns": 10, "vehicles_ew": 0, "queue_ratio": 1.0, "phase": 0})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])