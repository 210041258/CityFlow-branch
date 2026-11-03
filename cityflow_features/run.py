#!/usr/bin/env python3
"""Runnable entry point for the traffic-signal control pipeline.

The runner supports the project's dependency-free mock simulator and an
optional CityFlow adapter. It includes reproducible policies (random,
max-pressure, fixed-time, and hold), repeatable episodes, a smoke test, and
JSON/CSV summaries. It is an experiment/demo runner, not a deep-RL trainer.

Run from the repository root or this directory, for example:

    python cityflow_features/run.py --smoke-test
    python cityflow_features/run.py --training-env --steps 200 --policy max-pressure
    python cityflow_features/run.py --training-env --episodes 5 --compare --output results.csv
    python cityflow_features/run.py --training-env --simulator cityflow \\
        --cityflow-config examples/3-4/cityflow/config.json --steps 100

CityFlow, when selected, must be installed separately. Its roadnet must have
signalized, non-virtual intersections with compatible phase counts. The
training environment accepts phase indices; `phase_duration` actions are
accepted for compatibility, but their duration component is not currently
consumed by `TrafficSignalEnv`.
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Works both as `python run.py` and as a package module.
    from .action_space import ActionSpace
    from .training_env import TrafficSignalEnv, MockTrafficSimulator
except ImportError:  # pragma: no cover - import route depends on invocation
    from action_space import ActionSpace
    from training_env import TrafficSignalEnv, MockTrafficSimulator


POLICIES = ("random", "max-pressure", "fixed-time", "hold")


def _resolve_project_path(path: str) -> Path:
    """Resolve a user path against cwd, then against this file's directory."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    relative_to_module = Path(__file__).resolve().parent / candidate
    if relative_to_module.is_file():
        return relative_to_module.resolve()
    return candidate.resolve()


def discover_cityflow_intersections(config_path: str) -> Tuple[List[str], Dict[str, Any]]:
    """Read signalized junctions and map metadata from a CityFlow config.

    Returns ``(intersection_ids, roadnet)``. Virtual boundary junctions and
    junctions without usable road links/phases are excluded.
    """
    config_file = _resolve_project_path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"CityFlow config not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    roadnet_name = config.get("roadnetFile")
    if not roadnet_name:
        raise ValueError("CityFlow config must define roadnetFile")
    roadnet_path = Path(roadnet_name)
    if not roadnet_path.is_absolute():
        roadnet_path = (config_file.parent / roadnet_path).resolve()
    if not roadnet_path.is_file():
        raise FileNotFoundError(f"CityFlow roadnet not found: {roadnet_path}")
    with roadnet_path.open("r", encoding="utf-8") as handle:
        roadnet = json.load(handle)

    intersections = roadnet.get("intersections", [])
    ids = []
    for intersection in intersections:
        light = intersection.get("trafficLight") or {}
        phases = light.get("lightphases", [])
        if (
            not intersection.get("virtual", False)
            and intersection.get("roadLinks")
            and len(phases) >= 2
        ):
            ids.append(intersection["id"])
    if not ids:
        raise ValueError(
            f"No non-virtual signalized intersections with road links found in {roadnet_path}"
        )
    return ids, roadnet


class CityFlowTrafficSimulator:
    """Adapt a CityFlow engine to the `TrafficSignalEnv` simulator interface.

    Vehicle counts are aggregated over incoming lanes at each selected
    intersection. Road orientation is inferred from road endpoints in the
    roadnet; if metadata is incomplete, the road is assigned to EW. Since the
    CityFlow API does not expose a universal per-intersection discharge count,
    ``cleared_total`` is an explicitly approximate queue-reduction estimate.
    """

    def __init__(
        self,
        config_path: str,
        intersection_ids: Sequence[str],
        num_phases: int = 4,
        threads: int = 1,
        seed: Optional[int] = None,
    ):
        try:
            import cityflow
        except ImportError as exc:
            raise ImportError(
                "CityFlow is not installed. Install it separately or use --simulator mock."
            ) from exc
        if threads < 1:
            raise ValueError("threads must be at least 1")

        self._cityflow = cityflow
        self.config_path = str(_resolve_project_path(config_path))
        self.threads = threads
        self.seed = seed
        self.intersection_ids = list(intersection_ids)
        self.num_intersections = len(self.intersection_ids)
        self.num_phases = num_phases

        _, self._roadnet = discover_cityflow_intersections(self.config_path)
        self._junctions = {
            item["id"]: item for item in self._roadnet.get("intersections", [])
        }
        self._lane_groups = self._build_lane_groups()
        self._phase_cache = [0] * self.num_intersections
        self._cleared_estimate = [0] * self.num_intersections
        self._previous_queue = [0] * self.num_intersections
        self._lane_cache: Optional[Tuple[Dict[str, int], Dict[str, int]]] = None
        self._engine = None
        self._interval = self._read_interval()
        self._create_engine()
        self._validate_intersections()

    def _read_interval(self) -> float:
        with open(self.config_path, "r", encoding="utf-8") as handle:
            interval = float(json.load(handle).get("interval", 1.0))
        if interval <= 0:
            raise ValueError("CityFlow config interval must be greater than zero")
        return interval

    def _create_engine(self):
        self._engine = self._cityflow.Engine(self.config_path, thread_num=self.threads)
        self._lane_cache = None

    def _build_lane_groups(self) -> List[Dict[str, List[str]]]:
        roads = {road["id"]: road for road in self._roadnet.get("roads", [])}
        points = {
            item["id"]: item.get("point", {})
            for item in self._roadnet.get("intersections", [])
        }
        groups = []
        for intersection_id in self.intersection_ids:
            junction = self._junctions.get(intersection_id)
            if junction is None:
                raise ValueError(f"Intersection {intersection_id!r} is absent from roadnet")
            ns_lanes: List[str] = []
            ew_lanes: List[str] = []
            incoming_roads = {
                link.get("startRoad")
                for link in junction.get("roadLinks", [])
                if link.get("startRoad")
            }
            for road_id in incoming_roads:
                road = roads.get(road_id)
                if road is None:
                    continue
                start = points.get(road.get("startIntersection"), {})
                end = points.get(road.get("endIntersection"), {})
                try:
                    dx = float(end["x"]) - float(start["x"])
                    dy = float(end["y"]) - float(start["y"])
                    target = ns_lanes if abs(dy) >= abs(dx) else ew_lanes
                except (KeyError, TypeError, ValueError):
                    target = ew_lanes
                lane_count = len(road.get("lanes", []))
                target.extend(f"{road_id}_{lane_index}" for lane_index in range(lane_count))
            groups.append({"ns": ns_lanes, "ew": ew_lanes})
        return groups

    def _validate_intersections(self):
        for intersection_id in self.intersection_ids:
            junction = self._junctions[intersection_id]
            phase_count = len((junction.get("trafficLight") or {}).get("lightphases", []))
            if phase_count < self.num_phases:
                raise ValueError(
                    f"Intersection {intersection_id!r} has {phase_count} phases; "
                    f"--num-phases={self.num_phases} is incompatible"
                )

    def _lane_counts(self):
        if self._lane_cache is None:
            self._lane_cache = (
                self._engine.get_lane_vehicle_count(),
                self._engine.get_lane_waiting_vehicle_count(),
            )
        return self._lane_cache

    def _read_intersection_counts(self, index: int) -> Tuple[int, int, int]:
        lane_counts, waiting_counts = self._lane_counts()
        groups = self._lane_groups[index]
        ns = sum(int(lane_counts.get(lane_id, 0)) for lane_id in groups["ns"])
        ew = sum(int(lane_counts.get(lane_id, 0)) for lane_id in groups["ew"])
        stopped = sum(
            int(waiting_counts.get(lane_id, 0))
            for lane_id in groups["ns"] + groups["ew"]
        )
        return ns, ew, stopped

    def reset(self):
        # Recreate the engine instead of depending on CityFlow-version-specific
        # reset() behavior. This also guarantees a fresh episode state.
        self._create_engine()
        self._phase_cache = [0] * self.num_intersections
        self._cleared_estimate = [0] * self.num_intersections
        self._previous_queue = [0] * self.num_intersections

    def step(self, actions: Iterable[int], dt: float = 1.0):
        phases = [int(action) for action in actions]
        if len(phases) != self.num_intersections:
            raise ValueError(
                f"Expected {self.num_intersections} phases, got {len(phases)}"
            )
        for index, (intersection_id, phase) in enumerate(zip(self.intersection_ids, phases)):
            if phase < 0 or phase >= self.num_phases:
                raise ValueError(f"Invalid phase {phase} for {intersection_id}")
            # CityFlow's public method takes one junction and one phase per call.
            self._engine.set_tl_phase(intersection_id, phase)
            self._phase_cache[index] = phase

        ticks = max(1, int(round(float(dt) / self._interval)))
        self._lane_cache = None
        for _ in range(ticks):
            self._engine.next_step()
            self._lane_cache = None

        # Estimate discharge from reductions on the currently served approach.
        # This is not a detector-grade throughput measurement.
        for index, phase in enumerate(phases):
            ns, ew, _ = self._read_intersection_counts(index)
            current_queue = ns + ew
            previous_queue = self._previous_queue[index]
            if phase % 2 == 0:
                reduction = max(0, self._previous_queue[index] - current_queue)
            else:
                reduction = max(0, self._previous_queue[index] - current_queue)
            self._cleared_estimate[index] += reduction
            self._previous_queue[index] = current_queue

    def get_intersection_state(self, intersection_id: str) -> Dict[str, int]:
        try:
            index = self.intersection_ids.index(intersection_id)
        except ValueError as exc:
            raise KeyError(f"Unknown intersection: {intersection_id}") from exc
        ns, ew, stopped = self._read_intersection_counts(index)
        return {
            "vehicles_ns": ns,
            "vehicles_ew": ew,
            "cleared_total": self._cleared_estimate[index],
            "stopped": stopped,
            "phase_index": self._phase_cache[index],
        }

    def close(self):
        # CityFlow's Python wrapper does not consistently expose close().
        self._engine = None
        self._lane_cache = None


def _intersection_ids_from_args(args) -> List[str]:
    discovered, _ = discover_cityflow_intersections(args.cityflow_config)
    if args.intersection_ids:
        requested = [part.strip() for part in args.intersection_ids.split(",") if part.strip()]
        missing = [item for item in requested if item not in discovered]
        if missing:
            raise ValueError(f"Unknown or non-controllable intersection IDs: {missing}")
        return requested
    if args.num_intersections > len(discovered):
        raise ValueError(
            f"Requested {args.num_intersections} intersections, but only "
            f"{len(discovered)} controllable intersections were found"
        )
    return discovered[: args.num_intersections]


def build_simulator(args):
    """Create the selected simulator and return it with its intersection count."""
    if args.simulator == "mock":
        simulator = MockTrafficSimulator(
            num_intersections=args.num_intersections,
            num_phases=args.num_phases,
            arrival_rate=args.arrival_rate,
            departure_rate=args.departure_rate,
            seed=args.seed,
        )
        return simulator, args.num_intersections

    ids = _intersection_ids_from_args(args)
    simulator = CityFlowTrafficSimulator(
        config_path=args.cityflow_config,
        intersection_ids=ids,
        num_phases=args.num_phases,
        threads=args.threads,
        seed=args.seed,
    )
    return simulator, len(ids)


def _allowed_masks(action_space: ActionSpace, env: TrafficSignalEnv) -> List[List[int]]:
    phases = list(env._current_phase)
    green_satisfied = [
        env._phase_hold_steps[index] >= env.min_green_steps
        for index in range(env.num_intersections)
    ]
    return action_space.get_action_mask(phases, green_satisfied)


def choose_action(
    policy: str,
    action_space: ActionSpace,
    env: TrafficSignalEnv,
    observation: Dict[str, Dict[str, Any]],
    step: int,
    rng: random.Random,
) -> List[int]:
    """Choose a flat action for the project's three action encodings."""
    masks = _allowed_masks(action_space, env)
    if policy == "random":
        return action_space.sample(masks)

    desired_phases = []
    for index in range(env.num_intersections):
        current = int(env._current_phase[index])
        if policy == "hold":
            desired = current
        elif policy == "fixed-time":
            desired = (step // max(1, env.min_green_steps * 2)) % env.num_phases
        else:  # max-pressure: choose the phase serving the longer queue.
            state = observation.get(f"int_{index}", {})
            ns = float(state.get("vehicles_ns", 0))
            ew = float(state.get("vehicles_ew", 0))
            desired = current
            if ns > ew:
                desired = 0 if current % 2 else current
            elif ew > ns:
                desired = 1 if current % 2 == 0 else current
            if desired >= env.num_phases:
                desired %= env.num_phases
        desired_phases.append(desired)

    if action_space.mode == "switch_hold":
        actions = []
        for index, desired in enumerate(desired_phases):
            current = int(env._current_phase[index])
            allowed = masks[index]
            next_phase = (current + 1) % env.num_phases
            should_switch = desired != current and next_phase % 2 == desired % 2
            actions.append(1 if should_switch and 1 in allowed else 0)
        return actions

    if action_space.mode == "phase_duration":
        actions = []
        durations = action_space._duration_grid
        for index, desired in enumerate(desired_phases):
            allowed = masks[index]
            phase = desired if desired in allowed else int(env._current_phase[index])
            actions.extend([phase, rng.choice(durations)])
        return actions

    return [
        desired if desired in masks[index] else int(env._current_phase[index])
        for index, desired in enumerate(desired_phases)
    ]


def _episode_summary(
    policy: str,
    episode: int,
    seed: int,
    return_value: float,
    steps: int,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    reward_metrics = info.get("reward_metrics", {})
    signal_metrics = info.get("signal_metrics", {})
    return {
        "policy": policy,
        "episode": episode,
        "seed": seed,
        "steps": steps,
        "return": return_value,
        "phase_changes": info.get("phase_changes", 0),
        "avg_queue_pressure": reward_metrics.get("reward_avg_queue_pressure", 0.0),
        "avg_wait_penalty": reward_metrics.get("reward_avg_wait_penalty", 0.0),
        "avg_throughput_gain": reward_metrics.get("reward_avg_throughput_gain", 0.0),
        "tracked_intersections": signal_metrics.get("signal_intersections_tracked", 0),
        "reward_metrics": reward_metrics,
        "signal_metrics": signal_metrics,
    }


def run_training(args, policy_override: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run the configured policy for one or more independent episodes."""
    policies = [policy_override or args.policy]
    if args.compare:
        policies = ["random", "max-pressure", "fixed-time", "hold"]

    results: List[Dict[str, Any]] = []
    for policy_index, policy in enumerate(policies):
        for episode in range(args.episodes):
            episode_seed = args.seed + policy_index * 100_000 + episode
            # Re-seed per episode to make repeats independent and repeatable.
            args_for_episode = argparse.Namespace(**vars(args))
            args_for_episode.seed = episode_seed
            simulator, intersection_count = build_simulator(args_for_episode)

            action_space = ActionSpace(
                num_intersections=intersection_count,
                num_phases=args.num_phases,
                mode=args.mode,
                max_duration_steps=args.max_duration_steps,
                duration_step=args.duration_step,
                seed=episode_seed,
            )
            env = TrafficSignalEnv(
                num_intersections=intersection_count,
                num_phases=args.num_phases,
                simulator=simulator,
                max_steps=args.steps,
                sim_dt=args.sim_dt,
                min_green_steps=args.min_green_steps,
                seed=episode_seed,
            )
            observation, info = env.reset(seed=episode_seed)
            rng = random.Random(episode_seed + 1)
            episode_return = 0.0
            completed_steps = 0

            try:
                done = False
                while not done:
                    action = choose_action(
                        policy, action_space, env, observation,
                        completed_steps, rng,
                    )
                    action_space.validate(action)
                    applied = action_space.resolve(action, env._current_phase)
                    observation, reward, terminated, truncated, info = env.step(applied)
                    episode_return += float(reward)
                    completed_steps += 1
                    if args.render_every > 0 and completed_steps % args.render_every == 0:
                        env.render()
                    done = terminated or truncated
            finally:
                env.close()
                if hasattr(simulator, "close"):
                    simulator.close()

            result = _episode_summary(
                policy, episode, episode_seed, episode_return,
                completed_steps, info,
            )
            results.append(result)
            print(
                f"policy={policy:12s} episode={episode + 1:>3}/{args.episodes} "
                f"steps={completed_steps:>5} return={episode_return:>10.3f} "
                f"phase_changes={result['phase_changes']}"
            )

    return results


def run_smoke_test():
    """Exercise every action encoding and policy against the mock simulator."""
    print("Running traffic-signal pipeline smoke tests...")
    for mode in ActionSpace.VALID_MODES:
        for policy in POLICIES:
            args = argparse.Namespace(
                simulator="mock", num_intersections=3, num_phases=4,
                arrival_rate=0.4, departure_rate=0.9, seed=7,
                cityflow_config=None, intersection_ids=None, threads=1,
                mode=mode, max_duration_steps=20, duration_step=5,
                steps=5, sim_dt=1.0, min_green_steps=2, episodes=1,
                compare=False, policy=policy, render_every=0,
            )
            run_training(args)
    print("Smoke test passed (12 mode/policy combinations).")


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Write summaries to JSON or a flattened CSV file."""
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".csv":
        columns = [
            "policy", "episode", "seed", "steps", "return", "phase_changes",
            "avg_queue_pressure", "avg_wait_penalty", "avg_throughput_gain",
            "tracked_intersections",
        ]
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for result in results:
                writer.writerow({key: result.get(key) for key in columns})
    else:
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")
    print(f"Results written to {destination.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run traffic-signal control experiments with mock or CityFlow traffic."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--training-env", action="store_true", help="Run episodes.")
    modes.add_argument("--smoke-test", action="store_true", help="Test all policies and action modes.")

    parser.add_argument("--simulator", choices=("mock", "cityflow"), default="mock")
    parser.add_argument("--cityflow-config", default="config.json", help="CityFlow config JSON path.")
    parser.add_argument("--intersection-ids", help="Comma-separated controllable CityFlow junction IDs.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--num-intersections", type=int, default=5)
    parser.add_argument("--num-phases", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--sim-dt", type=float, default=1.0)
    parser.add_argument("--min-green-steps", type=int, default=3)
    parser.add_argument("--mode", choices=ActionSpace.VALID_MODES, default="switch_hold")
    parser.add_argument("--policy", choices=POLICIES, default="random")
    parser.add_argument("--compare", action="store_true", help="Run all four policies.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=0.4, help="Mock arrivals factor.")
    parser.add_argument("--departure-rate", type=float, default=0.9, help="Mock discharge factor.")
    parser.add_argument("--max-duration-steps", type=int, default=60)
    parser.add_argument("--duration-step", type=int, default=5)
    parser.add_argument("--render-every", type=int, default=0, help="Render every N steps; 0 disables.")
    parser.add_argument("--output", help="Optional results path (.json or .csv).")
    return parser


def validate_args(parser: argparse.ArgumentParser, args):
    if args.num_intersections < 1:
        parser.error("--num-intersections must be at least 1")
    if args.num_phases < 2:
        parser.error("--num-phases must be at least 2")
    if args.steps < 1 or args.episodes < 1:
        parser.error("--steps and --episodes must be at least 1")
    if args.sim_dt <= 0:
        parser.error("--sim-dt must be greater than zero")
    if args.min_green_steps < 0:
        parser.error("--min-green-steps cannot be negative")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.render_every < 0:
        parser.error("--render-every cannot be negative")
    if args.arrival_rate < 0 or args.departure_rate < 0:
        parser.error("Mock arrival/departure rates cannot be negative")
    if args.mode == "phase_duration":
        print(
            "Note: phase-duration actions are parsed, but TrafficSignalEnv currently "
            "uses only their phase component; durations are not scheduled."
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    if args.smoke_test:
        run_smoke_test()
        return 0
    if not args.training_env:
        parser.print_help()
        print("\nHint: add --training-env or --smoke-test.")
        return 0

    try:
        results = run_training(args)
        if args.output:
            save_results(results, args.output)
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
