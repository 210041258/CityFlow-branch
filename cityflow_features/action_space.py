#!/usr/bin/env python3
"""
Action Space Module
===================
Defines the action representation for the multi-intersection traffic-signal
RL agent. Sits alongside the other pipeline modules:

    signal_characterization.py  ->  observation space
    reward_calculator.py        ->  scalar reward
    action_space.py             ->  action space   (this file)
    training_env.py             ->  gym-style env wiring them together

Used by: run.py --training-env, RL wrappers, and any policy that needs to
know the action contract.

Design notes
------------
* stdlib only -- no hard dependency on gymnasium. Consumers that need a
  real gymnasium space can call `to_multidiscrete_nvec()` and wrap it.
* Three action modes are supported so experiments don't require changes to
  training_env.py's step() signature (which always receives a list of ints):

    - 'discrete_phase'  : action[i] in [0, num_phases)
                          -> phase index to apply at intersection i.
                          (default; matches training_env.py today)

    - 'switch_hold'     : action[i] in {0, 1}
                          -> 0 = hold current phase, 1 = advance to next.
                          The env/simulator maintains the per-intersection
                          phase; this module provides `resolve()` to turn
                          the switch/hold action into a concrete phase.

    - 'phase_duration'  : action is a flat list of length 2*N:
                          [phase_0, dur_0, phase_1, dur_1, ...]
                          -> phase index + duration (in steps) per intersection.
                          Useful for fixed-time / semi-actuated baselines.

* Action masking: `get_action_mask(current_phases, min_green_satisfied)` returns
  a per-intersection list of allowed action indices. The env uses this to
  enforce min-green (in 'switch_hold' mode, only hold is allowed until
  min-green is satisfied) and to prevent no-op "switch" actions.

* `resolve()` turns a flat action into the concrete phase list the simulator
  should apply, given the current per-intersection phases. This keeps the
  env's step() simple and makes the mode switch a one-liner.

Action conventions (matching signal_characterization.py / training_env.py):
    phase 0, 2 -> NS green
    phase 1, 3 -> EW green
"""

import random


# ---------------------------------------------------------------------- #
# Action space
# ---------------------------------------------------------------------- #
class ActionSpace:
    """Discrete action space for N signalized intersections.

    Args:
        num_intersections: number of intersections (N).
        num_phases: number of discrete phases per intersection (P). Ignored
            for 'switch_hold' (which is always 2 actions per intersection).
        mode: one of {'discrete_phase', 'switch_hold', 'phase_duration'}.
        max_duration_steps: upper bound for the duration component when
            mode == 'phase_duration'. Must be >= 1.
        duration_step: granularity (in sim steps) of the duration component.
            Defaults to 1 (every integer step allowed).
        seed: optional RNG seed for `sample()`.
    """

    VALID_MODES = ('discrete_phase', 'switch_hold', 'phase_duration')

    def __init__(self, num_intersections=10, num_phases=4,
                 mode='discrete_phase', max_duration_steps=60,
                 duration_step=1, seed=None):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of {self.VALID_MODES}"
            )
        if num_intersections < 1:
            raise ValueError("num_intersections must be >= 1")
        if mode != 'switch_hold' and num_phases < 2:
            raise ValueError("num_phases must be >= 2")
        if max_duration_steps < 1:
            raise ValueError("max_duration_steps must be >= 1")
        if duration_step < 1:
            raise ValueError("duration_step must be >= 1")

        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.mode = mode
        self.max_duration_steps = max_duration_steps
        self.duration_step = duration_step
        self._rng = random.Random(seed)

        # Precompute the discrete duration grid for 'phase_duration'.
        self._duration_grid = list(
            range(duration_step, max_duration_steps + 1, duration_step)
        )

        # Bookkeeping for sample()/validate() stats (get_metrics).
        self._sample_count = 0
        self._validate_count = 0
        self._validate_failures = 0

    # ------------------------------------------------------------------ #
    # Core properties
    # ------------------------------------------------------------------ #
    @property
    def action_len(self):
        """Length of the flat action list the env consumes."""
        if self.mode == 'phase_duration':
            return 2 * self.num_intersections
        return self.num_intersections

    @property
    def per_intersection_choices(self):
        """Cardinality of each intersection's action (for logging)."""
        if self.mode == 'switch_hold':
            return 2
        if self.mode == 'phase_duration':
            # phase x duration
            return self.num_phases * len(self._duration_grid)
        return self.num_phases

    @property
    def num_actions(self):
        """Total number of joint actions (for tabular / Q-table sizing)."""
        return self.per_intersection_choices ** self.num_intersections

    # ------------------------------------------------------------------ #
    # Spec / conversion helpers
    # ------------------------------------------------------------------ #
    def to_multidiscrete_nvec(self):
        """Return `nvec` for gymnasium.spaces.MultiDiscrete."""
        if self.mode == 'phase_duration':
            nvec = []
            for _ in range(self.num_intersections):
                nvec.extend([self.num_phases, len(self._duration_grid)])
            return nvec
        return [self.per_intersection_choices] * self.num_intersections

    def to_dict(self):
        """Return a plain-dict spec (used by training_env.get_action_space)."""
        return {
            'type': 'MultiDiscrete',
            'mode': self.mode,
            'nvec': self.to_multidiscrete_nvec(),
            'shape': (self.action_len,),
            'num_intersections': self.num_intersections,
            'num_phases': self.num_phases,
            'action_len': self.action_len,
            'per_intersection_choices': self.per_intersection_choices,
            'num_actions': self.num_actions,
        }

    def unflatten(self, flat_action):
        """Convert the flat list the env consumes into a structured form.

        Returns:
            list[tuple]  if mode == 'phase_duration': [(phase, dur), ...]
            list[int]    otherwise (a copy of the flat action)
        """
        if self.mode != 'phase_duration':
            return list(flat_action)
        if len(flat_action) != 2 * self.num_intersections:
            raise ValueError(
                f"phase_duration action must have length "
                f"{2 * self.num_intersections}, got {len(flat_action)}"
            )
        return [
            (int(flat_action[2 * i]), int(flat_action[2 * i + 1]))
            for i in range(self.num_intersections)
        ]

    def flatten(self, structured_action):
        """Inverse of unflatten(). Accepts list[int] or list[(phase, dur)]."""
        if self.mode != 'phase_duration':
            return list(structured_action)
        flat = []
        for phase, dur in structured_action:
            flat.extend([int(phase), int(dur)])
        return flat

    # ------------------------------------------------------------------ #
    # Action resolution (mode-aware)
    # ------------------------------------------------------------------ #
    def resolve(self, action, current_phases):
        """Turn a flat action into the concrete per-intersection phase list.

        Args:
            action: flat action list (length == self.action_len).
            current_phases: list[int] of current phase per intersection.
                Required for 'switch_hold' (and harmless to pass otherwise).

        Returns:
            list[int] of phase indices the simulator should apply.
        """
        if len(action) != self.action_len:
            raise ValueError(
                f"action length {len(action)} != expected {self.action_len}"
            )

        if self.mode == 'switch_hold':
            resolved = []
            for i, a in enumerate(action):
                a = int(a)
                if a == 0:
                    resolved.append(int(current_phases[i]))
                elif a == 1:
                    resolved.append((int(current_phases[i]) + 1) % self.num_phases)
                else:
                    raise ValueError(
                        f"switch_hold action[{i}]={a} not in {{0, 1}}"
                    )
            return resolved

        if self.mode == 'phase_duration':
            return [int(action[2 * i]) for i in range(self.num_intersections)]

        # discrete_phase
        return [int(a) for a in action]

    def get_action_mask(self, current_phases, min_green_satisfied):
        """Return per-intersection allowed-action masks.

        Args:
            current_phases: list[int] of current phase per intersection.
            min_green_satisfied: list[bool] -- True if the intersection has
                held its current phase for at least the min-green duration.

        Returns:
            list[list[int]] where mask[i] is the list of allowed action
            indices at intersection i. For 'discrete_phase' the mask is
            always all phases (min-green is enforced by the env's hold logic).
            For 'switch_hold', only [0] is allowed until min-green is met,
            then [0, 1]. For 'phase_duration', only the phase component is
            constrained to the current phase until min-green is met.
        """
        if self.mode == 'discrete_phase':
            return [list(range(self.num_phases))
                    for _ in range(self.num_intersections)]

        if self.mode == 'switch_hold':
            masks = []
            for i in range(self.num_intersections):
                if min_green_satisfied[i]:
                    masks.append([0, 1])
                else:
                    masks.append([0])
            return masks

        # phase_duration: mask the phase component; durations always allowed
        masks = []
        for i in range(self.num_intersections):
            if min_green_satisfied[i]:
                phases = list(range(self.num_phases))
            else:
                phases = [int(current_phases[i])]
            masks.append(phases)
        return masks

    # ------------------------------------------------------------------ #
    # Sampling / validation
    # ------------------------------------------------------------------ #
    def sample(self, action_masks=None):
        """Return a uniformly random valid flat action list.

        Args:
            action_masks: optional per-intersection masks from
                get_action_mask(); if provided, samples respect them.
        """
        self._sample_count += 1
        rng = self._rng

        if self.mode == 'switch_hold':
            if action_masks is None:
                return [rng.randint(0, 1)
                        for _ in range(self.num_intersections)]
            return [rng.choice(action_masks[i])
                    for i in range(self.num_intersections)]

        if self.mode == 'phase_duration':
            flat = []
            for i in range(self.num_intersections):
                if action_masks is not None:
                    phase = rng.choice(action_masks[i])
                else:
                    phase = rng.randrange(self.num_phases)
                dur = rng.choice(self._duration_grid)
                flat.extend([phase, dur])
            return flat

        # discrete_phase
        if action_masks is None:
            return [rng.randrange(self.num_phases)
                    for _ in range(self.num_intersections)]
        return [rng.choice(action_masks[i])
                for i in range(self.num_intersections)]

    def validate(self, action):
        """Raise ValueError on malformed actions; return True otherwise."""
        self._validate_count += 1
        try:
            self._validate(action)
        except ValueError:
            self._validate_failures += 1
            raise
        return True

    def _validate(self, action):
        if len(action) != self.action_len:
            raise ValueError(
                f"action length {len(action)} != expected {self.action_len} "
                f"for mode {self.mode!r}"
            )
        for i, a in enumerate(action):
            a_int = int(a)
            if a_int != a:
                raise ValueError(
                    f"action[{i}]={a!r} is not an integer"
                )
            if self.mode == 'switch_hold':
                if a_int not in (0, 1):
                    raise ValueError(
                        f"action[{i}]={a_int} not in {{0, 1}} "
                        f"(switch_hold mode)"
                    )
            elif self.mode == 'phase_duration':
                is_phase = (i % 2 == 0)
                if is_phase:
                    if not (0 <= a_int < self.num_phases):
                        raise ValueError(
                            f"action[{i}] (phase)={a_int} not in "
                            f"[0, {self.num_phases})"
                        )
                else:
                    if a_int not in self._duration_grid:
                        raise ValueError(
                            f"action[{i}] (duration)={a_int} not in "
                            f"{self._duration_grid[0]}..{self._duration_grid[-1]} "
                            f"step {self.duration_step}"
                        )
            else:  # discrete_phase
                if not (0 <= a_int < self.num_phases):
                    raise ValueError(
                        f"action[{i}]={a_int} not in [0, {self.num_phases})"
                    )

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def get_metrics(self):
        """Logging hooks consistent with the other pipeline modules."""
        return {
            'action_mode': self.mode,
            'action_num_intersections': self.num_intersections,
            'action_num_phases': self.num_phases,
            'action_action_len': self.action_len,
            'action_per_intersection_choices': self.per_intersection_choices,
            'action_num_joint_actions': self.num_actions,
            'action_samples_drawn': self._sample_count,
            'action_validations': self._validate_count,
            'action_validation_failures': self._validate_failures,
        }

    def __repr__(self):
        return (
            f"ActionSpace(mode={self.mode!r}, N={self.num_intersections}, "
            f"P={self.num_phases}, action_len={self.action_len}, "
            f"per_intersection_choices={self.per_intersection_choices})"
        )


# ---------------------------------------------------------------------- #
# Smoke test -- mirrors the style of the other pipeline modules' __main__
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    import random as _r

    print("=== discrete_phase (default; matches training_env.py) ===")
    a = ActionSpace(num_intersections=5, num_phases=4, seed=0)
    print(a)
    print("dict spec:", a.to_dict())
    print("sample:    ", a.sample())
    print("validate:  ", a.validate([0, 1, 2, 3, 0]))
    print("resolve:   ", a.resolve([0, 1, 2, 3, 0], [0, 0, 0, 0, 0]))
    print("metrics:   ", a.get_metrics())
    print()

    print("=== switch_hold ===")
    a = ActionSpace(num_intersections=5, num_phases=4,
                    mode='switch_hold', seed=0)
    print(a)
    print("sample:  ", a.sample())
    print("validate:", a.validate([1, 0, 1, 1, 0]))
    print("resolve: ", a.resolve([1, 0, 1, 1, 0], [0, 1, 2, 3, 0]))
    masks = a.get_action_mask([0, 1, 2, 3, 0], [True, False, True, False, True])
    print("masks:   ", masks)
    print("masked sample:", a.sample(masks))
    try:
        a.validate([2, 0, 0, 0, 0])
    except ValueError as e:
        print("expected failure:", e)
    print()

    print("=== phase_duration ===")
    a = ActionSpace(num_intersections=3, num_phases=4,
                    mode='phase_duration', max_duration_steps=30,
                    duration_step=5, seed=0)
    print(a)
    print("duration grid:", a._duration_grid)
    flat = a.sample()
    print("sample flat:  ", flat)
    structured = a.unflatten(flat)
    print("unflattened:  ", structured)
    print("round-trip ok:", a.flatten(structured) == flat)
    print("resolve:      ", a.resolve(flat, [0, 0, 0]))
    print("validate:     ", a.validate(flat))
    print("metrics:      ", a.get_metrics())