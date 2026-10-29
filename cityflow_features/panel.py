#!/usr/bin/env python3
"""
Metrics Panel Module
====================
Final step in the traffic-signal RL pipeline. Recovers, aggregates, and
renders the metrics produced by every prior stage into one coherent panel
suitable for:

    - logging / tensorboard (flatten())
    - a dashboard (snapshot(), render())
    - a quick verbal summary ("pitch")
    - recovering a paused or crashed run (save() / load())

Pipeline:
    signal_characterization.py  ->  observation space
    reward_calculator.py        ->  scalar reward
    action_space.py             ->  action space
    training_env.py             ->  gym-style env
    run.py                      ->  entry point
    panel.py                    ->  metrics panel      (this file)

Design notes
------------
* stdlib only; degrades gracefully if some stages aren't wired in.
* Panel history is a bounded deque, so a long run doesn't blow memory.
* save()/load() round-trips the panel state to JSON -- this is the
  "recovery" path: kill a run at step N, load the panel, keep logging.
* pitch() returns a single line suitable for a demo / README / chat.
* flatten() returns tensorboard-friendly keys (no nesting, '/' separators).
* render() prints a full multi-section dashboard.
"""

import json
import statistics
from collections import deque


class MetricsPanel:
    """Aggregated dashboard + recovery container for the RL pipeline.

    Args:
        history_size: max number of snapshots retained (rolling).
    """

    def __init__(self, history_size=200):
        self.history_size = int(history_size)
        self._history = deque(maxlen=self.history_size)
        self._step = 0

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record(self, env=None, action_space=None, step=None, extra=None):
        """Pull a snapshot from whatever stages are provided.

        Args:
            env: TrafficSignalEnv instance (or None).
            action_space: ActionSpace instance (or None).
            step: explicit step number; defaults to env._step_count.
            extra: arbitrary dict merged under snapshot['extra'].

        Returns:
            The snapshot dict that was appended to history.
        """
        snap = self._build_snapshot(env, action_space, step, extra)
        self._history.append(snap)
        self._step = snap['step']
        return snap

    def _build_snapshot(self, env, action_space, step, extra):
        snap = {
            'step': int(step) if step is not None
                    else int(getattr(env, '_step_count', self._step)),
            'env': {},
            'signal': {},
            'reward': {},
            'action': {},
            'extra': dict(extra or {}),
        }

        if env is not None:
            snap['env'] = {
                'step_count': getattr(env, '_step_count', 0),
                'episode_phase_changes': getattr(env, '_episode_phase_changes', 0),
                'num_intersections': getattr(env, 'num_intersections', 0),
                'num_phases': getattr(env, 'num_phases', 0),
                'min_green_steps': getattr(env, 'min_green_steps', 0),
                'sim_dt': getattr(env, 'sim_dt', 0.0),
            }
            try:
                snap['signal'] = dict(env.signal_char.get_metrics())
            except Exception:
                pass
            try:
                snap['reward'] = dict(env.reward_calc.get_metrics())
            except Exception:
                pass

        if action_space is not None:
            try:
                snap['action'] = dict(action_space.get_metrics())
            except Exception:
                pass

        return snap

    # ------------------------------------------------------------------ #
    # Read-only views
    # ------------------------------------------------------------------ #
    def latest(self):
        """Most recent snapshot, or None if nothing recorded."""
        return self._history[-1] if self._history else None

    def first(self):
        return self._history[0] if self._history else None

    def snapshot(self):
        """Lightweight summary of the panel state."""
        if not self._history:
            return {'num_records': 0, 'step_range': (0, 0), 'latest': None}
        return {
            'num_records': len(self._history),
            'step_range': (self._history[0]['step'], self._history[-1]['step']),
            'latest': self.latest(),
        }

    def trend(self, section='reward', key='reward_avg_episode_return'):
        """First-vs-last delta for one metric, plus a direction word."""
        if len(self._history) < 2:
            return {'delta': 0.0, 'direction': 'flat',
                    'first': None, 'last': None}
        first = self._history[0].get(section, {}).get(key)
        last = self._history[-1].get(section, {}).get(key)
        if first is None or last is None:
            return {'delta': 0.0, 'direction': 'flat',
                    'first': first, 'last': last}
        delta = float(last) - float(first)
        direction = ('improving' if delta > 0
                     else 'degrading' if delta < 0 else 'flat')
        return {'delta': delta, 'direction': direction,
                'first': first, 'last': last}

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def pitch(self):
        """One-line elevator-pitch summary of the current panel."""
        if not self._history:
            return "No metrics recorded yet."
        latest = self._history[-1]
        reward = latest.get('reward', {})
        env = latest.get('env', {})
        action = latest.get('action', {})

        avg_return = reward.get('reward_avg_episode_return', 0.0)
        queue = reward.get('reward_avg_queue_pressure', 0.0)
        throughput = reward.get('reward_avg_throughput_gain', 0.0)
        fairness = reward.get('reward_avg_fairness_penalty', 0.0)
        step = latest.get('step', 0)
        n = env.get('num_intersections', 0)
        changes = env.get('episode_phase_changes', 0)
        mode = action.get('action_mode', '?')

        tr = self.trend()
        arrow = {'improving': '↑', 'degrading': '↓', 'flat': '→'}[tr['direction']]

        return (
            f"[step {step:>4} | {n} intersections | mode={mode}] "
            f"avg return {avg_return:+.3f} {arrow} "
            f"queue {queue:.3f} · throughput {throughput:.3f} · "
            f"fairness {fairness:.3f} · {changes} phase changes"
        )

    def flatten(self):
        """Tensorboard-friendly flat dict from the latest snapshot."""
        latest = self.latest()
        if latest is None:
            return {}
        flat = {}
        for section in ('env', 'signal', 'reward', 'action'):
            for k, v in latest.get(section, {}).items():
                flat[f"panel/{section}/{k}"] = v
        for k, v in latest.get('extra', {}).items():
            flat[f"panel/extra/{k}"] = v
        return flat

    def compare(self, other):
        """Numeric delta of the latest snapshots of two panels."""
        a, b = self.latest(), other.latest()
        if a is None or b is None:
            return {}
        deltas = {}
        for section in ('env', 'signal', 'reward', 'action'):
            keys = set(a.get(section, {})) & set(b.get(section, {}))
            for k in keys:
                va, vb = a[section][k], b[section][k]
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    deltas[f"{section}/{k}"] = float(va) - float(vb)
        return deltas

    def render(self, mode='full'):
        """Print the panel. mode in {'full', 'pitch', 'compact'}."""
        latest = self.latest()
        if latest is None:
            print("[panel] no metrics recorded")
            return

        if mode == 'pitch':
            print(self.pitch())
            return

        if mode == 'compact':
            print(self.pitch())
            self._print_section('reward', latest.get('reward', {}))
            return

        print("=" * 74)
        print(f" TRAFFIC-SIGNAL RL PANEL   step={latest.get('step')}   "
              f"records={len(self._history)}")
        print("=" * 74)
        for section in ('env', 'signal', 'reward', 'action'):
            self._print_section(section, latest.get(section, {}))
        if latest.get('extra'):
            self._print_section('extra', latest['extra'])
        print("-" * 74)
        print(self.pitch())
        print("=" * 74)

    @staticmethod
    def _print_section(name, data):
        if not data:
            return
        print(f"\n[{name.upper()}]")
        for k, v in data.items():
            print(f"  {k:<46s} {MetricsPanel._fmt(v)}")

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:+.4f}"
        return str(v)

    # ------------------------------------------------------------------ #
    # Recovery (the "recovered panel" path)
    # ------------------------------------------------------------------ #
    def save(self, path):
        """Serialize panel state to JSON so a run can be resumed later."""
        with open(path, 'w') as f:
            json.dump({
                'history_size': self.history_size,
                'step': self._step,
                'history': list(self._history),
            }, f, indent=2, default=str)

    @classmethod
    def load(cls, path):
        """Restore a panel from disk (inverse of save())."""
        with open(path) as f:
            data = json.load(f)
        panel = cls(history_size=data.get('history_size', 200))
        panel._step = int(data.get('step', 0))
        panel._history = deque(
            data.get('history', []), maxlen=panel.history_size)
        return panel

    # ------------------------------------------------------------------ #
    # Pipeline-style metrics + lifecycle
    # ------------------------------------------------------------------ #
    def reset(self):
        self._history.clear()
        self._step = 0

    def get_metrics(self):
        """Consistent with the other pipeline modules' get_metrics()."""
        latest = self.latest() or {}
        return {
            'panel_num_records': len(self._history),
            'panel_history_size': self.history_size,
            'panel_last_step': self._step,
            'panel_last_avg_return':
                latest.get('reward', {}).get('reward_avg_episode_return', 0.0),
            'panel_trend_direction': self.trend()['direction'],
            'panel_trend_delta': self.trend()['delta'],
        }

    def __repr__(self):
        return (f"MetricsPanel(records={len(self._history)}, "
                f"step={self._step}, size={self.history_size})")


# ---------------------------------------------------------------------- #
# Smoke test -- exercises the full pipeline via run.py, then recovers.
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    import os
    import tempfile

    from action_space import ActionSpace
    from training_env import TrafficSignalEnv

    print("=== building env + action space + panel ===")
    env = TrafficSignalEnv(num_intersections=3, num_phases=4,
                           max_steps=15, sim_dt=1.0,
                           min_green_steps=2, seed=7)
    act = ActionSpace(num_intersections=3, num_phases=4,
                      mode='switch_hold', seed=7)
    panel = MetricsPanel(history_size=50)

    obs, info = env.reset(seed=7)
    panel.record(env=env, action_space=act)

    # Run a short episode, recording a snapshot every few steps.
    done = False
    while not done:
        masks = act.get_action_mask(
            env._current_phase,
            [env._phase_hold_steps[i] >= env.min_green_steps
             for i in range(env.num_intersections)])
        action = act.sample(masks)
        applied = act.resolve(action, env._current_phase)
        obs, reward, term, trunc, info = env.step(applied)
        if env._step_count % 3 == 0:
            panel.record(env=env, action_space=act,
                         extra={'last_reward': float(reward)})
        done = term or trunc

    print("\n=== full render ===")
    panel.render(mode='full')

    print("\n=== compact render ===")
    panel.render(mode='compact')

    print("\n=== pitch ===")
    print(panel.pitch())

    print("\n=== flattened (tensorboard keys) ===")
    for k, v in panel.flatten().items():
        print(f"  {k}: {v}")

    print("\n=== save / load round-trip (recovery) ===")
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    try:
        panel.save(path)
        recovered = MetricsPanel.load(path)
        print(f"  original:  {panel}")
        print(f"  recovered: {recovered}")
        print(f"  pitch equal: {panel.pitch() == recovered.pitch()}")
        print(f"  compare delta: {panel.compare(recovered)}")
    finally:
        os.unlink(path)

    print("\n=== pipeline-style metrics ===")
    print(panel.get_metrics())

    env.close()
    print("\nSmoke test passed.")