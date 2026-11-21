#!/usr/bin/env python3
"""
main.py -- Improved DQN traffic-signal training on a flow file.

Fixes over previous version
---------------------------
* [FIX] total_steps now counts *env-steps* (standard SB3 convention), so a
        100k run over 128 envs gives 781 iterations, not 66.
* [FIX] avgEpR no longer stays NaN when episodes outlast the run: we log
        avg_dec_R (rolling per-decision reward) every iteration, plus
        avg_ep_R once episodes actually complete.
* [FIX] evaluate() returns mean queue *per approach* (not raw sums).
* [FIX] target_update_steps, episode_decisions, eps_decay_steps, eval_every
        and log_every are auto-clamped to the run length.
* [NEW] Best checkpoint saved to <ckpt>.best when eval_R improves.
* [NEW] Final comparison print: agent vs fixed / random / max-pressure.
* [NEW] Robust flow-file resolution (falls back to basename if the given
        path doesn't exist, e.g. Colab /content/ on Windows).
* [NEW] Progress line shows ETA, %, and buffer transitions.
* [NEW] Plot handles NaN columns, adds baselines automatically, adds a
        per-decision-reward panel.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import errno
import json
import math
import ntpath
import os
import random
import socket
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================================ #
# AdvancedTrainingLock (unchanged)
# ============================================================================ #
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None


class AdvancedTrainingLock:
    def __init__(self, path: str, stale_after_s: int = 3600, retries: int = 5):
        self.path = path
        self.stale_after_s = stale_after_s
        self.retries = retries
        self._fp = None

    def _read_meta(self):
        try:
            with open(self.path) as f:
                return dict(l.split("=", 1) for l in f.read().splitlines() if "=" in l)
        except FileNotFoundError:
            return {}

    def _pid_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError as e:
            return e.errno == errno.EPERM
        return True

    def _is_stale(self) -> bool:
        meta = self._read_meta()
        if not meta:
            return True
        if meta.get("host") != socket.gethostname():
            return False
        try:
            pid = int(meta["pid"])
        except (KeyError, ValueError):
            return True
        if not self._pid_alive(pid):
            return True
        try:
            age = time.time() - float(meta.get("ts", 0))
        except ValueError:
            age = self.stale_after_s + 1
        return age > self.stale_after_s

    def _try_lock(self) -> bool:
        try:
            if fcntl is not None:
                fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:
                msvcrt.locking(self._fp.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except (BlockingIOError, OSError):
            return False

    def _acquire(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._fp = open(self.path, "a+")
        for attempt in range(self.retries):
            if self._try_lock():
                break
            if self._is_stale():
                print(f"[lock] stale lock at {self.path}; reclaiming "
                      f"(attempt {attempt + 1}/{self.retries})")
                time.sleep(0.1 * (attempt + 1))
                continue
            self._fp.close(); self._fp = None
            raise RuntimeError(f"[lock] another run holds {self.path}")
        else:
            self._fp.close(); self._fp = None
            raise RuntimeError(f"[lock] could not acquire {self.path}")

        self._fp.seek(0); self._fp.truncate()
        self._fp.write(
            f"pid={os.getpid()}\nhost={socket.gethostname()}\n"
            f"ts={time.time():.0f}\n"
            f"started={_dt.datetime.now().isoformat(timespec='seconds')}\n"
        )
        self._fp.flush()

    def touch(self):
        if self._fp is None:
            return
        try:
            self._fp.seek(0, os.SEEK_END)
            self._fp.write(f"heartbeat={time.time():.0f}\n")
            self._fp.flush(); os.utime(self.path, None)
        except Exception:
            pass

    def release(self):
        if self._fp is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                try:
                    self._fp.seek(0)
                    msvcrt.locking(self._fp.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        finally:
            self._fp.close(); self._fp = None
            try: os.remove(self.path)
            except OSError: pass

    def __enter__(self): self._acquire(); return self
    def __exit__(self, *exc): self.release(); return False


# ============================================================================ #
# Flow parsing and network reconstruction (unchanged)
# ============================================================================ #
APPROACH_NAMES = ("N", "S", "E", "W")


def parse_road(road_id: str):
    p = road_id.split("_")
    return int(p[1]), int(p[2]), int(p[3])


def infer_intersections(flows) -> dict:
    roads = set()
    for flow in flows:
        roads.update(flow["route"])
    candidates = set()
    for r in roads:
        x, y, d = parse_road(r)
        if   d == 0: candidates.add((x + 1, y))
        elif d == 2: candidates.add((x - 1, y))
        elif d == 1: candidates.add((x, y + 1))
        elif d == 3: candidates.add((x, y - 1))
    out = {}
    for (x, y) in candidates:
        needs = {
            "W": f"road_{x - 1}_{y}_0",
            "E": f"road_{x + 1}_{y}_2",
            "N": f"road_{x}_{y - 1}_1",
            "S": f"road_{x}_{y + 1}_3",
        }
        if all(v in roads for v in needs.values()):
            out[(x, y)] = needs
    return out


def build_network(flows, intersections):
    inter_list = sorted(intersections.keys())
    idx_of = {it: i for i, it in enumerate(inter_list)}
    n_inter = len(inter_list)
    UP_DX = (0, 0, 1, -1); UP_DY = (-1, 1, 0, 0)
    DN_DX = (0, 0, -1, 1); DN_DY = (1, -1, 0, 0)
    upstream   = np.full((n_inter, 4), -1, dtype=np.int64)
    downstream = np.full((n_inter, 4), -1, dtype=np.int64)
    for i, (x, y) in enumerate(inter_list):
        for a in range(4):
            ux, uy = x + UP_DX[a], y + UP_DY[a]
            if (ux, uy) in idx_of:
                upstream[i, a] = idx_of[(ux, uy)]
            dx_, dy_ = x + DN_DX[a], y + DN_DY[a]
            if (dx_, dy_) in idx_of:
                downstream[i, a] = idx_of[(dx_, dy_)]
    ext_rate = np.zeros((n_inter, 4), dtype=np.float32)
    for flow in flows:
        if not flow.get("route"):
            continue
        c, r, d = parse_road(flow["route"][0])
        if   d == 0: ti, ta = (c + 1, r), 3
        elif d == 2: ti, ta = (c - 1, r), 2
        elif d == 1: ti, ta = (c, r + 1), 0
        elif d == 3: ti, ta = (c, r - 1), 1
        else: continue
        if ti in idx_of:
            ext_rate[idx_of[ti], ta] += 1.0 / float(flow["interval"])
    is_boundary = upstream == -1
    return inter_list, ext_rate, upstream, downstream, is_boundary


# ============================================================================ #
# Simulator (unchanged)
# ============================================================================ #
@dataclass
class SimConfig:
    dt: float = 1.0
    discharge_rate: float = 2.0
    action_steps: int = 5
    episode_decisions: int = 100
    min_green: int = 2
    queue_capacity: float = 40.0
    w_pressure: float = 0.05
    w_throughput: float = 1.0
    w_switch: float = 0.5


class VecFlowSim:
    def __init__(self, flows, num_envs: int, seed: int, cfg: SimConfig):
        self.cfg = cfg
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        inters = infer_intersections(flows)
        (self.inter_list, self.ext_rate, self.upstream,
         self.downstream, self.is_boundary) = build_network(flows, inters)
        self.n_inter = len(self.inter_list)
        self.n_dir = 4
        self.action_dim = 2
        self.obs_dim = self.n_dir * 2 + 2 + 1
        self.queues         = np.zeros((num_envs, self.n_inter, self.n_dir), dtype=np.float32)
        self.phases         = np.zeros((num_envs, self.n_inter), dtype=np.int64)
        self.elapsed        = np.zeros((num_envs, self.n_inter), dtype=np.int64)
        self.decision_steps = np.zeros(num_envs, dtype=np.int64)
        self._served = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
        self._up_j    = np.where(self.upstream >= 0, self.upstream, 0)
        self._up_mask = (self.upstream >= 0)
        self._app_idx = np.arange(self.n_dir)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.queues[:] = 0.0; self.phases[:] = 0
            self.elapsed[:] = 0; self.decision_steps[:] = 0
        else:
            self.queues[env_ids] = 0.0; self.phases[env_ids] = 0
            self.elapsed[env_ids] = 0; self.decision_steps[env_ids] = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        qn = np.clip(self.queues / self.cfg.queue_capacity, 0.0, 2.0)
        up_q = self.queues[:, self._up_j, self._app_idx[None, :]]
        up_q = np.where(self._up_mask[None], up_q, 0.0)
        up_qn = np.clip(up_q / self.cfg.queue_capacity, 0.0, 2.0)
        ph_oh = np.eye(2, dtype=np.float32)[self.phases]
        elapsed_n = np.clip(self.elapsed / max(1, self.cfg.min_green), 0.0, 1.0)[..., None]
        return np.concatenate([qn, up_qn, ph_oh, elapsed_n], axis=-1).astype(np.float32)

    def valid_action_mask(self) -> np.ndarray:
        keep = self.phases
        switch = 1 - self.phases
        can_switch = self.elapsed >= self.cfg.min_green
        mask = np.zeros((self.num_envs, self.n_inter, 2), dtype=bool)
        mask[np.arange(self.num_envs)[:, None], np.arange(self.n_inter)[None, :], keep] = True
        mask[np.arange(self.num_envs)[:, None], np.arange(self.n_inter)[None, :], switch] = can_switch
        return mask

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        mask = self.valid_action_mask()
        allowed = mask[np.arange(self.num_envs)[:, None],
                       np.arange(self.n_inter)[None, :], actions]
        actions = np.where(~allowed, self.phases, actions)
        switched = (actions != self.phases)
        self.phases  = actions.copy()
        self.elapsed = np.where(switched, 1, self.elapsed + 1)
        served = self._served[actions]
        dis = self.cfg.discharge_rate * self.cfg.dt
        total_throughput = np.zeros((self.num_envs, self.n_inter), dtype=np.float32)
        pressure_acc     = np.zeros((self.num_envs, self.n_inter), dtype=np.float32)
        for _ in range(self.cfg.action_steps):
            arrivals = self.rng.poisson(self.ext_rate[None] * self.cfg.dt,
                                        size=(self.num_envs, self.n_inter, self.n_dir)).astype(np.float32)
            cap = served * dis
            avail = self.queues + arrivals
            outflow = np.minimum(cap, avail)
            outflow = np.where(served > 0, outflow, 0.0)
            up_out = outflow[:, self._up_j, self._app_idx[None, :]]
            inflow = np.where(self._up_mask[None], up_out, 0.0)
            self.queues = np.maximum(self.queues + arrivals - outflow + inflow, 0.0)
            total_throughput += outflow.sum(axis=-1)
            pressure_acc     += self.queues.sum(axis=-1)
        self.decision_steps += 1
        cfg = self.cfg
        reward = (cfg.w_throughput * total_throughput
                  - cfg.w_pressure * pressure_acc
                  - cfg.w_switch * switched.astype(np.float32))
        dones = self.decision_steps >= cfg.episode_decisions
        done_idx = np.where(dones)[0]
        if len(done_idx):
            self.reset(done_idx)
        infos = {"throughput": total_throughput,
                 "pressure": pressure_acc / cfg.action_steps}
        return self._obs(), reward.astype(np.float32), dones, infos


# ============================================================================ #
# Q-network, Replay, n-step collector (unchanged)
# ============================================================================ #
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.reset_parameters(); self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    @staticmethod
    def _f(x): return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        eps_in  = self._f(torch.randn(self.in_features,  device=self.weight_mu.device))
        eps_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


def linear_layer(in_f, out_f, noisy: bool):
    return NoisyLinear(in_f, out_f) if noisy else nn.Linear(in_f, out_f)


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128, noisy: bool = True):
        super().__init__()
        self.noisy = noisy
        self.trunk = nn.Sequential(
            linear_layer(obs_dim, hidden, noisy), nn.ReLU(),
            linear_layer(hidden, hidden, noisy), nn.ReLU(),
        )
        self.value     = linear_layer(hidden, 1, noisy)
        self.advantage = linear_layer(hidden, action_dim, noisy)
        self.action_dim = action_dim

    def forward(self, x):
        h = self.trunk(x); v = self.value(h); a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)

    def reset_noise(self):
        if not self.noisy: return
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = 1
        while self.capacity < capacity: self.capacity *= 2
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.max_priority = 1.0
    def total(self): return float(self.tree[0])
    def update(self, idx, priority):
        tree_idx = idx + self.capacity - 1
        diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += diff
        if priority > self.max_priority: self.max_priority = priority
    def get(self, value):
        tree_idx = 0
        while tree_idx < self.capacity - 1:
            left = 2 * tree_idx + 1; right = left + 1
            if value <= self.tree[left]: tree_idx = left
            else: value -= self.tree[left]; tree_idx = right
        return tree_idx - self.capacity + 1, float(self.tree[tree_idx])


class PrioritizedReplay:
    def __init__(self, capacity, obs_dim, alpha=0.6, beta=0.4, beta_end=1.0,
                 beta_steps=1_000_000, n_step=3, gamma=0.99):
        self.capacity, self.alpha, self.beta = capacity, alpha, beta
        self.beta_end, self.beta_steps = beta_end, beta_steps
        self.n_step, self.gamma = n_step, gamma
        self.tree = SumTree(capacity)
        self.obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act  = np.zeros(capacity, dtype=np.int64)
        self.ret  = np.zeros(capacity, dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0; self.size = 0; self._step = 0

    def push(self, obs, act, ret, nobs, done):
        idx = self.ptr
        self.obs[idx] = obs; self.act[idx] = act; self.ret[idx] = ret
        self.nobs[idx] = nobs; self.done[idx] = done
        self.tree.update(idx, self.tree.max_priority)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity: self.size += 1

    def sample(self, batch):
        total = self.tree.total(); seg = total / batch
        idxs = np.zeros(batch, dtype=np.int64)
        ps   = np.zeros(batch, dtype=np.float64)
        for i in range(batch):
            val = seg * i + np.random.random() * seg
            idx, p = self.tree.get(val)
            idxs[i] = idx; ps[i] = p
        ps = np.maximum(ps, 1e-8)
        probs = ps / max(total, 1e-8)
        beta = self.beta + (self.beta_end - self.beta) * min(1.0, self._step / max(1, self.beta_steps))
        weights = (self.size * probs) ** (-beta)
        weights = weights / weights.max()
        self._step += 1
        return (self.obs[idxs], self.act[idxs], self.ret[idxs],
                self.nobs[idxs], self.done[idxs],
                idxs, weights.astype(np.float32))

    def update_priorities(self, idxs, prios):
        for idx, p in zip(idxs, prios):
            self.tree.update(int(idx), (abs(float(p)) + 1e-8) ** self.alpha)


class UniformReplay:
    def __init__(self, capacity, obs_dim, n_step=3, gamma=0.99):
        self.capacity, self.n_step, self.gamma = capacity, n_step, gamma
        self.obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act  = np.zeros(capacity, dtype=np.int64)
        self.ret  = np.zeros(capacity, dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0; self.size = 0
    def push(self, obs, act, ret, nobs, done):
        idx = self.ptr
        self.obs[idx] = obs; self.act[idx] = act; self.ret[idx] = ret
        self.nobs[idx] = nobs; self.done[idx] = done
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity: self.size += 1
    def sample(self, batch):
        idxs = np.random.randint(0, self.size, size=batch)
        return (self.obs[idxs], self.act[idxs], self.ret[idxs],
                self.nobs[idxs], self.done[idxs], idxs,
                np.ones(batch, dtype=np.float32))
    def update_priorities(self, idxs, prios): pass


class NStepCollector:
    def __init__(self, n_step, gamma):
        self.n, self.gamma = n_step, gamma
        self.obs  = deque(maxlen=n_step)
        self.act  = deque(maxlen=n_step)
        self.rew  = deque(maxlen=n_step)
        self.done = deque(maxlen=n_step)
        self.last_nobs = None
    def push(self, obs, act, rew, nobs, done, buffer):
        self.obs.append(obs); self.act.append(act)
        self.rew.append(rew); self.done.append(done)
        self.last_nobs = nobs
        if len(self.obs) == self.n: self._emit(buffer)
        if done:
            while len(self.obs) > 0: self._emit(buffer)
            self.last_nobs = None
    def _emit(self, buffer):
        R = sum((self.gamma ** k) * r for k, r in enumerate(self.rew))
        buffer.push(self.obs[0], self.act[0], R, self.last_nobs,
                    float(any(self.done)))
        self.obs.popleft(); self.act.popleft()
        self.rew.popleft(); self.done.popleft()


# ============================================================================ #
# Baseline policies (unchanged)
# ============================================================================ #
def fixed_time_policy(phases, elapsed, decision_steps, period=5):
    want = (decision_steps // period) % 2
    return np.broadcast_to(want[:, None], phases.shape).copy()

def random_policy(phases, elapsed, num_envs, n_inter, rng):
    return rng.integers(0, 2, size=(num_envs, n_inter), dtype=np.int64)

def max_pressure_policy(queues, phases, elapsed, min_green):
    ns = queues[..., 0] + queues[..., 1]
    ew = queues[..., 2] + queues[..., 3]
    # Preserve the current phase when pressure is tied to avoid needless switching.
    want = np.where(ew > ns, 1, np.where(ns > ew, 0, phases))
    can_switch = elapsed >= min_green
    return np.where(can_switch, want, phases)


# [FIX] evaluate returns a dict with normalised metrics
def evaluate(sim: VecFlowSim, policy_fn, n_steps: int, device, q_net=None,
             eps: float = 0.0, seed: int = 1000) -> dict:
    sim.reset()
    rng = np.random.default_rng(seed)
    total_r = 0.0
    total_q = 0.0
    total_t = 0.0
    count_agents = 0
    obs = sim._obs()
    for _ in range(n_steps):
        if q_net is not None:
            with torch.no_grad():
                q_net.eval()
                flat = torch.as_tensor(obs.reshape(-1, sim.obs_dim), device=device)
                q = q_net(flat).view(sim.num_envs, sim.n_inter, 2).cpu().numpy()
                q_net.train()
            mask = sim.valid_action_mask()
            q_masked = np.where(mask, q, -1e9)
            if eps > 0 and rng.random() < eps:
                actions = np.array([
                    rng.choice(np.where(mask[e, i])[0])
                    for e in range(sim.num_envs) for i in range(sim.n_inter)
                ]).reshape(sim.num_envs, sim.n_inter)
            else:
                actions = q_masked.argmax(-1)
        else:
            actions = policy_fn(sim, rng)
        obs, rew, dones, infos = sim.step(actions)
        total_r += float(rew.sum())
        # sim.step() resets completed environments before returning. Use the
        # pressure measured during the step so terminal queues aren't counted
        # as zero in the evaluation metric.
        total_q += float(infos["pressure"].sum())
        total_t += float(infos["throughput"].sum())
        count_agents += sim.num_envs * sim.n_inter
    n_cells = n_steps * sim.num_envs * sim.n_inter * sim.n_dir
    return {
        "R": total_r / max(1, count_agents),          # mean per-agent reward / decision
        "Q": total_q / max(1, n_cells),               # mean queue length per approach  [FIX]
        "T": total_t / max(1, count_agents),          # mean throughput per agent / decision
    }


# ============================================================================ #
# Plotting
# ============================================================================ #
def plot_training_curves(csv_path, out_png="training_curves.png",
                         smooth=0.85, baselines=None):
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: missing dependency ({e})"); return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[plot] {csv_path} is empty"); return None

    def column(name):
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    def ema(xs, alpha):
        out, prev = [], None
        for v in xs:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out.append(prev if prev is not None else 0.0); continue
            prev = v if prev is None else alpha * prev + (1 - alpha) * v
            out.append(prev)
        return out

    if baselines is None and {"fix_R", "rand_R", "mp_R"}.issubset(df.columns):
        baselines = {}
        for col, name in [("fix_R", "fixed"), ("rand_R", "random"), ("mp_R", "max-pressure")]:
            s = df[col].dropna()
            if len(s): baselines[name] = float(s.iloc[-1])

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Training dashboard — {os.path.basename(csv_path)}",
                 fontsize=14, fontweight="bold")

    # Panel 1: per-decision reward (always available) + episode reward (if any)
    y = df.get("avg_dec_R")
    if y is not None and y.notna().any():
        ax[0, 0].plot(df["step"], y, alpha=0.25, color="C0", lw=0.8)
        ax[0, 0].plot(df["step"], ema(y.values, smooth), color="C0", lw=2,
                      label=f"EMA α={smooth}")
    ep = df.get("avg_ep_R")
    if ep is not None and ep.notna().any():
        ax[0, 0].plot(df["step"], ep, color="C3", lw=1.5, label="episode", alpha=0.7)
    ax[0, 0].set_title("reward (per decision / episode)")
    ax[0, 0].legend(fontsize=8, loc="lower right"); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(column("step"), column("loss"), color="C1", lw=1)
    ax[0, 1].set_title("TD loss"); ax[0, 1].grid(alpha=0.3)

    ax[0, 2].plot(column("step"), column("avg_q"), color="C2", lw=1)
    ax[0, 2].set_title("mean Q"); ax[0, 2].grid(alpha=0.3)

    ev = column("eval_R").dropna()
    if len(ev):
        ax[1, 0].plot(column("step"), column("eval_R"), color="C3", lw=1.8,
                      marker="o", ms=3, label="agent")
    if baselines:
        for (name, val), c in zip(baselines.items(), ["C4", "C5", "C6"]):
            ax[1, 0].axhline(val, ls="--", lw=1.2, color=c, label=name)
    ax[1, 0].set_title("eval reward vs baselines")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    eq = column("eval_Q").dropna()
    if len(eq):
        ax[1, 1].plot(column("step"), column("eval_Q"), color="C9", lw=1.8, marker="o", ms=3)
    ax[1, 1].set_title("eval mean queue / approach"); ax[1, 1].grid(alpha=0.3)

    ax[1, 2].plot(column("step"), column("eps"), color="black", lw=1.2)
    ax[1, 2].set_title("ε"); ax[1, 2].grid(alpha=0.3)

    for a in ax.flat: a.set_xlabel("step")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"[plot] wrote {out_png}")

    tail = df.tail(max(5, len(df) // 20))
    return {
        "final_avg_dec_R": float(tail["avg_dec_R"].dropna().mean()) if "avg_dec_R" in df else None,
        "final_avg_ep_R":  float(tail["avg_ep_R"].dropna().mean())  if "avg_ep_R"  in df else None,
        "final_loss":      float(tail["loss"].mean()),
        "final_avg_q":     float(tail["avg_q"].mean()),
        "final_eval_R":    float(tail["eval_R"].dropna().mean())    if "eval_R"    in df else None,
        "final_eval_Q":    float(tail["eval_Q"].dropna().mean())    if "eval_Q"    in df else None,
    }


# ============================================================================ #
# Training loop
# ============================================================================ #
def _autoscale(args, n_inter):
    """Clamp hyper-parameters so they actually fire inside the run."""
    # The vectorized loop completes whole batches, so its iteration count is
    # the ceiling of the requested environment-step budget.
    total_iters = max(1, math.ceil(args.total_steps / args.num_envs))
    changes = []
    def clamp(attr, new_val):
        old = getattr(args, attr)
        new = min(old, new_val)
        if new != old:
            changes.append(f"{attr}: {old}->{new}")
            setattr(args, attr, new)

    clamp("episode_decisions", max(1, total_iters // 3))
    clamp("target_update_steps", max(1, total_iters // 5))
    clamp("eps_decay_steps", max(1, args.total_steps))
    clamp("eval_every", max(1, total_iters // 5))
    clamp("log_every", max(1, total_iters // 12))
    if changes:
        print("[autoscale] " + ", ".join(changes))
    return total_iters


def train(args, lock=None):
    args.noisy = not args.no_noisy
    args.per   = not args.no_per
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with open(args.flow_file) as f:
        flows = json.load(f)

    sim_cfg = SimConfig(dt=args.dt, discharge_rate=args.discharge_rate,
                        action_steps=args.action_steps,
                        episode_decisions=args.episode_decisions,
                        min_green=args.min_green,
                        queue_capacity=args.queue_capacity,
                        w_pressure=args.w_pressure,
                        w_throughput=args.w_throughput,
                        w_switch=args.w_switch)
    sim = VecFlowSim(flows, num_envs=args.num_envs, seed=args.seed, cfg=sim_cfg)
    n_inter = sim.n_inter
    if n_inter == 0:
        raise ValueError("flow file contains no complete four-way intersections")
    n_agents = args.num_envs * n_inter

    total_iters = _autoscale(args, n_inter)
    # re-apply autoscaled values into sim_cfg
    sim_cfg.episode_decisions = args.episode_decisions

    print(f"[init] flows={len(flows)}  intersections={n_inter}  "
          f"obs_dim={sim.obs_dim}  envs={args.num_envs}  device={device}")
    print(f"[init] inter grid: x∈[{min(x for x,y in sim.inter_list)}.."
          f"{max(x for x,y in sim.inter_list)}]  "
          f"y∈[{min(y for x,y in sim.inter_list)}.."
          f"{max(y for x,y in sim.inter_list)}]")
    print(f"[init] boundary approaches: {int(sim.is_boundary.sum())}/{n_inter*4}  "
          f"total demand={float(sim.ext_rate.sum()):.2f} veh/s")
    print(f"[init] total_steps={args.total_steps} env-steps = {total_iters} iters "
          f"(= {args.total_steps * n_inter:,} agent-transitions)")
    print(f"[init] net={'Dueling+Noisy' if args.noisy else 'Dueling'}  "
          f"replay={'PER' if args.per else 'uniform'}  n_step={args.n_step}")
    grad_steps = args.grad_steps if args.grad_steps > 0 else max(1, args.num_envs // 8)
    print(f"[init] grad_steps/iter={grad_steps}  batch={args.batch_size}  "
          f"buffer={args.buffer_size}  min_green={args.min_green}")

    if total_iters < 100:
        print(f"[WARN] Only {total_iters} iterations — very short for a 12-intersection "
              f"task. Try --num-envs {max(8, args.num_envs // 8)} "
              f"or --total-steps {args.total_steps * 8:,}")

    q_net   = DuelingQNet(sim.obs_dim, sim.action_dim, args.hidden, noisy=args.noisy).to(device)
    tgt_net = DuelingQNet(sim.obs_dim, sim.action_dim, args.hidden, noisy=args.noisy).to(device)
    tgt_net.load_state_dict(q_net.state_dict()); tgt_net.eval()
    optimiser = optim.Adam(q_net.parameters(), lr=args.lr)

    if args.per:
        buffer = PrioritizedReplay(args.buffer_size, sim.obs_dim,
                                   alpha=args.per_alpha, beta=args.per_beta,
                                   beta_end=args.per_beta_end,
                                   beta_steps=args.total_steps,
                                   n_step=args.n_step, gamma=args.gamma)
    else:
        buffer = UniformReplay(args.buffer_size, sim.obs_dim,
                               n_step=args.n_step, gamma=args.gamma)

    collectors = [[NStepCollector(args.n_step, args.gamma) for _ in range(n_inter)]
                  for _ in range(args.num_envs)]

    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)), exist_ok=True)
    log_csv = open(args.log_csv, "w", newline="")
    writer  = csv.writer(log_csv)
    writer.writerow(["step", "iter", "eps", "buf_size", "loss", "avg_q",
                     "avg_dec_R", "avg_ep_R", "sps",
                     "eval_R", "eval_Q", "eval_T",
                     "fix_R", "rand_R", "mp_R"])
    log_jsonl = open(args.log_jsonl, "w") if args.log_jsonl else None

    try:
        import torch.utils.tensorboard as tb
        tb_writer = tb.SummaryWriter(log_dir=args.tb_dir) if args.tb_dir else None
    except Exception:
        tb_writer = None

    # ---- baselines ----
    print("[eval] baselines ...")
    base_sim = VecFlowSim(flows, num_envs=max(8, min(args.num_envs, 32)),
                          seed=args.seed + 1, cfg=sim_cfg)
    ev_steps = args.eval_steps
    fix_m  = evaluate(base_sim, lambda s, r: fixed_time_policy(s.phases, s.elapsed,
                                                               s.decision_steps, args.fix_period),
                      ev_steps, device)
    rand_m = evaluate(base_sim, lambda s, r: random_policy(s.phases, s.elapsed,
                                                            s.num_envs, s.n_inter, r),
                      ev_steps, device)
    mp_m   = evaluate(base_sim, lambda s, r: max_pressure_policy(s.queues, s.phases,
                                                                  s.elapsed, args.min_green),
                      ev_steps, device)
    fix_R, rand_R, mp_R = fix_m["R"], rand_m["R"], mp_m["R"]
    print(f"[eval] fixed={fix_R:+.3f}  random={rand_R:+.3f}  "
          f"max-pressure={mp_R:+.3f}   "
          f"(queues: {fix_m['Q']:.2f} / {rand_m['Q']:.2f} / {mp_m['Q']:.2f})")

    # ---- training ----
    states = sim.reset()
    total_env_steps = 0
    it = 0
    ep_rewards = np.zeros((args.num_envs, n_inter), dtype=np.float32)
    reward_window_ep  = deque(maxlen=50)     # episode returns (fires only when episodes end)
    reward_window_dec = deque(maxlen=200)    # per-decision mean reward (always fires) [NEW]
    last_losses = deque(maxlen=20)
    last_qs     = deque(maxlen=20)
    best_eval_R = -float("inf")
    best_eval_step = 0
    t0 = time.time()

    try:
        while total_env_steps < args.total_steps:
            it += 1
            frac = total_env_steps / max(1, args.eps_decay_steps)
            eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * frac)

            mask = sim.valid_action_mask()
            # Keep NoisyLinear active while acting. Switching to eval() here
            # disables its learned parameter noise and leaves exploration to
            # epsilon alone.
            with torch.no_grad():
                q_net.train()
                flat = torch.as_tensor(states.reshape(-1, sim.obs_dim), device=device)
                q = q_net(flat).view(args.num_envs, n_inter, 2).cpu().numpy()
            q_masked = np.where(mask, q, -1e9)
            actions = q_masked.argmax(-1).reshape(n_agents)

            # Draw epsilon independently for each intersection; a single
            # random decision for the entire vector synchronizes exploration
            # across all environments and signals.
            explore = np.random.random(n_agents) < eps
            if np.any(explore):
                flat_mask = mask.reshape(n_agents, 2)
                both = flat_mask[:, 0] & flat_mask[:, 1]
                only1 = flat_mask[:, 1] & ~flat_mask[:, 0]
                random_actions = np.where(
                    both, np.random.randint(0, 2, size=n_agents), only1.astype(np.int64))
                actions[explore] = random_actions[explore]
            actions = actions.reshape(args.num_envs, n_inter)

            next_obs, rewards, dones, infos = sim.step(actions)
            total_env_steps += args.num_envs        # [FIX] env-steps, not agent-decisions

            env_done = dones.astype(bool)
            for e in range(args.num_envs):
                d_e = bool(env_done[e])
                for i in range(n_inter):
                    collectors[e][i].push(states[e, i], int(actions[e, i]),
                                          float(rewards[e, i]), next_obs[e, i],
                                          d_e, buffer)

            # [NEW] rolling per-decision reward
            reward_window_dec.append(float(rewards.mean()))
            ep_rewards += rewards
            for e in np.where(env_done)[0]:
                reward_window_ep.append(float(ep_rewards[e].mean()))
                ep_rewards[e] = 0.0
            states = next_obs

            if buffer.size >= max(1, args.train_start):
                for _ in range(grad_steps):
                    s, a, r, ns, d, idxs, w = buffer.sample(args.batch_size)
                    s  = torch.as_tensor(s,  device=device)
                    a  = torch.as_tensor(a,  device=device)
                    r  = torch.as_tensor(r,  device=device)
                    ns = torch.as_tensor(ns, device=device)
                    d  = torch.as_tensor(d,  device=device)
                    w  = torch.as_tensor(w,  device=device)
                    qv = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        # Select the Double-DQN next action only from actions
                        # that are legal in the next state. Observation layout:
                        # queues, upstream queues, phase one-hot, elapsed-green.
                        phase_start = sim.n_dir * 2
                        next_phase = ns[:, phase_start:phase_start + sim.action_dim].argmax(1)
                        can_switch = (torch.ones_like(next_phase, dtype=torch.bool)
                                      if args.min_green <= 0 else
                                      ns[:, phase_start + sim.action_dim] >= 1.0 - 1e-6)
                        next_mask = F.one_hot(next_phase, num_classes=sim.action_dim).bool()
                        next_mask |= can_switch.unsqueeze(1)
                        next_q = q_net(ns).masked_fill(~next_mask, -torch.inf)
                        na = next_q.argmax(1)
                        nq = tgt_net(ns).gather(1, na.unsqueeze(1)).squeeze(1)
                        tq = r + (args.gamma ** args.n_step) * nq * (1.0 - d)
                    td = qv - tq
                    loss = (w * F.smooth_l1_loss(qv, tq, reduction="none")).mean()
                    optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                    optimiser.step()
                    if args.per:
                        buffer.update_priorities(idxs,
                            td.detach().abs().cpu().numpy() + 1e-6)
                    last_losses.append(float(loss.item()))
                    last_qs.append(float(qv.detach().mean().item()))
                q_net.reset_noise()

            if it % args.target_update_steps == 0:
                # Polyak averaging updates the bootstrap target smoothly
                # rather than replacing it with the online network abruptly.
                with torch.no_grad():
                    for target_param, online_param in zip(
                            tgt_net.parameters(), q_net.parameters()):
                        target_param.lerp_(online_param, args.target_tau)

            eval_m = {"R": float("nan"), "Q": float("nan"), "T": float("nan")}
            if ((args.eval_every > 0 and it % args.eval_every == 0)
                    or total_env_steps >= args.total_steps):
                q_net.eval()
                eval_m = evaluate(base_sim, None, ev_steps, device,
                                  q_net=q_net, eps=0.0, seed=args.seed + 99)
                q_net.train()
                if eval_m["R"] > best_eval_R:
                    best_eval_R = eval_m["R"]
                    best_eval_step = total_env_steps
                    torch.save({"q_net": q_net.state_dict(),
                                "args": vars(args),
                                "step": total_env_steps,
                                "eval_R": eval_m["R"]},
                               args.ckpt + ".best")

            if it % args.log_every == 0 or total_env_steps >= args.total_steps:
                avg_dec = float(np.mean(reward_window_dec)) if reward_window_dec else float("nan")
                avg_ep  = float(np.mean(reward_window_ep))  if reward_window_ep  else float("nan")
                sps     = total_env_steps / max(time.time() - t0, 1e-6)
                elapsed = time.time() - t0
                pct     = 100.0 * total_env_steps / args.total_steps
                eta     = max(0.0, (elapsed / max(1e-6, total_env_steps))
                              * (args.total_steps - total_env_steps))
                loss_avg = float(np.mean(last_losses)) if last_losses else float("nan")
                q_avg    = float(np.mean(last_qs))     if last_qs     else float("nan")

                writer.writerow([
                    total_env_steps, it, f"{eps:.4f}", buffer.size,
                    f"{loss_avg:.5f}", f"{q_avg:.4f}",
                    f"{avg_dec:.4f}", f"{avg_ep:.4f}", f"{sps:.1f}",
                    f"{eval_m['R']:.4f}", f"{eval_m['Q']:.4f}", f"{eval_m['T']:.4f}",
                    f"{fix_R:.4f}", f"{rand_R:.4f}", f"{mp_R:.4f}"])
                log_csv.flush()
                if log_jsonl:
                    log_jsonl.write(json.dumps({
                        "step": total_env_steps, "iter": it, "eps": eps,
                        "buf_size": buffer.size, "loss": loss_avg,
                        "avg_q": q_avg, "avg_dec_R": avg_dec, "avg_ep_R": avg_ep,
                        "sps": sps, "eval_R": eval_m["R"], "eval_Q": eval_m["Q"],
                        "eval_T": eval_m["T"],
                        "fix_R": fix_R, "rand_R": rand_R, "mp_R": mp_R,
                    }) + "\n")
                    log_jsonl.flush()
                if tb_writer:
                    tb_writer.add_scalar("train/avg_dec_R", avg_dec, total_env_steps)
                    if not math.isnan(avg_ep):
                        tb_writer.add_scalar("train/avg_ep_R", avg_ep, total_env_steps)
                    tb_writer.add_scalar("train/loss", loss_avg, total_env_steps)
                    tb_writer.add_scalar("train/avg_q", q_avg, total_env_steps)
                    tb_writer.add_scalar("train/eps", eps, total_env_steps)
                    tb_writer.add_scalar("train/sps", sps, total_env_steps)
                    if not math.isnan(eval_m["R"]):
                        tb_writer.add_scalar("eval/R",  eval_m["R"], total_env_steps)
                        tb_writer.add_scalar("eval/Q",  eval_m["Q"], total_env_steps)
                        tb_writer.add_scalar("eval/T",  eval_m["T"], total_env_steps)

                print(f"[{total_env_steps:>7d}/{args.total_steps}  {pct:5.1f}%  "
                      f"it={it:<5d}  eps={eps:.2f}  loss={loss_avg:7.3f}  "
                      f"q={q_avg:7.3f}  R_dec={avg_dec:+7.3f}  "
                      f"R_ep={(f'{avg_ep:+7.3f}' if not math.isnan(avg_ep) else '    n/a')}"
                      f"  buf={buffer.size:>7d}  sps={sps:5.0f}  ETA={eta:5.0f}s"
                      + (f"  |  eval R={eval_m['R']:+.3f} Q={eval_m['Q']:.2f}"
                         if not math.isnan(eval_m["R"]) else ""))

                if lock is not None: lock.touch()

            if args.ckpt_every > 0 and it % args.ckpt_every == 0:
                torch.save({"q_net": q_net.state_dict(),
                            "args": vars(args),
                            "step": total_env_steps}, args.ckpt)

        torch.save({"q_net": q_net.state_dict(), "args": vars(args),
                    "step": total_env_steps, "eval_R": best_eval_R}, args.ckpt)
        print(f"[done] final ckpt: {args.ckpt}")
        print(f"[done] best  ckpt: {args.ckpt}.best  (eval R={best_eval_R:+.3f} "
              f"@ step {best_eval_step})")
        print(f"[done] log csv:  {args.log_csv}")

        # [NEW] final comparison
        q_net.eval()
        final_m = evaluate(base_sim, None, ev_steps, device,
                           q_net=q_net, eps=0.0, seed=args.seed + 999)
        q_net.train()
        print("\n" + "=" * 68)
        print("FINAL COMPARISON  (mean per-decision reward, higher is better)")
        print("=" * 68)
        for name, val in [("fixed-time", fix_R), ("random", rand_R),
                          ("max-pressure", mp_R), ("DQN agent (best)", best_eval_R),
                          ("DQN agent (final)", final_m["R"])]:
            print(f"  {name:<22} {val:+8.4f}")
        delta_mp   = (best_eval_R - mp_R)   / abs(mp_R)   * 100 if mp_R   else 0.0
        delta_fix  = (best_eval_R - fix_R)  / abs(fix_R)  * 100 if fix_R  else 0.0
        print("-" * 68)
        print(f"  agent vs fixed-time:   {delta_fix:+6.2f}%")
        print(f"  agent vs max-pressure: {delta_mp:+6.2f}%")
        if best_eval_R <= max(fix_R, mp_R):
            print("[note] agent did not beat the strongest baseline. "
                  "Try --total-steps 1_000_000 or --num-envs 32 for a longer run.")
        print("=" * 68 + "\n")
    finally:
        if log_jsonl: log_jsonl.close()
        log_csv.close()
        if tb_writer: tb_writer.close()


# ============================================================================ #
# CLI
# ============================================================================ #
def resolve_flow_file(path: str) -> str:
    """Try the given path; fall back to its basename in the cwd.  [NEW]"""
    path = os.path.expanduser(path)
    if os.path.exists(path):
        return path
    # ntpath handles Windows-style input paths even when running on POSIX.
    base = ntpath.basename(path)
    if base != path and os.path.isfile(base):
        print(f"[warn] {path!r} not found; using {base!r} from cwd")
        return base
    raise FileNotFoundError(
        f"flow file not found: {path!r}\n"
        f"  (looked in cwd: {os.getcwd()})")


def validate_args(args):
    positive_ints = (
        "num_envs", "total_steps", "action_steps", "episode_decisions",
        "batch_size", "buffer_size", "n_step", "hidden", "log_every",
        "target_update_steps", "eval_steps", "fix_period",
    )
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.buffer_size < args.batch_size:
        raise ValueError("--buffer-size must be at least --batch-size")
    if args.train_start < 0 or args.train_start > args.buffer_size:
        raise ValueError("--train-start must be between 0 and --buffer-size")
    if args.min_green < 0 or args.grad_steps < 0 or args.ckpt_every < 0:
        raise ValueError("--min-green, --grad-steps, and --ckpt-every cannot be negative")
    if args.eval_every < 0:
        raise ValueError("--eval-every cannot be negative")
    for name in ("dt", "discharge_rate", "queue_capacity", "lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be between 0 and 1")
    if not (0.0 <= args.eps_end <= args.eps_start <= 1.0):
        raise ValueError("epsilon values must satisfy 0 <= --eps-end <= --eps-start <= 1")
    if not math.isfinite(args.target_tau) or not 0.0 < args.target_tau <= 1.0:
        raise ValueError("--target-tau must be finite and in (0, 1]")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow-file", default="flow_3_4.json")
    p.add_argument("--action-steps", type=int, default=5)
    p.add_argument("--episode-decisions", type=int, default=100)
    p.add_argument("--discharge-rate", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--min-green", type=int, default=2)
    p.add_argument("--queue-capacity", type=float, default=40.0)
    p.add_argument("--w-pressure",   type=float, default=0.05)
    p.add_argument("--w-throughput", type=float, default=1.0)
    p.add_argument("--w-switch",     type=float, default=0.5)
    p.add_argument("--num-envs",    type=int, default=210)
    p.add_argument("--total-steps", type=int, default=1_000_000,
                   help="total ENV-steps (decisions per env).  100k over 128 envs "
                        "= 781 iterations.  [semantics clarified]")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--train-start", type=int, default=10_000)
    p.add_argument("--grad-steps",  type=int, default=0)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--hidden",      type=int, default=128)
    p.add_argument("--eps-start",   type=float, default=1.0)
    p.add_argument("--eps-end",     type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=300_000)
    p.add_argument("--target-update-steps", type=int, default=1_000)
    p.add_argument("--target-tau", type=float, default=0.005,
                   help="Polyak target-network update rate (0 < tau <= 1)")
    p.add_argument("--no-noisy", action="store_true")
    p.add_argument("--no-per",   action="store_true")
    p.add_argument("--per-alpha",    type=float, default=0.6)
    p.add_argument("--per-beta",     type=float, default=0.4)
    p.add_argument("--per-beta-end", type=float, default=1.0)
    p.add_argument("--n-step",       type=int,   default=3)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--fix-period", type=int, default=5)
    p.add_argument("--log-every",   type=int, default=50)
    p.add_argument("--log-csv",     default="train_log.csv")
    p.add_argument("--log-jsonl",   default=None)
    p.add_argument("--tb-dir",      default=None)
    p.add_argument("--ckpt",        default="final.pt")
    p.add_argument("--ckpt-every",  type=int, default=0)
    p.add_argument("--plot-after", action="store_true")
    p.add_argument("--plot-out",   default=None)
    p.add_argument("--plot-smooth", type=float, default=0.85)
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        p.error(str(exc))
    return args


def main():
    args = parse_args()
    if args.dry_run:
        args.num_envs          = min(8, args.num_envs)
        args.total_steps       = min(2_000, args.total_steps)
        args.episode_decisions = min(15, args.episode_decisions)
        args.train_start       = 200
        args.batch_size        = 64
        args.log_every         = 5
        args.eval_every        = 50
        args.eval_steps        = 30
        args.grad_steps        = 2
        args.ckpt              = "dry_run.pt"
        args.log_csv           = "dry_run_log.csv"
        args.target_update_steps = 50
        args.eps_decay_steps   = max(1, args.total_steps)
        args.min_green         = min(2, args.min_green)
        print("[dry-run] overrides applied")

    args.flow_file = resolve_flow_file(args.flow_file)     # [NEW]

    lock_path = os.path.splitext(args.ckpt)[0] + ".lock"
    lock = AdvancedTrainingLock(lock_path)

    try:
        with lock:
            train(args, lock=lock)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return

    if args.plot_after:
        out_png = args.plot_out or (os.path.splitext(args.ckpt)[0] + "_curves.png")
        plot_training_curves(args.log_csv, out_png=out_png,
                             smooth=args.plot_smooth)


if __name__ == "__main__":
    main()

    