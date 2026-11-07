#!/usr/bin/env python3
"""
improved_datagram_generator.py
==============================
Secure-V2X datagram + adaptive traffic-signal simulator.

Improvements over the previous version
--------------------------------------
1.  **Dedicated RNG** — the generator threads a single ``random.Random``
    instance through every stochastic component, so ``--seed`` is
    reproducible regardless of global ``random`` state.
2.  **Validated configs** — dataclasses reject bad parameters in
    ``__post_init__``.
3.  **Named constants** — magic numbers now live at module level.
4.  **``RowRecord`` / ``RunStats`` dataclasses** — cleaner row assembly,
    summary printing, and an explicit mapping from Python attribute →
    CSV column.
5.  **Robust visualiser** — handles empty / short data, non-finite
    residual-BER values, and rolling windows longer than the dataset.
6.  **CLI knobs** — the key ``SimulationConfig`` tunables are exposed.
7.  **``pathlib.Path``** for the output file and no unused imports.

Usage
-----
    python improved_datagram_generator.py --rows 3121 --plot
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. CONSTANTS (named magic numbers)
# ============================================================

PAYLOAD_CHOICES: Tuple[int, ...] = (64, 256, 1024, 2048)
ECC_CHOICES: Tuple[int, ...] = (0, 1, 2)
DEFAULT_SEED: int = 42

# Arrival model
ARRIVAL_BASE: float = 2.5
ARRIVAL_WAVE_AMPLITUDE: float = 0.3
ARRIVAL_WAVE_PERIOD: int = 50
ARRIVAL_NOISE_STD: float = 0.8
ARRIVAL_FLOOR: float = 0.5

# Service model
SWITCHED_SERVICE: float = 10.0
HELD_SERVICE: float = 2.0
LATENCY_PENALTY_PER_MS: float = 0.08
LATENCY_PENALTY_FLOOR: float = 0.6

# Reward model
R_BASE: float = 10.0
R_QUEUE: float = 0.5
R_SPILL: float = 5.0
R_INVALID: float = 50.0
R_SWITCH: float = 0.5
R_VARIANCE: float = 0.02

# Crypto
HMAC_KEY: bytes = b"secure_key_3121"
NONCE_HEX_LEN: int = 12
SIG_HEX_LEN: int = 16
MIN_LATENCY_MS: float = 0.001

# Progress logging
PROGRESS_EVERY: int = 500


# ============================================================
# 2. CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CryptoConfig:
    """One crypto-algorithm profile."""
    name: str
    flops_per_byte: float
    base_latency_ms: float
    jitter_ms: float = 0.01

    def __post_init__(self) -> None:
        for fname, val in (
            ("flops_per_byte", self.flops_per_byte),
            ("base_latency_ms", self.base_latency_ms),
            ("jitter_ms", self.jitter_ms),
        ):
            if val < 0:
                raise ValueError(f"{fname} must be >= 0 (got {val})")


@dataclass(frozen=True)
class NaturalConfig:
    """One natural-condition profile."""
    name: str
    capacity: float
    snr_db: float
    demand_multiplier: float

    def __post_init__(self) -> None:
        if not (0 < self.capacity <= 2.0):
            raise ValueError(f"capacity must be in (0, 2], got {self.capacity}")
        if not (0 < self.demand_multiplier <= 2.0):
            raise ValueError(f"demand_multiplier must be in (0, 2], "
                             f"got {self.demand_multiplier}")


@dataclass
class SimulationConfig:
    """Tunables for the whole simulation."""
    target_rows: int = 3121
    num_intersections: int = 10
    max_queue_per_intersection: float = 100.0
    min_green_time: int = 3
    initial_queue: float = 15.0
    initial_queue_jitter: float = 3.0
    epsilon: float = 0.10
    pressure_threshold: float = 15.0
    realtime_latency_limit_ms: float = 5.0
    seed: int = DEFAULT_SEED

    crypto_methods: List[CryptoConfig] = field(default_factory=lambda: [
        CryptoConfig("None",            0.0,   0.010, 0.001),
        CryptoConfig("AES-128-GCM",    25.0,   0.050, 0.005),
        CryptoConfig("AES-256-GCM",    45.0,   0.080, 0.008),
        CryptoConfig("ECC-P256-ECDSA", 200.0,  0.250, 0.020),
    ])

    naturals: List[NaturalConfig] = field(default_factory=lambda: [
        NaturalConfig("N-clear",    1.00, 20.0, 1.00),
        NaturalConfig("N-rain",     0.85, 12.0, 0.95),
        NaturalConfig("N-snow",     0.70,  8.0, 0.75),
        NaturalConfig("N-accident", 0.60, 10.0, 0.70),
    ])

    def __post_init__(self) -> None:
        if self.target_rows <= 0:
            raise ValueError("target_rows must be > 0")
        if self.num_intersections <= 0:
            raise ValueError("num_intersections must be > 0")
        if self.max_queue_per_intersection <= 0:
            raise ValueError("max_queue_per_intersection must be > 0")
        if self.min_green_time < 0:
            raise ValueError("min_green_time must be >= 0")
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        if self.realtime_latency_limit_ms <= 0:
            raise ValueError("realtime_latency_limit_ms must be > 0")
        if not self.crypto_methods:
            raise ValueError("crypto_methods must not be empty")
        if not self.naturals:
            raise ValueError("naturals must not be empty")


# ============================================================
# 3. CRYPTO LAYER
# ============================================================

class CryptoLayer:
    """Deterministic-but-jittered crypto latency + HMAC signing."""

    @staticmethod
    def process_payload(
        method: CryptoConfig,
        payload_size: int,
        rng: random.Random,
    ) -> Tuple[float, float, str, str]:
        base = method.base_latency_ms + (
            payload_size * 0.0001
        ) * (1 + math.log1p(payload_size))
        latency = max(
            MIN_LATENCY_MS,
            base + rng.uniform(-method.jitter_ms, method.jitter_ms),
        )
        flops = method.flops_per_byte * payload_size

        nonce = hashlib.sha256(
            str(rng.random()).encode()
        ).hexdigest()[:NONCE_HEX_LEN]

        payload = f"{method.name}-{payload_size}-{nonce}".encode("utf-8")
        signature = hmac.new(HMAC_KEY, payload, hashlib.sha256).hexdigest()[
            :SIG_HEX_LEN
        ]
        return round(latency, 4), round(flops, 2), nonce, signature


# ============================================================
# 4. PHYSICAL LAYER
# ============================================================

class PhysicalLayer:
    """Realistic ECC — Hamming and RS both fail at high BER."""

    @staticmethod
    def calculate_ber(snr_db: float) -> float:
        snr_linear = 10.0 ** (snr_db / 10.0)
        if snr_linear < 1e-6:
            return 0.5
        return 0.5 * math.erfc(math.sqrt(snr_linear))

    @classmethod
    def simulate_ecc(
        cls,
        snr_db: float,
        ecc_strength: int,
        payload_bytes: int,
        rng: random.Random,
    ) -> Tuple[bool, float, float]:
        """Return (success, effective_ber, packet_error_prob)."""
        base_ber = cls.calculate_ber(snr_db)
        total_bits = payload_bytes * 8

        if ecc_strength == 0:
            effective_ber = base_ber
        elif ecc_strength == 1:
            effective_ber = base_ber * 0.5
        else:
            effective_ber = base_ber ** 2

        packet_error_prob = 1.0 - (1.0 - effective_ber) ** total_bits
        is_success = rng.random() > packet_error_prob
        return is_success, effective_ber, packet_error_prob


# ============================================================
# 5. PER-INTERSECTION TRAFFIC PHYSICS
# ============================================================

class TrafficPhysics:
    """Each intersection tracks its own queue; switch = priority surge."""

    def __init__(self, num_intersections: int, max_queue: float) -> None:
        self.num_intersections = num_intersections
        self.max_queue = max_queue

    def update(
        self,
        queues: List[float],
        arrivals: List[float],
        capacity: float,
        latency_ms: float,
        actions: List[int],
    ) -> Tuple[List[float], float]:
        latency_penalty = max(
            LATENCY_PENALTY_FLOOR,
            1.0 - latency_ms * LATENCY_PENALTY_PER_MS,
        )
        next_queues: List[float] = []
        spilled = 0.0

        for i in range(self.num_intersections):
            service_mult = SWITCHED_SERVICE if actions[i] == 1 else HELD_SERVICE
            service = service_mult * capacity * latency_penalty
            new_q = queues[i] + arrivals[i] - service
            if new_q > self.max_queue:
                spilled += new_q - self.max_queue
                new_q = self.max_queue
            elif new_q < 0.0:
                new_q = 0.0
            next_queues.append(new_q)

        return next_queues, spilled


# ============================================================
# 6. PRESSURE-BASED AGENT
# ============================================================

class TrafficAgent:
    """Switches when local pressure > threshold; ε-greedy exploration."""

    def __init__(
        self,
        num_intersections: int,
        min_green: int,
        epsilon: float,
        threshold: float,
        rng: random.Random,
    ) -> None:
        self.N = num_intersections
        self.min_green = min_green
        self.epsilon = epsilon
        self.threshold = threshold
        self.rng = rng
        self.phases: List[int] = [0] * num_intersections
        self.time_since_switch: List[int] = [0] * num_intersections

    def get_action_mask(self) -> List[List[int]]:
        return [
            [0, 1] if self.time_since_switch[i] >= self.min_green else [0]
            for i in range(self.N)
        ]

    def select_action(
        self,
        queues: List[float],
        masks: List[List[int]],
    ) -> List[int]:
        actions: List[int] = []
        for i in range(self.N):
            if queues[i] > self.threshold and 1 in masks[i]:
                actions.append(1)
            elif self.rng.random() < self.epsilon:
                actions.append(self.rng.choice(masks[i]))
            else:
                actions.append(0)

        for i, a in enumerate(actions):
            if a == 1:
                self.phases[i] = (self.phases[i] + 1) % 2
                self.time_since_switch[i] = 0
            else:
                self.time_since_switch[i] += 1
        return actions


# ============================================================
# 7. ROW + STATS
# ============================================================

CSV_HEADERS: Tuple[str, ...] = (
    "Step", "Natural_Condition", "SNR_dB", "Encryption_Method", "Payload_Bytes",
    "Crypto_Latency_MS", "Crypto_Flops", "Nonce", "Signature_HMAC",
    "Action_Taken", "Switches", "ECC_Strength", "ECC_Success", "Residual_BER",
    "Packet_Error_Prob", "Queues_Per_Intersection", "Phases_Per_Intersection",
    "Total_Queue", "Spilled_Vehicles", "Reward", "Datagram_Valid",
)


@dataclass(frozen=True)
class RowRecord:
    """A single row of the dataset. Serialised via :meth:`to_csv_dict`."""
    step: int
    natural_condition: str
    snr_db: float
    encryption_method: str
    payload_bytes: int
    crypto_latency_ms: float
    crypto_flops: float
    nonce: str
    signature_hmac: str
    action_taken: Tuple[int, ...]
    switches: int
    ecc_strength: int
    ecc_success: bool
    residual_ber: float
    packet_error_prob: float
    queues: Tuple[float, ...]
    phases: Tuple[int, ...]
    total_queue: float
    spilled_vehicles: float
    reward: float
    is_valid: bool

    def to_csv_dict(self) -> Dict[str, object]:
        return {
            "Step": self.step,
            "Natural_Condition": self.natural_condition,
            "SNR_dB": self.snr_db,
            "Encryption_Method": self.encryption_method,
            "Payload_Bytes": self.payload_bytes,
            "Crypto_Latency_MS": self.crypto_latency_ms,
            "Crypto_Flops": self.crypto_flops,
            "Nonce": self.nonce,
            "Signature_HMAC": self.signature_hmac,
            "Action_Taken": json.dumps(list(self.action_taken)),
            "Switches": self.switches,
            "ECC_Strength": self.ecc_strength,
            "ECC_Success": self.ecc_success,
            "Residual_BER": f"{self.residual_ber:.6e}",
            "Packet_Error_Prob": f"{self.packet_error_prob:.4f}",
            "Queues_Per_Intersection": json.dumps(
                [round(q, 2) for q in self.queues]
            ),
            "Phases_Per_Intersection": json.dumps(list(self.phases)),
            "Total_Queue": round(self.total_queue, 2),
            "Spilled_Vehicles": round(self.spilled_vehicles, 2),
            "Reward": round(self.reward, 2),
            "Datagram_Valid": self.is_valid,
        }


@dataclass
class RunStats:
    """Rolling counters accumulated while generating the dataset."""
    valid: int = 0
    dropped: int = 0
    reward_sum: float = 0.0
    switches_total: int = 0
    final_q: float = 0.0

    def record(self, row: RowRecord) -> None:
        if row.is_valid:
            self.valid += 1
        else:
            self.dropped += 1
        self.reward_sum += row.reward
        self.switches_total += row.switches
        self.final_q = row.total_queue


# ============================================================
# 8. GENERATOR
# ============================================================

class DatagramGenerator:
    """End-to-end simulator: crypto → channel → traffic → reward."""

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config
        # Isolated RNG — does NOT touch the module-level ``random`` state.
        self.rng = random.Random(config.seed)

        self.physics = TrafficPhysics(
            config.num_intersections,
            config.max_queue_per_intersection,
        )
        self.agent = TrafficAgent(
            config.num_intersections,
            config.min_green_time,
            config.epsilon,
            config.pressure_threshold,
            self.rng,
        )
        self.queues: List[float] = [
            max(
                0.0,
                config.initial_queue + self.rng.uniform(
                    -config.initial_queue_jitter,
                    config.initial_queue_jitter,
                ),
            )
            for _ in range(config.num_intersections)
        ]

    # -- internals --------------------------------------------------------

    def _sample_arrivals(
        self,
        nat: NaturalConfig,
        step: int,
    ) -> List[float]:
        wave = 1.0 + ARRIVAL_WAVE_AMPLITUDE * math.sin(
            step / ARRIVAL_WAVE_PERIOD
        )
        base = ARRIVAL_BASE * nat.demand_multiplier * wave
        return [
            max(ARRIVAL_FLOOR, self.rng.gauss(base, ARRIVAL_NOISE_STD))
            for _ in range(self.cfg.num_intersections)
        ]

    @staticmethod
    def _compute_reward(
        next_queues: List[float],
        spilled: float,
        is_valid: bool,
        switches: int,
    ) -> float:
        n = max(1, len(next_queues))
        avg_q = sum(next_queues) / n
        var = sum((q - avg_q) ** 2 for q in next_queues) / n
        return (
            R_BASE
            - avg_q * R_QUEUE
            - spilled * R_SPILL
            - (0.0 if is_valid else R_INVALID)
            + switches * R_SWITCH
            - var * R_VARIANCE
        )

    def _build_row(
        self,
        step: int,
        nat: NaturalConfig,
        enc: CryptoConfig,
        payload: int,
        ecc: int,
        latency: float,
        flops: float,
        nonce: str,
        sig: str,
        ecc_ok: bool,
        ber: float,
        pkt_err: float,
        action: List[int],
        next_queues: List[float],
        spilled: float,
        reward: float,
    ) -> RowRecord:
        is_valid = (
            ecc_ok
            and latency < self.cfg.realtime_latency_limit_ms
        )
        return RowRecord(
            step=step,
            natural_condition=nat.name,
            snr_db=nat.snr_db,
            encryption_method=enc.name,
            payload_bytes=payload,
            crypto_latency_ms=latency,
            crypto_flops=flops,
            nonce=nonce,
            signature_hmac=sig,
            action_taken=tuple(action),
            switches=sum(action),
            ecc_strength=ecc,
            ecc_success=ecc_ok,
            residual_ber=ber,
            packet_error_prob=pkt_err,
            queues=tuple(next_queues),
            phases=tuple(self.agent.phases),
            total_queue=sum(next_queues),
            spilled_vehicles=spilled,
            reward=reward,
            is_valid=is_valid,
        )

    # -- public API -------------------------------------------------------

    def generate(self, out_path: Path) -> RunStats:
        """Stream ``cfg.target_rows`` datagrams into ``out_path``."""
        n = self.cfg.target_rows
        print(f"Generating {n} datagrams -> {out_path}")

        stats = RunStats()
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

            for step in range(n):
                nat = self.rng.choice(self.cfg.naturals)
                enc = self.rng.choice(self.cfg.crypto_methods)
                payload = self.rng.choice(PAYLOAD_CHOICES)
                ecc = self.rng.choice(ECC_CHOICES)

                latency, flops, nonce, sig = CryptoLayer.process_payload(
                    enc, payload, self.rng
                )
                ecc_ok, ber, pkt_err = PhysicalLayer.simulate_ecc(
                    nat.snr_db, ecc, payload, self.rng
                )

                arrivals = self._sample_arrivals(nat, step)
                masks = self.agent.get_action_mask()
                action = self.agent.select_action(self.queues, masks)

                next_queues, spilled = self.physics.update(
                    self.queues, arrivals, nat.capacity, latency, action,
                )
                reward = self._compute_reward(
                    next_queues, spilled, ecc_ok, sum(action),
                )

                row = self._build_row(
                    step=step + 1,
                    nat=nat, enc=enc, payload=payload, ecc=ecc,
                    latency=latency, flops=flops, nonce=nonce, sig=sig,
                    ecc_ok=ecc_ok, ber=ber, pkt_err=pkt_err,
                    action=action, next_queues=next_queues,
                    spilled=spilled, reward=reward,
                )
                writer.writerow(row.to_csv_dict())
                stats.record(row)
                self.queues = next_queues

                if (step + 1) % PROGRESS_EVERY == 0 or step + 1 == n:
                    pct = (step + 1) / n * 100.0
                    print(f"  {step + 1}/{n} ({pct:.1f}%)", flush=True)

        self._print_summary(out_path, stats)
        return stats

    def _print_summary(self, out_path: Path, stats: RunStats) -> None:
        n = self.cfg.target_rows
        print("\n" + "=" * 60)
        print(f"  FILE:            {out_path}")
        print(f"  Valid:           {stats.valid} ({stats.valid / n * 100:.1f}%)")
        print(f"  Dropped:         {stats.dropped} "
              f"({stats.dropped / n * 100:.1f}%)")
        print(f"  Total switches:  {stats.switches_total}")
        print(f"  Final queue:     {stats.final_q:.2f}")
        print(f"  Avg reward:      {stats.reward_sum / n:.2f}")
        print("=" * 60)


# ============================================================
# 9. VISUALIZATION
# ============================================================

class DatasetVisualizer:
    """Six-panel diagnostic view of the generated dataset."""

    @staticmethod
    def plot_all(filename: Path) -> None:
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return

        if df.empty:
            print("Dataset is empty — nothing to plot.")
            return

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle(f"Improved Datagram Dataset — {len(df)} rows", fontsize=16)

        DatasetVisualizer._plot_queue(axes[0, 0], df)
        DatasetVisualizer._plot_reward(axes[0, 1], df)
        DatasetVisualizer._plot_ber(axes[1, 0], df)
        DatasetVisualizer._plot_validity(axes[1, 1], df)
        DatasetVisualizer._plot_switches(axes[2, 0], df)
        DatasetVisualizer._plot_ecc(axes[2, 1], df)

        plt.tight_layout()
        out = filename.with_name("improved_datagram_analysis.png")
        plt.savefig(out, dpi=100)
        print(f"Plots -> {out}")
        plt.show()

    # -- panels -----------------------------------------------------------

    @staticmethod
    def _plot_queue(ax, df: pd.DataFrame) -> None:
        ax.plot(df["Step"], df["Total_Queue"], color="blue", linewidth=0.7)
        ax.set_title("Total Queue over Step")
        ax.set_xlabel("Step")
        ax.set_ylabel("Vehicles")
        ax.grid(True)

    @staticmethod
    def _plot_reward(ax, df: pd.DataFrame) -> None:
        ax.plot(df["Step"], df["Reward"], alpha=0.3, color="red")
        window = min(50, max(1, len(df)))
        rolling = df["Reward"].rolling(window, min_periods=1).mean()
        ax.plot(
            df["Step"], rolling, color="darkred", linewidth=2,
            label=f"{window}-step rolling mean",
        )
        ax.set_title("Reward over Step")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True)

    @staticmethod
    def _plot_ber(ax, df: pd.DataFrame) -> None:
        ber = df["Residual_BER"]
        # Replace non-positive values so log scale has something to show.
        ber = ber.where(ber > 0, 1e-12)
        sc = ax.scatter(
            df["SNR_dB"], ber, c=df["ECC_Strength"],
            cmap="viridis", alpha=0.5, s=10,
        )
        ax.set_yscale("log")
        ax.set_title("SNR vs Residual BER (by ECC)")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Residual BER")
        plt.colorbar(sc, ax=ax, label="ECC Strength")
        ax.grid(True, which="both", ls="--")

    @staticmethod
    def _plot_validity(ax, df: pd.DataFrame) -> None:
        vc = df["Datagram_Valid"].value_counts()
        ax.bar(
            ["Dropped", "Valid"],
            [vc.get(False, 0), vc.get(True, 0)],
            color=["crimson", "forestgreen"],
        )
        ax.set_title("Datagram Validity")
        ax.set_ylabel("Count")

    @staticmethod
    def _plot_switches(ax, df: pd.DataFrame) -> None:
        max_sw = int(df["Switches"].max()) if len(df) else 0
        ax.hist(
            df["Switches"], bins=range(0, max_sw + 2),
            color="steelblue", edgecolor="black", alpha=0.7,
        )
        ax.set_title("Distribution of Switches per Step")
        ax.set_xlabel("Intersections Switching")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_ecc(ax, df: pd.DataFrame) -> None:
        es = df.groupby("ECC_Strength")["ECC_Success"].mean() * 100.0
        ax.bar(
            es.index.astype(str), es.values,
            color=["crimson", "orange", "forestgreen"][: len(es)],
        )
        ax.set_title("ECC Success Rate by Strength")
        ax.set_xlabel("ECC (0=None, 1=Hamming, 2=RS)")
        ax.set_ylabel("Success %")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)


# ============================================================
# 10. CLI
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a secure-V2X datagram dataset.",
    )
    p.add_argument("--rows", type=int, default=3121,
                   help="number of datagrams to emit")
    p.add_argument("--out", type=Path, default=Path("improved_datagrams.csv"),
                   help="output CSV path")
    p.add_argument("--plot", action="store_true",
                   help="render the diagnostic PNG after generation")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed (default: %(default)s)")
    p.add_argument("--num-intersections", type=int, default=10,
                   help="number of controlled intersections")
    p.add_argument("--epsilon", type=float, default=0.10,
                   help="ε-greedy exploration rate")
    p.add_argument("--threshold", type=float, default=15.0,
                   help="queue threshold above which the agent switches")
    p.add_argument("--max-queue", type=float, default=100.0,
                   help="per-intersection queue capacity")
    p.add_argument("--min-green", type=int, default=3,
                   help="minimum green time (steps between switches)")
    p.add_argument("--latency-limit", type=float, default=5.0,
                   help="crypto latency budget (ms) before datagram is stale")
    return p


def main(argv: List[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    cfg = SimulationConfig(
        target_rows=args.rows,
        num_intersections=args.num_intersections,
        max_queue_per_intersection=args.max_queue,
        min_green_time=args.min_green,
        epsilon=args.epsilon,
        pressure_threshold=args.threshold,
        realtime_latency_limit_ms=args.latency_limit,
        seed=args.seed,
    )

    DatagramGenerator(cfg).generate(args.out)
    if args.plot:
        DatasetVisualizer.plot_all(args.out)


if __name__ == "__main__":
    main()