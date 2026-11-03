#!/usr/bin/env python3
"""Demonstrate people/software/hardware factor normalization on synthetic data.

No employee records or production telemetry are read. The default 3,000
samples are generated deterministically to provide a repeatable, runnable demo.

    python examples/factor_normalization_demo.py
    python examples/factor_normalization_demo.py --samples 100 --seed 7
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

# Allow running the script directly from a checkout without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cityflow_features.factor_normalization import default_normalizers


def make_synthetic_record(rng: random.Random):
    """Generate bounded illustrative values, not empirical measurements."""
    return {
        "staffing_coverage": rng.uniform(0.55, 1.0),
        "workload_ratio": rng.uniform(0.55, 1.6),
        "skill_match": rng.uniform(0.45, 1.0),
        "team_retention_rate": rng.uniform(0.65, 0.99),
        "service_availability": rng.uniform(0.96, 0.9999),
        "latency_ms": rng.uniform(8.0, 350.0),
        "error_rate": rng.uniform(0.0001, 0.08),
        "throughput_rps": rng.uniform(100.0, 9500.0),
        "hardware_availability": rng.uniform(0.95, 0.9999),
        "cpu_utilization_pct": rng.uniform(10.0, 98.0),
        "memory_utilization_pct": rng.uniform(15.0, 96.0),
        "temperature_c": rng.uniform(25.0, 90.0),
    }


def run_demo(samples: int = 3000, seed: int = 42):
    if samples < 1:
        raise ValueError("samples must be at least one")
    rng = random.Random(seed)
    normalizers = default_normalizers()
    scores = {group: [] for group in normalizers}

    for _ in range(samples):
        record = make_synthetic_record(rng)
        for group, normalizer in normalizers.items():
            score = normalizer.score(record)
            scores[group].append(score["composite"])

    print("Operational factor normalization demo")
    print(f"Synthetic samples: {samples} | seed: {seed}")
    print("Scores are illustrative normalized weighted means on [0, 1].")
    for group, values in scores.items():
        print(
            f"{group:18s} mean={statistics.fmean(values):.3f} "
            f"min={min(values):.3f} max={max(values):.3f}"
        )

    example = make_synthetic_record(random.Random(seed))
    print("\nOne sample (normalized factor values):")
    for group, normalizer in normalizers.items():
        result = normalizer.score(example)
        formatted = ", ".join(
            f"{name}={value:.3f}" for name, value in result["normalized"].items()
        )
        print(f"{group}: composite={result['composite']:.3f}; {formatted}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3000, help="Synthetic records to score.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic demo seed.")
    args = parser.parse_args(argv)
    try:
        run_demo(args.samples, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
