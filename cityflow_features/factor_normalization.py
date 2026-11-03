#!/usr/bin/env python3
"""Reusable, dependency-free normalization for operational factors.

This module provides explainable 0..1 scores for aggregate people-management,
software, and hardware indicators. It uses synthetic or organization-approved
aggregate inputs only; it is not an access-control/authorization system, an
employee assessment tool, or a substitute for validating metric definitions.

Run the demo from the repository root with:
    python examples/factor_normalization_demo.py --samples 3000
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence
import math


@dataclass(frozen=True)
class FactorSpec:
    """Definition of one numeric factor and its normalization behavior.

    ``direction`` can be ``benefit`` (higher is better), ``cost`` (lower is
    better), or ``target`` (near ``target`` is best). Raw values are clipped to
    [minimum, maximum] before scoring, so outliers cannot produce scores outside
    [0, 1]. Weights are nonnegative and are renormalized over present values.
    """

    name: str
    minimum: float
    maximum: float
    direction: str = "benefit"
    weight: float = 1.0
    target: Optional[float] = None

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("factor name must not be empty")
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError(f"{self.name}: bounds must be finite")
        if self.maximum <= self.minimum:
            raise ValueError(f"{self.name}: maximum must be greater than minimum")
        if self.direction not in {"benefit", "cost", "target"}:
            raise ValueError(f"{self.name}: unsupported direction {self.direction!r}")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError(f"{self.name}: weight must be finite and nonnegative")
        if self.direction == "target":
            if self.target is None or not math.isfinite(self.target):
                raise ValueError(f"{self.name}: target direction requires a finite target")
            if not self.minimum <= self.target <= self.maximum:
                raise ValueError(f"{self.name}: target must lie within the bounds")

    def normalize(self, value: float) -> float:
        """Map a raw value to a bounded desirability score in [0, 1]."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{self.name}: value must be finite")
        bounded = min(self.maximum, max(self.minimum, value))
        span = self.maximum - self.minimum
        if self.direction == "benefit":
            score = (bounded - self.minimum) / span
        elif self.direction == "cost":
            score = (self.maximum - bounded) / span
        else:
            # Linear decline from the optimum to the furthest bound.
            distance = abs(bounded - self.target)
            furthest = max(self.target - self.minimum, self.maximum - self.target)
            score = 1.0 - distance / furthest if furthest else 1.0
        return min(1.0, max(0.0, score))


class FactorNormalizer:
    """Normalize a factor group and calculate a weighted composite score."""

    def __init__(self, specs: Sequence[FactorSpec], *, missing: str = "skip"):
        if not specs:
            raise ValueError("at least one factor specification is required")
        if missing not in {"skip", "error"}:
            raise ValueError("missing must be either 'skip' or 'error'")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("factor names must be unique within a normalizer")
        if not any(spec.weight > 0 for spec in specs):
            raise ValueError("at least one factor must have a positive weight")
        self.specs = tuple(specs)
        self.missing = missing
        self._by_name = {spec.name: spec for spec in specs}

    @property
    def factor_names(self):
        return tuple(self._by_name)

    def normalize(self, values: Mapping[str, float]) -> Dict[str, Optional[float]]:
        """Return normalized values; absent factors are None when skip is used."""
        unknown = set(values) - set(self._by_name)
        if unknown:
            raise ValueError(f"unknown factor(s): {', '.join(sorted(unknown))}")
        result: Dict[str, Optional[float]] = {}
        for spec in self.specs:
            if spec.name not in values or values[spec.name] is None:
                if self.missing == "error":
                    raise ValueError(f"missing required factor: {spec.name}")
                result[spec.name] = None
            else:
                result[spec.name] = spec.normalize(values[spec.name])
        return result

    def score(self, values: Mapping[str, float]) -> Dict[str, object]:
        """Return per-factor scores and a weighted mean over available factors."""
        normalized = self.normalize(values)
        weighted_sum = 0.0
        present_weight = 0.0
        for spec in self.specs:
            score = normalized[spec.name]
            if score is not None and spec.weight > 0:
                weighted_sum += score * spec.weight
                present_weight += spec.weight
        return {
            "normalized": normalized,
            "composite": weighted_sum / present_weight if present_weight else None,
            "factors_used": sum(value is not None for value in normalized.values()),
        }


# These are example operational indicators, not universal policy thresholds.
# Set ranges and targets to values justified by the local data, context, and
# stakeholders before using these scores in any decision process.
PEOPLE_MANAGEMENT_FACTORS = (
    FactorSpec("staffing_coverage", 0.0, 1.0, "benefit", 1.0),
    FactorSpec("workload_ratio", 0.25, 2.0, "target", 1.0, target=1.0),
    FactorSpec("skill_match", 0.0, 1.0, "benefit", 1.0),
    FactorSpec("team_retention_rate", 0.0, 1.0, "benefit", 1.0),
)

SOFTWARE_FACTORS = (
    FactorSpec("service_availability", 0.0, 1.0, "benefit", 1.0),
    FactorSpec("latency_ms", 0.0, 1000.0, "cost", 1.0),
    FactorSpec("error_rate", 0.0, 1.0, "cost", 1.0),
    FactorSpec("throughput_rps", 0.0, 10000.0, "benefit", 1.0),
)

HARDWARE_FACTORS = (
    FactorSpec("hardware_availability", 0.0, 1.0, "benefit", 1.0),
    FactorSpec("cpu_utilization_pct", 0.0, 100.0, "target", 1.0, target=70.0),
    FactorSpec("memory_utilization_pct", 0.0, 100.0, "target", 1.0, target=70.0),
    FactorSpec("temperature_c", 20.0, 100.0, "cost", 1.0),
)


def default_normalizers(*, missing: str = "skip") -> Dict[str, FactorNormalizer]:
    """Create the standard people/software/hardware normalizers."""
    return {
        "people_management": FactorNormalizer(PEOPLE_MANAGEMENT_FACTORS, missing=missing),
        "software": FactorNormalizer(SOFTWARE_FACTORS, missing=missing),
        "hardware": FactorNormalizer(HARDWARE_FACTORS, missing=missing),
    }


def normalize_operational_record(record: Mapping[str, float]) -> Dict[str, Dict[str, object]]:
    """Score available standard factor groups from one flat input mapping."""
    normalizers = default_normalizers()
    return {
        group: normalizer.score({key: record[key] for key in normalizer.factor_names if key in record})
        for group, normalizer in normalizers.items()
    }


__all__ = [
    "FactorSpec",
    "FactorNormalizer",
    "PEOPLE_MANAGEMENT_FACTORS",
    "SOFTWARE_FACTORS",
    "HARDWARE_FACTORS",
    "default_normalizers",
    "normalize_operational_record",
]
