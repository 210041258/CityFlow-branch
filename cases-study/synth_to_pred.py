#!/usr/bin/env python3
"""
synthetic_dataset.py
====================
A closed-form, "made-up" dataset laid out as a 3-D tensor of shape
(128, 64, 12).  No traffic simulator is invoked — every cell is computed
analytically from the same multiplicative factors used by
``analytical_surrogate.py`` and perturbed by the same logistic signature
used by ``scenario_advanced.py``.

    Axis 0 — 128 controllers      CONTROLLERS[i]
    Axis 1 —  64 scenarios        SCENARIO_KEYS[j]   (8 lat × 8 loss)
    Axis 2 —  12 conditions       CONDITION_KEYS[k]

    total = 128 × 64 × 12 = 98 304 cells

Presets
-------
    quick      16 ×  8 ×  4       ≈    512 cells
    standard   64 × 32 ×  8       ≈ 16 384 cells
    full      128 × 64 × 12       = 98 304 cells     (default)

CLI
---
    python synthetic_dataset.py                    # full default run
    python synthetic_dataset.py --preset quick     # 16 × 8 × 4
    python synthetic_dataset.py --export           # write CSV
    python synthetic_dataset.py --dump-npy         # write .npy tensor
    python synthetic_dataset.py --slice 7          # inspect controller 7
    python synthetic_dataset.py --quiet --export   # silent batch mode

Programmatic
------------
    from synthetic_dataset import SyntheticTrafficDataset

    ds = SyntheticTrafficDataset()                     # full preset
    ds = SyntheticTrafficDataset(preset='quick')       # lighter
    ds = SyntheticTrafficDataset.default(seed=7)       # factory
    ds = SyntheticTrafficDataset(auto_build=True)      # build on init

    M = ds.build()                    # (n1, n2, n3) object array of dicts
    T = ds.as_tensor()                # (n1, n2, n3, F) numeric tensor
    rec = ds[7, 3, 11]                # one record
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
__all__ = [
    "SyntheticTrafficDataset",
    "Config",
    "PRESETS",
    "CONTROLLERS",
    "SCENARIO_KEYS",
    "CONDITION_KEYS",
    "CF_MODELS",
]


# ===========================================================================
#  CONFIG  —  every tunable in one place, validated on construction
# ===========================================================================
PRESETS: dict[str, dict[str, int | None]] = {
    "quick":    {"n_ctrl": 16,  "n_scen": 8,  "n_cond": 4,  "n_steps": 20},
    "standard": {"n_ctrl": 64,  "n_scen": 32, "n_cond": 8,  "n_steps": 50},
    "full":     {"n_ctrl": 128, "n_scen": 64, "n_cond": 12, "n_steps": None},
}

# Constants used by the factors — kept here so the class never refers to
# global magic numbers.
_CF_BENCHMARK = {
    "SAFE":  {"free_flow": {"in_system_s": 100.0},
              "congested": {"in_system_s": 230.0}},
    "GIPPS": {"free_flow": {"in_system_s": 105.0},
              "congested": {"in_system_s": 250.0}},
    "IDM":   {"free_flow": {"in_system_s": 102.0},
              "congested": {"in_system_s": 240.0}},
}
CF_MODELS = tuple(_CF_BENCHMARK)

SATURATION_FLOW_VPS = 0.5

K_CTRL_FAMILY = {
    "baseline":    1.26,
    "adaptive":    1.06,
    "coordinated": 1.00,
    "field":       0.98,
    "learning":    1.02,
    "resilient":   0.99,
}

RESILIENCE_FLATTENING = {
    "baseline":    1.00,
    "adaptive":    0.95,
    "coordinated": 0.90,
    "field":       0.85,
    "learning":    0.75,
    "resilient":   0.35,
}

_K_COMM_BASE_K     = 1.43 * 100.0
_K_COMM_BASE_X0    = 101.54 / 600.0
_K_COMM_BASE_L     = 0.60

_NUMERIC_COLS = (
    "travel_time_s", "queue_veh", "waiting_time_s", "throughput_vph",
    "mean_aoi_ms", "delivery_ratio",
    "K_ctrl", "K_comm", "K_nat", "K_dem",
    "severity", "true_value",
)


@dataclass
class Config:
    """All tunable parameters, validated at construction."""

    seed:          int   = 42
    car_following: str   = "SAFE"
    sample_cf:     bool  = False
    noise_frac:    float = 0.06
    outlier_frac:  float = 0.02
    outlier_mult:  float = 2.2
    preset:        str   = "full"
    csv_path:      str   = "synthetic_dataset.csv"
    npy_path:      str   = "synthetic_tensor.npy"
    progress:      bool  = True
    auto_build:    bool  = False
    verbose:       bool  = True

    def __post_init__(self) -> None:
        if self.preset not in PRESETS:
            raise ValueError(
                f"Unknown preset {self.preset!r}. "
                f"Choose from {tuple(PRESETS)}."
            )
        if self.car_following not in CF_MODELS:
            raise ValueError(
                f"Unknown car_following {self.car_following!r}. "
                f"Choose from {CF_MODELS}."
            )
        if not 0.0 <= self.noise_frac < 1.0:
            raise ValueError(f"noise_frac must be in [0, 1); got {self.noise_frac}.")
        if not 0.0 <= self.outlier_frac <= 1.0:
            raise ValueError(f"outlier_frac must be in [0, 1]; got {self.outlier_frac}.")
        if self.outlier_mult <= 0.0:
            raise ValueError(f"outlier_mult must be > 0; got {self.outlier_mult}.")


# ===========================================================================
#  AXIS 0 — 128 CONTROLLERS, built family-by-family
# ===========================================================================
_FAMILY_SPEC = {
    "baseline": {
        "count": 16,
        "bases": ["fixed-time", "actuated", "semi-actuated", "pretimed",
                  "gap-out", "cycle-72", "phase-split", "dual-ring"],
    },
    "adaptive": {
        "count": 16,
        "bases": ["adaptive", "adapt-p", "adapt-pi", "adapt-pid",
                  "adapt-q", "adapt-a3c", "adapt-lin", "adapt-nonlin"],
    },
    "coordinated": {
        "count": 16,
        "bases": ["coord", "coord-green", "offset-opt", "platoon",
                  "green-wave", "green-sync", "flow-sync", "bidir-wave"],
    },
    "field": {
        "count": 32,
        "bases": ["max-pressure", "scats", "scoot", "multiband",
                  "mpc", "fuzzy", "weather", "bike",
                  "transit", "emergency", "ped-adapt", "rideshare",
                  "max-flow", "split-opt", "cycle-opt", "offset-p"],
    },
    "learning": {
        "count": 24,
        "bases": ["idqn", "coop-marl", "ppo-marl", "sac-marl",
                  "td3-marl", "a2c-marl", "ddpg-marl", "rainbow",
                  "iqn", "cql", "iql", "bc-marl"],
    },
    "resilient": {
        "count": 24,
        "bases": ["res-marl", "res-dqn", "res-ppo", "byz-robust",
                  "fed-marl", "sec-agg", "robust-td3", "res-sac",
                  "res-td3", "res-a2c", "res-ddpg", "res-iqn"],
    },
}
assert sum(s["count"] for s in _FAMILY_SPEC.values()) == 128


def _build_controllers() -> dict[str, str]:
    """Return an ordered mapping ``{controller_name: family}``."""
    out: dict[str, str] = {}
    for family, spec in _FAMILY_SPEC.items():
        bases = list(spec["bases"])
        while len(bases) < spec["count"]:
            i = len(bases) - len(spec["bases"])
            bases.append(
                f"{spec['bases'][i % len(spec['bases'])]}"
                f"-v{2 + (i // len(spec['bases']))}"
            )
        for name in bases[: spec["count"]]:
            out[name] = family
    return out


CONTROLLER_FAMILY: dict[str, str] = _build_controllers()
CONTROLLERS: tuple[str, ...] = tuple(CONTROLLER_FAMILY)
assert len(CONTROLLERS) == 128


# ===========================================================================
#  AXIS 1 — 64 SCENARIOS (8 latencies × 8 loss rates)
# ===========================================================================
_LATENCY_GRID = np.linspace(10.0, 1000.0, 8)
_LOSS_GRID    = np.linspace(0.005, 0.650, 8)

SCENARIOS: dict[str, dict[str, float]] = {}
SCENARIO_KEYS: list[str] = []
for _i, _lat in enumerate(_LATENCY_GRID):
    for _j, _loss in enumerate(_LOSS_GRID):
        _key = f"S{_i * 8 + _j + 1:02d}"
        SCENARIOS[_key] = {"latency_ms": float(_lat), "loss_rate": float(_loss)}
        SCENARIO_KEYS.append(_key)
SCENARIO_KEYS = tuple(SCENARIO_KEYS)
assert len(SCENARIO_KEYS) == 64


# ===========================================================================
#  AXIS 2 — 12 CONDITIONS
# ===========================================================================
CONDITIONS: dict[str, dict[str, float]] = {
    "C01-clear":        {"capacity": 1.00, "demand_mult": 1.00, "direction_bias": 0.00, "ns": 0.30, "ew": 0.30},
    "C02-rain":         {"capacity": 0.85, "demand_mult": 0.95, "direction_bias": 0.05, "ns": 0.28, "ew": 0.32},
    "C03-snow":         {"capacity": 0.60, "demand_mult": 0.80, "direction_bias": 0.10, "ns": 0.26, "ew": 0.24},
    "C04-hailstorm":    {"capacity": 0.45, "demand_mult": 0.55, "direction_bias": 0.00, "ns": 0.22, "ew": 0.22},
    "C05-flooding":     {"capacity": 0.35, "demand_mult": 0.50, "direction_bias": 0.05, "ns": 0.20, "ew": 0.20},
    "C06-accident":     {"capacity": 0.55, "demand_mult": 0.90, "direction_bias": 0.25, "ns": 0.35, "ew": 0.25},
    "C07-major-crash":  {"capacity": 0.40, "demand_mult": 0.85, "direction_bias": 0.30, "ns": 0.36, "ew": 0.24},
    "C08-rush-am":      {"capacity": 0.95, "demand_mult": 1.35, "direction_bias": 0.30, "ns": 0.40, "ew": 0.18},
    "C09-rush-pm":      {"capacity": 0.95, "demand_mult": 1.35, "direction_bias": 0.30, "ns": 0.18, "ew": 0.40},
    "C10-oversat":      {"capacity": 1.00, "demand_mult": 1.60, "direction_bias": 0.00, "ns": 0.60, "ew": 0.50},
    "C11-blackout":     {"capacity": 1.00, "demand_mult": 0.85, "direction_bias": 0.00, "ns": 0.28, "ew": 0.28},
    "C12-power-outage": {"capacity": 0.90, "demand_mult": 0.75, "direction_bias": 0.00, "ns": 0.26, "ew": 0.26},
}
CONDITION_KEYS = tuple(CONDITIONS)
assert len(CONDITION_KEYS) == 12


# ===========================================================================
#  LOGISTIC SIGNATURE  —  f(x) = L / (1 + exp(-k (x - x0)))
# ===========================================================================
@dataclass(frozen=True, slots=True)
class LogisticSignature:
    L:  float
    k:  float
    x0: float

    def __call__(self, x: float) -> float:
        return self.L / (1.0 + math.exp(-self.k * (x - self.x0)))


SCENARIO_SIGNATURES: dict[str, LogisticSignature] = {
    f"S{i:02d}": LogisticSignature(
        L  = 140.0 + 2.4 * (i - 1),
        k  = 0.0143 + 0.00030 * (i - 1),
        x0 = 0.05 + 0.0050 * (i - 1),
    )
    for i in range(1, 65)
}


# ===========================================================================
#  THE DATASET CLASS
# ===========================================================================
@dataclass
class SyntheticTrafficDataset:
    """
    Generates the tensor ``M`` with shape ``(n_ctrl, n_scen, n_cond)``:

        M[i, j, k]  ==  record for
                        CONTROLLERS[i]
                      × SCENARIO_KEYS[j]
                      × CONDITION_KEYS[k]

    All parameters fall back to :class:`Config` defaults.  Presets
    (``'quick'`` / ``'standard'`` / ``'full'``) shrink the axes.
    """

    # ---- config-replicated fields (so dataclass introspection works) ----
    seed:          int   = 42
    car_following: str   = "SAFE"
    sample_cf:     bool  = False
    noise_frac:    float = 0.06
    outlier_frac:  float = 0.02
    outlier_mult:  float = 2.2
    csv_path:      str   = "synthetic_dataset.csv"
    npy_path:      str   = "synthetic_tensor.npy"
    preset:        str   = "full"
    progress:      bool  = True
    auto_build:    bool  = False
    verbose:       bool  = True

    # ---- internal state (not part of the dataclass surface) ------------
    _cfg:       Config               = field(init=False, repr=False, compare=False)
    _ctrls:     tuple[str, ...]      = field(init=False, repr=False, compare=False)
    _scens:     tuple[str, ...]      = field(init=False, repr=False, compare=False)
    _conds:     tuple[str, ...]      = field(init=False, repr=False, compare=False)
    _shape:     tuple[int, int, int] = field(init=False, repr=False, compare=False)
    _arrays:    Optional[dict[str, np.ndarray]] = field(init=False, repr=False, compare=False, default=None)
    _M:         Optional[np.ndarray] = field(init=False, repr=False, compare=False, default=None)
    _T:         Optional[np.ndarray] = field(init=False, repr=False, compare=False, default=None)

    NUMERIC_COLS: tuple[str, ...] = _NUMERIC_COLS

    # ------------------------------------------------------------------ init
    def __post_init__(self) -> None:
        self._cfg = Config(
            seed=self.seed,
            car_following=self.car_following,
            sample_cf=self.sample_cf,
            noise_frac=self.noise_frac,
            outlier_frac=self.outlier_frac,
            outlier_mult=self.outlier_mult,
            preset=self.preset,
            csv_path=self.csv_path,
            npy_path=self.npy_path,
            progress=self.progress,
            auto_build=self.auto_build,
            verbose=self.verbose,
        )
        p = PRESETS[self._cfg.preset]
        self._ctrls = CONTROLLERS[: p["n_ctrl"]]
        self._scens = SCENARIO_KEYS[: p["n_scen"]]
        self._conds = CONDITION_KEYS[: p["n_cond"]]
        self._shape = (len(self._ctrls), len(self._scens), len(self._conds))
        if self._cfg.auto_build:
            self.build()

    # ---------------------------------------------------------- factories
    @classmethod
    def default(cls, **overrides: Any) -> "SyntheticTrafficDataset":
        """Factory returning a dataset with the module's default config."""
        cfg = Config(**overrides) if overrides else Config()
        return cls(**cfg.__dict__)

    # -------------------------------------------------------- introspection
    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the underlying tensor."""
        return self._shape

    @property
    def size(self) -> int:
        """Total number of cells."""
        return self._shape[0] * self._shape[1] * self._shape[2]

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        state = "built" if self._M is not None else "empty"
        return (f"SyntheticTrafficDataset(preset={self.preset!r}, "
                f"shape={self._shape}, seed={self.seed}, {state})")

    # ------------------------------------------------------------- access
    def __getitem__(self, key: tuple[int, int, int]) -> dict[str, Any]:
        i, j, k = key
        return self._ensure_records()[i, j, k]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        M = self._ensure_records()
        for i, j, k in product(range(self._shape[0]),
                               range(self._shape[1]),
                               range(self._shape[2])):
            yield M[i, j, k]

    # =========================================================================
    #  FACTORS — vectorised, closed-form (mirror the scalar versions
    #  that used to live in `cell()`).
    # =========================================================================
    @staticmethod
    def _K_comm_vector(family_flat: np.ndarray,
                       latency: np.ndarray,
                       loss: np.ndarray) -> np.ndarray:
        rho = 0.5 * (latency / 600.0) + 0.5 * loss
        k   = _K_COMM_BASE_K / family_flat
        x0  = _K_COMM_BASE_X0 * (1.0 + 0.5 * (1.0 - family_flat))
        L   = _K_COMM_BASE_L * family_flat
        return 1.0 + L / (1.0 + np.exp(-k * (rho - x0)))

    @staticmethod
    def _K_nat_vector(capacity: np.ndarray, demand_mult: np.ndarray,
                      bias: np.ndarray) -> np.ndarray:
        return ((1.0 / np.maximum(capacity, 0.1))
                * (1.0 + 0.35 * (demand_mult - 1.0))
                * (1.0 + 0.15 * np.abs(bias)))

    @staticmethod
    def _K_dem_vector(ns: np.ndarray, ew: np.ndarray,
                      capacity: np.ndarray) -> np.ndarray:
        arrival = ns + ew
        service = 2 * SATURATION_FLOW_VPS * capacity * 10.0
        rho = np.minimum(0.95, arrival / np.maximum(service, 1e-6))
        return 1.0 / (1.0 - rho)

    # =========================================================================
    #  VECTORISED CORE — computes every numeric field as an array of shape
    #  (n1, n2, n3) in a single pass.
    # =========================================================================
    def _compute_arrays(self, force: bool = False) -> dict[str, np.ndarray]:
        if self._arrays is not None and not force:
            return self._arrays

        n1, n2, n3 = self._shape
        rng = np.random.default_rng(self._cfg.seed)

        # --- axis vectors -----------------------------------------------
        ctrl_family = np.array([CONTROLLER_FAMILY[c] for c in self._ctrls])
        kc   = np.array([K_CTRL_FAMILY[f] for f in ctrl_family])           # (n1,)
        flat = np.array([RESILIENCE_FLATTENING[f] for f in ctrl_family])   # (n1,)

        latency = np.array([SCENARIOS[s]["latency_ms"] for s in self._scens])
        loss    = np.array([SCENARIOS[s]["loss_rate"]  for s in self._scens])

        sig_L  = np.array([SCENARIO_SIGNATURES[s].L  for s in self._scens])
        sig_k  = np.array([SCENARIO_SIGNATURES[s].k  for s in self._scens])
        sig_x0 = np.array([SCENARIO_SIGNATURES[s].x0 for s in self._scens])

        cap   = np.array([CONDITIONS[c]["capacity"]       for c in self._conds])
        dmult = np.array([CONDITIONS[c]["demand_mult"]    for c in self._conds])
        bias  = np.array([CONDITIONS[c]["direction_bias"] for c in self._conds])
        ns    = np.array([CONDITIONS[c]["ns"]             for c in self._conds])
        ew    = np.array([CONDITIONS[c]["ew"]             for c in self._conds])

        # --- broadcast to (n1, n2, n3) ----------------------------------
        kc_b    = kc[:, None, None]
        flat_b  = flat[:, None, None]
        lat_b   = latency[None, :, None]
        loss_b  = loss[None, :, None]
        sigL_b  = sig_L[None, :, None]
        sigk_b  = sig_k[None, :, None]
        sigx0_b = sig_x0[None, :, None]
        cap_b   = cap[None, None, :]
        dmult_b = dmult[None, None, :]
        bias_b  = bias[None, None, :]
        ns_b    = ns[None, None, :]
        ew_b    = ew[None, None, :]

        # --- severities --------------------------------------------------
        sev_comm   = 0.5 * (lat_b / 600.0) + 0.5 * loss_b
        sev_nat    = 1.0 - cap_b
        sev_dem    = np.minimum(0.5, (ns_b + ew_b - 0.6) / 1.2)
        sev        = 0.5 * sev_comm + 0.3 * sev_nat + 0.2 * sev_dem
        congestion = np.minimum(0.99,
                                0.4 * sev_comm + 0.4 * sev_nat + 0.2 * sev_dem)

        # --- car-following base time ------------------------------------
        ff_lut = np.array([_CF_BENCHMARK[c]["free_flow"]["in_system_s"]
                           for c in CF_MODELS])
        cg_lut = np.array([_CF_BENCHMARK[c]["congested"]["in_system_s"]
                           for c in CF_MODELS])
        if self._cfg.sample_cf:
            cf_idx = rng.integers(0, len(CF_MODELS), size=(n1, n2, n3))
            ff = ff_lut[cf_idx]
            cg = cg_lut[cf_idx]
        else:
            idx = CF_MODELS.index(self._cfg.car_following)
            ff, cg = ff_lut[idx], cg_lut[idx]

        w    = np.clip(congestion, 0.0, 1.0)
        base = (1.0 - w) * ff + w * cg

        # --- multiplicative factors -------------------------------------
        K_comm = self._K_comm_vector(flat_b, lat_b, loss_b)   # (n1, n2, 1)
        K_nat  = self._K_nat_vector(cap_b, dmult_b, bias_b)   # (1, 1, n3)
        K_dem  = self._K_dem_vector(ns_b * dmult_b,
                                    ew_b * dmult_b,
                                    cap_b)                    # (1, 1, n3)

        raw = base * kc_b * K_comm * K_nat * K_dem

        # --- logistic correction ----------------------------------------
        # log_c = sig(sev) / sig(x0) = 2 / (1 + exp(-k (sev - x0)))
        log_c      = 2.0 / (1.0 + np.exp(-sigk_b * (sev - sigx0_b)))
        true_value = raw * (0.85 + 0.30 * log_c)

        # --- noise + outliers -------------------------------------------
        noisy = true_value * (1.0 + rng.normal(0.0, self._cfg.noise_frac,
                                               size=true_value.shape))
        outlier = rng.random(size=true_value.shape) < self._cfg.outlier_frac
        noisy = np.where(outlier, noisy * self._cfg.outlier_mult, noisy)

        # --- derived metrics --------------------------------------------
        travel = np.maximum(5.0, noisy)
        excess = np.maximum(0.0, travel / np.maximum(base, 1.0) - 1.0)

        queue = np.maximum(0.0, 3.0 + 12.0 * excess
                                + rng.normal(0.0, 0.6, size=travel.shape))
        wait  = np.maximum(0.0, 5.0 + 20.0 * excess
                                + rng.normal(0.0, 1.2, size=travel.shape))
        thr   = np.maximum(50.0, 950.0 / (1.0 + 0.9 * excess))
        aoi   = np.maximum(0.0, 40.0 + 350.0 * sev_comm
                                + 20.0 * np.abs(rng.normal(0.0, 1.0,
                                                           size=travel.shape)))
        deliver = np.clip(
            0.99 - 0.55 * sev_comm - 0.20 * sev_nat
                 + rng.normal(0.0, 0.01, size=travel.shape),
            0.0, 1.0,
        )

        # --- pack into a dict; broadcast singletons so every array
        #     shares the same (n1, n2, n3) shape -------------------------
        def _broadcast(x: np.ndarray) -> np.ndarray:
            return np.broadcast_to(x, (n1, n2, n3)).copy()

        arrays: dict[str, np.ndarray] = {
            "travel_time_s":  _broadcast(travel),
            "queue_veh":      _broadcast(queue),
            "waiting_time_s": _broadcast(wait),
            "throughput_vph": _broadcast(thr),
            "mean_aoi_ms":    _broadcast(aoi),
            "delivery_ratio": _broadcast(deliver),
            "K_ctrl":         _broadcast(kc_b * np.ones_like(base)),
            "K_comm":         _broadcast(K_comm * np.ones_like(base)),
            "K_nat":          _broadcast(K_nat  * np.ones_like(base)),
            "K_dem":          _broadcast(K_dem  * np.ones_like(base)),
            "severity":       _broadcast(sev),
            "true_value":     _broadcast(true_value),
        }
        self._arrays = arrays
        return arrays

    # =========================================================================
    #  BUILDERS
    # =========================================================================
    def _ensure_records(self) -> np.ndarray:
        if self._M is None:
            self.build()
        assert self._M is not None
        return self._M

    def build(self, force: bool = False) -> np.ndarray:
        """
        Build and return the ``(n1, n2, n3)`` object array of dict records.

        Results are cached — pass ``force=True`` to rebuild.
        """
        if self._M is not None and not force:
            return self._M

        arrays = self._compute_arrays(force=force)
        n1, n2, n3 = self._shape

        if self._cfg.progress and self._cfg.verbose and self.size > 1000:
            print(f"  building M with shape {(n1, n2, n3)} "
                  f"= {self.size:,} cells ...")

        # Pre-extract per-axis names for fast dict construction.
        ctrl_names = np.array(self._ctrls)[:, None, None]
        scen_names = np.array(self._scens)[None, :, None]
        cond_names = np.array(self._conds)[None, None, :]

        M = np.empty((n1, n2, n3), dtype=object)
        for i, j, k in product(range(n1), range(n2), range(n3)):
            M[i, j, k] = {
                "i": i, "j": j, "k": k,
                "controller":     ctrl_names[i, 0, 0],
                "car_following":  self._cfg.car_following,
                "scenario":       scen_names[0, j, 0],
                "condition":      cond_names[0, 0, k],
                "travel_time_s":  round(float(arrays["travel_time_s"][i, j, k]), 3),
                "queue_veh":      round(float(arrays["queue_veh"][i, j, k]), 3),
                "waiting_time_s": round(float(arrays["waiting_time_s"][i, j, k]), 3),
                "throughput_vph": round(float(arrays["throughput_vph"][i, j, k]), 2),
                "mean_aoi_ms":    round(float(arrays["mean_aoi_ms"][i, j, k]), 2),
                "delivery_ratio": round(float(arrays["delivery_ratio"][i, j, k]), 4),
                "K_ctrl":         round(float(arrays["K_ctrl"][i, j, k]), 4),
                "K_comm":         round(float(arrays["K_comm"][i, j, k]), 4),
                "K_nat":          round(float(arrays["K_nat"][i, j, k]), 4),
                "K_dem":          round(float(arrays["K_dem"][i, j, k]), 4),
                "severity":       round(float(arrays["severity"][i, j, k]), 4),
                "true_value":     round(float(arrays["true_value"][i, j, k]), 3),
            }
        self._M = M
        return M

    def as_tensor(self, M: Optional[np.ndarray] = None,
                  cols: Optional[tuple[str, ...]] = None) -> np.ndarray:
        """
        Return a dense numeric tensor of shape ``(n1, n2, n3, F)``.

        If ``M`` is omitted, the cached arrays are used directly
        (no re-derivation from dict records).
        """
        cols = cols or self.NUMERIC_COLS
        arrays = self._compute_arrays()
        return np.stack([arrays[c] for c in cols], axis=-1)

    build_numeric = as_tensor  # explicit alias

    # =========================================================================
    #  SLICES — read-only views over the object array
    # =========================================================================
    def slice_controller(self, i: int, M: Optional[np.ndarray] = None) -> np.ndarray:
        """``M[i, :, :]`` → shape ``(n2, n3)``."""
        return (self._ensure_records() if M is None else M)[i]

    def slice_scenario(self, j: int, M: Optional[np.ndarray] = None) -> np.ndarray:
        """``M[:, j, :]`` → shape ``(n1, n3)``."""
        return (self._ensure_records() if M is None else M)[:, j, :]

    def slice_condition(self, k: int, M: Optional[np.ndarray] = None) -> np.ndarray:
        """``M[:, :, k]`` → shape ``(n1, n2)``."""
        return (self._ensure_records() if M is None else M)[:, :, k]

    # =========================================================================
    #  STATISTICS
    # =========================================================================
    def stats(self, M: Optional[np.ndarray] = None) -> dict[str, Any]:
        """Return mean / stdev / min / max for every numeric column."""
        _ = M  # M is accepted for API symmetry but the cached arrays are used
        arrays = self._compute_arrays()
        out: dict[str, Any] = {"shape": self._shape, "n_cells": self.size}
        for name in self.NUMERIC_COLS:
            x = arrays[name]
            out[name] = {
                "mean":  round(float(x.mean()), 3),
                "stdev": round(float(x.std()), 3),
                "min":   round(float(x.min()), 3),
                "max":   round(float(x.max()), 3),
            }
        return out

    # =========================================================================
    #  EXPORTERS
    # =========================================================================
    def to_csv(self, M: Optional[np.ndarray] = None,
               path: Optional[str] = None) -> str:
        """Write every record to ``path`` (defaults to ``self.csv_path``)."""
        path = path or self.csv_path
        M = self._ensure_records() if M is None else M
        n1, n2, n3 = M.shape
        sample = M[0, 0, 0]
        cols = list(sample.keys())
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for i, j, k in product(range(n1), range(n2), range(n3)):
                w.writerow(M[i, j, k])
        if self._cfg.verbose:
            print(f"  wrote {n1 * n2 * n3:,} rows "
                  f"({n1} × {n2} × {n3}) → {path}")
        return path

    def to_npy(self, M: Optional[np.ndarray] = None,
               path: Optional[str] = None) -> str:
        """Save the numeric tensor to ``path`` (defaults to ``self.npy_path``)."""
        path = path or self.npy_path
        T = self.as_tensor(M) if M is not None else self.as_tensor()
        np.save(path, T)
        if self._cfg.verbose:
            print(f"  wrote tensor {T.shape} → {path}")
        return path

    # =========================================================================
    #  REPORTS
    # =========================================================================
    def summary(self, M: Optional[np.ndarray] = None) -> str:
        """Return (and optionally print) a multi-line report string."""
        s = self.stats(M)
        n1, n2, n3 = self._shape
        lines = [
            "=" * 90,
            f"  SYNTHETIC DATASET  —  preset={self._cfg.preset!r}",
            "=" * 90,
            f"  shape          : {self._shape}",
            f"  total cells    : {self.size:,}",
            f"  axis 0 (ctrl)  : {n1}",
            f"  axis 1 (scen)  : {n2}",
            f"  axis 2 (cond)  : {n3}",
            f"  seed           : {self._cfg.seed}",
            f"  car-following  : {self._cfg.car_following}"
            + ("  (sampled per cell)" if self._cfg.sample_cf else ""),
            f"  noise frac     : {self._cfg.noise_frac}",
            "",
            f"  {'metric':<18}{'mean':>10}{'stdev':>10}"
            f"{'min':>10}{'max':>10}",
            "  " + "-" * 58,
        ]
        for k, v in s.items():
            if k in ("shape", "n_cells"):
                continue
            lines.append(f"  {k:<18}{v['mean']:>10.3f}{v['stdev']:>10.3f}"
                         f"{v['min']:>10.3f}{v['max']:>10.3f}")
        text = "\n".join(lines)
        if self._cfg.verbose:
            print(text)
        return text

    def print_slice_controller(self, i: int,
                               M: Optional[np.ndarray] = None,
                               max_rows: int = 8) -> None:
        """Print the ``(n2 × n3)`` travel-time slice for controller ``i``."""
        M = self._ensure_records() if M is None else M
        n2, n3 = self._shape[1], self._shape[2]
        ctrl = self._ctrls[i]
        print("=" * 100)
        print(f"  M[{i}, :, :]  —  controller = {ctrl}  "
              f"(family = {CONTROLLER_FAMILY[ctrl]})")
        print("=" * 100)
        hdr = f"{'scenario':<10}" + "".join(f"{k[:8]:>9}"
                                            for k in self._conds[:n3])
        print(hdr)
        print("-" * len(hdr))
        for j in range(min(max_rows, n2)):
            sc = self._scens[j]
            line = f"{sc:<10}"
            for k in range(n3):
                line += f"{M[i, j, k]['travel_time_s']:>9.1f}"
            print(line)
        if n2 > max_rows:
            print(f"  …   ({n2 - max_rows} more scenario rows)")
        print()


# ===========================================================================
#  CLI
# ===========================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the synthetic (128, 64, 12) traffic dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset", choices=tuple(PRESETS), default="full",
                   help="quick | standard | full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.06,
                   help="fractional Gaussian noise added to the true value")
    p.add_argument("--car-following", default="SAFE", choices=CF_MODELS)
    p.add_argument("--sample-cf", action="store_true",
                   help="sample a car-following model per cell")
    p.add_argument("--export", action="store_true",
                   help="write CSV (default path: synthetic_dataset.csv)")
    p.add_argument("--dump-npy", action="store_true",
                   help="write .npy tensor (default: synthetic_tensor.npy)")
    p.add_argument("--slice", type=int, default=None, metavar="I",
                   help="print M[i, :, :] for controller i")
    p.add_argument("--quiet", action="store_true",
                   help="suppress progress + summary printing")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    quiet = args.quiet
    if not quiet:
        print("#" * 90)
        print("#  SYNTHETIC DATASET  —  128 × 64 × 12 tensor")
        print("#" * 90)

    ds = SyntheticTrafficDataset(
        seed=args.seed,
        car_following=args.car_following,
        sample_cf=args.sample_cf,
        noise_frac=args.noise,
        preset=args.preset,
        verbose=not quiet,
    )
    n1, n2, n3 = ds.shape
    if not quiet:
        print(f"\n[1] Building M with shape ({n1}, {n2}, {n3}) "
              f"[preset={ds.preset!r}] ...")
    M = ds.build()
    if not quiet:
        print(f"    M.shape = {M.shape}   M.size = {M.size:,}")

        print("\n[2] Summary:")
        ds.summary(M)

        print("\n[3] Numeric tensor:")
        T = ds.as_tensor()
        print(f"    T.shape = {T.shape}  "
              f"(×{len(ds.NUMERIC_COLS)} numeric columns)")
        print(f"    columns = {ds.NUMERIC_COLS}")

    if args.slice is not None and not quiet:
        print(f"\n[4] Slice for controller {args.slice}:")
        ds.print_slice_controller(args.slice, M)

    if args.export:
        if not quiet:
            print("\n[5] CSV export ...")
        ds.to_csv(M)

    if args.dump_npy:
        if not quiet:
            print("\n[6] .npy export ...")
        ds.to_npy(M)

    if not quiet:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())