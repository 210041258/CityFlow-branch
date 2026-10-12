#!/usr/bin/env python3
"""
tensor_pipeline.py
==================
Downstream consumer of `synthetic_tensor.npy` (written by
`synthetic_dataset.py --dump-npy`).

The .npy file has shape (n_ctrl, n_scen, n_cond, F) where F is the
number of numeric columns of the dataset (default 12, see
`SyntheticTrafficDataset.NUMERIC_COLS`).

Pipeline
--------
  1. LOAD     — read the .npy, restore the (n1, n2, n3) grid + feature axis
  2. SPLIT    — row-wise train / test split with a fixed seed
  3. BASELINE — mean predictor, Ridge, kNN
  4. SURROGATE— GP on residuals + EI-based active-learning loop
  5. REPORT   — MAE / RMSE / R² for every model, side by side
  6. EXPORT   — predictions back to .npy + CSV

CLI
---
    python tensor_pipeline.py                              # full run
    python tensor_pipeline.py --tensor my.npy              # custom file
    python tensor_pipeline.py --target travel_time_s       # other column
    python tensor_pipeline.py --budget 200 --init 20       # AL loop
    python tensor_pipeline.py --export-preds               # save predictions

Programmatic
------------
    from tensor_pipeline import TensorPipeline

    tp = TensorPipeline('synthetic_tensor.npy')
    tp.summary()
    report = tp.run(budget=200, n_init=20)
    tp.export('preds.npy')
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
__all__ = ["TensorPipeline", "PipelineReport"]

# Column names emitted by SyntheticTrafficDataset.NUMERIC_COLS — the
# canonical ordering used by the .npy feature axis.
DEFAULT_COLS: tuple[str, ...] = (
    "travel_time_s", "queue_veh", "waiting_time_s", "throughput_vph",
    "mean_aoi_ms", "delivery_ratio",
    "K_ctrl", "K_comm", "K_nat", "K_dem",
    "severity", "true_value",
)

# Which column to regress on by default.
DEFAULT_TARGET = "travel_time_s"

# Which columns are *labels* (never used as inputs when predicting).
LEAKY_COLS = ("true_value",)


# ===========================================================================
#  REPORT CONTAINER
# ===========================================================================
@dataclass
class PipelineReport:
    target:      str
    n_train:     int
    n_test:      int
    n_features:  int
    scores:      dict[str, dict[str, float]] = field(default_factory=dict)
    al_trace:    list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 78,
            f"  PIPELINE REPORT  —  target = {self.target!r}",
            "=" * 78,
            f"  train / test   : {self.n_train:,} / {self.n_test:,}",
            f"  feature count  : {self.n_features}",
            "",
            f"  {'model':<18}{'MAE':>10}{'RMSE':>10}{'R²':>10}",
            "  " + "-" * 46,
        ]
        for name, s in self.scores.items():
            lines.append(f"  {name:<18}{s['mae']:>10.4f}"
                         f"{s['rmse']:>10.4f}{s['r2']:>10.4f}")
        if self.al_trace:
            lines.append("")
            lines.append("  Active-learning trace (EI loop)")
            lines.append(f"  {'round':>6}{'MAE':>10}{'RMSE':>10}{'R²':>10}")
            lines.append("  " + "-" * 36)
            for row in self.al_trace:
                lines.append(f"  {row['round']:>6}"
                             f"{row['mae']:>10.4f}"
                             f"{row['rmse']:>10.4f}"
                             f"{row['r2']:>10.4f}")
        return "\n".join(lines)


# ===========================================================================
#  THE PIPELINE
# ===========================================================================
@dataclass
class TensorPipeline:
    """
    Loads a (n1, n2, n3, F) numeric tensor and runs a small ML pipeline.

    All parameters have sensible defaults so `TensorPipeline(path)` works.
    """

    path:        str  = "synthetic_tensor.npy"
    cols:        tuple[str, ...] = DEFAULT_COLS
    target:      str  = DEFAULT_TARGET
    seed:        int  = 42
    test_frac:   float = 0.2
    verbose:     bool = True

    # --- internal state -------------------------------------------------
    X:            Optional[np.ndarray] = field(init=False, default=None, repr=False)
    y:            Optional[np.ndarray] = field(init=False, default=None, repr=False)
    grid_shape:   Optional[tuple]      = field(init=False, default=None, repr=False)
    _rng:         np.random.Generator  = field(init=False, repr=False)
    _split:       Optional[tuple]      = field(init=False, default=None, repr=False)
    _preds:       Optional[np.ndarray] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.load()

    # ------------------------------------------------------------------ load
    def load(self, force: bool = False) -> "TensorPipeline":
        """Read the .npy file and flatten it to (N, F)."""
        if self.X is not None and not force:
            return self

        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(
                f"Tensor file not found: {path}. "
                f"Run `python synthetic_dataset.py --dump-npy` first."
            )
        T = np.load(path)
        if T.ndim != 4:
            raise ValueError(
                f"Expected a 4-D tensor (n1, n2, n3, F); got shape {T.shape}."
            )

        self.grid_shape = T.shape[:3]
        F = T.shape[-1]
        if F != len(self.cols):
            raise ValueError(
                f"Feature axis is {F} but {len(self.cols)} column names "
                f"were supplied.  Pass `cols=...` matching your tensor."
            )
        if self.target not in self.cols:
            raise ValueError(
                f"target {self.target!r} not in {self.cols}"
            )

        # flatten (n1, n2, n3, F) → (N, F)
        flat = T.reshape(-1, F)
        tgt_idx = self.cols.index(self.target)

        # feature matrix excludes the target and any leaky columns
        drop = {tgt_idx} | {self.cols.index(c)
                            for c in LEAKY_COLS
                            if c in self.cols}
        keep = [i for i in range(F) if i not in drop]

        self.X = flat[:, keep].astype(np.float64)
        self.y = flat[:, tgt_idx].astype(np.float64)

        if self.verbose:
            print(f"  loaded {path}  shape={T.shape}  "
                  f"features={self.X.shape[1]}  rows={self.X.shape[0]:,}")
        return self

    # ------------------------------------------------------------------ split
    def split(self, test_frac: Optional[float] = None,
              force: bool = False) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
        """Return (X_tr, X_te, y_tr, y_te).  Cached unless force=True."""
        if self._split is not None and not force and test_frac is None:
            return self._split
        assert self.X is not None and self.y is not None

        frac = self.test_frac if test_frac is None else test_frac
        n = len(self.X)
        idx = self._rng.permutation(n)
        cut = int(round(n * (1 - frac)))
        tr, te = idx[:cut], idx[cut:]
        out = (self.X[tr], self.X[te], self.y[tr], self.y[te])
        self._split = out
        return out

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        err  = y_pred - y_true
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return {"mae": mae, "rmse": rmse, "r2": r2}

    @staticmethod
    def _ridge_fit(X: np.ndarray, y: np.ndarray,
                   alpha: float = 1.0) -> np.ndarray:
        """Closed-form ridge regression.  Returns coefficient vector."""
        n, d = X.shape
        # normalise to keep the ridge penalty comparable across features
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-9
        Xn = (X - mu) / sd
        Xb = np.hstack([np.ones((n, 1)), Xn])
        A = Xb.T @ Xb + alpha * np.eye(d + 1)
        # intercept should not be regularised
        A[0, 0] -= alpha
        w = np.linalg.solve(A, Xb.T @ y)
        return np.concatenate([mu.ravel() * 0.0, w])  # placeholder for shape

    # ------------------------------------------------------------------ models
    def _ridge_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                       X_te: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-9
        Xn_tr = (X_tr - mu) / sd
        Xn_te = (X_te - mu) / sd
        Xb_tr = np.hstack([np.ones((len(Xn_tr), 1)), Xn_tr])
        Xb_te = np.hstack([np.ones((len(Xn_te), 1)), Xn_te])
        A = Xb_tr.T @ Xb_tr + alpha * np.eye(Xb_tr.shape[1])
        A[0, 0] -= alpha
        w = np.linalg.solve(A, Xb_tr.T @ y_tr)
        return Xb_te @ w

    def _knn_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                     X_te: np.ndarray, k: int = 5) -> np.ndarray:
        # normalise so distance is not dominated by one column
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-9
        A = (X_tr - mu) / sd
        B = (X_te - mu) / sd
        # chunked distance to keep memory bounded
        out = np.empty(len(B))
        chunk = 2048
        for s in range(0, len(B), chunk):
            b = B[s:s + chunk]
            d2 = ((b[:, None, :] - A[None, :, :]) ** 2).sum(-1)
            idx = np.argpartition(d2, k, axis=1)[:, :k]
            out[s:s + chunk] = y_tr[idx].mean(axis=1)
        return out

    def _gp_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray,
                    lengthscale: float = 1.0,
                    noise: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Tiny ARD-RBF GP; returns (mean, variance) on the test set."""
        # scale inputs to unit variance
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-9
        A = (X_tr - mu) / sd
        B = (X_te - mu) / sd

        def kern(P, Q):
            Pn = P / lengthscale
            Qn = Q / lengthscale
            p = (Pn ** 2).sum(1)[:, None]
            q = (Qn ** 2).sum(1)[None, :]
            return np.exp(-0.5 * np.maximum(p + q - 2 * Pn @ Qn.T, 0.0))

        K   = kern(A, A) + (noise ** 2 + 1e-6) * np.eye(len(A))
        Ks  = kern(A, B)
        Kss = np.diag(kern(B, B))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
        mu_pred = Ks.T @ alpha
        v = np.linalg.solve(L, Ks)
        var = np.maximum(Kss - (v ** 2).sum(0), 1e-9)
        return mu_pred, var

    # ------------------------------------------------------------------ run
    def run(self, budget: int = 200, n_init: int = 20,
            gp_lengthscale: float = 1.0,
            gp_noise: float = 0.1) -> PipelineReport:
        """
        Train four baselines + run an EI-based active-learning loop on
        the residuals of a Ridge regressor.
        """
        X_tr, X_te, y_tr, y_te = self.split()
        n_feat = X_tr.shape[1]

        report = PipelineReport(
            target=self.target,
            n_train=len(X_tr), n_test=len(X_te),
            n_features=n_feat,
        )

        # --- baselines ---------------------------------------------------
        baselines = {
            "mean":   np.full_like(y_te, y_tr.mean()),
            "ridge":  self._ridge_predict(X_tr, y_tr, X_te),
            "knn5":   self._knn_predict(X_tr, y_tr, X_te, k=5),
        }
        for name, pred in baselines.items():
            report.scores[name] = self._score(y_te, pred)

        # --- GP on residuals of Ridge -----------------------------------
        ridge_tr = self._ridge_predict(X_tr, y_tr, X_tr)
        resid_tr = y_tr - ridge_tr
        gp_mu, gp_var = self._gp_predict(X_tr, resid_tr, X_te,
                                         gp_lengthscale, gp_noise)
        gp_pred = self._ridge_predict(X_tr, y_tr, X_te) + gp_mu
        report.scores["ridge+gp"] = self._score(y_te, gp_pred)

        # --- EI-based active-learning loop ------------------------------
        report.al_trace = self._active_learn(
            X_tr, y_tr, X_te, y_te,
            budget=budget, n_init=n_init,
            gp_lengthscale=gp_lengthscale, gp_noise=gp_noise,
        )

        if self.verbose:
            print(report)
        return report

    def _active_learn(self, X_tr: np.ndarray, y_tr: np.ndarray,
                      X_te: np.ndarray, y_te: np.ndarray,
                      budget: int, n_init: int,
                      gp_lengthscale: float, gp_noise: float) -> list[dict]:
        """
        Expected-Improvement loop on the Ridge residuals.

        Round 0            → n_init random points
        Rounds 1 … budget  → pick argmax EI from a random pool
        """
        n_pool = min(2000, max(200, len(X_tr) // 10))
        seen   = self._rng.choice(len(X_tr), size=n_init, replace=False)
        seen_set = set(seen.tolist())
        trace = []

        for rnd in range(budget):
            # fit surrogate on current observations
            ridge_obs = self._ridge_predict(X_tr[seen], y_tr[seen],
                                            X_tr[seen])
            resid_obs = y_tr[seen] - ridge_obs
            mu, var = self._gp_predict(X_tr[seen], resid_obs, X_tr,
                                        gp_lengthscale, gp_noise)

            # pool = unobserved rows
            pool_idx = np.array([i for i in range(len(X_tr))
                                 if i not in seen_set])
            if len(pool_idx) == 0:
                break
            sample = self._rng.choice(
                pool_idx, size=min(n_pool, len(pool_idx)), replace=False,
            )
            mu_s, var_s = mu[sample], var[sample]
            sigma = np.sqrt(np.maximum(var_s, 1e-12))
            best = y_tr[seen].min()
            xi = 0.01
            z = (best - mu_s - xi) / sigma
            # Φ and φ via erf / exp
            phi = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
            pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
            ei = (best - mu_s - xi) * phi + sigma * pdf
            pick = sample[int(np.argmax(ei))]
            seen_set.add(int(pick))
            seen = np.append(seen, pick)

            # evaluate on held-out test set
            pred_te = self._ridge_predict(X_tr[seen], y_tr[seen], X_te)
            s = self._score(y_te, pred_te)
            trace.append({"round": rnd + 1, **s})

        return trace

    # ------------------------------------------------------------------ export
    def export(self, path: str = "predictions.npy") -> str:
        """Write the ridge+GP prediction for every row to `path`."""
        if self._preds is None:
            X_tr, X_te, y_tr, y_te = self.split()
            ridge_all = self._ridge_predict(X_tr, y_tr, self.X)
            resid_tr = y_tr - self._ridge_predict(X_tr, y_tr, X_tr)
            mu, _ = self._gp_predict(X_tr, resid_tr, self.X)
            self._preds = ridge_all + mu
        np.save(path, self._preds)
        if self.verbose:
            print(f"  wrote {len(self._preds):,} predictions → {path}")
        return path

    def export_csv(self, path: str = "predictions.csv") -> str:
        if self._preds is None:
            self.export()  # populate predictions
        assert self._preds is not None and self.y is not None
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "y_true", "y_pred", "residual"])
            for i, (t, p) in enumerate(zip(self.y, self._preds)):
                w.writerow([i, f"{t:.4f}", f"{p:.4f}", f"{t - p:.4f}"])
        if self.verbose:
            print(f"  wrote {len(self._preds):,} rows → {path}")
        return path

    # ------------------------------------------------------------------ summary
    def summary(self) -> str:
        """One-screen overview of what was loaded."""
        assert self.X is not None and self.y is not None
        lines = [
            "=" * 78,
            "  TENSOR PIPELINE — LOADED",
            "=" * 78,
            f"  file          : {self.path}",
            f"  grid shape    : {self.grid_shape}",
            f"  flat rows     : {self.X.shape[0]:,}",
            f"  features      : {self.X.shape[1]}",
            f"  target        : {self.target!r}",
            f"  target mean   : {self.y.mean():.3f}",
            f"  target stdev  : {self.y.std():.3f}",
            f"  target range  : [{self.y.min():.3f}, {self.y.max():.3f}]",
        ]
        text = "\n".join(lines)
        if self.verbose:
            print(text)
        return text


# ===========================================================================
#  CLI
# ===========================================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Downstream consumer of synthetic_tensor.npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tensor", default="synthetic_tensor.npy",
                   help="path to the .npy file written by synthetic_dataset")
    p.add_argument("--target", default=DEFAULT_TARGET,
                   choices=DEFAULT_COLS,
                   help="numeric column to regress on")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--budget", type=int, default=200,
                   help="active-learning rounds (EI)")
    p.add_argument("--init", type=int, default=20,
                   help="initial random observations for the AL loop")
    p.add_argument("--export-preds", action="store_true",
                   help="write predictions.npy (and .csv)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    tp = TensorPipeline(
        path=args.tensor,
        target=args.target,
        seed=args.seed,
        test_frac=args.test_frac,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print()
        tp.summary()

    report = tp.run(budget=args.budget, n_init=args.init)

    if args.export_preds:
        if not args.quiet:
            print()
        tp.export("predictions.npy")
        tp.export_csv("predictions.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())