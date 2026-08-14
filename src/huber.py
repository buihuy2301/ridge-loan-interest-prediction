"""The Huber objective, the second function this project minimises.

Ridge is a quadratic, which hands Newton the solution in one step and lets
backtracking accept the full step at every iteration. Section 4.4 of
KE_HOACH_TRIEN_KHAI.md asks for a function on the same data that takes those two
privileges away without leaving the class of smooth convex problems:

    f(w) = 1/n sum_i H_delta(x_i^T w - y_i) + lam/2 ||w||^2

    H_delta(r) = r^2 / 2                    if |r| <= delta
                 delta (|r| - delta / 2)    otherwise

The function is convex and continuously differentiable. Its Hessian exists
wherever no residual sits exactly on the kink; rows on the kink are counted as
quadratic by convention, which is what `<=` in the mask below encodes.

Nothing here is a quadratic, so three things Ridge gets for free have to be
worked for: the Hessian changes with w, the minimiser is found numerically, and
the suboptimality gap cannot be written as a quadratic form.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .objective import RidgeObjective

REFERENCE_PATH = Path("data/processed/huber_reference.json")


class HuberObjective:
    """f, its derivatives, and the constants that bound its convergence rate."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lam: float,
        delta: float,
        w_star: np.ndarray | None = None,
    ) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
        if lam <= 0.0:
            raise ValueError(f"lam must be positive, got {lam}")
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta}")
        if X.dtype != np.float64 or y.dtype != np.float64:
            raise ValueError("X and y must be float64")

        self.X = X
        self.y = y
        self.lam = float(lam)
        self.delta = float(delta)
        self.n, self.d = X.shape

        # Set when the reference solution is read from disk, so a rerun does not
        # pay for the Newton solve again.
        self._w_star: np.ndarray | None = None
        if w_star is not None:
            self._w_star = np.asarray(w_star, dtype=np.float64)
        self.reference_grad_norm = float("nan")
        self.reference_iters = -1
        self._frozen_factor = None

    # ---------------------------------------------------------------- values

    @property
    def n_rows(self) -> int:
        return self.n

    def _residual(self, w: np.ndarray) -> np.ndarray:
        return self.X @ w - self.y

    def _loss(self, residual: np.ndarray) -> np.ndarray:
        """H_delta applied row by row."""
        size = np.abs(residual)
        return np.where(
            size <= self.delta,
            0.5 * residual * residual,
            self.delta * (size - 0.5 * self.delta),
        )

    def value(self, w: np.ndarray) -> float:
        return float(self._loss(self._residual(w)).sum() / self.n + self.lam / 2 * (w @ w))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        residual = self._residual(w)
        return self.X.T @ np.clip(residual, -self.delta, self.delta) / self.n + self.lam * w

    def value_and_gradient(self, w: np.ndarray) -> tuple[float, np.ndarray]:
        """Both at once, reusing the single expensive product X @ w."""
        residual = self._residual(w)
        value = float(self._loss(residual).sum() / self.n + self.lam / 2 * (w @ w))
        gradient = self.X.T @ np.clip(residual, -self.delta, self.delta) / self.n + self.lam * w
        return value, gradient

    # --------------------------------------------------------------- Hessian

    def compute_hessian(self, w: np.ndarray | None = None) -> np.ndarray:
        """(1/n) X_S^T X_S + lam I, with S the rows inside the quadratic zone.

        Rows outside it have a linear loss and contribute no curvature, so the
        Hessian is the Ridge one built from a subset of the rows. Unlike the Ridge
        case this genuinely changes with w, which is why the argument is required
        here and ignored there.
        """
        if w is None:
            raise ValueError("the Huber Hessian depends on w, so w is required")
        inside = np.flatnonzero(np.abs(self._residual(w)) <= self.delta)
        active = self.X[inside]
        hessian = active.T @ active / self.n
        hessian.flat[:: self.d + 1] += self.lam
        return hessian
