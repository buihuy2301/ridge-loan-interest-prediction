"""Axis 2 of the experiment matrix: how far to travel along a proposed direction.

A step rule sees only a `Proposal`, never the algorithm that produced it, which
is what lets the same `Armijo` object serve gradient descent, Nesterov and
Newton without knowing the difference.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Protocol

from .direction import Proposal


class StepResult(NamedTuple):
    size: float
    n_evals: int
    """Objective evaluations spent choosing this step; zero for rules that do not
    probe the function."""
    accepted: bool
    """False when a line search ran out of trials without meeting its condition.
    The loop uses this to stop instead of burning the remaining iterations on a
    search that cannot succeed."""


class StepRule(Protocol):
    label: str

    def __call__(self, proposal: Proposal, k: int) -> StepResult: ...


class Fixed:
    """A constant step, the baseline the assignment asks to compare against.

    Theory says gradient descent converges for 0 < t < 2/L and that the optimal
    constant for a strongly convex quadratic is 2/(L + mu). The experiment grid
    deliberately includes a value above 2/L to show divergence, so this class
    does not clamp anything.
    """

    def __init__(self, size: float, label: str | None = None) -> None:
        if size <= 0.0:
            raise ValueError(f"step size must be positive, got {size}")
        self.size = float(size)
        self.label = label or f"t = {size:.4g}"

    def __call__(self, proposal: Proposal, k: int) -> StepResult:
        return StepResult(self.size, 0, True)


class Armijo:
    """Backtracking line search on the sufficient decrease condition

        f(y + t p) <= f(y) + c t grad f(y)^T p

    written in the general form rather than the f(w - t g) <= f(w) - c t ||g||^2
    form of the brief. The two agree when p = -grad f, but only the general one
    stays valid for the Newton direction, where grad^T p = -grad^T H^-1 grad.

    The probed function is `proposal.local`, which is the full objective for
    deterministic methods and the current mini-batch for SGD. In the SGD case an
    accepted step only guarantees decrease on that batch, never on f itself.
    """

    def __init__(
        self,
        c: float = 1e-4,
        rho: float = 0.5,
        t0: float = 1.0,
        max_evals: int = 50,
    ) -> None:
        if not 0.0 < c < 1.0:
            raise ValueError(f"c must lie in (0, 1), got {c}")
        if not 0.0 < rho < 1.0:
            raise ValueError(f"rho must lie in (0, 1), got {rho}")
        if t0 <= 0.0:
            raise ValueError(f"t0 must be positive, got {t0}")
        self.c = c
        self.rho = rho
        self.t0 = t0
        self.max_evals = max_evals
        self.label = f"Armijo (c = {c:g}, rho = {rho:g}, t0 = {t0:.3g})"

    def __call__(self, proposal: Proposal, k: int) -> StepResult:
        slope = proposal.directional_derivative()
        if slope >= 0.0:
            # Not a descent direction, so no positive step can satisfy the
            # condition. Report failure rather than looping to max_evals.
            return StepResult(0.0, 0, False)

        f0 = proposal.value
        probe = proposal.local.value
        point, direction = proposal.point, proposal.direction

        t = self.t0
        for n_evals in range(1, self.max_evals + 1):
            if probe(point + t * direction) <= f0 + self.c * t * slope:
                return StepResult(t, n_evals, True)
            t *= self.rho

        # Every trial failed. Near the numerical floor this is the normal
        # outcome: the true decrease drops below the resolution of f, so the
        # comparison is decided by rounding noise. Section 4.3 of the plan.
        return StepResult(t, self.max_evals, False)


class Decay:
    """Deterministic step-size schedules for SGD, all functions of k alone.

        constant : eta_0
        inverse  : eta_0 / (1 + gamma k)
        sqrt     : eta_0 / sqrt(k + 1)
        staircase: eta_0 * 2^(-floor(k / period))

    The constant rule is included because it does not converge to w*: it settles
    into a neighbourhood of radius O(eta sigma^2 / mu), which shows up on a log
    scale as a horizontal line. That plateau is the point of the comparison, not
    a bug to be tuned away.
    """

    KINDS = ("constant", "inverse", "sqrt", "staircase")

    def __init__(
        self,
        kind: str,
        eta0: float,
        gamma: float = 1.0,
        period: int = 1000,
    ) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        if eta0 <= 0.0:
            raise ValueError(f"eta0 must be positive, got {eta0}")
        if kind == "staircase" and period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self.kind = kind
        self.eta0 = float(eta0)
        self.gamma = float(gamma)
        self.period = period

        self.label = {
            "constant": f"eta = {eta0:.3g}",
            "inverse": f"eta = {eta0:.3g}/(1 + {gamma:g}k)",
            "sqrt": f"eta = {eta0:.3g}/sqrt(k+1)",
            "staircase": f"eta = {eta0:.3g}, halved every {period} steps",
        }[kind]

    def __call__(self, proposal: Proposal, k: int) -> StepResult:
        if self.kind == "constant":
            size = self.eta0
        elif self.kind == "inverse":
            size = self.eta0 / (1.0 + self.gamma * k)
        elif self.kind == "sqrt":
            size = self.eta0 / math.sqrt(k + 1)
        else:
            size = self.eta0 * 2.0 ** (-(k // self.period))
        return StepResult(size, 0, True)
