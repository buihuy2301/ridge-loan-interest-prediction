"""Experiment groups as data, and one executor that expands and runs them.

Each figure group in section 5.3 of the plan is a list of `RunSpec`, built by a
function that receives the objective and returns the specs. Adding a value to a
grid is a change to that list, not a new function, and the eight mandatory
configurations of the brief come out of `product` rather than being written by
hand eight times.

Groups write `results/raw/<name>.json` as soon as they finish and are skipped on
a later pass, because the full sweep takes hours and a machine that goes to
sleep in the middle should cost one group, not all of them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .direction import Direction, MiniBatch, Nesterov, NewtonStep, SteepestDescent
from .iterate import iterate, warm_up
from .objective import RidgeObjective, batch_smoothness
from .record import RunRecord, load_records, save_records
from .stepsize import Armijo, Decay, Fixed, StepRule

MakeDirection = Callable[[RidgeObjective], Direction]
MakeStep = Callable[[RidgeObjective], StepRule]

RESULTS_DIR = Path("results/raw")


@dataclass
class RunSpec:
    """One cell of a grid: how to build the pair, and how long to run it.

    The two factories take the objective rather than being ready-made objects
    because every `Direction` except `SteepestDescent` carries state, and because
    step sizes are almost always expressed through L or mu.
    """

    make_direction: MakeDirection
    make_step: MakeStep
    label: str | None = None
    max_iter: int = 3000
    max_epochs: int | None = None
    tol: float = 1e-10
    record_every: int | None = None
    repeats: int = 1
    params: dict = field(default_factory=dict)


@dataclass
class ExperimentGroup:
    name: str
    """Doubles as the JSON filename and the figure filename stem."""
    title: str
    build: Callable[[RidgeObjective], list[RunSpec]]


def product(
    directions: dict[str, MakeDirection],
    steps: dict[str, MakeStep],
    **common,
) -> list[RunSpec]:
    """Every direction against every step rule.

    The brief asks for four algorithms with two step rules each; passing those
    two dictionaries here is what makes the count checkable rather than a matter
    of reading the code carefully.
    """
    return [
        RunSpec(
            make_direction=make_direction,
            make_step=make_step,
            label=f"{d_name}, {s_name}",
            params={"direction": d_name, "step": s_name},
            **common,
        )
        for d_name, make_direction in directions.items()
        for s_name, make_step in steps.items()
    ]


# ----------------------------------------------------------------- execution


def run_spec(objective: RidgeObjective, spec: RunSpec, warmup: bool = False) -> RunRecord:
    """Run one cell, repeated if asked, keeping the median-time repetition.

    Repetitions exist for the clock alone. A deterministic pair produces the same
    trajectory every time, so the gaps and iteration counts are identical and
    only `times` differs; taking the median repetition rather than averaging the
    series keeps one internally consistent record.
    """
    runs = [
        iterate(
            objective,
            spec.make_direction(objective),
            spec.make_step(objective),
            max_iter=spec.max_iter,
            max_epochs=spec.max_epochs,
            tol=spec.tol,
            record_every=spec.record_every,
            warmup=warmup,
            label=spec.label,
        )
        for _ in range(max(1, spec.repeats))
    ]
    chosen = sorted(runs, key=lambda r: r.total_time)[len(runs) // 2]
    chosen.meta.update(spec.params)
    chosen.meta["repeats"] = len(runs)
    chosen.meta["times_all"] = [r.total_time for r in runs]
    return chosen


def run_group(
    group: ExperimentGroup,
    objective: RidgeObjective,
    out_dir: str | Path = RESULTS_DIR,
    force: bool = False,
    verbose: bool = True,
) -> list[RunRecord]:
    """Run a group, or load it back if its file is already there."""
    path = Path(out_dir) / f"{group.name}.json"
    log = print if verbose else (lambda *a, **k: None)

    if path.exists() and not force:
        records, _ = load_records(path)
        log(f"{group.name}: loaded {len(records)} runs from {path}")
        return records

    specs = group.build(objective)
    log(f"{group.name}: {len(specs)} runs")
    warm_up(objective)

    records = []
    started = time.perf_counter()
    for spec in specs:
        record = run_spec(objective, spec)
        records.append(record)
        log(
            f"   {record.label:<46} {record.status:<18} "
            f"iters={record.iters[-1]:>7}  gap={record.final_gap:.2e}  "
            f"t={record.total_time:6.2f}s"
        )
    log(f"{group.name}: done in {time.perf_counter() - started:.0f}s")

    save_records(records, path, meta={"group": group.name, "title": group.title, **objective.summary()})
    return records


def run_all(
    groups: Iterable[ExperimentGroup],
    objective: RidgeObjective,
    out_dir: str | Path = RESULTS_DIR,
    force: bool = False,
) -> dict[str, list[RunRecord]]:
    return {g.name: run_group(g, objective, out_dir, force) for g in groups}


# ------------------------------------------------------------------ the grids


def group_step_fixed(objective: RidgeObjective) -> list[RunSpec]:
    """Constant steps expressed through L, including one that diverges."""
    L, mu = objective.L, objective.mu
    choices = {
        "t = 2.1/L": 2.1 / L,  # above the 2/L threshold, on purpose
        "t = 2/L": 2.0 / L,
        "t = 1.9/L": 1.9 / L,
        "t = 2/(L+mu)": 2.0 / (L + mu),
        "t = 1/L": 1.0 / L,
        "t = 0.5/L": 0.5 / L,
        "t = 0.1/L": 0.1 / L,
        "t = 0.01/L": 0.01 / L,
    }
    return [
        RunSpec(
            make_direction=SteepestDescent,
            # Both loop variables are bound as defaults: the lambda runs later,
            # inside run_spec, when `name` would otherwise be the last key.
            make_step=lambda _, t=t, name=name: Fixed(t, label=name),
            label=f"GD ({name})",
            max_iter=5000,
            params={"t": t, "t_label": name},
        )
        for name, t in choices.items()
    ]


def group_step_blind(objective: RidgeObjective) -> list[RunSpec]:
    """The step sizes the brief suggests trying by hand, without computing L.

    Kept as its own group so chapter 4 can put the hand-picked values next to
    2/(L+mu) and say how far the search would have had to go to find it.
    """
    return [
        RunSpec(
            make_direction=SteepestDescent,
            make_step=lambda _, t=t: Fixed(t),
            label=f"GD (t = {t:g}, chosen without L)",
            max_iter=5000,
            params={"t": t, "blind": True},
        )
        for t in (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    ]


def group_step_armijo(objective: RidgeObjective) -> list[RunSpec]:
    """The Armijo grid of section 5.2, plus the probe count each setting costs."""
    L = objective.L
    specs = []
    for c in (1e-4, 0.1, 0.3):
        for rho in (0.5, 0.8, 0.9):
            for t0_name, t0 in (("t0 = 1", 1.0), ("t0 = 10/L", 10.0 / L)):
                specs.append(
                    RunSpec(
                        make_direction=SteepestDescent,
                        make_step=lambda _, c=c, rho=rho, t0=t0: Armijo(c=c, rho=rho, t0=t0),
                        label=f"GD (Armijo c={c:g}, rho={rho:g}, {t0_name})",
                        max_iter=2000,
                        params={"c": c, "rho": rho, "t0": t0, "t0_label": t0_name},
                    )
                )
    return specs


def group_momentum_formula(objective: RidgeObjective) -> list[RunSpec]:
    """Three momentum rules against plain gradient descent at the same step."""
    step = 1.0 / objective.L
    directions: dict[str, MakeDirection] = {
        "GD": SteepestDescent,
        "AGD (beta from t, mu)": lambda o: Nesterov(o, "strongly_convex"),
        "AGD (beta = (k-1)/(k+2))": lambda o: Nesterov(o, "nesterov"),
        "AGD (beta = 0.9)": lambda o: Nesterov(o, "constant", beta=0.9),
    }
    return [
        RunSpec(
            make_direction=make,
            make_step=lambda _: Fixed(step, label="t = 1/L"),
            label=f"{name}, t = 1/L",
            max_iter=3000,
            params={"direction": name, "t": step},
        )
        for name, make in directions.items()
    ]


def group_momentum_restart(objective: RidgeObjective) -> list[RunSpec]:
    """Adaptive restart on and off, for each momentum rule."""
    step = 1.0 / objective.L
    specs = []
    for rule in ("strongly_convex", "nesterov", "constant"):
        for restart in (False, True):
            specs.append(
                RunSpec(
                    make_direction=lambda o, r=rule, rs=restart: Nesterov(o, r, restart=rs),
                    make_step=lambda _: Fixed(step, label="t = 1/L"),
                    max_iter=3000,
                    params={"rule": rule, "restart": restart},
                )
            )
    return specs


def group_batch_size(objective: RidgeObjective) -> list[RunSpec]:
    """Batch sizes, each with the step its own smoothness constant allows.

    A single step shared across batch sizes is a separate comparison, included
    here as the `shared step` runs: L_max is far larger than L, so a step that is
    safe for the full gradient makes the smallest batches diverge.
    """
    specs = []
    for batch in (32, 256, 512, 1024, 2048):
        step = 1.0 / batch_smoothness(objective, batch)
        specs.append(
            RunSpec(
                make_direction=lambda o, b=batch: MiniBatch(o, b, seed=0),
                make_step=lambda _, s=step: Decay("constant", s),
                label=f"SGD (B = {batch}, eta = 1/L_B)",
                max_epochs=40,
                params={"batch_size": batch, "eta0": step, "step_rule": "1/L_B"},
            )
        )
    shared = 1.0 / objective.L
    for batch in (32, 256, 2048):
        specs.append(
            RunSpec(
                make_direction=lambda o, b=batch: MiniBatch(o, b, seed=0),
                make_step=lambda _: Decay("constant", shared),
                label=f"SGD (B = {batch}, eta = 1/L shared)",
                max_epochs=40,
                params={"batch_size": batch, "eta0": shared, "step_rule": "1/L shared"},
            )
        )
    return specs


def group_batch_eta(objective: RidgeObjective, batch: int = 256) -> list[RunSpec]:
    """A logarithmic grid of initial step sizes at one batch size.

    Section 5.2 asks for this sweep, and it is the evidence for whether the
    theoretical value 1/L_B is worth computing. Measured here, 1/L_B lands at
    0.0013 and the empirical optimum at 0.001, so the bound is well calibrated
    despite L_max exceeding L by four orders of magnitude.
    """
    reference = 1.0 / batch_smoothness(objective, batch)
    grid = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1]
    specs = [
        RunSpec(
            make_direction=lambda o, b=batch: MiniBatch(o, b, seed=0),
            make_step=lambda _, e=eta: Decay("constant", e),
            label=f"SGD (B = {batch}, eta = {eta:g})",
            max_epochs=40,
            params={"batch_size": batch, "eta0": eta},
        )
        for eta in grid
    ]
    specs.append(
        RunSpec(
            make_direction=lambda o, b=batch: MiniBatch(o, b, seed=0),
            make_step=lambda _, e=reference: Decay("constant", e),
            label=f"SGD (B = {batch}, eta = 1/L_B = {reference:.4g})",
            max_epochs=40,
            params={"batch_size": batch, "eta0": reference, "is_reference": True},
        )
    )
    return specs


def group_batch_schedule(objective: RidgeObjective, batch: int = 256, seeds=(0, 1, 2, 3, 4)) -> list[RunSpec]:
    """The four deterministic schedules of section 4.3, over five seeds.

    Five seeds because the curves are random: the report shows the median with a
    min-max band, and a single seed would invite the reader to over-read one run.
    """
    base = 1.0 / batch_smoothness(objective, batch)
    steps_per_epoch = objective.n // batch

    # The decay rate is the parameter that decides this comparison, and picking
    # it by eye gets the answer backwards. With gamma = 1/steps_per_epoch, which
    # looks like a natural choice, eta falls to 3e-6 by epoch 400 and every
    # decaying rule freezes above the constant step's noise floor. Theory for a
    # strongly convex objective asks for eta_k of order 1/(mu k), which fixes
    # gamma at mu * eta_0; that value is 30 times smaller here and reverses the
    # ranking. Both are kept so the report can show the sensitivity rather than
    # assert a conclusion that depends on an arbitrary constant.
    gamma_theory = objective.mu * base
    rules: dict[str, Callable[[], StepRule]] = {
        "constant": lambda: Decay("constant", base),
        "inverse (gamma = mu*eta0)": lambda: Decay("inverse", base, gamma=gamma_theory),
        "inverse (gamma = 1/epoch)": lambda: Decay("inverse", base, gamma=1.0 / steps_per_epoch),
        "sqrt": lambda: Decay("sqrt", base),
        "staircase": lambda: Decay("staircase", base, period=40 * steps_per_epoch),
    }
    return [
        RunSpec(
            make_direction=lambda o, s=seed: MiniBatch(o, batch, seed=s),
            make_step=lambda _, make=make: make(),
            label=f"SGD (B = {batch}, {name}, seed {seed})",
            # 40 epochs is too short to see the crossover: the constant step is
            # still ahead there and a report written on that budget would say
            # the wrong thing.
            max_epochs=200,
            params={"batch_size": batch, "schedule": name, "seed": seed, "seed_group": name},
        )
        for name, make in rules.items()
        for seed in seeds
    ]


def group_newton_damping(objective: RidgeObjective) -> list[RunSpec]:
    """Full step against damped, and refactoring against reusing the Hessian."""
    directions: dict[str, MakeDirection] = {
        "Newton": lambda o: NewtonStep(o),
        "Newton (Hessian reused)": lambda o: NewtonStep(o, reuse_factorization=True),
    }
    steps: dict[str, MakeStep] = {
        "t = 1": lambda _: Fixed(1.0, label="t = 1"),
        "Armijo": lambda _: Armijo(c=1e-4, rho=0.5, t0=1.0),
    }
    return product(directions, steps, max_iter=50, repeats=3)


def group_headline(objective: RidgeObjective) -> list[RunSpec]:
    """The best configuration of each method, timed carefully.

    The winners come from the sweep groups above, read off by `best_of` on time
    to reach 1e-6: t = 1.9/L for a constant step, Armijo with rho = 0.5 and
    c = 0.3, Nesterov with beta from t and mu, batch 2048 for SGD, and Newton
    reusing its factorisation. Fixing them here rather than rediscovering them
    means the headline figure can be regenerated at any scale.

    Timing is the whole point of this group, so every run is repeated three times
    and the median kept. Iteration budgets are the counts each configuration
    needed on the sweep problem, rounded up.
    """
    L = objective.L
    return [
        RunSpec(SteepestDescent, lambda o: Fixed(1.9 / o.L, label="t = 1.9/L"),
                label="GD (t = 1.9/L)", max_iter=2500, repeats=3,
                params={"method": "GD, fixed step"}),
        RunSpec(SteepestDescent, lambda o: Armijo(c=0.3, rho=0.5, t0=1.0),
                label="GD (Armijo c=0.3, rho=0.5, t0=1)", max_iter=700, repeats=3,
                params={"method": "GD, backtracking"}),
        RunSpec(lambda o: Nesterov(o, "strongly_convex", restart=True),
                lambda o: Fixed(1.0 / o.L, label="t = 1/L"),
                label="AGD (beta from t, mu, restart)", max_iter=400, repeats=3,
                params={"method": "AGD"}),
        RunSpec(lambda o: MiniBatch(o, 2048, seed=0),
                lambda o: Decay("constant", 1.0 / batch_smoothness(o, 2048)),
                label="SGD (B = 2048, eta = 1/L_B)", max_epochs=40, repeats=3,
                params={"method": "SGD"}),
        RunSpec(lambda o: NewtonStep(o, reuse_factorization=True),
                lambda o: Fixed(1.0, label="t = 1"),
                label="Newton (Hessian reused, t = 1)", max_iter=10, repeats=3,
                params={"method": "Newton"}),
    ]


BUILDERS: dict[str, tuple[str, Callable[[RidgeObjective], list[RunSpec]]]] = {
    "step-fixed": ("Constant step sizes for gradient descent", group_step_fixed),
    "step-blind": ("Step sizes picked without computing L", group_step_blind),
    "step-armijo": ("Backtracking configurations", group_step_armijo),
    "momentum-formula": ("Three momentum rules", group_momentum_formula),
    "momentum-restart": ("Adaptive restart on and off", group_momentum_restart),
    "batch-size": ("Mini-batch sizes", group_batch_size),
    "batch-eta": ("Initial step sizes for SGD", group_batch_eta),
    "batch-schedule": ("Step-size schedules for SGD", group_batch_schedule),
    "newton-damping": ("Newton: damping and Hessian reuse", group_newton_damping),
    "headline": ("Best configuration of each method", group_headline),
}

GROUPS = [ExperimentGroup(name, title, build) for name, (title, build) in BUILDERS.items()]


# ------------------------------------------------------ picking and reporting


def best_of(
    records: Sequence[RunRecord],
    threshold: float = 1e-6,
    key: str = "time",
) -> RunRecord:
    """The configuration that reaches `threshold` fastest, by time or by iterations.

    Runs that never reach the threshold are ranked last, which is the honest
    treatment of a diverging configuration or of SGD sitting on its plateau.
    """
    if key not in ("time", "iters"):
        raise ValueError(f"key must be 'time' or 'iters', got {key!r}")

    def rank(record: RunRecord) -> tuple[int, float]:
        reached = record.time_to_reach(threshold) if key == "time" else record.iters_to_reach(threshold)
        return (1, record.final_gap) if reached is None else (0, float(reached))

    return min(records, key=rank)


def headline_group(chosen: dict[str, RunSpec]) -> ExperimentGroup:
    """The summary comparison, built after the sweeps have named a winner each.

    Timing is what this group is for, so every configuration is repeated three
    times and the median kept.
    """
    def build(objective: RidgeObjective) -> list[RunSpec]:
        specs = []
        for name, spec in chosen.items():
            spec.repeats = max(spec.repeats, 3)
            spec.label = spec.label or name
            specs.append(spec)
        return specs

    return ExperimentGroup("headline", "Best configuration of each method", build)


def summary_table(records: Sequence[RunRecord], threshold: float = 1e-6) -> list[dict]:
    """Rows for the table of section 5.4, ready for pandas."""
    rows = []
    for record in records:
        rows.append(
            {
                "method": record.direction_label,
                "configuration": record.step_label,
                "status": record.status,
                f"iters_to_{threshold:g}": record.iters_to_reach(threshold),
                f"seconds_to_{threshold:g}": record.time_to_reach(threshold),
                "final_gap": record.final_gap,
                "iterations_run": record.iters[-1],
                "total_seconds": record.total_time,
                "fevals_per_iter": record.fevals_per_iter,
                "epochs": float(record.epochs[-1]) if record.is_stochastic else None,
            }
        )
    return rows


# -------------------------------- experiments that vary the objective itself


def _rmse(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    residual = X @ w - y
    return float(np.sqrt(residual @ residual / len(y)))


def run_lambda_sweep(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambdas: Sequence[float] = (1e-3, 1e-2, 3.1623e-2, 1e-1, 1.0, 10.0),
    max_iter: int = 4000,
    out_dir: str | Path = RESULTS_DIR,
    force: bool = False,
    verbose: bool = True,
) -> tuple[list[RunRecord], list[dict]]:
    """Section 5.6: what lambda does to convergence speed and to test error.

    This group varies the objective rather than the algorithm, so it sits outside
    the Direction-by-StepRule product and gets its own runner. Every run uses the
    step 2/(L+mu) of its own problem, otherwise the comparison would measure the
    step size instead of lambda.
    """
    path = Path(out_dir) / "lambda-kappa.json"
    log = print if verbose else (lambda *a, **k: None)
    if path.exists() and not force:
        records, meta = load_records(path)
        log(f"lambda-kappa: loaded {len(records)} runs from {path}")
        return records, meta.get("table", [])

    records, table = [], []
    for lam in lambdas:
        objective = RidgeObjective(X, y, lam)
        warm_up(objective)
        row = {
            "lam": lam,
            "mu": objective.mu,
            "kappa": objective.kappa,
            "f_star": objective.f_star,
            "rmse_test": _rmse(X_test, y_test, objective.w_star),
        }
        for name, make_direction in (
            ("GD", SteepestDescent),
            ("AGD", lambda o: Nesterov(o, "strongly_convex", restart=True)),
        ):
            step = 2.0 / (objective.L + objective.mu) if name == "GD" else 1.0 / objective.L
            record = iterate(
                objective, make_direction(objective), Fixed(step),
                max_iter=max_iter, warmup=False,
                label=f"{name} (lambda = {lam:g}, kappa = {objective.kappa:.0f})",
            )
            record.meta.update({"lam": lam, "kappa": objective.kappa, "method": name})
            records.append(record)
            row[f"iters_{name}"] = record.iters_to_reach(1e-6)
            row[f"seconds_{name}"] = record.time_to_reach(1e-6)
        table.append(row)
        log(
            f"   lambda={lam:<9.4g} mu={row['mu']:.5f}  kappa={row['kappa']:8.1f}  "
            f"GD={row['iters_GD']}  AGD={row['iters_AGD']}  rmse={row['rmse_test']:.4f}"
        )

    save_records(records, path, meta={"group": "lambda-kappa",
                                      "title": "Effect of the regularisation constant",
                                      "table": table})
    return records, table


def run_scaling_comparison(
    variants: dict,
    lam: float,
    max_iter: int = 4000,
    out_dir: str | Path = RESULTS_DIR,
    force: bool = False,
    verbose: bool = True,
) -> tuple[list[RunRecord], list[dict]]:
    """Section 5.7: the same rows encoded three ways, run with the same method.

    Each variant gets the step 1/L of its own problem, so what the figure shows
    is the effect of the encoding on kappa, not a badly chosen step. The three
    are genuinely different objectives: the Ridge penalty means something else
    when the columns carry different units, so the solutions differ too.
    """
    path = Path(out_dir) / "scaling-kappa.json"
    log = print if verbose else (lambda *a, **k: None)
    if path.exists() and not force:
        records, meta = load_records(path)
        log(f"scaling-kappa: loaded {len(records)} runs from {path}")
        return records, meta.get("table", [])

    records, table = [], []
    for name, (design, held) in variants.items():
        objective = RidgeObjective(design.X, design.y, lam)
        warm_up(objective)
        record = iterate(
            objective, SteepestDescent(objective), Fixed(1.0 / objective.L, label="t = 1/L"),
            max_iter=max_iter, warmup=False,
            label=f"GD, {name} (kappa = {objective.kappa:.3g})",
        )
        record.meta.update({"scaling": name, "kappa": objective.kappa})
        records.append(record)
        table.append(
            {
                "scaling": name,
                "L": objective.L,
                "mu": objective.mu,
                "kappa": objective.kappa,
                "iters_to_1e-6": record.iters_to_reach(1e-6),
                "final_gap": record.final_gap,
                "status": record.status,
                "rmse_test": _rmse(held.X, held.y, objective.w_star),
            }
        )
        log(
            f"   {name:<12} L={objective.L:11.4g}  mu={objective.mu:9.5g}  "
            f"kappa={objective.kappa:11.4g}  iters={table[-1]['iters_to_1e-6']}  "
            f"gap={record.final_gap:.2e}"
        )

    save_records(records, path, meta={"group": "scaling-kappa",
                                      "title": "Effect of column scaling on the conditioning",
                                      "table": table})
    return records, table


def main() -> None:
    """Run every group at one scale, resuming whatever is already on disk."""
    import argparse

    from .dataset import FULL, SWEEP, load_processed

    parser = argparse.ArgumentParser(description="Run the experiment grids.")
    parser.add_argument("--scale", choices=("sweep", "full"), default="sweep")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--only", nargs="*", help="group names; default is all of them")
    parser.add_argument("--force", action="store_true", help="rerun groups already on disk")
    args = parser.parse_args()

    scale = SWEEP if args.scale == "sweep" else FULL
    objective, X_test, y_test, _ = load_processed(scale=scale)
    print(
        f"scale={args.scale}  n={objective.n:,}  d={objective.d}  "
        f"lam={objective.lam:.5g}  L={objective.L:.4f}  mu={objective.mu:.6f}  "
        f"kappa={objective.kappa:.1f}\n"
    )

    groups = [g for g in GROUPS if not args.only or g.name in args.only]
    started = time.perf_counter()
    for group in groups:
        run_group(group, objective, args.out, force=args.force)
        print()

    if not args.only or "library" in args.only:
        from .reference import run_baselines, save_baselines

        path = Path(args.out) / "library.json"
        if path.exists() and not args.force:
            print(f"library: already at {path}")
        else:
            print("library: scikit-learn baselines")
            results = run_baselines(objective, X_test, y_test)
            save_baselines(results, objective, path)

    print(f"\nall groups done in {(time.perf_counter() - started) / 60:.1f} min")


def seed_band(records: Sequence[RunRecord], group_key: str) -> dict[str, dict]:
    """Median with a min-max band across seeds, for the stochastic figures.

    Seeds of one configuration share `meta['seed_group']`; runs are aligned on
    the checkpoint index, which is safe because they share a schedule and budget.
    """
    bands: dict[str, list[RunRecord]] = {}
    for record in records:
        bands.setdefault(record.meta.get(group_key, record.label), []).append(record)

    summary: dict[str, dict] = {}
    for name, group in bands.items():
        length = min(len(r.gaps) for r in group)
        gaps = np.array([r.gaps[:length] for r in group])
        summary[name] = {
            "x_iters": group[0].iters[:length],
            "x_epochs": group[0].epochs[:length].tolist() if group[0].is_stochastic else None,
            "x_time": np.median([r.times[:length] for r in group], axis=0).tolist(),
            "median": np.median(gaps, axis=0).tolist(),
            "low": gaps.min(axis=0).tolist(),
            "high": gaps.max(axis=0).tolist(),
            "n_seeds": len(group),
        }
    return summary

if __name__ == "__main__":
    main()
