"""Turn the Lending Club dump into the one fixed pair (X, y) every experiment shares.

The optimisation questions this project asks only mean something if every method
minimises the same function, so this module runs once and its output in
`data/processed/` is then frozen.

Two scales are produced from one cleaned frame, for the reason set out in section
3.5 of the plan: a full-size problem that meets the brief's request for over a
million samples, and a smaller one on which the parameter grids are affordable.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .objective import RidgeObjective

TARGET = "int_rate"

# Known at underwriting time, which is when the rate is set.
NUMERIC_COLUMNS = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_inq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_sats", "num_tl_op_past_12m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens",
    "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit",
]

CATEGORICAL_COLUMNS = [
    "emp_length", "home_ownership", "verification_status", "purpose",
    "addr_state", "application_type", "initial_list_status", "disbursement_method",
]

DATE_COLUMNS = ["earliest_cr_line", "issue_d"]

# Strings carrying units or an ordinal scale, converted to numbers in `clean`.
UNIT_COLUMNS = ["term"]

READ_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + DATE_COLUMNS + UNIT_COLUMNS + [TARGET]

# Never read. Two different problems, both fatal to the exercise if ignored.
#
# Lending Club sets the interest rate from a lookup table indexed by `sub_grade`,
# so those columns are the target under another name. Measured on this file:
# sub_grade alone explains 95.5% of the variance of int_rate across all years and
# 98.3% within 2016, where the lookup table is fixed. `installment` is computed
# from loan_amnt, term and int_rate by the amortisation formula.
#
# The rest are recorded after the money is disbursed, so they are the future
# relative to the moment the rate is chosen.
LEAKING_COLUMNS = {
    "grade", "sub_grade", "installment",
    "loan_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "debt_settlement_flag", "settlement_amount", "settlement_percentage",
}

# Columns that duplicate another one up to an affine map. They add no
# information and push the smallest Gram eigenvalue to zero, inflating kappa for
# a reason that belongs to the encoding rather than to the data.
#
#   funded_amnt, funded_amnt_inv : nearly equal to loan_amnt
#   fico_range_high              : equal to fico_range_low + 4 on 2,260,227 of
#                                  2,260,668 rows, correlation 0.99999991. Kept
#                                  together they produce a Gram eigenvalue of
#                                  1.07e-7 whose eigenvector is exactly
#                                  (fico_low - fico_high)/sqrt(2).
REDUNDANT_COLUMNS = {"funded_amnt", "funded_amnt_inv", "fico_range_high"}

EMP_LENGTH_MAP = {"< 1 year": 0.0, "10+ years": 10.0}


@dataclass
class ScaleSpec:
    """One of the two problem sizes."""

    name: str
    n_train: int
    n_test: int
    suffix: str = ""

    def path(self, out_dir: Path, array: str) -> Path:
        return out_dir / f"{array}{self.suffix}.npy"


SWEEP = ScaleSpec("sweep", n_train=200_000, n_test=50_000)
FULL = ScaleSpec("full", n_train=1_200_000, n_test=300_000, suffix="_full")


# --------------------------------------------------------------------- loading


def load_raw(
    csv_path: str | Path = "data/raw/accepted_2007_to_2018Q4.csv.gz",
    cache_path: str | Path | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read the accepted-loans file, caching a Parquet copy after the first pass.

    Reading the gzipped CSV takes about 45 seconds even with `usecols`; the
    Parquet copy reloads in a couple of seconds, which matters because the
    notebooks reread this during development.
    """
    csv_path = Path(csv_path)
    # `with_suffix` would only replace `.gz` and leave `.csv` in the middle.
    cache_path = Path(cache_path) if cache_path else csv_path.parent / (
        csv_path.name.split(".")[0] + ".parquet"
    )
    columns = columns or READ_COLUMNS

    if cache_path.exists():
        cached = set(pd.read_parquet(cache_path, columns=[]).columns) or set(
            pd.read_parquet(cache_path).columns
        )
        # A cache written before a column was added to READ_COLUMNS would raise a
        # confusing error deep inside pyarrow, so rebuild it instead.
        if set(columns) <= cached:
            return pd.read_parquet(cache_path, columns=columns)

    frame = pd.read_csv(csv_path, usecols=columns, low_memory=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame


# -------------------------------------------------------------------- cleaning


def _strip_percent(series: pd.Series) -> pd.Series:
    """Some releases store rates as '13.99%' strings, others as floats already."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    return pd.to_numeric(
        series.astype("string").str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def _month_index(series: pd.Series) -> pd.Series:
    """Turn 'Aug-2003' into months since year zero, so differences are meaningful."""
    parsed = pd.to_datetime(series, format="%b-%Y", errors="coerce")
    return parsed.dt.year * 12 + parsed.dt.month


def clean(frame: pd.DataFrame, rare_level_threshold: int = 1000) -> pd.DataFrame:
    """Minimal cleaning: units, dates, rare levels, missing values.

    Deliberately shallow. Feature engineering is outside the scope set in section
    7 of CLAUDE.md, and every extra transformation is one more thing that would
    have to be re-justified when the optimisation results are discussed.
    """
    frame = frame.copy()

    frame[TARGET] = _strip_percent(frame[TARGET])
    frame = frame.dropna(subset=[TARGET])

    frame["revol_util"] = _strip_percent(frame["revol_util"])

    # ' 36 months' -> 36. Ordinal with two levels, so a number rather than a
    # one-hot pair; after standardisation the two encodings are the same column.
    frame["term_months"] = pd.to_numeric(
        frame["term"].astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )

    # '< 1 year' -> 0, '10+ years' -> 10, 'n years' -> n.
    emp = frame["emp_length"].astype("string")
    frame["emp_length_years"] = (
        emp.map(EMP_LENGTH_MAP)
        .fillna(pd.to_numeric(emp.str.extract(r"(\d+)", expand=False), errors="coerce"))
        .astype("float64")
    )

    issued = _month_index(frame["issue_d"])
    frame["credit_history_months"] = issued - _month_index(frame["earliest_cr_line"])
    frame["issue_month_index"] = issued - issued.min()

    for column in CATEGORICAL_COLUMNS:
        if column not in frame:
            continue
        values = frame[column].astype("string").fillna("unknown")
        # home_ownership has ANY, OTHER and NONE with a few hundred rows between
        # them. Kept as their own columns they are nearly constant, which adds a
        # direction the data barely constrains.
        counts = values.value_counts()
        rare = counts[counts < rare_level_threshold].index
        frame[column] = values.where(~values.isin(rare), "other")

    frame = frame.drop(columns=DATE_COLUMNS + UNIT_COLUMNS + ["emp_length"])
    return frame


def numeric_feature_names(frame: pd.DataFrame) -> list[str]:
    derived = ["term_months", "emp_length_years", "credit_history_months", "issue_month_index"]
    return [c for c in NUMERIC_COLUMNS if c in frame] + [c for c in derived if c in frame]


# ------------------------------------------------------------- design matrix


@dataclass
class DesignMatrix:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    y_mean: float
    column_means: np.ndarray
    column_stds: np.ndarray
    medians: np.ndarray
    """Training medians used to fill missing numeric entries, kept so the test
    frame is imputed with the same values rather than with its own."""


def _one_hot(frame: pd.DataFrame, categories: dict[str, list[str]]) -> np.ndarray:
    """One-hot with the first level dropped, using categories fixed on the train set.

    Dropping a level per variable is not cosmetic. Keeping all of them makes the
    columns of each variable sum to the all-ones vector, an exact degenerate
    direction in the Gram matrix; the Ridge term keeps the problem solvable but
    mu collapses to lambda for a reason that belongs to the encoding, not the
    data.
    """
    blocks = []
    for column, levels in categories.items():
        values = frame[column].astype("string").to_numpy()
        for level in levels[1:]:
            blocks.append((values == level).astype(np.float64))
    return np.column_stack(blocks) if blocks else np.empty((len(frame), 0))


def category_levels(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Levels in a fixed order, taken once from the training set of the full scale."""
    return {
        column: sorted(frame[column].astype("string").dropna().unique().tolist())
        for column in CATEGORICAL_COLUMNS
        if column in frame
    }


def build_design_matrix(
    train: pd.DataFrame,
    categories: dict[str, list[str]],
    reference: DesignMatrix | None = None,
    scaling: str = "standardize",
) -> DesignMatrix:
    """Assemble, impute, then rescale the columns.

    Statistics come from the training frame and are reused for the test frame via
    `reference`, so nothing about the test set reaches the training problem.

    `scaling` selects one of three treatments, which is what the experiment of
    section 5.7 compares:

        "standardize"  mean 0, standard deviation 1; the frozen problem
        "center"       mean 0 only, so column spreads stay as they are
        "raw"          neither; annual_inc near 1e6 sits beside dti near 10

    Rescaling is the preprocessing step with the most direct effect on the
    optimisation, because it decides how far apart the Gram eigenvalues sit and
    therefore what kappa is. Splitting centring from scaling separates the two
    effects instead of attributing both to one step.
    """
    if scaling not in ("standardize", "center", "raw"):
        raise ValueError(f"scaling must be standardize, center or raw, got {scaling!r}")
    numeric_names = numeric_feature_names(train)
    numeric = train[numeric_names].to_numpy(dtype=np.float64, na_value=np.nan)

    medians = np.nanmedian(numeric, axis=0) if reference is None else reference.medians
    numeric = np.where(np.isnan(numeric), medians, numeric)

    categorical = _one_hot(train, categories)
    X = np.hstack([numeric, categorical])

    if reference is None:
        means = X.mean(axis=0) if scaling in ("standardize", "center") else np.zeros(X.shape[1])
        if scaling == "standardize":
            stds = X.std(axis=0)
            # A column that is constant on the training set carries no
            # information; dividing by its zero spread would give NaNs.
            stds[stds < 1e-12] = 1.0
        else:
            stds = np.ones(X.shape[1])
    else:
        means, stds = reference.column_means, reference.column_stds

    X = (X - means) / stds

    y = train[TARGET].to_numpy(dtype=np.float64)
    y_mean = float(y.mean()) if reference is None else reference.y_mean
    y = y - y_mean

    names = numeric_names + [
        f"{column}={level}"
        for column, levels in categories.items()
        for level in levels[1:]
    ]
    return DesignMatrix(X, y, names, y_mean, means, stds, medians)


# ------------------------------------------------------- regularisation choice


@dataclass
class LambdaChoice:
    grid: list[float]
    mean_mse: list[float]
    se_mse: list[float]
    lam_min: float
    lam_chosen: float
    rule: str = "one-standard-error"
    note: str = ""


def choose_lambda(
    X: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> LambdaChoice:
    """Five-fold cross-validation with the one-standard-error rule.

    Taking the outright minimum is tempting but unreliable here: the curve is
    usually flat across several decades of lambda, and the minimum of a flat
    curve lands wherever the noise puts it, typically at the smallest lambda in
    the grid and therefore at the largest kappa. The one-standard-error rule
    trades a fraction of a percent of predictive accuracy for a far better
    conditioned optimisation problem, which is the trade section 5.6 examines.

    Each fold forms its Gram matrix once and reuses it for every lambda, so the
    grid costs almost nothing beyond the first solve.
    """
    lambdas = np.logspace(-6, 2, 33) if grid is None else np.asarray(grid, dtype=float)
    n, d = X.shape
    folds = np.array_split(np.random.default_rng(seed).permutation(n), n_folds)

    errors = np.empty((n_folds, len(lambdas)))
    for f, validation in enumerate(folds):
        mask = np.ones(n, dtype=bool)
        mask[validation] = False
        X_tr, y_tr = X[mask], y[mask]
        X_va, y_va = X[validation], y[validation]

        gram = X_tr.T @ X_tr / len(X_tr)
        rhs = X_tr.T @ y_tr / len(X_tr)
        for j, lam in enumerate(lambdas):
            regularised = gram.copy()
            regularised.flat[:: d + 1] += lam
            w = np.linalg.solve(regularised, rhs)
            residual = X_va @ w - y_va
            errors[f, j] = residual @ residual / len(y_va)

    mean = errors.mean(axis=0)
    se = errors.std(axis=0, ddof=1) / np.sqrt(n_folds)

    best = int(np.argmin(mean))
    threshold = mean[best] + se[best]
    within = np.flatnonzero(mean <= threshold)
    chosen = int(within[-1])  # largest lambda still statistically indistinguishable

    return LambdaChoice(
        grid=[float(v) for v in lambdas],
        mean_mse=[float(v) for v in mean],
        se_mse=[float(v) for v in se],
        lam_min=float(lambdas[best]),
        lam_chosen=float(lambdas[chosen]),
        note=(
            f"CV minimum at lambda={lambdas[best]:.3e} with MSE {mean[best]:.6f}; "
            f"one-SE rule picks lambda={lambdas[chosen]:.3e} with MSE {mean[chosen]:.6f}"
        ),
    )


# ----------------------------------------------------------------- the pipeline


@dataclass
class PreparationReport:
    seed: int
    rows_read: int
    rows_after_cleaning: int
    d: int
    lam: LambdaChoice | None = None
    scales: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        payload = {
            "seed": self.seed,
            "rows_read": self.rows_read,
            "rows_after_cleaning": self.rows_after_cleaning,
            "d": self.d,
            "scales": self.scales,
        }
        if self.lam is not None:
            payload["lambda"] = self.lam.__dict__
        return payload


def prepare(
    csv_path: str | Path,
    out_dir: str | Path = "data/processed",
    seed: int = 0,
    scales: tuple[ScaleSpec, ...] = (SWEEP, FULL),
    verbose: bool = True,
) -> PreparationReport:
    """Build both problems and freeze them on disk.

    The larger scale is drawn first and the smaller one is carved out of it, so
    the sweep problem really is a sub-problem of the headline one rather than an
    independent sample. Train and test stay disjoint at both sizes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    started = time.perf_counter()
    raw = load_raw(csv_path)
    log(f"read {len(raw):,} rows x {raw.shape[1]} columns in {time.perf_counter() - started:.0f}s")

    frame = clean(raw)
    del raw
    log(f"after cleaning: {len(frame):,} rows")

    largest = max(scales, key=lambda s: s.n_train + s.n_test)
    needed = largest.n_train + largest.n_test
    if len(frame) < needed:
        raise ValueError(f"need {needed:,} rows, cleaning left {len(frame):,}")

    order = np.random.default_rng(seed).permutation(len(frame))[:needed]
    frame = frame.iloc[order].reset_index(drop=True)
    train_pool = frame.iloc[: largest.n_train]
    test_pool = frame.iloc[largest.n_train :]

    categories = category_levels(train_pool)
    log(f"categorical levels: {sum(len(v) - 1 for v in categories.values())} one-hot columns")

    if not scales:
        raise ValueError("at least one scale is required")

    report = PreparationReport(seed=seed, rows_read=needed, rows_after_cleaning=len(frame), d=0)
    feature_names: list[str] = []

    for scale in scales:
        train = train_pool.iloc[: scale.n_train]
        test = test_pool.iloc[: scale.n_test]

        design = build_design_matrix(train, categories)
        held_out = build_design_matrix(test, categories, reference=design)
        report.d = design.X.shape[1]
        feature_names = design.feature_names

        if report.lam is None:
            # Chosen once, on the smaller problem, then shared. Cross-validation
            # is a model-selection question and does not need the full sample.
            started = time.perf_counter()
            report.lam = choose_lambda(design.X, design.y, seed=seed)
            log(f"lambda: {report.lam.note}  ({time.perf_counter() - started:.0f}s)")

        lam = report.lam.lam_chosen
        objective = RidgeObjective(design.X, design.y, lam)
        summary = objective.summary()

        residual = held_out.X @ objective.w_star - held_out.y
        summary["rmse_test"] = float(np.sqrt(residual @ residual / len(held_out.y)))
        summary["n_test"] = int(len(held_out.y))
        summary["y_mean"] = design.y_mean
        report.scales[scale.name] = summary
        log(
            f"{scale.name:>6}: n={summary['n']:,} d={summary['d']} "
            f"L={summary['L']:.4f} mu={summary['mu']:.6f} kappa={summary['kappa']:.1f} "
            f"f*={summary['f_star']:.6f} rmse={summary['rmse_test']:.4f}"
        )

        np.save(scale.path(out_dir, "X_train"), design.X)
        np.save(scale.path(out_dir, "y_train"), design.y)
        np.save(scale.path(out_dir, "X_test"), held_out.X)
        np.save(scale.path(out_dir, "y_test"), held_out.y)

    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=1))
    (out_dir / "problem_config.json").write_text(json.dumps(report.to_json(), indent=1))
    log(f"wrote {out_dir}/problem_config.json")
    return report


def build_scaling_variants(
    csv_path: str | Path = "data/raw/accepted_2007_to_2018Q4.csv.gz",
    scale: ScaleSpec = SWEEP,
    seed: int = 0,
    variants: tuple[str, ...] = ("raw", "center", "standardize"),
) -> dict[str, tuple[DesignMatrix, DesignMatrix]]:
    """The same rows encoded three ways, for the experiment of section 5.7.

    `data/processed/` holds only the standardised problem, deliberately: it is
    the one every other experiment shares and it must not change. The comparison
    against unscaled columns therefore rebuilds from the raw file rather than
    undoing the scaling, which would not be reversible for the imputed entries.
    """
    frame = clean(load_raw(csv_path))
    needed = scale.n_train + scale.n_test
    order = np.random.default_rng(seed).permutation(len(frame))[:needed]
    frame = frame.iloc[order].reset_index(drop=True)
    train, test = frame.iloc[: scale.n_train], frame.iloc[scale.n_train :]
    categories = category_levels(train)

    out = {}
    for variant in variants:
        design = build_design_matrix(train, categories, scaling=variant)
        held = build_design_matrix(test, categories, reference=design, scaling=variant)
        out[variant] = (design, held)
    return out


def load_processed(
    out_dir: str | Path = "data/processed",
    scale: ScaleSpec = SWEEP,
) -> tuple[RidgeObjective, np.ndarray, np.ndarray, dict]:
    """Rebuild one frozen problem plus its test set."""
    out_dir = Path(out_dir)
    config = json.loads((out_dir / "problem_config.json").read_text())
    lam = config["lambda"]["lam_chosen"]
    objective = RidgeObjective(
        np.load(scale.path(out_dir, "X_train")),
        np.load(scale.path(out_dir, "y_train")),
        lam,
    )
    return (
        objective,
        np.load(scale.path(out_dir, "X_test")),
        np.load(scale.path(out_dir, "y_test")),
        config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/raw/accepted_2007_to_2018Q4.csv.gz")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep-only", action="store_true",
                        help="skip the 1.2M-row problem, useful while iterating")
    args = parser.parse_args()

    prepare(
        args.csv,
        args.out,
        seed=args.seed,
        scales=(SWEEP,) if args.sweep_only else (SWEEP, FULL),
    )


if __name__ == "__main__":
    main()
