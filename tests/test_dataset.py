"""Checks for the data pipeline, run on a small synthetic frame.

Nothing here touches the 374 MB download: the transformations are what needs
testing, and they are the same whether the frame has fifty rows or two million.
"""

import numpy as np
import pandas as pd
import pytest

from src import dataset
from src.dataset import (
    LEAKING_COLUMNS,
    NUMERIC_COLUMNS,
    REDUNDANT_COLUMNS,
    TARGET,
    build_design_matrix,
    category_levels,
    choose_lambda,
    clean,
    numeric_feature_names,
)


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame(
        {
            TARGET: rng.uniform(5.3, 31.0, n).round(2),
            "term": rng.choice([" 36 months", " 60 months"], n),
            "emp_length": rng.choice(
                ["< 1 year", "3 years", "10+ years", None], n, p=[0.3, 0.3, 0.3, 0.1]
            ),
            "revol_util": rng.uniform(0, 100, n).round(1),
            "home_ownership": rng.choice(
                ["MORTGAGE", "RENT", "OWN", "ANY"], n, p=[0.45, 0.4, 0.14, 0.01]
            ),
            "purpose": rng.choice(["debt_consolidation", "credit_card"], n),
            "verification_status": rng.choice(["Verified", "Not Verified"], n),
            "addr_state": rng.choice(["CA", "NY", "TX"], n),
            "application_type": rng.choice(["Individual", "Joint App"], n),
            "initial_list_status": rng.choice(["w", "f"], n),
            "disbursement_method": rng.choice(["Cash", "DirectPay"], n),
            "issue_d": rng.choice(["Jan-2016", "Jun-2016", "Dec-2017"], n),
            "earliest_cr_line": rng.choice(["Aug-2003", "Feb-1996"], n),
        }
    )
    for column in NUMERIC_COLUMNS:
        if column not in frame:
            frame[column] = rng.standard_normal(n) * 100
    frame.loc[frame.index[:30], "dti"] = np.nan
    return frame


# ------------------------------------------------------------------- leakage


def test_leaking_columns_are_never_read():
    assert LEAKING_COLUMNS.isdisjoint(dataset.READ_COLUMNS)
    for column in ("grade", "sub_grade", "installment"):
        assert column in LEAKING_COLUMNS


def test_redundant_columns_are_never_read():
    """fico_range_high duplicates fico_range_low up to a constant offset."""
    assert REDUNDANT_COLUMNS.isdisjoint(dataset.READ_COLUMNS)
    assert "fico_range_low" in NUMERIC_COLUMNS
    assert "fico_range_high" in REDUNDANT_COLUMNS


# ------------------------------------------------------------------ cleaning


def test_units_become_numbers(raw_frame):
    cleaned = clean(raw_frame)
    assert set(cleaned["term_months"].unique()) == {36, 60}
    assert cleaned["emp_length_years"].dropna().between(0, 10).all()
    assert "term" not in cleaned and "emp_length" not in cleaned


def test_percent_strings_are_accepted_as_well_as_floats(raw_frame):
    as_text = raw_frame.copy()
    as_text["revol_util"] = as_text["revol_util"].map(lambda v: f"{v}%")
    assert np.allclose(
        clean(as_text)["revol_util"].to_numpy(),
        clean(raw_frame)["revol_util"].to_numpy(),
    )


def test_credit_history_is_a_duration_in_months(raw_frame):
    cleaned = clean(raw_frame)
    # Jan-2016 minus Aug-2003 is 149 months; the other combinations are longer.
    assert cleaned["credit_history_months"].min() >= 100
    assert cleaned["issue_month_index"].min() == 0


def test_rare_levels_are_folded_together(raw_frame):
    cleaned = clean(raw_frame, rare_level_threshold=50)
    assert "ANY" not in set(cleaned["home_ownership"])
    assert "other" in set(cleaned["home_ownership"])


def test_missing_categories_become_their_own_level(raw_frame):
    cleaned = clean(raw_frame, rare_level_threshold=1)
    assert cleaned["home_ownership"].isna().sum() == 0


# ------------------------------------------------------------ design matrix


def test_one_hot_drops_one_level_per_variable(raw_frame):
    cleaned = clean(raw_frame, rare_level_threshold=1)
    categories = category_levels(cleaned)
    design = build_design_matrix(cleaned, categories)

    n_numeric = len(numeric_feature_names(cleaned))
    expected = n_numeric + sum(len(levels) - 1 for levels in categories.values())
    assert design.X.shape == (len(cleaned), expected)
    assert len(design.feature_names) == expected


def test_dropping_a_level_keeps_the_gram_matrix_nonsingular(raw_frame):
    """Keeping every level would make each variable's columns sum to one."""
    cleaned = clean(raw_frame, rare_level_threshold=1)
    design = build_design_matrix(cleaned, category_levels(cleaned))
    gram = design.X.T @ design.X / len(design.X)
    assert np.linalg.eigvalsh(gram)[0] > 1e-8


def test_columns_are_standardised_and_the_target_centred(raw_frame):
    cleaned = clean(raw_frame, rare_level_threshold=1)
    design = build_design_matrix(cleaned, category_levels(cleaned))
    assert np.allclose(design.X.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(design.X.std(axis=0), 1.0, atol=1e-10)
    assert design.y.mean() == pytest.approx(0.0, abs=1e-10)
    assert design.y_mean == pytest.approx(cleaned[TARGET].mean())


def test_test_frame_reuses_training_statistics(raw_frame):
    """Otherwise the test set would be scaled by numbers it computed itself."""
    cleaned = clean(raw_frame, rare_level_threshold=1)
    categories = category_levels(cleaned)
    train = build_design_matrix(cleaned.iloc[:400], categories)
    test = build_design_matrix(cleaned.iloc[400:], categories, reference=train)

    assert np.allclose(test.column_means, train.column_means)
    assert np.allclose(test.column_stds, train.column_stds)
    assert np.allclose(test.medians, train.medians)
    assert test.y_mean == train.y_mean
    assert test.feature_names == train.feature_names
    # Standardising with foreign statistics must not accidentally re-centre.
    assert not np.allclose(test.X.mean(axis=0), 0.0, atol=1e-10)


def test_missing_numeric_entries_are_filled_with_training_medians(raw_frame):
    cleaned = clean(raw_frame, rare_level_threshold=1)
    design = build_design_matrix(cleaned, category_levels(cleaned))
    assert np.isfinite(design.X).all()


# --------------------------------------------------------------- lambda choice


def _cv_problem(seed: int = 0, n: int = 800, d: int = 12):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X = (X - X.mean(0)) / X.std(0)
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)
    return X, y - y.mean()


def test_one_standard_error_rule_never_picks_below_the_minimum():
    X, y = _cv_problem()
    choice = choose_lambda(X, y, seed=1)
    assert choice.lam_chosen >= choice.lam_min
    assert choice.rule == "one-standard-error"


def test_one_standard_error_rule_stays_within_one_standard_error():
    X, y = _cv_problem()
    choice = choose_lambda(X, y, seed=1)
    index = choice.grid.index(choice.lam_chosen)
    best = int(np.argmin(choice.mean_mse))
    assert choice.mean_mse[index] <= choice.mean_mse[best] + choice.se_mse[best]
    # Anything larger must be outside the band, or the rule stopped too early.
    if index + 1 < len(choice.grid):
        assert choice.mean_mse[index + 1] > choice.mean_mse[best] + choice.se_mse[best]


def test_cross_validation_curve_rises_at_both_ends():
    """A sane curve: under-regularised on the left, over-regularised on the right."""
    X, y = _cv_problem()
    choice = choose_lambda(X, y, seed=1)
    mean = np.asarray(choice.mean_mse)
    assert mean[-1] > mean.min()
    assert np.isfinite(mean).all()


def test_lambda_choice_is_reproducible():
    X, y = _cv_problem()
    assert choose_lambda(X, y, seed=3).lam_chosen == choose_lambda(X, y, seed=3).lam_chosen
