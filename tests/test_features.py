"""
Tests for part5_feature_engineering.py.

These matter more than any other tests in this suite: the whole project
repeatedly flags lag/rolling data leakage as a real risk (see Part 5's
docstring and feature_engineering_notes.md), so this file exists to prove
that risk didn't actually materialise in the code, not just to assert it
in prose.
"""

import numpy as np
import pandas as pd
import pytest

from part5_feature_engineering import (
    LAGS,
    ROLLING_WINDOWS,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    build_feature_table,
)

TARGET = "value"


def _make_ramp(n_hours: int = 300) -> pd.DataFrame:
    """A strictly increasing target (0, 1, 2, ...) — deliberately simple so
    that any lag/rolling feature touching the "wrong" row is immediately
    obvious from its value, rather than needing a statistical test.
    """
    index = pd.date_range("2016-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({TARGET: np.arange(n_hours, dtype=float)}, index=index)


def test_lag_features_use_only_past_target_values():
    """lag_N at row t must equal the target value exactly N hours before
    t — not t itself, and not anything after t.
    """
    df = _make_ramp()
    out = add_lag_features(df, TARGET, lags=[1, 24])

    # Row 50: lag_1 should be the value from row 49, lag_24 from row 26.
    row = out.iloc[50]
    assert row["lag_1"] == df[TARGET].iloc[49]
    assert row["lag_24"] == df[TARGET].iloc[26]
    # And explicitly NOT the current row's own value (the leakage bug this
    # guards against: forgetting .shift() entirely).
    assert row["lag_1"] != row[TARGET]


def test_rolling_features_exclude_the_current_row():
    """roll_mean_3 at row t must be the mean of the THREE PRIOR rows, never
    including row t's own value — this is exactly the shift(1)-before-
    rolling() pattern the module's docstring calls out as the single most
    common leakage bug in time-series feature engineering.
    """
    df = _make_ramp()
    out = add_rolling_features(df, TARGET, windows=[3])

    # Row 10 (value=10): the three prior rows are 7, 8, 9 -> mean 8.0.
    # If the current row leaked in, the mean would include 10 and be higher.
    assert out.iloc[10]["roll_mean_3"] == pytest.approx(8.0)
    assert out.iloc[10]["roll_mean_3"] != pytest.approx(9.0)  # would indicate leakage


def test_feature_table_has_no_missing_target_values():
    """The brief's suggested test list asks explicitly for this: the final,
    dropna()-ed feature table must have zero missing target values.
    """
    df = _make_ramp(400)
    features = build_feature_table(df, target=TARGET)
    assert features[TARGET].isna().sum() == 0


def test_feature_table_drops_exactly_the_warmup_rows():
    """The longest lag/rolling window is 168 hours (one week) — the feature
    table should drop exactly that many leading rows, not more, not fewer.
    """
    df = _make_ramp(400)
    features = build_feature_table(df, target=TARGET)
    longest_window = max(max(LAGS), max(ROLLING_WINDOWS))
    assert len(features) == len(df) - longest_window


def test_time_features_are_never_null_and_correctly_cyclical():
    """Time-based features are the one feature group that's always
    genuinely known at any forecast origin — this checks the sin/cos
    encoding is actually cyclical (hour 0 and hour 24 should encode
    identically) rather than just present.
    """
    df = _make_ramp(48)
    out = add_time_features(df)

    assert out["hour_sin"].isna().sum() == 0
    assert out["hour_cos"].isna().sum() == 0
    # Hour 0 (row 0) and hour 0 again 24 hours later (row 24) must encode
    # to the same sin/cos pair — that's the entire point of the encoding.
    assert out.iloc[0]["hour_sin"] == pytest.approx(out.iloc[24]["hour_sin"])
    assert out.iloc[0]["hour_cos"] == pytest.approx(out.iloc[24]["hour_cos"])
