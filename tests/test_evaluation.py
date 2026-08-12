"""
Tests for part2_problem_definition.py — the shared metrics and train/test
split every other part relies on. If these are wrong, every downstream
MAE/RMSE/MASE/Bias number in the project is wrong too, so this is the
highest-value place to have tests.
"""

import numpy as np
import pandas as pd
import pytest

from part2_problem_definition import (
    DAILY_PERIOD,
    bias,
    evaluate_forecast,
    mae,
    mase,
    rmse,
    train_test_split_by_days,
)


def _make_series(n_hours: int, start: str = "2016-01-01") -> pd.Series:
    """A simple deterministic hourly series for tests — a smooth daily sine
    wave plus a small linear trend, so it has enough structure for MASE's
    seasonal-naive scale to be non-zero without needing real data.
    """
    index = pd.date_range(start, periods=n_hours, freq="h")
    hours = np.arange(n_hours)
    values = 500 + 300 * np.sin(2 * np.pi * hours / 24) + 0.1 * hours
    return pd.Series(values, index=index)


def test_perfect_forecast_scores_zero_on_every_metric():
    """A forecast identical to the actuals should score exactly zero on
    every metric — this is the sanity check the supplementary README asks
    for explicitly ('test that MASE is zero for a perfect forecast').
    """
    series = _make_series(24 * 20)
    train, test = train_test_split_by_days(series, test_days=6)

    perfect_forecast = test.copy()
    result = evaluate_forecast("perfect", test, perfect_forecast, train)

    assert result["MAE"] == pytest.approx(0.0, abs=1e-9)
    assert result["RMSE"] == pytest.approx(0.0, abs=1e-9)
    assert result["MASE"] == pytest.approx(0.0, abs=1e-9)
    assert result["Bias"] == pytest.approx(0.0, abs=1e-9)


def test_forecast_length_matches_test_period():
    """evaluate_forecast's n_points should equal the number of hours in the
    test period — a forecast that's the wrong length is a bug that should
    fail loudly, not silently truncate or pad.
    """
    series = _make_series(24 * 30)
    train, test = train_test_split_by_days(series, test_days=14)
    assert len(test) == 14 * DAILY_PERIOD

    forecast = pd.Series(train.mean(), index=test.index)
    result = evaluate_forecast("mean_baseline", test, forecast, train)
    assert result["n_points"] == len(test)


def test_train_test_split_is_chronological_not_random():
    """Every timestamp in train must come before every timestamp in test —
    this is what makes the split a genuine hold-out rather than a leak."""
    series = _make_series(24 * 25)
    train, test = train_test_split_by_days(series, test_days=10)

    assert len(train) + len(test) == len(series)
    assert train.index.max() < test.index.min()


def test_mase_correctly_ranks_a_better_forecast_below_a_worse_one():
    """A forecast that's very close to the actuals should score a lower
    (better) MASE than a poor forecast (the training mean, which ignores
    all seasonal structure) — this is guaranteed by construction, unlike
    asserting an absolute MASE threshold against an arbitrary synthetic
    series, which depends fragilely on the series' own trend/noise ratio.
    """
    series = _make_series(24 * 20)
    train, test = train_test_split_by_days(series, test_days=5)

    good_forecast = test + np.random.default_rng(0).normal(0, 1.0, size=len(test))
    bad_forecast = pd.Series(train.mean(), index=test.index)

    good_result = evaluate_forecast("near_perfect", test, good_forecast, train)
    bad_result = evaluate_forecast("mean_only", test, bad_forecast, train)

    assert good_result["MASE"] < bad_result["MASE"]


def test_metric_helper_functions_agree_with_evaluate_forecast():
    """mae/rmse/mase/bias should each match the corresponding value
    evaluate_forecast reports — guards against the wrapper and the
    individual functions drifting apart if one is edited without the other.
    """
    series = _make_series(24 * 15)
    train, test = train_test_split_by_days(series, test_days=5)
    forecast = pd.Series(train.mean(), index=test.index)

    result = evaluate_forecast("mean_baseline", test, forecast, train)

    assert mae(test, forecast) == result["MAE"]
    assert rmse(test, forecast) == result["RMSE"]
    assert mase(test, forecast, train) == result["MASE"]
    assert bias(test, forecast) == result["Bias"]
