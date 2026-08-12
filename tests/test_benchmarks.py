"""
Tests for part3_benchmark_models.py — the five benchmark forecast
functions every later model is compared against.
"""

import numpy as np
import pandas as pd
import pytest

from part3_benchmark_models import (
    drift_forecast,
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)


def _make_train(n_hours: int = 100) -> pd.Series:
    index = pd.date_range("2016-01-01", periods=n_hours, freq="h")
    return pd.Series(np.arange(n_hours, dtype=float), index=index)


def _future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(train.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")


@pytest.mark.parametrize("horizon", [1, 24, 100])
def test_all_benchmarks_return_the_requested_horizon_length(horizon):
    train = _make_train()
    index = _future_index(train, horizon)

    assert len(mean_forecast(train, horizon, index)) == horizon
    assert len(naive_forecast(train, horizon, index)) == horizon
    assert len(seasonal_naive_forecast(train, horizon, index, seasonality=24)) == horizon
    assert len(drift_forecast(train, horizon, index)) == horizon


def test_mean_forecast_is_constant_at_the_training_mean():
    train = _make_train(100)  # values 0..99, mean = 49.5
    index = _future_index(train, 10)
    forecast = mean_forecast(train, 10, index)
    assert (forecast == train.mean()).all()
    assert forecast.iloc[0] == pytest.approx(49.5)


def test_naive_forecast_is_constant_at_the_last_training_value():
    train = _make_train(100)  # last value = 99
    index = _future_index(train, 10)
    forecast = naive_forecast(train, 10, index)
    assert (forecast == train.iloc[-1]).all()
    assert forecast.iloc[0] == 99.0


def test_seasonal_naive_first_cycle_matches_the_last_training_season_exactly():
    """The first `seasonality` forecast values must exactly reproduce the
    last `seasonality` training values — this is the defining property of
    a seasonal-naive forecast, and the thing that would break if the lag
    indexing were off by one.
    """
    train = _make_train(200)
    horizon = 24
    index = _future_index(train, horizon)
    forecast = seasonal_naive_forecast(train, horizon, index, seasonality=24)

    expected = train.iloc[-24:].values
    np.testing.assert_array_equal(forecast.values, expected)


def test_seasonal_naive_tiles_beyond_one_full_cycle():
    """Beyond one season, the recursive implementation should repeat the
    same last-season pattern — forecast step 24 (second cycle, hour 0)
    should equal forecast step 0 (first cycle, hour 0).
    """
    train = _make_train(200)
    horizon = 48  # two full daily cycles
    index = _future_index(train, horizon)
    forecast = seasonal_naive_forecast(train, horizon, index, seasonality=24)

    assert forecast.iloc[0] == forecast.iloc[24]
    assert forecast.iloc[5] == forecast.iloc[29]


def test_drift_forecast_extrapolates_the_correct_linear_slope():
    """A perfectly linear training series (0, 1, 2, ..., 9) has slope 1 —
    the one-step-ahead drift forecast should be exactly 10.
    """
    index = pd.date_range("2016-01-01", periods=10, freq="h")
    train = pd.Series(np.arange(10, dtype=float), index=index)
    future_index = _future_index(train, 3)

    forecast = drift_forecast(train, 3, future_index)
    np.testing.assert_array_almost_equal(forecast.values, [10.0, 11.0, 12.0])
