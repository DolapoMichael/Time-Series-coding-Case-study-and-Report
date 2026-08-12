"""
Tests for part1_data_prep_eda.py — missingness checking and the sum/mean
hourly resampling logic (the choice documented and deliberately deviated
from the supplementary demo in Part 1/Part 2).
"""

import numpy as np
import pandas as pd
import pytest

from part1_data_prep_eda import check_missingness, resample_hourly


def _make_raw_10min(n_rows: int = 60, start: str = "2016-01-01") -> pd.DataFrame:
    """Six rows per hour (10-minute resolution), matching the real
    dataset's structure: Appliances/lights as per-interval Wh readings,
    one sensor column standing in for the many temperature/humidity ones.
    """
    index = pd.date_range(start, periods=n_rows, freq="10min")
    return pd.DataFrame(
        {
            "Appliances": np.full(n_rows, 10.0),  # 10 Wh every 10 minutes
            "lights": np.full(n_rows, 2.0),
            "T1": np.full(n_rows, 20.0),  # constant "temperature" reading
            "rv1": np.random.default_rng(0).normal(size=n_rows),  # dropped by resample_hourly
        },
        index=index,
    )


def test_check_missingness_reports_no_gaps_on_clean_data():
    df = _make_raw_10min(60)  # exactly 10 hours, no gaps
    na_counts = check_missingness(df)
    # No column should show a nonzero count on clean data — the function
    # returns a Series with '(none)' as the only key in that case.
    assert list(na_counts.index) in ([], ["(none)"])


def test_check_missingness_detects_a_genuinely_missing_value():
    df = _make_raw_10min(60)
    df.loc[df.index[5], "T1"] = np.nan
    na_counts = check_missingness(df)
    assert "T1" in na_counts.index
    assert na_counts["T1"] == 1


def test_resample_hourly_sums_energy_and_averages_sensors():
    """The core, deliberate design decision from Part 1: Appliances/lights
    are summed (six 10-Wh readings -> 60 Wh for the hour), while sensor
    columns are averaged (constant 20.0 stays 20.0).
    """
    df = _make_raw_10min(60)  # exactly 10 complete hours
    hourly = resample_hourly(df)

    assert len(hourly) == 10
    assert (hourly["Appliances"] == 60.0).all()  # 6 x 10 Wh summed
    assert (hourly["lights"] == 12.0).all()  # 6 x 2 Wh summed
    assert (hourly["T1"] == 20.0).all()  # constant, so mean == 20.0 regardless


def test_resample_hourly_drops_the_random_benchmark_columns():
    df = _make_raw_10min(60)
    hourly = resample_hourly(df)
    assert "rv1" not in hourly.columns


def test_resampled_data_has_no_missing_target_values():
    df = _make_raw_10min(120)
    hourly = resample_hourly(df)
    assert hourly["Appliances"].isna().sum() == 0
