"""Test the feature engineering functions in the src/features directory."""

import pandas as pd
from pandas.testing import assert_series_equal

from src.transformations.features import hours_since_last_event


def test_hours_since_last_event_single_group() -> None:
    """Test the hours since last event function with a single group."""
    data = {
        "datetime": pd.date_range("2024-01-01", periods=6, freq="H"),
        "machineID": [1] * 6,
        "code_event": [0, 0, 1, 0, 0, 1],
    }
    df = pd.DataFrame(data)

    expected = pd.Series([pd.NA, pd.NA, 0, 1, 2, 0], dtype="Float64")
    result = hours_since_last_event(df, "code_event")

    assert_series_equal(
        result.reset_index(drop=True), expected, check_dtype=True, check_names=False
    )


def test_hours_since_last_event_multiple_groups() -> None:
    """Test the hours since last event function with multiple groups."""
    data = {
        "datetime": pd.date_range("2024-01-01", periods=6, freq="H").tolist() * 2,
        "machineID": [1] * 6 + [2] * 6,
        "code_event": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    }
    df = pd.DataFrame(data)

    expected = pd.Series(
        [pd.NA, pd.NA, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2],
        dtype="Float64",
    )

    result = hours_since_last_event(df, "code_event")

    assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False, check_dtype=True
    )


def test_hours_since_last_event_all_zero() -> None:
    """Test the hours since last event function with all zeros."""
    data = {
        "datetime": pd.date_range("2024-01-01", periods=4, freq="H"),
        "machineID": [1] * 4,
        "code_event": [0, 0, 0, 0],
    }
    df = pd.DataFrame(data)

    expected = pd.Series([pd.NA, pd.NA, pd.NA, pd.NA], dtype="Float64")
    result = hours_since_last_event(df, "code_event")

    assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False, check_dtype=True
    )


def test_hours_since_last_event_first_row_event() -> None:
    """Test the hours since last event function with the first row being an event."""
    data = {
        "datetime": pd.date_range("2024-01-01", periods=4, freq="H"),
        "machineID": [1] * 4,
        "code_event": [1, 0, 0, 0],
    }
    df = pd.DataFrame(data)

    expected = pd.Series([0, 1, 2, 3], dtype="Float64")
    result = hours_since_last_event(df, "code_event")

    assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False, check_dtype=True
    )
