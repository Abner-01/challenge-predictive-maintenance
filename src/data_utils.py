"""Data validation functions for time series data."""

import pandas as pd


def validate_no_leakage(
    train_df: pd.DataFrame,
    original_df: pd.DataFrame,
    time_col: str = "datetime",
    label_col: str = "failure",
    horizon: int = 24,
) -> bool:
    """Check that no failure events happen within `horizon` after the last training timestamp.

    Args:
        train_df (pd.DataFrame): The training set returned by the splitter.
        original_df (pd.DataFrame): The full original dataset.
        time_col (str): Name of the datetime column.
        label_col (str): Name of the failure label column (should be binary).
        horizon (int): Buffer window to check for future failure leakage.

    Returns:
        bool: True if no leakage detected, False if there is a leakage.
    """
    train = train_df.copy()
    original = original_df.copy()
    train[time_col] = pd.to_datetime(train[time_col])
    original[time_col] = pd.to_datetime(original[time_col])

    train_end_time = train[time_col].max()
    leakage_window_end = train_end_time + pd.Timedelta(hours=horizon + 1)

    leakage_df = original[
        (original[time_col] > train_end_time)
        & (original[time_col] < leakage_window_end)
        & (original[label_col] == 1)
    ]

    if not leakage_df.empty:
        print("❌ Data leakage detected: failure(s) found within horizon window.")
        print(leakage_df[[time_col, "machineID", label_col]])
        return False

    expected_failure_time = train_end_time + pd.Timedelta(hours=(horizon + 1))
    expected_failures = original[(original[time_col] == expected_failure_time)]

    if expected_failures.empty:
        print(
            "❌ No expected failure found at horizon time. \n Expected failure time:",
            expected_failure_time,
        )
        return False

    print("✅ No data leakage detected.")
    print("✅ Expected failure found at:", expected_failure_time)
    return True


def find_first_positive(
    data: list[tuple[pd.DataFrame, float]],
) -> tuple[pd.DataFrame, float] | None:
    """Return the first tuple where the second element is 1."""
    return next(((df, label) for df, label in data if label == 1), None)


def find_last_positive(
    data: list[tuple[pd.DataFrame, float]],
) -> tuple[pd.DataFrame, float] | None:
    """Return the last tuple where the second element is 1."""
    return next(((df, label) for df, label in reversed(data) if label == 1), None)
