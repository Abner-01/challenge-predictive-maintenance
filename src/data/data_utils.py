"""Data util functions for time series data."""

import glob
import os

import pandas as pd

from consts import DATA_FOLDER, DATETIME_COLUMN


def load_data() -> dict[str, pd.DataFrame]:
    """Load and merge CSV files from the data folder."""
    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

    dataframes = {}
    for file in csv_files:
        name = os.path.basename(file).split(".")[0]
        df = pd.read_csv(file)
        if DATETIME_COLUMN in df.columns:
            df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
        dataframes[name] = df

    return dataframes


def pivot_categorical_events(
    df: pd.DataFrame,
    index_cols: list[str],
    pivot_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Pivot a categorical column into binary indicator columns."""
    df = df.copy()
    df["val"] = 1

    pivot = df.pivot_table(
        index=index_cols,
        columns=pivot_col,
        values="val",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # Rename pivoted columns with prefix
    pivot.columns = index_cols + [
        f"{prefix}_{col}" for col in pivot.columns[len(index_cols) :]
    ]  # type: ignore
    return pivot


def merge_multiple_dataframes(
    dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge multiple DataFrames on common keys.

    Args:
        dataframes (dict[str, pd.DataFrame]): Dictionary containing DataFrames to merge,
            identified by their names as keys.

    Returns:
        pd.DataFrame: Final merged DataFrame containing combined data from all provided DataFrames.
    """
    merged_df = dataframes["PdM_telemetry"].merge(
        dataframes["PdM_errors"], on=["machineID", DATETIME_COLUMN], how="left"
    )
    merged_df = merged_df.merge(
        dataframes["PdM_failures"],
        on=["machineID", DATETIME_COLUMN],
        how="left",
        suffixes=("_error", "_failure"),
    )

    for df_key, pivot_col, prefix in [
        ("PdM_maint", "comp", "maint"),
        ("PdM_failures", "failure", "failure"),
        ("PdM_errors", "errorID", "code"),
    ]:
        pivoted = pivot_categorical_events(
            dataframes[df_key],
            index_cols=["machineID", DATETIME_COLUMN],
            pivot_col=pivot_col,
            prefix=prefix,
        )
        merged_df = merged_df.merge(
            pivoted, on=["machineID", DATETIME_COLUMN], how="left"
        )

    merged_df = merged_df.merge(dataframes["PdM_machines"], on="machineID", how="left")

    numeric_cols = merged_df.select_dtypes(include=["number"]).columns
    categorical_cols = merged_df.select_dtypes(include=["object", "category"]).columns

    merged_df[categorical_cols] = merged_df[categorical_cols].fillna("none")
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)

    # Label generated for a classification task.
    merged_df["failure_flag"] = merged_df["failure"].apply(
        lambda x: 0 if x == "none" else 1
    )
    return merged_df


def validate_no_label_leakage(
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
        label_col (str): Name of the failure label column.
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


def find_timestamps(data, set_name: str):
    """Find the earliest and latest timestamps in the dataset.

    Args:
        data: List of tuples containing DataFrames and labels.
    """
    timestamps: list[pd.DatetimeIndex] = []
    for df, label in data:
        if label == 1:
            if "datetime" in df.columns:
                timestamps.extend(df["datetime"].tolist())

    timestamps = pd.to_datetime(timestamps)  # type: ignore

    print(f"📅 Earliest timestamp in {set_name}: {timestamps.min()}")  # type: ignore
    print(f"📅 Latest timestamp in {set_name}: {timestamps.max()}")  # type: ignore
    return timestamps.min(), timestamps.max()  # type: ignore


def compare_timestamps(train_split, val_split) -> None:
    """Compare timestamps between training and validation splits."""
    _, train_max = find_timestamps(train_split, "train")
    val_min, _ = find_timestamps(val_split, "validation")

    print(f"Is there any time leakage: {not train_max < val_min}")
