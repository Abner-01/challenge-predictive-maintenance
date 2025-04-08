"""Module to extract features from the dataset."""

import numpy as np
import pandas as pd

# Get EMA for variace of sensor readings


def hours_since_last_event(
    df: pd.DataFrame,
    event_col: str,
    group_col: str = "machineID",
    time_col: str = "datetime",
) -> pd.Series:
    """Get the number of rows since the last occurrence of a given event, grouped by a machine ID.

    Returns NaN before the first occurrence (unknown history) and continues counting after.
    """
    df = df.sort_values(by=[group_col, time_col])

    def _compute(group: pd.DataFrame) -> pd.Series:
        idxs = group[event_col] == 1
        last_seen = idxs.cumsum()
        result = group.groupby(last_seen).cumcount()
        # Where there’s been no previous event (cumsum == 0), return NaN
        result[last_seen == 0] = pd.NA
        return result

    return (
        df.groupby(group_col, group_keys=False)
        .apply(_compute)
        .astype("Float64")
        .squeeze(axis=0)
    )


def add_time_since_last_event_features(
    df: pd.DataFrame,
    group_col: str = "machineID",
    time_col: str = "datetime",
    prefix: str = "time_since_last_",
    impute_value: float = 10_000.0,
) -> pd.DataFrame:
    """Add time-since-last-event features for all binary columns with underscores."""
    event_cols = [
        col for col in df.columns if "_" in col and df[col].dropna().isin([0, 1]).all()
    ]

    for col in event_cols:
        new_col = f"{prefix}{col}"
        mask_col = f"{new_col}_known"

        time_since = hours_since_last_event(
            df, event_col=col, group_col=group_col, time_col=time_col
        )

        df[mask_col] = time_since.notna().astype("int")
        df[new_col] = time_since.fillna(impute_value)

    return df


def encode_cyclical_feature(
    df: pd.DataFrame, col: str, max_val: int, prefix: str
) -> pd.DataFrame:
    """Encode a cyclical time feature (e.g. hour, day, month) using sine and cosine.

    Args:
        df: DataFrame with the feature.
        col: Name of the column to encode.
        max_val: Maximum value of the feature (e.g., 24 for hours, 7 for weekday).
        prefix: Prefix for new columns.

    Returns:
        df with two new columns: prefix_sin, prefix_cos
    """
    radians = 2.0 * np.pi * df[col] / max_val
    df[f"{prefix}_sin"] = np.sin(radians)
    df[f"{prefix}_cos"] = np.cos(radians)
    return df


def encode_time(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    """Encode a time feature (e.g. datetime) into hour and day of week."""
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df = encode_cyclical_feature(df, "hour", 24, prefix="hour")
    df = encode_cyclical_feature(df, "dayofweek", 7, prefix="dayofweek")
    df = df.drop(columns=["hour", "dayofweek"])

    return df


def encode_machine_model(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    """Encode the model column by extracting the numeric part."""
    df[model_col] = df[model_col].str.extract(r"(\d+)").astype(int)
    df = pd.get_dummies(df, columns=[model_col], prefix=model_col, dtype="float")
    return df


if __name__ == "main":
    from src.data.data_utils import load_data, merge_multiple_dataframes

    dfs = load_data()
    final_df = merge_multiple_dataframes(dfs)
    final_df = add_time_since_last_event_features(final_df)
    final_df = encode_time(final_df)
    final_df = encode_machine_model(final_df)
