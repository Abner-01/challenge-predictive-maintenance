"""Data Processing Module."""

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore

from consts import DATA_FOLDER, LABEL_COLUMN, NOT_SCALED_COLUMNS


def load_data() -> dict[str, pd.DataFrame]:
    """Load and merge CSV files from the data folder."""
    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

    dataframes = {}
    for file in csv_files:
        name = os.path.basename(file).split(".")[0]
        df = pd.read_csv(file)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
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
        dataframes["PdM_errors"], on=["machineID", "datetime"], how="left"
    )
    merged_df = merged_df.merge(
        dataframes["PdM_failures"],
        on=["machineID", "datetime"],
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
            index_cols=["machineID", "datetime"],
            pivot_col=pivot_col,
            prefix=prefix,
        )
        merged_df = merged_df.merge(pivoted, on=["machineID", "datetime"], how="left")

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


def _generate_failure_examples(
    machine_data: pd.DataFrame,
    seq_len: int,
    horizon: int,
    label_col: str,
) -> list[tuple[pd.DataFrame, float]]:
    """Generate windowed failure examples for a machine. This labels are used for classification.

    Args:
        machine_data (pd.DataFrame): Time-series data for one machine, sorted by datetime index.
        seq_len (int): Length of the sliding window.
        horizon (int): How many time steps in the future the failure occurs.
        label_col (str): Name of the column indicating failures (1 for failure, 0 for no failure)
    Returns:
        list[tuple[pd.DataFrame, float]]: list of (window, label=1) tuples.
    """
    examples = []
    failure_indices = machine_data.index[machine_data[label_col] == 1].tolist()
    for fi in failure_indices:
        end_idx = fi - horizon
        start_idx = end_idx - seq_len
        if start_idx >= 0:
            window_df = machine_data.iloc[start_idx:end_idx]
            window_df = window_df.drop(columns=[label_col], errors="ignore")
            examples.append((window_df, 1.0))

    return examples


def _generate_non_failure_examples(
    machine_data: pd.DataFrame,
    seq_len: int,
    horizon: int,
    num_examples: int,
    label_col: str,
    random_state: Optional[int],
) -> list[tuple[pd.DataFrame, float]]:
    """Generate windowed non-failure examples to balance the dataset. Labels for classification.

    Args:
        machine_data (pd.DataFrame): Time-series data for one machine.
        seq_len (int): Length of the sliding window.
        horizon (int): How many time steps in the future the failure occurs.
        num_examples (int): Number of negative examples to generate.
        label_col (str): Name of the column indicating failures.
        random_state (Optional[int]): Random seed for reproducibility.

    Returns:
        list[tuple[pd.DataFrame, float]]: list of (window, label=0) tuples.
    """
    if random_state is not None:
        np.random.seed(random_state)

    non_failure_indices = machine_data.index[machine_data[label_col] == 0].tolist()
    valid_non_failure_indices = [
        idx for idx in non_failure_indices if idx >= (horizon + seq_len)
    ]

    if len(non_failure_indices) < num_examples:
        num_examples = len(valid_non_failure_indices)

    chosen_indices = np.random.choice(
        valid_non_failure_indices, size=num_examples, replace=False
    )
    examples = []

    for nfi in chosen_indices:
        end_idx = nfi - horizon
        start_idx = end_idx - seq_len
        if start_idx >= 0:
            window_df = machine_data.iloc[start_idx:end_idx]
            window_df = window_df.drop(columns=[label_col], errors="ignore")
            examples.append((window_df, 0.0))

    return examples


def create_examples_for_machine(
    machine_data: pd.DataFrame,
    seq_len: int,
    horizon: int,
    random_state: Optional[int] = None,
    label_col: str = LABEL_COLUMN,
) -> list[tuple[pd.DataFrame, float]]:
    """Create balanced windowed examples for a single machine's time-series data.

    Args:
        machine_data (pd.DataFrame): Time-series data for one machine, sorted by datetime index.
        seq_len (int): Length of each sliding window.
        horizon (int): How many time steps in the future the failure occurs.
        random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        label_col (str, optional): Column name that marks failures. Defaults to LABEL_COLUMN.

    Returns:
        list[tuple[pd.DataFrame, int]]: A list of (window, label) pairs, where label is 0
        (no failure) or 1 (failure).
    """
    failure_examples = _generate_failure_examples(
        machine_data,
        seq_len,
        horizon,
        label_col,
    )
    non_failure_examples = _generate_non_failure_examples(
        machine_data,
        seq_len,
        horizon,
        len(failure_examples),
        label_col,
        random_state,
    )
    return failure_examples + non_failure_examples


def create_examples_for_split(
    split_df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    random_state: Optional[int] = None,
    label_col: str = LABEL_COLUMN,
) -> list[tuple[pd.DataFrame, float]]:
    """Generate examples for all machines in a data split.

    Args:
        split_df (pd.DataFrame): Data containing multiple machines' time-series records.
        seq_len (int): Length of each sliding window.
        horizon (int): How many time steps in the future the failure occurs.
        random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        label_col (str, optional): Column name that marks failures. Defaults to LABEL_COLUMN.

    Returns:
        list[tuple[pd.DataFrame, float]]: list of (window, label) examples for all machines.
    """
    all_examples = []
    for _, machine_data in split_df.groupby("machineID"):
        machine_data = machine_data.reset_index(drop=True)
        examples = create_examples_for_machine(
            machine_data,
            seq_len,
            horizon,
            random_state,
            label_col,
        )
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(examples)  # type: ignore

        all_examples.extend(examples)
    return all_examples


def _split_by_machine_ids(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: Optional[int] = None,
    machine_id_col: str = "machineID",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the unique machine IDs into train/val/test subsets.

    Args:
        df (pd.DataFrame): The full DataFrame containing a "machineID" column.
        train_ratio (float): Fraction of machines for training.
        val_ratio (float): Fraction of machines for validation.
        test_ratio (float): Fraction of machines for testing.
        random_state (Optional[int]): Seed for reproducibility.

    Returns:
        (train_machines, val_machines, test_machines)
        Each is a NumPy array of machine IDs.
    """
    machine_ids = df[machine_id_col].unique()

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must be 1.0, got {total}"
        )
    if random_state is not None:
        np.random.seed(random_state)

    np.random.shuffle(machine_ids)

    train_machines = machine_ids[:60]
    val_machines = machine_ids[60:80]
    test_machines = machine_ids[80:100]

    train_df = df[df[machine_id_col].isin(train_machines)].copy()
    val_df = df[df[machine_id_col].isin(val_machines)].copy()
    test_df = df[df[machine_id_col].isin(test_machines)].copy()

    return train_df, val_df, test_df


def _split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame by datetime, preserving chronological order across all machineIDs.

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime column.
        time_col (str): Name of the datetime column.
        train_ratio (float): Fraction of time for training.
        val_ratio (float): Fraction of time for validation.
        test_ratio (float): Fraction of time for testing.

    Returns:
        Tuple of train_df, val_df, test_df.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    min_time = df[time_col].min()
    max_time = df[time_col].max()
    total_duration = max_time - min_time

    train_end = min_time + total_duration * train_ratio
    val_end = train_end + total_duration * val_ratio

    train_df = df[df[time_col] <= train_end]
    val_df = df[(df[time_col] > train_end) & (df[time_col] <= val_end)]
    test_df = df[df[time_col] > val_end]

    return train_df, val_df, test_df


def _scale_df(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a StandardScaler on train_df[scale_columns], then transform train, val, test.

    Args:
        train_df, val_df, test_df (pd.DataFrame): DataFrames to scale.
        scale_columns (Optional[List[str]]): Columns to scale. If None, pick numeric columns.

    Returns:
        (train_df_scaled, val_df_scaled, test_df_scaled)
    """
    if scale_columns is None:
        numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
        # Remove "machineID" and other columns if present
        for col in NOT_SCALED_COLUMNS:
            if col in numeric_cols:
                numeric_cols.remove(col)
        scale_columns = numeric_cols

    columns_to_scale = [
        col
        for col in scale_columns
        if train_df[col].min() < 0.0 or train_df[col].max() > 1.0
    ]
    scaler = StandardScaler()
    scaler.fit(train_df[columns_to_scale])

    train_df[columns_to_scale] = scaler.transform(train_df[columns_to_scale])
    val_df[columns_to_scale] = scaler.transform(val_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])

    return train_df, val_df, test_df


def select_columns(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    ignore_columns: Optional[list[str]] = None,
    label_col: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Select only the columns to keep, such as feature columns + label columns.

    Args:
        df (pd.DataFrame): The original DataFrame.
        features (Optional[List[str]]): List of feature columns to keep. If None, keeps all.
        ignore_columns (Optional[List[str]]): List of columns to exclude from the final output.
        label_cols (str): Label columns to keep. If None, keeps all.

    Returns:
        pd.DataFrame: DataFrame with only the specified features + label columns.
    """
    if features is None:
        features = [col for col in df.columns if col not in [label_col]]
    elif missing_feats := [col for col in features if col not in df.columns]:
        raise ValueError(f"Missing feature columns: {missing_feats}")

    elif missing_labels := [col for col in [label_col] if col not in df.columns]:
        raise ValueError(f"Missing label column: {missing_labels}")

    keep_cols = set(features + [label_col])
    if ignore_columns:
        keep_cols -= set(ignore_columns)

    final_cols = [col for col in df.columns if col in keep_cols]
    return df[final_cols].copy()


def create_splits(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    random_state: Optional[int] = None,
    scale_columns: Optional[list[str]] = None,
    features: Optional[list[str]] = None,
    ignore_columns: Optional[list[str]] = None,
    label_col: str = LABEL_COLUMN,
    split_by_time: bool = False,
    enable_scaling: bool = True,
) -> tuple[
    list[tuple[pd.DataFrame, float]],
    list[tuple[pd.DataFrame, float]],
    list[tuple[pd.DataFrame, float]],
]:
    """Split the dataset into train, validation, and test sets and generate examples.

    Args:
        df (pd.DataFrame): Full dataset containing all machines.
        seq_len (int): Length of each sliding window.
        horizon (int): How many time steps in the future the failure occurs.
        random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        scale_columns (Optional[list[str]], optional): List of columns to scale.
            If None, automatically select numeric columns.
        label_col (str): Column name that marks failures. Defaults to "failure_flag".


    Returns:
        tuple: A tuple containing training, validation, and test examples as:
            - train_examples (list[tuple[pd.DataFrame, float]])
            - val_examples (list[tuple[pd.DataFrame, float]])
            - test_examples (list[tuple[pd.DataFrame, float]])
    """
    if split_by_time:
        train_df, val_df, test_df = _split_by_time(
            df,
            time_col="datetime",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
    else:
        train_df, val_df, test_df = _split_by_machine_ids(
            df,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=random_state,
        )
    train_df = select_columns(train_df, features, ignore_columns, label_col)
    val_df = select_columns(val_df, features, ignore_columns, label_col)
    test_df = select_columns(test_df, features, ignore_columns, label_col)

    if enable_scaling:
        train_df, val_df, test_df = _scale_df(
            train_df,
            val_df,
            test_df,
            scale_columns=scale_columns,
        )
    else:
        print("Warning skipping scaling. Set enable_scaling=True to enable scaling.")

    train_examples = create_examples_for_split(
        train_df,
        seq_len,
        horizon,
        random_state,
        label_col,
    )
    val_examples = create_examples_for_split(
        val_df,
        seq_len,
        horizon,
        random_state,
        label_col,
    )
    test_examples = create_examples_for_split(
        test_df,
        seq_len,
        horizon,
        random_state,
        label_col,
    )

    return train_examples, val_examples, test_examples


if __name__ == "__main__":
    dfs = load_data()
    final_df = merge_multiple_dataframes(dfs)
    train_split, val_split, test_split = create_splits(
        final_df, seq_len=25, horizon=24, random_state=42
    )

    #! TODO:
    # Automatically download the data
    # Change the balancing function to avoid balance test and val
    # Enforce positve horizonts

    # EDA:
    # - Check for missing values
    # - Check for duplicates
    # - Check correlations
    # - Check stationarity
    # - Check for seasonality
    # - Check for trends
    # - Check distribution
    # - Checkk autocorlation for lags
    # - Check causality
