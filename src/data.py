"""Data Processing Module."""

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore

from consts import DATA_FOLDER, LABEL_COLUMN, NOT_SCALED_COLUMNS


def load_csv_files() -> dict[str, pd.DataFrame]:
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

    df_maint = dataframes["PdM_maint"].copy()
    df_maint["val"] = 1
    df_maint_pivoted = df_maint.pivot_table(
        index=["machineID", "datetime"],
        columns="comp",
        values="val",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    merged_df = merged_df.merge(
        df_maint_pivoted, on=["machineID", "datetime"], how="left"
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


def _generate_failure_examples(
    machine_data: pd.DataFrame,
    seq_len: int,
    horizont: int,
    failure_col: str,
) -> list[tuple[pd.DataFrame, float]]:
    """Generate windowed failure examples for a machine. This labels are used for classification.

    Args:
        machine_data (pd.DataFrame): Time-series data for one machine, sorted by datetime index.
        seq_len (int): Length of the sliding window.
        horizont (int): How many time steps in the future the failure occurs.
        failure_col (str): Name of the column indicating failures (1 for failure, 0 for no failure)
    Returns:
        list[tuple[pd.DataFrame, float]]: list of (window, label=1) tuples.
    """
    examples = []
    failure_indices = machine_data.index[machine_data[failure_col] == 1].tolist()
    for fi in failure_indices:
        end_idx = fi - horizont
        start_idx = end_idx - seq_len
        if start_idx >= 0:
            window_df = machine_data.iloc[start_idx:end_idx]
            examples.append((window_df, 1.0))

    return examples


def _generate_non_failure_examples(
    machine_data: pd.DataFrame,
    seq_len: int,
    horizont: int,
    num_examples: int,
    failure_col: str,
    random_state: Optional[int],
) -> list[tuple[pd.DataFrame, float]]:
    """Generate windowed non-failure examples to balance the dataset. Labels for classification.

    Args:
        machine_data (pd.DataFrame): Time-series data for one machine.
        seq_len (int): Length of the sliding window.
        horizont (int): How many time steps in the future the failure occurs.
        num_examples (int): Number of negative examples to generate.
        failure_col (str): Name of the column indicating failures.
        random_state (Optional[int]): Random seed for reproducibility.

    Returns:
        list[tuple[pd.DataFrame, float]]: list of (window, label=0) tuples.
    """
    if random_state is not None:
        np.random.seed(random_state)

    non_failure_indices = machine_data.index[machine_data[failure_col] == 0].tolist()
    valid_non_failure_indices = [
        idx for idx in non_failure_indices if idx >= (horizont + seq_len)
    ]

    if len(non_failure_indices) < num_examples:
        num_examples = len(valid_non_failure_indices)

    chosen_indices = np.random.choice(
        valid_non_failure_indices, size=num_examples, replace=False
    )
    examples = []

    for nfi in chosen_indices:
        end_idx = nfi - horizont
        start_idx = end_idx - seq_len
        if start_idx >= 0:
            window_df = machine_data.iloc[start_idx:end_idx]
            examples.append((window_df, 0.0))

    return examples


def create_examples_for_machine(
    machine_data: pd.DataFrame,
    seq_len: int,
    horizont: int,
    random_state: Optional[int] = None,
    label_col: str = LABEL_COLUMN,
) -> list[tuple[pd.DataFrame, float]]:
    """Create balanced windowed examples for a single machine's time-series data.

    Args:
        machine_data (pd.DataFrame): Time-series data for one machine, sorted by datetime index.
        seq_len (int): Length of each sliding window.
        horizont (int): How many time steps in the future the failure occurs.
        random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        label_col (str, optional): Column name that marks failures. Defaults to LABEL_COLUMN.

    Returns:
        list[tuple[pd.DataFrame, int]]: A list of (window, label) pairs, where label is 0
        (no failure) or 1 (failure).
    """
    failure_examples = _generate_failure_examples(
        machine_data,
        seq_len,
        horizont,
        label_col,
    )
    non_failure_examples = _generate_non_failure_examples(
        machine_data,
        seq_len,
        horizont,
        len(failure_examples),
        label_col,
        random_state,
    )
    return failure_examples + non_failure_examples


def create_examples_for_split(
    split_df: pd.DataFrame,
    seq_len: int,
    horizont: int,
    random_state: Optional[int] = None,
    label_col: str = LABEL_COLUMN,
) -> list[tuple[pd.DataFrame, float]]:
    """Generate examples for all machines in a data split.

    Args:
        split_df (pd.DataFrame): Data containing multiple machines' time-series records.
        seq_len (int): Length of each sliding window.
        horizont (int): How many time steps in the future the failure occurs.
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
            horizont,
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
    machine_ids = df["machineID"].unique()

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

    train_df = df[df["machineID"].isin(train_machines)].copy()
    val_df = df[df["machineID"].isin(val_machines)].copy()
    test_df = df[df["machineID"].isin(test_machines)].copy()

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

    scaler = StandardScaler()
    scaler.fit(train_df[scale_columns])

    train_df[scale_columns] = scaler.transform(train_df[scale_columns])
    val_df[scale_columns] = scaler.transform(val_df[scale_columns])
    test_df[scale_columns] = scaler.transform(test_df[scale_columns])

    return train_df, val_df, test_df


def select_columns(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    label_col: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Select only the columns to keep, such as feature columns + label columns.

    Args:
        df (pd.DataFrame): The original DataFrame.
        features (Optional[List[str]]): List of feature columns to keep. If None, keeps all.
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

    keep_cols = features + [label_col]
    final_cols = [col for col in df.columns if col in keep_cols]
    return df[final_cols].copy()


def create_splits(
    df: pd.DataFrame,
    seq_len: int,
    horizont: int,
    random_state: Optional[int] = None,
    scale_columns: Optional[list[str]] = None,
    features: Optional[list[str]] = None,
    label_col: str = LABEL_COLUMN,
) -> tuple[
    list[tuple[pd.DataFrame, float]],
    list[tuple[pd.DataFrame, float]],
    list[tuple[pd.DataFrame, float]],
]:
    """Split the dataset into train, validation, and test sets and generate examples.

    Args:
        df (pd.DataFrame): Full dataset containing all machines.
        seq_len (int): Length of each sliding window.
        horizont (int): How many time steps in the future the failure occurs.
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
    train_df, val_df, test_df = _split_by_machine_ids(
        df,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=random_state,
    )
    train_df = select_columns(train_df, features, label_col)
    val_df = select_columns(val_df, features, label_col)
    test_df = select_columns(test_df, features, label_col)

    train_df, val_df, test_df = _scale_df(
        train_df,
        val_df,
        test_df,
        scale_columns=scale_columns,
    )

    train_examples = create_examples_for_split(
        train_df,
        seq_len,
        horizont,
        random_state,
        label_col,
    )
    val_examples = create_examples_for_split(
        val_df,
        seq_len,
        horizont,
        random_state,
        label_col,
    )
    test_examples = create_examples_for_split(
        test_df,
        seq_len,
        horizont,
        random_state,
        label_col,
    )

    return train_examples, val_examples, test_examples


if __name__ == "__main__":
    dfs = load_csv_files()
    final_df = merge_multiple_dataframes(dfs)
    train_split, val_split, test_split = create_splits(
        final_df, seq_len=25, horizont=24, random_state=42
    )
    a = final_df[final_df["machineID"] == 1]
    a[a["failure_flag"] == 1]
    len(train_split)
    # train_split[3]
    train_split[0]


# Change the balancing function to avoid balance test and val

# EDA:
# - Check for missing values
# - Check for duplicates
# - Check correlations
# - Check distribution
# - Checkk autocorlation for lags
# - Check causality
