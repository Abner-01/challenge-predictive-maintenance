"""Data splitting and preprocessing utilities for time-series data."""

import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore

from consts import DATETIME_COLUMN, LABEL_COLUMN, NOT_SCALED_COLUMNS


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


def _generate_all_non_failure_windows(
    machine_data: pd.DataFrame,
    seq_len: int,
    horizon: int,
    label_col: str,
) -> list[tuple[pd.DataFrame, float]]:
    """Generate all possible non-failure windows without balancing."""
    all_non_failure_windows = []
    total_length = len(machine_data)
    for end_idx in range(seq_len + horizon, total_length):
        window = machine_data.iloc[end_idx - horizon - seq_len : end_idx - horizon]
        label = machine_data.iloc[end_idx][label_col]
        if label == 0:
            window = window.drop(columns=[label_col], errors="ignore")
            all_non_failure_windows.append((window, 0.0))

    return all_non_failure_windows


def create_examples_for_machine(
    machine_data: pd.DataFrame,
    seq_len: int,
    horizon: int,
    random_state: Optional[int] = None,
    label_col: str = LABEL_COLUMN,
    balance: bool = True,
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
    if balance:
        non_failure_examples = _generate_non_failure_examples(
            machine_data,
            seq_len,
            horizon,
            len(failure_examples),
            label_col,
            random_state,
        )
    else:
        non_failure_examples = _generate_all_non_failure_windows(
            machine_data,
            seq_len,
            horizon,
            label_col,
        )
    return failure_examples + non_failure_examples


def create_examples_for_split(
    split_df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    random_state: Optional[int] = None,
    label_col: str = LABEL_COLUMN,
    balance: bool = True,
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
            balance=balance,
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
    time_col: str = DATETIME_COLUMN,
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


class DataSplitter:
    """Split a dataset into train, validation, and test sets.

    Otionally by time or by machine IDs, and create sliding-window examples for time-series
    modeling. Optionally applies standard scaling to numeric columns.

    Args:
        seq_len (int): Length of each sliding window (number of time steps in the past).
        horizon (int): How many time steps in the future until a "failure" event is considered.
        random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        scale_columns (Optional[list[str]], optional): Columns to scale using `StandardScaler`.
            If None, it will pick numeric columns by default, excluding columns in
            ``NOT_SCALED_COLUMNS``. Defaults to None.
        features (Optional[list[str]], optional): Subset of columns to keep for training (besides
            the label). If None, keep all columns (except those in `ignore_columns`). Defaults to
            None.
        ignore_columns (Optional[list[str]], optional): Columns to exclude from the dataset.
            Defaults to None.
        label_col (str, optional): Name of the label column for the failure flag. Defaults to
            ``failure_flag``.
        split_by_time (bool, optional): Whether to split the dataset based on time (True) or by
            machine IDs (False). Defaults to False.
        enable_scaling (bool, optional): Whether to apply standard scaling to numeric columns.
            Defaults to True.
    """

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        random_state: Optional[int] = None,
        scale_columns: Optional[list[str]] = None,
        features: Optional[list[str]] = None,
        ignore_columns: Optional[list[str]] = None,
        label_col: str = LABEL_COLUMN,
        split_by_time: bool = False,
        enable_scaling: bool = True,
        balance_val_test: bool = False,
    ) -> None:
        """Initialize the DataSplitter with parameters for splitting and scaling."""
        self.seq_len = seq_len
        self.horizon = horizon
        self.random_state = random_state
        self.scale_columns = scale_columns
        self.features = features
        self.ignore_columns = ignore_columns
        self.label_col = label_col
        self.split_by_time = split_by_time
        self.enable_scaling = enable_scaling
        self.scaler: Optional[StandardScaler] = None
        self.balance_val_test = balance_val_test

    def create_splits(self, df: pd.DataFrame) -> tuple[
        list[tuple[pd.DataFrame, float]],
        list[tuple[pd.DataFrame, float]],
        list[tuple[pd.DataFrame, float]],
    ]:
        """Split the dataset into train, validation, and test sets, then create sliding-window.

        The split can be performed either by time or by machine IDs, depending on the value of
        ``split_by_time``. After splitting, only columns specified in ``features`` are retained
        (unless explicitly ignored in ``ignore_columns``).
        If ``enable_scaling`` is True, numeric columns are scaled using a :class:`StandardScaler`
        fitted on the training set.

        Args:
            df (pd.DataFrame): The full dataset containing data from all machines.

        Returns:
            tuple: A tuple of three lists (train_examples, val_examples, test_examples). Each list
            contains tuples of the form (window_dataframe, label), where:

            - ``window_dataframe`` is a `pd.DataFrame` of shape (seq_len, number_of_features).
            - ``label`` is a float (0.0 or 1.0) indicating whether the window is a failure window
                or not.
        """
        if self.split_by_time:
            train_df, val_df, test_df = _split_by_time(
                df,
                time_col=DATETIME_COLUMN,
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
                random_state=self.random_state,
            )

        train_df = select_columns(
            train_df, self.features, self.ignore_columns, self.label_col
        )
        val_df = select_columns(
            val_df, self.features, self.ignore_columns, self.label_col
        )
        test_df = select_columns(
            test_df, self.features, self.ignore_columns, self.label_col
        )

        if self.enable_scaling:
            train_df, val_df, test_df = self._scale_df(train_df, val_df, test_df)
        else:
            print(
                "Warning: skipping scaling. Set enable_scaling=True to enable scaling."
            )

        train_examples = create_examples_for_split(
            train_df,
            self.seq_len,
            self.horizon,
            self.random_state,
            self.label_col,
        )
        val_examples = create_examples_for_split(
            val_df,
            self.seq_len,
            self.horizon,
            self.random_state,
            self.label_col,
            balance=self.balance_val_test,
        )
        test_examples = create_examples_for_split(
            test_df,
            self.seq_len,
            self.horizon,
            self.random_state,
            self.label_col,
            balance=self.balance_val_test,
        )

        return train_examples, val_examples, test_examples

    def _scale_df(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit a `StandardScaler` on ``train_df[scale_columns]`` and transform sets.

        The columns to be scaled are determined by ``self.scale_columns`` if provided. If
        ``self.scale_columns`` is None, then numeric columns (excluding those in
        ``NOT_SCALED_COLUMNS``) are automatically selected. Only columns with a min below 0.0 or
        a max above 1.0 are actually scaled.

        Args:
            train_df (pd.DataFrame): Training split.
            val_df (pd.DataFrame): Validation split.
            test_df (pd.DataFrame): Test split.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                The scaled (train_df, val_df, test_df).
        """
        if self.scale_columns is None:
            numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
            for col in NOT_SCALED_COLUMNS:
                if col in numeric_cols:
                    numeric_cols.remove(col)
            self.scale_columns = numeric_cols

        columns_to_scale = [
            col
            for col in self.scale_columns
            if train_df[col].min() < 0.0 or train_df[col].max() > 1.0
        ]
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[columns_to_scale])

        train_df[columns_to_scale] = self.scaler.transform(train_df[columns_to_scale])
        val_df[columns_to_scale] = self.scaler.transform(val_df[columns_to_scale])
        test_df[columns_to_scale] = self.scaler.transform(test_df[columns_to_scale])

        return train_df, val_df, test_df

    def get_scaler(self) -> Optional[StandardScaler]:
        """Return the fitted `StandardScaler`.

        Raises:
            ValueError: If the scaler has not been fitted yet (i.e., ``create_splits``
            has not been called).

        Returns:
            Optional[StandardScaler]: The fitted scaler if available, otherwise None.
        """
        if self.scaler is None:
            raise ValueError(
                "Scaler has not been fitted yet. Call create_splits first."
            )
        return self.scaler

    def save_scaler(self, filepath: str) -> None:
        """Save the fitted scaler to a pickle file.

        Args:
            filepath (str): Path to the file where the scaler should be saved.

        Raises:
            ValueError: If the scaler has not been fitted yet.
        """
        if self.scaler is None:
            raise ValueError(
                "Scaler has not been fitted yet. Call create_splits first."
            )
        with open(filepath, "wb") as f:
            pickle.dump(self.scaler, f)


if __name__ == "__main__":
    from src.data.data_utils import load_data, merge_multiple_dataframes

    dfs = load_data()
    final_df = merge_multiple_dataframes(dfs)
    splitter = DataSplitter(seq_len=25, horizon=24, random_state=42)
    train_split, val_split, test_split = splitter.create_splits(final_df)
