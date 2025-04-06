"""Module for creating datasets for training models."""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LSTMFailureDataset(Dataset):
    """Dataset for the LSTM model."""

    def __init__(
        self, sequence_label_pairs, drop_cols: Optional[list[str]] = None
    ) -> None:
        """Initialize the dataset.

        Args:
            sequence_label_pairs: list of tuples (window_df, label)
                where window_df is a DataFrame containing the features for the sequence
                and label is the target variable (0 or 1).
        """
        self.sequence_label_pairs = sequence_label_pairs
        self.drop_cols = drop_cols or []

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequence_label_pairs)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        window_df, label = self.sequence_label_pairs[idx]

        x = window_df.drop(columns=self.drop_cols, errors="ignore").values.astype(
            "float32"
        )
        y = torch.tensor(label, dtype=torch.long)

        return torch.tensor(x), y


class XGBoostFailureDataset:
    """Prepare windowed sequence-label pairs for XGBoost by flattening sequences."""

    def __init__(
        self,
        sequence_label_pairs: list[tuple[pd.DataFrame, float]],
        drop_cols: Optional[list[str]] = None,
        flatten_method: str = "stack",
    ) -> None:
        """Initialize the dataset.

        Args:
            sequence_label_pairs: list of (window_df, label) pairs.
            drop_cols: columns to drop from the window_df before flattening.
            flatten_method: 'stack' flattens all values row-wise,
                            'mean' takes the mean across time (useful for static models).
        """
        self.sequence_label_pairs = sequence_label_pairs
        self.drop_cols = drop_cols or []
        self.flatten_method = flatten_method

        self.x, self.y = self._flatten_sequences()

    def _flatten_sequences(self) -> tuple[np.ndarray, np.ndarray]:
        """Flatten sequences and prepare labels for XGBoost."""
        x_list = []
        y_list = []

        for window_df, label in self.sequence_label_pairs:
            df = window_df.drop(columns=self.drop_cols, errors="ignore")

            if self.flatten_method == "stack":
                # Flatten sequence row-wise: (seq_len, n_features) → (seq_len * n_features,)
                x_flat = df.values.flatten()
            elif self.flatten_method == "mean":
                # Take average over time: (seq_len, n_features) → (n_features,)
                x_flat = np.asarray(df.mean(axis=0).values, dtype=np.float32)
            else:
                raise ValueError(f"Unknown flatten_method: {self.flatten_method}")

            x_list.append(x_flat)
            y_list.append(label)

        return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns x and y arrays for training in XGBoost."""
        return self.x, self.y


def generate_flat_feature_names(
    sample_df: pd.DataFrame, drop_cols: list[str]
) -> list[str]:
    """Generate readable feature names like 'feature_t0', 'feature_t1', ... after flattening."""
    df = sample_df.drop(columns=drop_cols, errors="ignore")
    time_steps = df.shape[0]
    features = df.columns
    return [f"{feat}_t{t}" for t in range(time_steps) for feat in features]
