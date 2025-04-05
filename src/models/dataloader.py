"""Module for creating datasets for training models."""

from typing import Optional

import torch
from torch.utils.data import Dataset


class FailureDataset(Dataset):
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
