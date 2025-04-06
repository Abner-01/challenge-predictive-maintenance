"""LSTM Classifier."""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM Classifier for time series data."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the LSTM classifier.

        Args:
            input_size: Number of features per time step
            hidden_size: Number of hidden units in the LSTM
            num_layers: Number of stacked LSTM layers
            num_classes: Number of output classes (for classification)
            dropout: Dropout probability between LSTM layers (if num_layers > 1)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        lstm_output_size = hidden_size

        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM and fully connected layer."""
        # x: [batch, seq_len, input_size]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, seq_len, hidden]
        last_out = lstm_out[:, -1, :]  # last time step
        return self.fc(last_out)  # [batch, num_classes]
