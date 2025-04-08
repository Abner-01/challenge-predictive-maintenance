"""LSTM Network for time series classification."""

from typing import Optional

import numpy as np
import torch
from mlflow.entities import RunInfo
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.architectures.lstm_architecture import LSTMClassifier
from src.models.trainer import evaluate, train_model


class LSTMNetwork:
    """LSTM Network for time series classification."""

    def __init__(
        self,
        input_dim: int,
        device: Optional[torch.device] = None,
        experiment_name: Optional[str] = "failure-detection",
        run_name: Optional[str] = None,
        log_model: bool = True,
        epochs: int = 10,
        batch_size: int = 64,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
        lr=0.001,
    ) -> None:
        """Initialize the LSTM network.

        Args:
            input_dim (int): Input feature dimension (excluding dropped columns).
            model_class (type): The PyTorch model class (e.g., LSTMClassifier).
            model_kwargs (dict): Arguments to instantiate the model.
            device (torch.device, optional): Device to train on. Defaults to CUDA if available.
            experiment_name (str, optional): MLflow experiment name.
            run_name (str, optional): MLflow run name.
            log_model (bool, optional): Whether to log model with MLflow.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_model = log_model
        self.model = model = LSTMClassifier(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        ).to(device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.run_info: Optional[RunInfo] = None
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> None:
        """Train the LSTM model on the provided dataset."""
        self.run_info = train_model(
            model=self.model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=self.epochs,
            device=self.device,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            batch_size=self.batch_size,
            experiment_name=self.experiment_name,
        )

    def evaluate(
        self,
        test_dataset: Dataset,
        return_confusion_matrix: bool = True,
    ) -> tuple[float, float, np.ndarray]:
        """Evaluate the trained model on a test dataset."""
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, drop_last=False
        )
        return evaluate(
            model=self.model,
            data_iter=test_loader,
            device=self.device,
            loss_fn=self.loss_fn,
            return_confusion_matrix=return_confusion_matrix,
        )

    def save_model(self, path: str) -> None:
        """Save model weights to disk."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load model weights from disk."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def get_run(self) -> Optional[RunInfo]:
        """Return the MLflow run associated with this training."""
        return self.run_info
