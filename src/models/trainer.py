"""Utility functions for training and evaluating the LSTM model."""

from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import confusion_matrix  # type: ignore
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange


def train_epoch(
    model: nn.Module,
    data_iter: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for a single epoch.

    Args:
        model (nn.Module): The model to train.
        data_iter (DataLoader): DataLoader providing training data.
        loss_fn (nn.Module): Loss function used for optimization.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model weights.
        device (torch.device): Device to perform training on (e.g., "cuda" or "cpu").

    Returns:
        tuple[float, float]: Average training loss and accuracy for the epoch.
    """
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for x, y in tqdm(data_iter, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_count += y.size(0)

    avg_loss = total_loss / total_count
    acc = total_correct / total_count
    return avg_loss, acc


def evaluate(
    model: nn.Module,
    data_iter: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    return_confusion_matrix: bool = False,
) -> tuple[float, float, np.ndarray]:
    """Evaluate the model on a validation or test set.

    Args:
        model (nn.Module): The model to evaluate.
        data_iter (DataLoader): DataLoader providing evaluation data.
        loss_fn (nn.Module): Loss function used to compute evaluation loss.
        device (torch.device): Device to perform evaluation on (e.g., "cuda" or "cpu").
        return_confusion_matrix (bool): If True, also returns the confusion matrix.

    Returns:
        tuple:
            - avg_loss (float): Average evaluation loss.
            - accuracy (float): Evaluation accuracy.
            - conf_matrix (np.ndarray): Confusion matrix (optional).
    """
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(data_iter, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_count += y.size(0)

            if return_confusion_matrix:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count

    if return_confusion_matrix:
        conf_matrix = confusion_matrix(all_labels, all_preds)
        return avg_loss, accuracy, conf_matrix

    return avg_loss, accuracy, np.array([])


def _create_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoader instances for training and validation datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple[DataLoader, DataLoader]: DataLoaders for training and validation.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def _log_epoch_metrics(
    epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float
) -> None:
    """Log and print training and validation metrics for the current epoch.

    Args:
        epoch (int): Current epoch number.
        train_loss (float): Training loss.
        train_acc (float): Training accuracy.
        val_loss (float): Validation loss.
        val_acc (float): Validation accuracy.
    """
    tqdm.write(
        f"Epoch {epoch:02d} "
        f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
        f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
    )
    mlflow.log_metrics(
        {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        },
        step=epoch,
    )


def _log_hyperparameters(
    epochs: int, batch_size: int, loss_fn: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    """Log model hyperparameters to MLflow.

    Args:
        epochs (int): Total number of training epochs.
        batch_size (int): Number of samples per batch.
        loss_fn (nn.Module): Loss function used.
        optimizer (torch.optim.Optimizer): Optimizer instance.
    """
    mlflow.log_params(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "loss_fn": loss_fn.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "lr": optimizer.param_groups[0]["lr"],
        }
    )


def _log_trained_model(
    model: nn.Module, train_loader: DataLoader, device: torch.device
) -> None:
    """Log the trained model to MLflow with input signature and example.

    Args:
        model (nn.Module): Trained model instance.
        train_loader (DataLoader): DataLoader used for accessing a sample input.
        device (torch.device): Device on which the model runs (CPU/GPU).
    """
    mlflow.log_params(
        {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
            "num_classes": model.num_classes,
        }
    )
    example_batch = next(iter(train_loader))[0]
    input_example = example_batch[:1].to(device)

    signature = mlflow.models.infer_signature(
        input_example.cpu().numpy(),
        model(input_example).detach().cpu().numpy(),
    )

    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        input_example=input_example.cpu().numpy(),
        signature=signature,
    )


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    batch_size: int = 32,
    num_workers: int = 0,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    log_model: bool = True,
) -> None:
    """Train and evaluate a model with MLflow tracking and logging.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_dataset (Dataset): Dataset for training.
        val_dataset (Dataset): Dataset for validation.
        loss_fn (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the model on.
        epochs (int): Number of epochs to train.
        batch_size (int, optional): Batch size for training and validation. Defaults to 32.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        experiment_name (Optional[str], optional): Name of the MLflow experiment. Defaults to None.
        run_name (Optional[str], optional): Name of the MLflow run. Defaults to None.
        log_model (bool, optional): Whether to log the trained model to MLflow. Defaults to True.
    """
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        _log_hyperparameters(epochs, batch_size, loss_fn, optimizer)
        train_loader, val_loader = _create_dataloaders(
            train_dataset, val_dataset, batch_size, num_workers
        )

        for epoch in trange(1, epochs + 1, desc="Training Epochs"):
            train_loss, train_acc = train_epoch(
                model, train_loader, loss_fn, optimizer, device
            )
            val_loss, val_acc, _ = evaluate(model, val_loader, loss_fn, device)
            _log_epoch_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

        if log_model:
            _log_trained_model(model, train_loader, device)
