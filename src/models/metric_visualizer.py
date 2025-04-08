"""Plot training and validation metrics (loss and accuracy) for a given MLflow run."""

import pandas as pd
import matplotlib.pyplot as plt  # type: ignore


def fetch_metric_history(metric_name: str, client, run_id) -> pd.DataFrame:
    """Fetch metric history from MLflow and return as DataFrame."""
    try:
        history = client.get_metric_history(run_id, metric_name)
        return pd.DataFrame([{"step": m.step, "value": m.value} for m in history])
    except Exception as e:
        print(f"Could not fetch '{metric_name}': {e}")
        return pd.DataFrame(columns=["step", "value"])


def plot_metric(ax, train_df, val_df, title, ylabel, labels=("Train", "Validation")):
    """Plot a metric (e.g., loss or accuracy) on the given axis."""
    if not train_df.empty:
        ax.plot(train_df["step"], train_df["value"], label=f"{labels[0]}", linewidth=2)
    if not val_df.empty:
        ax.plot(val_df["step"], val_df["value"], label=f"{labels[1]}", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)


def plot_loss_accuracy(client, run_id) -> None:
    """Plot training and validation loss and accuracy for a given run."""
    metrics_to_plot = {
        "Loss Curve": ("train_loss", "val_loss"),
        "Accuracy Curve": ("train_accuracy", "val_accuracy"),
    }

    fig, axs = plt.subplots(
        1, len(metrics_to_plot), figsize=(7 * len(metrics_to_plot), 5)
    )

    # If only one subplot, wrap in list for uniform handling
    if len(metrics_to_plot) == 1:
        axs = [axs]

    for ax, (title, (train_name, val_name)) in zip(axs, metrics_to_plot.items()):
        train_df = fetch_metric_history(train_name, client, run_id)
        val_df = fetch_metric_history(val_name, client, run_id)
        plot_metric(ax, train_df, val_df, title, ylabel=title.split()[0])

    plt.tight_layout()
    plt.show()


def print_rec_pres_acc(result: tuple) -> None:
    """Print evaluation metrics including accuracy, recall, and precision."""
    avg_loss, accuracy, conf_matrix = result
    tn, fp, fn, tp = conf_matrix.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print("LSTM Evaluation:")
    print(conf_matrix)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"Precision:{precision:.4f}")
    print(f"Loss:     {avg_loss:.4f}")
