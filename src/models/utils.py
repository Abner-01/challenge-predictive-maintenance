import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class FailureDataset(Dataset):
    def __init__(self, examples, feature_cols):
        """
        examples: list of (window_df, label)
        feature_cols: columns used as features in the LSTM
        """
        self.examples = examples
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        window_df, label = self.examples[idx]
        # x shape: (seq_len, num_features)
        x = window_df[self.feature_cols].values

        x = torch.from_numpy(x)
        y = torch.from_numpy(label)
        return x, y


def train_epoch(model, data_iter: DataLoader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for x, y in tqdm(data_iter, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_count += y.size(0)

    return total_loss / total_count, total_correct / total_count


def evaluate(model, data_iter, device):
    """Evaluate the model on the validation set."""
    model.eval()
    total_correct, total_count = 0, 0
    with torch.no_grad():
        for x, y in tqdm(data_iter, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_count += y.size(0)
    return total_correct / total_count
