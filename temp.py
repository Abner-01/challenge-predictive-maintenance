import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import create_splits, load_data, merge_multiple_dataframes
from src.features import (
    add_time_since_last_event_features,
    encode_machine_model,
    encode_time,
)
from src.models.dataloader import FailureDataset
from src.models.lstm_architecture import LSTMClassifier
from src.models.trainer import evaluate, train_model
from src.data_utils import validate_no_leakage, find_first_positive, find_last_positive

ignored_columns = [
    "errorID",
    "failure",
    "maint_comp1",
    "maint_comp2",
    "maint_comp3",
    "maint_comp4",
    "failure_comp1",
    "failure_comp2",
    "failure_comp3",
    "failure_comp4",
    "code_error1",
    "code_error2",
    "code_error3",
    "code_error4",
    "code_error5",
    "model",
]
HORIZON = 24
dfs = load_data()
dfs.keys()

final_df = merge_multiple_dataframes(dfs)

final_df = add_time_since_last_event_features(final_df)
final_df = encode_time(final_df)
final_df = encode_machine_model(final_df)


train_split, val_split, test_split = create_splits(
    final_df,
    seq_len=25,
    horizon=HORIZON,
    random_state=42,
    ignore_columns=ignored_columns,
    split_by_time=True,
    enable_scaling=True,
)

if first_positive := find_first_positive(train_split):
    validate_no_leakage(
        first_positive[0],
        dfs["PdM_failures"],
        horizon=HORIZON,
    )

if last_postive := find_last_positive(train_split):
    validate_no_leakage(
        last_postive[0],
        dfs["PdM_failures"],
        horizon=HORIZON,
    )

drop_cols = ["datetime", "machineID"]
train_dataset = FailureDataset(train_split, drop_cols=drop_cols)
val_dataset = FailureDataset(val_split, drop_cols=drop_cols)
test_dataset = FailureDataset(test_split, drop_cols=drop_cols)

input_dim = train_split[0][0].shape[1] - len(drop_cols)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(
    input_size=input_dim,
    hidden_size=128,
    num_layers=2,
    num_classes=2,
    dropout=0.3,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=10,
    device=device,
    loss_fn=loss_fn,
    optimizer=optimizer,
    batch_size=64,
    experiment_name="failure-detection",
)

test_loader = DataLoader(
    test_dataset,
    drop_last=False,
)

result = evaluate(
    model=model,
    data_iter=test_loader,
    device=device,
    loss_fn=loss_fn,
    return_confusion_matrix=True,
)

if len(result) == 3:
    avg_loss, accuracy, conf_matrix = result
    print(conf_matrix)
