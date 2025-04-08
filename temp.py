"""Experiments script for LSTM and XGBoost models on the PdM dataset."""

import matplotlib.pyplot as plt
import shap  # type: ignore
import torch
import xgboost as xgb
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.data_spliter import DataSplitter
from src.data.data_utils import (
    find_first_positive,
    find_last_positive,
    load_data,
    merge_multiple_dataframes,
    validate_no_label_leakage,
)
from src.models.datasets import LSTMFailureDataset, XGBoostFailureDataset
from src.models.lstm import LSTMNetwork
from src.transformations.features import (
    add_time_since_last_event_features,
    encode_machine_model,
    encode_time,
)

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
final_df = merge_multiple_dataframes(dfs)

final_df = add_time_since_last_event_features(final_df)
final_df = encode_time(final_df)
final_df = encode_machine_model(final_df)

splitter = DataSplitter(
    seq_len=25,
    horizon=HORIZON,
    random_state=42,
    ignore_columns=ignored_columns,
    split_by_time=True,
    enable_scaling=True,
)
train_split, val_split, test_split = splitter.create_splits(final_df)

if first_positive := find_first_positive(train_split):
    validate_no_label_leakage(
        first_positive[0],
        dfs["PdM_failures"],
        horizon=HORIZON,
    )

if last_postive := find_last_positive(train_split):
    validate_no_label_leakage(
        last_postive[0],
        dfs["PdM_failures"],
        horizon=HORIZON,
    )


drop_cols = ["datetime", "machineID"]
train_dataset = LSTMFailureDataset(train_split, drop_cols=drop_cols)
val_dataset = LSTMFailureDataset(val_split, drop_cols=drop_cols)
test_dataset = LSTMFailureDataset(test_split, drop_cols=drop_cols)

input_dim = train_split[0][0].shape[1] - len(drop_cols)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm = LSTMNetwork(
    input_dim=input_dim,
    device=device,
    hidden_size=128,
    num_layers=2,
    num_classes=2,
    dropout=0.3,
)
lstm.fit(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)
result = lstm.evaluate(
    test_dataset=test_dataset,
    return_confusion_matrix=True,
)

if len(result) == 3:
    avg_loss, accuracy, conf_matrix = result
    tn, fp, fn, tp = conf_matrix.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print(conf_matrix)
    print(accuracy)
    print(recall)

lstm.get_run()

xgbtrain_dataset = XGBoostFailureDataset(
    train_split, drop_cols=drop_cols, flatten_method="stack"
)
X_train, y_train = xgbtrain_dataset.get_data()
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

xgbtest_dataset = XGBoostFailureDataset(
    test_split, drop_cols=drop_cols, flatten_method="stack"
)
X_test, y_test = xgbtest_dataset.get_data()

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

print("Evaluation Metrics on Test Set:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(
    shap_values, X_test, feature_names=[f"f{i}" for i in range(X_test.shape[1])]
)

# Explain the first prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0], matplotlib=True)
plt.savefig("force_plot.png", bbox_inches="tight")
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    shap_values[0],
    feature_names=[f"f{i}" for i in range(X_test.shape[1])],
)
