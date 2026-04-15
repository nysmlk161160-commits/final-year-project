import json
import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.exceptions import InconsistentVersionWarning

from 五通道模型工具 import (
    C_IN,
    EXT_COLS,
    MODEL_DIR,
    TEST_RATIO_DEFAULT,
    WINDOW_LEN_DEFAULT,
    FIG_DIR,
    flatten_windows,
    format_size,
    ieee_plot_style,
    prepare_dataset,
)


CNN_METRICS_PATH = os.path.join(MODEL_DIR, "cnn_metrics.json")
KNN_METRICS_PATH = os.path.join(MODEL_DIR, "knn_metrics.json")
REG_METRICS_PATH = os.path.join(MODEL_DIR, "regression_metrics.json")

CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_window_model.pt")
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "knn_window_model.pkl")
REG_MODEL_PATH = os.path.join(MODEL_DIR, "regression_window_model.pkl")

WINDOW_LEN = WINDOW_LEN_DEFAULT
INPUT_CHANNELS = 12
FLOAT_BYTES = 4
RANDOM_SEED = 42
TEST_RATIO = TEST_RATIO_DEFAULT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIAG_FIG_PATH = os.path.join(FIG_DIR, "model_fit_diagnostics_ieee.png")

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, c_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_inference_ram_bytes(item):
    name = item["model_name"]
    if name == "CNN":
        input_bytes = INPUT_CHANNELS * WINDOW_LEN * FLOAT_BYTES
        conv1_out = 16 * WINDOW_LEN * FLOAT_BYTES
        conv2_out = 32 * WINDOW_LEN * FLOAT_BYTES
        gap_out = 32 * FLOAT_BYTES
        logits = item.get("n_classes", 25) * FLOAT_BYTES
        temp_overhead = 1024
        return input_bytes + conv1_out + conv2_out + gap_out + logits + temp_overhead

    if name == "Multinomial Logistic Regression":
        input_flat = INPUT_CHANNELS * WINDOW_LEN * FLOAT_BYTES
        logits = item.get("n_classes", 25) * FLOAT_BYTES
        temp_overhead = 512
        return input_flat + logits + temp_overhead

    if name == "kNN":
        input_flat = INPUT_CHANNELS * WINDOW_LEN * FLOAT_BYTES
        train_store = item.get("estimated_deploy_bytes", 0)
        distance_buf = item.get("n_train_windows", 0) * FLOAT_BYTES
        temp_overhead = 512
        return input_flat + train_store + distance_buf + temp_overhead

    return 0


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    y_onehot = np.eye(num_classes, dtype=np.float32)[y_true]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def summarize_confusions(cm: np.ndarray, id2label: dict, topk: int = 3):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), id2label[i], id2label[j]))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs[:topk]


def evaluate_split(y_true: np.ndarray, y_prob: np.ndarray, id2label: dict):
    y_pred = np.argmax(y_prob, axis=1)
    num_classes = y_prob.shape[1]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ll = float(log_loss(y_true, y_prob, labels=np.arange(num_classes)))
    brier = multiclass_brier_score(y_true, y_prob, num_classes)
    confidence = np.max(y_prob, axis=1)
    correct_mask = y_pred == y_true
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "log_loss": ll,
        "brier_score": brier,
        "mean_confidence": float(np.mean(confidence)),
        "mean_confidence_correct": float(np.mean(confidence[correct_mask])) if np.any(correct_mask) else 0.0,
        "mean_confidence_wrong": float(np.mean(confidence[~correct_mask])) if np.any(~correct_mask) else 0.0,
        "error_rate": float(1.0 - acc),
        "confusions": summarize_confusions(cm, id2label),
    }


def predict_proba_cnn(model: nn.Module, X: np.ndarray, batch_size: int = 256):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
            logits = model(xb)
            pb = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(pb)
    return np.vstack(probs)


def diagnose_fit(train_stats: dict, test_stats: dict):
    acc_gap = train_stats["accuracy"] - test_stats["accuracy"]
    f1_gap = train_stats["macro_f1"] - test_stats["macro_f1"]
    loss_gap = test_stats["log_loss"] - train_stats["log_loss"]

    if acc_gap < 0.02 and f1_gap < 0.02 and loss_gap < 0.1:
        verdict = "Good fit: train and test performance are close, with stable generalization."
    elif acc_gap < 0.05 and f1_gap < 0.05:
        verdict = "Acceptable fit: a small generalization gap is present."
    else:
        verdict = "Potential overfitting: training performance is noticeably better than test performance."

    return {
        "acc_gap": float(acc_gap),
        "macro_f1_gap": float(f1_gap),
        "log_loss_gap": float(loss_gap),
        "verdict": verdict,
    }


def plot_fit_diagnostics(items):
    import matplotlib.pyplot as plt

    ieee_plot_style()
    names = [item["model_name"] for item in items]
    train_acc = [item["fit_eval"]["train"]["accuracy"] for item in items]
    test_acc = [item["fit_eval"]["test"]["accuracy"] for item in items]
    train_loss = [item["fit_eval"]["train"]["log_loss"] for item in items]
    test_loss = [item["fit_eval"]["test"]["log_loss"] for item in items]

    x = np.arange(len(names))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    axes[0].bar(x - width / 2, train_acc, width, label="Train Accuracy", color="#0A6C74")
    axes[0].bar(x + width / 2, test_acc, width, label="Test Accuracy", color="#94D2BD")
    axes[0].set_title("Train vs Test Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=10)
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[0].legend(frameon=True, fancybox=False, edgecolor="black")

    axes[1].bar(x - width / 2, train_loss, width, label="Train Log Loss", color="#8C1D40")
    axes[1].bar(x + width / 2, test_loss, width, label="Test Log Loss", color="#E9C46A")
    axes[1].set_title("Train vs Test Log Loss")
    axes[1].set_ylabel("Log Loss")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=10)
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[1].legend(frameon=True, fancybox=False, edgecolor="black")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("black")

    fig.savefig(DIAG_FIG_PATH)
    plt.close(fig)


def score_rank_desc(items, scorer):
    ordered = sorted(items, key=scorer, reverse=True)
    return {item["model_name"]: idx + 1 for idx, item in enumerate(ordered)}


def score_rank_asc(items, scorer):
    ordered = sorted(items, key=scorer)
    return {item["model_name"]: idx + 1 for idx, item in enumerate(ordered)}


paths = [CNN_METRICS_PATH, KNN_METRICS_PATH, REG_METRICS_PATH]
missing = [p for p in paths if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        "The following metric files are missing. Please run the CNN, kNN, and regression training scripts first:\n"
        + "\n".join(missing)
    )

dataset = prepare_dataset(
    include_dynamic=False,
    window_len=WINDOW_LEN,
    test_ratio=TEST_RATIO,
    random_seed=RANDOM_SEED,
)
X_tr, y_tr = dataset["X_tr"], dataset["y_tr"]
X_te, y_te = dataset["X_te"], dataset["y_te"]
id2label = dataset["id2label"]
labels = dataset["labels"]
num_classes = len(labels)

X_tr_flat = flatten_windows(X_tr)
X_te_flat = flatten_windows(X_te)

items = [load_json(path) for path in paths]

cnn = SimpleCNN(num_classes=num_classes, c_in=C_IN).to(DEVICE)
cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
cnn_train_prob = predict_proba_cnn(cnn, X_tr)
cnn_test_prob = predict_proba_cnn(cnn, X_te)

with open(KNN_MODEL_PATH, "rb") as f:
    knn = pickle.load(f)
knn_train_prob = knn.predict_proba(X_tr_flat)
knn_test_prob = knn.predict_proba(X_te_flat)

with open(REG_MODEL_PATH, "rb") as f:
    reg = pickle.load(f)
reg_train_prob = reg.predict_proba(X_tr_flat)
reg_test_prob = reg.predict_proba(X_te_flat)

probs = {
    "CNN": (cnn_train_prob, cnn_test_prob),
    "kNN": (knn_train_prob, knn_test_prob),
    "Multinomial Logistic Regression": (reg_train_prob, reg_test_prob),
}

for item in items:
    item["estimated_infer_ram_bytes"] = estimate_inference_ram_bytes(item)
    train_prob, test_prob = probs[item["model_name"]]
    train_stats = evaluate_split(y_tr, train_prob, id2label)
    test_stats = evaluate_split(y_te, test_prob, id2label)
    item["fit_eval"] = {
        "train": train_stats,
        "test": test_stats,
        "gap": diagnose_fit(train_stats, test_stats),
    }

plot_fit_diagnostics(items)

acc_rank = score_rank_desc(items, scorer=lambda x: x["metrics"]["accuracy"])
f1_rank = score_rank_desc(items, scorer=lambda x: x["metrics"]["f1_macro"])
train_rank = score_rank_asc(items, scorer=lambda x: x["train_time_sec"])
infer_rank = score_rank_asc(items, scorer=lambda x: x["avg_infer_time_ms"])
fit_rank = score_rank_asc(items, scorer=lambda x: abs(x["fit_eval"]["gap"]["acc_gap"]) + x["fit_eval"]["gap"]["log_loss_gap"])

print("=== Three-Model Comparison Summary ===")
header = (
    f"{'Model':<32}"
    f"{'Acc':>8}"
    f"{'MacroF1':>10}"
    f"{'LogLoss':>10}"
    f"{'Brier':>10}"
    f"{'Train(s)':>12}"
    f"{'Infer(ms)':>12}"
    f"{'RAM(est)':>12}"
    f"{'Deploy':>14}"
)
print(header)
print("-" * len(header))
for item in items:
    print(
        f"{item['model_name']:<32}"
        f"{item['fit_eval']['test']['accuracy']:>8.4f}"
        f"{item['fit_eval']['test']['macro_f1']:>10.4f}"
        f"{item['fit_eval']['test']['log_loss']:>10.4f}"
        f"{item['fit_eval']['test']['brier_score']:>10.4f}"
        f"{item['train_time_sec']:>12.3f}"
        f"{item['avg_infer_time_ms']:>12.3f}"
        f"{format_size(item['estimated_infer_ram_bytes']):>12}"
        f"{format_size(item['estimated_deploy_bytes']):>14}"
    )

print("\n=== Fit Diagnostics ===")

for item in items:
    train_stats = item["fit_eval"]["train"]
    test_stats = item["fit_eval"]["test"]
    gap = item["fit_eval"]["gap"]
    print(f"\n[{item['model_name']}]")
    print(f"  Train Accuracy      : {train_stats['accuracy']:.4f}")
    print(f"  Test Accuracy       : {test_stats['accuracy']:.4f}")
    print(f"  Train Macro F1      : {train_stats['macro_f1']:.4f}")
    print(f"  Test Macro F1       : {test_stats['macro_f1']:.4f}")
    print(f"  Train Log Loss      : {train_stats['log_loss']:.4f}")
    print(f"  Test Log Loss       : {test_stats['log_loss']:.4f}")
    print(f"  Test Brier Score    : {test_stats['brier_score']:.4f}")
    print(f"  Mean Confidence     : {test_stats['mean_confidence']:.4f}")
    print(f"  Confidence(correct) : {test_stats['mean_confidence_correct']:.4f}")
    print(f"  Confidence(wrong)   : {test_stats['mean_confidence_wrong']:.4f}")
    print(f"  Accuracy Gap        : {gap['acc_gap']:.4f}")
    print(f"  Macro-F1 Gap        : {gap['macro_f1_gap']:.4f}")
    print(f"  LogLoss Gap         : {gap['log_loss_gap']:.4f}")
    print(f"  Fit assessment      : {gap['verdict']}")
    confusions = test_stats["confusions"]
    if confusions:
        print("  Most frequent confusions:")
        for count, true_lab, pred_lab in confusions:
            print(f"    {true_lab} -> {pred_lab}: {count} cases")
    else:
        print("  Most frequent confusions: none")

print("\n=== Diagnostic Summary ===")
for item in items:
    print(
        f"{item['model_name']}: "
        f"test accuracy {item['fit_eval']['test']['accuracy']:.4f}, "
        f"test log loss {item['fit_eval']['test']['log_loss']:.4f}, "
        f"fit stability rank {fit_rank[item['model_name']]}."
    )

print("\n=== Resource and Deployment Comparison ===")
for item in items:
    print(
        f"{item['model_name']}: "
        f"training time {item['train_time_sec']:.3f} s, "
        f"single-window inference {item['avg_infer_time_ms']:.3f} ms, "
        f"estimated inference RAM {format_size(item['estimated_infer_ram_bytes'])}, "
        f"deployment size {format_size(item['estimated_deploy_bytes'])}."
    )

print("\nDiagnostic figure saved:")
print("  ", DIAG_FIG_PATH)
