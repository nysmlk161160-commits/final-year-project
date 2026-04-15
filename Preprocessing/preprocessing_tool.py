import json
import io
import os
import pickle
import re
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)


INCLUDE_DYNAMIC_DEFAULT = False

STATIC_DIR = r"D:\gesture_project\cnn_version_test\五通道最终版本\windows_static_csv_6ch"
DYNAMIC_DIR = r"D:\gesture_project\cnn_version_test\windows_dynamic_csv_6ch"
BASE_OUT = r"D:\gesture_project\cnn_version_test\五通道最终版本"
MODEL_DIR = os.path.join(BASE_OUT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
FIG_DIR = os.path.join(BASE_OUT, "training_curves")
os.makedirs(FIG_DIR, exist_ok=True)

WINDOW_LEN_DEFAULT = 20
TEST_RATIO_DEFAULT = 0.2
RANDOM_SEED_DEFAULT = 42

SIGNAL_COLS = ["a0", "a1", "a2", "a3", "a4", "roll", "pitch"]
EXT_COLS = SIGNAL_COLS + ["da0", "da1", "da2", "da3", "da4"]
C_IN = len(EXT_COLS)

LABEL_RE = re.compile(r"_label_(.+)\.csv$", re.IGNORECASE)
EV_RE = re.compile(r"_event_(\d+)_label_", re.IGNORECASE)


def label_from_filename(fname: str) -> str:
    m = LABEL_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse label from filename: {fname} (expected pattern _label_xxx.csv)")
    return m.group(1)


def event_from_filename(fname: str):
    m = EV_RE.search(fname)
    return m.group(1) if m else None


def load_window(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    for c in SIGNAL_COLS:
        if c not in df.columns:
            raise ValueError(f"{os.path.basename(path)} is missing column: {c}")
    return df[SIGNAL_COLS].values.astype(np.float32)


def load_window_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in SIGNAL_COLS:
        if c not in df.columns:
            raise ValueError(f"{os.path.basename(path)} is missing column: {c}")
    return df[SIGNAL_COLS].copy()


def pad_or_trunc_hold_last(X: np.ndarray, target_len: int) -> np.ndarray:
    t, c = X.shape
    if t >= target_len:
        return X[:target_len]
    if t == 0:
        return np.zeros((target_len, c), dtype=np.float32)
    last = X[-1:, :]
    pad = np.repeat(last, repeats=(target_len - t), axis=0)
    return np.vstack([X, pad])


def add_flex_deltas(X7: np.ndarray) -> np.ndarray:
    if X7.shape[1] != 7:
        raise ValueError(f"add_flex_deltas expects input shape [T, 7], but got {X7.shape}")

    t = X7.shape[0]
    da = np.zeros((t, 5), dtype=np.float32)
    if t > 1:
        da[1:, 0] = X7[1:, 0] - X7[:-1, 0]
        da[1:, 1] = X7[1:, 1] - X7[:-1, 1]
        da[1:, 2] = X7[1:, 2] - X7[:-1, 2]
        da[1:, 3] = X7[1:, 3] - X7[:-1, 3]
        da[1:, 4] = X7[1:, 4] - X7[:-1, 4]
    return np.concatenate([X7, da], axis=1)


def collect_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        print(f"Warning: directory not found, skipped: {folder}")
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".csv")
    ]


def load_many(
    file_list: List[str],
    label2id: Dict[str, int],
    window_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    for path in file_list:
        x7 = load_window(path)
        x7 = pad_or_trunc_hold_last(x7, window_len)
        x12 = add_flex_deltas(x7)
        lab = label_from_filename(os.path.basename(path))
        x_list.append(x12)
        y_list.append(label2id[lab])
    x_arr = np.stack(x_list, axis=0)
    y_arr = np.array(y_list, dtype=np.int64)
    return x_arr, y_arr


def load_many_from_processed(
    file_list: List[str],
    processed_map: Dict[str, pd.DataFrame],
    label2id: Dict[str, int],
    window_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    for path in file_list:
        df = processed_map[path]
        x7 = df[SIGNAL_COLS].values.astype(np.float32)
        x7 = pad_or_trunc_hold_last(x7, window_len)
        x12 = add_flex_deltas(x7)
        lab = label_from_filename(os.path.basename(path))
        x_list.append(x12)
        y_list.append(label2id[lab])
    x_arr = np.stack(x_list, axis=0)
    y_arr = np.array(y_list, dtype=np.int64)
    return x_arr, y_arr


def flatten_windows(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def print_label_distribution(name: str, file_list: List[str]):
    counts = {}
    for path in file_list:
        lab = label_from_filename(os.path.basename(path))
        counts[lab] = counts.get(lab, 0) + 1
    print(f"\n{name} label distribution:")
    for lab in sorted(counts.keys()):
        print(f"  {lab}: {counts[lab]}")


def print_dataframe_info(df: pd.DataFrame, title: str):
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(f"\n=== {title} .info() ===")
    print(buffer.getvalue().strip())


def print_missing_values(df: pd.DataFrame, title: str):
    print(f"\n=== {title} Missing Values ===")
    missing = df.isna().sum()
    print(missing.to_string())
    print(f"Total missing values: {int(missing.sum())}")


def print_describe(df: pd.DataFrame, title: str):
    print(f"\n=== {title} describe() ===")
    print(df.describe().to_string())


def build_label_arrays(file_list: List[str], label2id: Dict[str, int]):
    y = np.array([label2id[label_from_filename(os.path.basename(p))] for p in file_list], dtype=np.int64)
    onehot = np.eye(len(label2id), dtype=np.float32)[y]
    return y, onehot


def print_onehot_summary(file_list: List[str], label2id: Dict[str, int]):
    y, onehot = build_label_arrays(file_list, label2id)
    print("\n=== Label Encoding Summary ===")
    print("Integer label mapping:")
    for lab, idx in label2id.items():
        print(f"  {lab} -> {idx}")
    print(f"One-hot label matrix shape: {onehot.shape}")
    preview_rows = min(5, len(file_list))
    print("Preview of the first few one-hot rows:")
    for i in range(preview_rows):
        print(f"  file={os.path.basename(file_list[i])}, class_id={int(y[i])}, onehot={onehot[i].astype(int).tolist()}")


def prepare_dataset(
    include_dynamic: bool = INCLUDE_DYNAMIC_DEFAULT,
    window_len: int = WINDOW_LEN_DEFAULT,
    test_ratio: float = TEST_RATIO_DEFAULT,
    random_seed: int = RANDOM_SEED_DEFAULT,
):
    files = []
    static_files = collect_files(STATIC_DIR)
    dyn_files = collect_files(DYNAMIC_DIR) if include_dynamic else []
    files += static_files
    files += dyn_files

    if len(files) == 0:
        raise RuntimeError("No window CSV files were found. Please check the directory paths.")

    print(f"\nTotal window files found: {len(files)}")
    print(f"  STATIC_DIR : {len(static_files)}")
    print(f"  DYNAMIC_DIR: {len(dyn_files)} (INCLUDE_DYNAMIC={include_dynamic})")

    rng = np.random.RandomState(random_seed)
    label2files = {}
    for path in files:
        lab = label_from_filename(os.path.basename(path))
        label2files.setdefault(lab, []).append(path)

    labels = sorted(label2files.keys())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    print("\nLabel mapping:")
    for k, v in label2id.items():
        print(f"  {k} -> {v} (files={len(label2files[k])})")

    train_files, test_files = [], []
    for _, flist in label2files.items():
        flist = list(flist)
        rng.shuffle(flist)
        n = len(flist)
        n_test = max(1, int(round(n * test_ratio))) if n >= 2 else 0
        test_part = flist[:n_test]
        train_part = flist[n_test:]
        train_files.extend(train_part)
        test_files.extend(test_part)

    print(f"\nTraining window files: {len(train_files)}")
    print(f"Test window files: {len(test_files)}")
    print_label_distribution("Training set", train_files)
    print_label_distribution("Test set", test_files)
    if len(test_files) == 0:
        print("Warning: the test set is empty. Each gesture should ideally have at least 5-10 window files.")

    train_events = set(event_from_filename(os.path.basename(p)) for p in train_files)
    test_events = set(event_from_filename(os.path.basename(p)) for p in test_files)
    leak = (train_events & test_events) - {None}
    print("\n[Leak Check] leaked events count =", len(leak))
    if len(leak) > 0:
        print("[Leak Check] example leaked events =", list(leak)[:10])

    overlap = set(train_files) & set(test_files)
    print("[Leak Check] train-test file overlap =", len(overlap))
    if len(overlap) > 0:
        print("[Leak Check] example overlapped files =", list(overlap)[:5])

    print("\n=== Step 1: Train/Test Split ===")
    train_raw_frames = [load_window_df(path) for path in train_files]
    train_raw_df = pd.concat(train_raw_frames, ignore_index=True)
    print_dataframe_info(train_raw_df, "Train Raw Data")
    print_missing_values(train_raw_df, "Train Raw Data")
    print_describe(train_raw_df, "Train Raw Data")

    print("\n=== Step 2: Missing-Value Check ===")
    print("No outlier detection or replacement is applied in the current preprocessing pipeline.")

    processed_train = {path: load_window_df(path) for path in train_files}
    processed_test = {path: load_window_df(path) for path in test_files}

    train_processed_df = pd.concat([processed_train[p] for p in train_files], ignore_index=True)
    test_processed_df = pd.concat([processed_test[p] for p in test_files], ignore_index=True)
    print_missing_values(train_processed_df, "Processed Train Data")
    print_missing_values(test_processed_df, "Processed Test Data")

    print("\n=== Step 3: Label Encoding ===")
    print("One-hot labels are generated for reference, while integer labels are retained for model training.")
    print_onehot_summary(train_files, label2id)

    print("\n=== Step 4: Differential Features ===")
    print("First-order differences da0-da4 are constructed from a0-a4 and concatenated with the original 7 channels.")

    X_tr, y_tr = load_many_from_processed(train_files, processed_train, label2id, window_len)
    X_te, y_te = load_many_from_processed(test_files, processed_test, label2id, window_len)

    print(f"\nX_tr: {X_tr.shape}, y_tr: {y_tr.shape}")
    print(f"X_te: {X_te.shape}, y_te: {y_te.shape}")

    print("\n=== Step 5: Z-Score Normalization ===")
    print("Mean and standard deviation are computed from the training set only and applied to both train and test sets.")
    mean = X_tr.mean(axis=(0, 1)).astype(np.float32)
    std = (X_tr.std(axis=(0, 1)) + 1e-8).astype(np.float32)
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    print("\nNormalization parameters:")
    for i, (m, s) in enumerate(zip(mean, std)):
        print(f"  Channel {EXT_COLS[i]}: mean={m:.4f}, std={s:.4f}")

    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_te": X_te,
        "y_te": y_te,
        "mean": mean,
        "std": std,
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "train_files": train_files,
        "test_files": test_files,
        "processed_train": processed_train,
        "processed_test": processed_test,
        "window_len": window_len,
        "include_dynamic": include_dynamic,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
    }


def run_preprocessing_only(
    include_dynamic: bool = INCLUDE_DYNAMIC_DEFAULT,
    window_len: int = WINDOW_LEN_DEFAULT,
    test_ratio: float = TEST_RATIO_DEFAULT,
    random_seed: int = RANDOM_SEED_DEFAULT,
):
    print("\n=== Preprocessing Only Mode ===")
    print(f"include_dynamic = {include_dynamic}")
    print(f"window_len      = {window_len}")
    print(f"test_ratio      = {test_ratio}")
    print(f"random_seed     = {random_seed}")
    data = prepare_dataset(
        include_dynamic=include_dynamic,
        window_len=window_len,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    print("\nPreprocessing completed.")
    print(f"Training set shape: X_tr={data['X_tr'].shape}, y_tr={data['y_tr'].shape}")
    print(f"Test set shape: X_te={data['X_te'].shape}, y_te={data['y_te'].shape}")
    print("Input channel order:", ", ".join(EXT_COLS))
    return data


def print_method_summary(model_name: str, params: Dict, key_functions: List[str], notes: List[str] = None):
    print(f"\n=== {model_name} Method Summary ===")
    print("Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    if key_functions:
        print("Key functions / modules:")
        for item in key_functions:
            print(f"  - {item}")
    if notes:
        print("Notes:")
        for item in notes:
            print(f"  - {item}")


def evaluate_predictions(y_true, y_pred, id2label: Dict[int, str], title: str):
    labels_sorted = list(range(len(id2label)))
    target_names = [id2label[i] for i in labels_sorted]
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )

    print(f"\n=== {title} Classification Report ===")
    print(report_text)
    print(f"=== {title} Confusion Matrix ===")
    print(cm)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
    }


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def save_pickle(path: str, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def save_metrics_json(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ieee_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_loss_curve(loss_history: List[float], title: str, out_path: str, color: str = "#005F73"):
    ieee_plot_style()
    epochs = np.arange(1, len(loss_history) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        epochs,
        loss_history,
        color=color,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="Training Loss",
    )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.set_xlim(1, len(loss_history))
    ax.legend(frameon=True, fancybox=False, edgecolor="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    fig.savefig(out_path)
    plt.close(fig)


def plot_not_applicable_loss_curve(title: str, message: str, out_path: str, color: str = "#7A0019"):
    ieee_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=10,
        color=color,
        bbox=dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor="black"),
    )
    for spine in ax.spines.values():
        spine.set_color("black")
    fig.savefig(out_path)
    plt.close(fig)


def compute_multiclass_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(log_loss(y_true, y_prob, labels=np.arange(y_prob.shape[1])))


if __name__ == "__main__":
    run_preprocessing_only()
