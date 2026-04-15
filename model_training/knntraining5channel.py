import os
import pickle
import time

import numpy as np
import psutil
from sklearn.neighbors import KNeighborsClassifier

from 五通道模型工具 import (
    BASE_OUT,
    C_IN,
    EXT_COLS,
    FIG_DIR,
    MODEL_DIR,
    RANDOM_SEED_DEFAULT,
    TEST_RATIO_DEFAULT,
    WINDOW_LEN_DEFAULT,
    evaluate_predictions,
    flatten_windows,
    format_size,
    get_file_size,
    now_str,
    plot_not_applicable_loss_curve,
    prepare_dataset,
    print_method_summary,
    save_metrics_json,
)


INCLUDE_DYNAMIC = False
WINDOW_LEN = WINDOW_LEN_DEFAULT
TEST_RATIO = TEST_RATIO_DEFAULT
RANDOM_SEED = RANDOM_SEED_DEFAULT

K_NEIGHBORS = 10
DISTANCE_METRIC = "euclidean"
WEIGHTS = "uniform"
ALGORITHM = "auto"

MODEL_PATH = os.path.join(MODEL_DIR, "knn_window_model.pkl")
LABEL_PATH = os.path.join(MODEL_DIR, "knn_label_map.npy")
NORM_PATH = os.path.join(MODEL_DIR, "knn_norm_stats.npz")
METRICS_PATH = os.path.join(MODEL_DIR, "knn_metrics.json")
LOSS_FIG_PATH = os.path.join(FIG_DIR, "knn_loss_curve_ieee.png")

print("Device: CPU")

data = prepare_dataset(
    include_dynamic=INCLUDE_DYNAMIC,
    window_len=WINDOW_LEN,
    test_ratio=TEST_RATIO,
    random_seed=RANDOM_SEED,
)

X_tr = data["X_tr"]
X_te = data["X_te"]
y_tr = data["y_tr"]
y_te = data["y_te"]
labels = data["labels"]
id2label = data["id2label"]
mean = data["mean"]
std = data["std"]

X_tr_flat = flatten_windows(X_tr)
X_te_flat = flatten_windows(X_te)

np.save(LABEL_PATH, data["label2id"])
np.savez(NORM_PATH, mean=mean, std=std)

print(f"\nX_tr_flat: {X_tr_flat.shape}, y_tr: {y_tr.shape}")
print(f"X_te_flat: {X_te_flat.shape}, y_te: {y_te.shape}")
print("Input channel order:", ", ".join(EXT_COLS))
print(f"Flattened feature dimension per window: {X_tr_flat.shape[1]}")

params = {
    "window_len": WINDOW_LEN,
    "input_channels_after_concat": C_IN,
    "flatten_feature_dim": int(X_tr_flat.shape[1]),
    "include_dynamic": INCLUDE_DYNAMIC,
    "test_ratio": TEST_RATIO,
    "random_seed": RANDOM_SEED,
    "n_neighbors": K_NEIGHBORS,
    "metric": DISTANCE_METRIC,
    "weights": WEIGHTS,
    "algorithm": ALGORITHM,
}
key_functions = []
notes = []
print_method_summary("kNN", params, key_functions, notes)

print("\nStart training...")
train_start_time = time.time()
model = KNeighborsClassifier(
    n_neighbors=K_NEIGHBORS,
    metric=DISTANCE_METRIC,
    weights=WEIGHTS,
    algorithm=ALGORITHM,
)
model.fit(X_tr_flat, y_tr)
train_time = time.time() - train_start_time
print(f"\nTraining finished, total time: {train_time:.2f} s")
plot_not_applicable_loss_curve(
    "kNN Training Loss",
    "kNN is a lazy learner.\nNo iterative epoch-wise loss is defined.",
    LOSS_FIG_PATH,
)
print("Training loss note:", LOSS_FIG_PATH)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\nModel saved:")
print("  Model file:", MODEL_PATH)
print("  Label map:", LABEL_PATH)
print("  Normalization stats:", NORM_PATH)

print("\nStart testing...")
y_pred = []
infer_times = []

for i in range(len(X_te_flat)):
    x_single = X_te_flat[i:i + 1]
    t0 = time.time()
    pred = model.predict(x_single)
    t1 = time.time()
    y_pred.append(int(pred[0]))
    infer_times.append(t1 - t0)

avg_infer_time = float(np.mean(infer_times) * 1000) if infer_times else 0.0
metrics = evaluate_predictions(y_te, y_pred, id2label, "kNN")

print(f"\nAverage single-window inference time: {avg_infer_time:.3f} ms")

process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1024 / 1024
print(f"Current Python process memory usage: {mem_mb:.2f} MB")

stored_train_feature_bytes = int(X_tr_flat.size * 4)
stored_train_label_bytes = int(y_tr.size * 8)
norm_bytes = int((mean.size + std.size) * 4)
estimated_deploy_bytes = stored_train_feature_bytes + stored_train_label_bytes + norm_bytes
model_disk_size = get_file_size(MODEL_PATH)

print("\n=== kNN Deployment Estimate ===")
print(f"Stored training features: {stored_train_feature_bytes} bytes ({format_size(stored_train_feature_bytes)})")
print(f"Stored training labels: {stored_train_label_bytes} bytes ({format_size(stored_train_label_bytes)})")
print(f"Normalization stats: {norm_bytes} bytes ({format_size(norm_bytes)})")
print(f"Estimated deployment size: {estimated_deploy_bytes} bytes ({format_size(estimated_deploy_bytes)})")
print(f"Model file size on disk: {model_disk_size} bytes ({format_size(model_disk_size)})")

payload = {
    "model_name": "kNN",
    "generated_at": now_str(),
    "params": params,
    "key_functions": key_functions,
    "notes": notes,
    "train_time_sec": float(train_time),
    "avg_infer_time_ms": avg_infer_time,
    "memory_rss_mb": float(mem_mb),
    "model_disk_size_bytes": int(model_disk_size),
    "estimated_deploy_bytes": int(estimated_deploy_bytes),
    "stored_feature_bytes": int(stored_train_feature_bytes),
    "stored_label_bytes": int(stored_train_label_bytes),
    "norm_bytes": int(norm_bytes),
    "n_classes": len(labels),
    "n_train_windows": int(len(y_tr)),
    "n_test_windows": int(len(y_te)),
    "loss_curve_path": LOSS_FIG_PATH,
    "loss_history": [],
    "metrics": metrics,
}
save_metrics_json(METRICS_PATH, payload)
print("\nMetrics summary saved:")
print("  ", METRICS_PATH)
