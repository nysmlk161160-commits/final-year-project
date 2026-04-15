import os
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from 五通道模型工具 import (
    C_IN,
    EXT_COLS,
    FIG_DIR,
    MODEL_DIR,
    RANDOM_SEED_DEFAULT,
    TEST_RATIO_DEFAULT,
    WINDOW_LEN_DEFAULT,
    evaluate_predictions,
    format_size,
    get_file_size,
    now_str,
    plot_loss_curve,
    prepare_dataset,
    print_method_summary,
    save_metrics_json,
)

INCLUDE_DYNAMIC = False
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_window_model.pt")
LABEL_PATH = os.path.join(MODEL_DIR, "label_map.npy")
NORM_PATH = os.path.join(MODEL_DIR, "norm_stats.npz")
H_PATH = os.path.join(MODEL_DIR, "model_weights.h")
METRICS_PATH = os.path.join(MODEL_DIR, "cnn_metrics.json")
LOSS_FIG_PATH = os.path.join(FIG_DIR, "cnn_loss_curve_ieee.png")
WINDOW_LEN = WINDOW_LEN_DEFAULT
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
TEST_RATIO = TEST_RATIO_DEFAULT
RANDOM_SEED = RANDOM_SEED_DEFAULT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


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
label2id = data["label2id"]
id2label = data["id2label"]
mean = data["mean"]
std = data["std"]

np.save(LABEL_PATH, label2id)
np.savez(NORM_PATH, mean=mean, std=std)


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].T, self.y[idx]


train_loader = DataLoader(WindowDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(WindowDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)


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


model = SimpleCNN(num_classes=len(labels), c_in=C_IN).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

param_count = sum(p.numel() for p in model.parameters())
print(f"\nModel parameter count: {param_count:,}")
print("Input channel order:", ", ".join(EXT_COLS))

params = {
    "window_len": WINDOW_LEN,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "input_channels_after_concat": C_IN,
    "include_dynamic": INCLUDE_DYNAMIC,
    "test_ratio": TEST_RATIO,
    "random_seed": RANDOM_SEED,
    "conv1": "Conv1d(12, 16, kernel_size=3, padding=1)",
    "conv2": "Conv1d(16, 32, kernel_size=3, padding=1)",
    "pooling": "AdaptiveAvgPool1d(1)",
    "classifier": "Linear(32, num_classes)",
    "loss": "CrossEntropyLoss",
    "optimizer": "Adam",
}
key_functions = []
notes = []
print_method_summary("CNN", params, key_functions, notes)

print("\nStart training...")
train_start_time = time.time()
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_epoch_loss = total_loss / max(1, len(train_loader))
    loss_history.append(float(avg_epoch_loss))

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, avg_loss={avg_epoch_loss:.4f}")

train_time = time.time() - train_start_time
print(f"\nTraining finished, total time: {train_time:.2f} s")
plot_loss_curve(loss_history, "CNN Training Loss", LOSS_FIG_PATH, color="#0A6C74")
print("Training loss curve:", LOSS_FIG_PATH)

torch.save(model.state_dict(), MODEL_PATH)

print("\nModel saved:")
print("  Model weights:", MODEL_PATH)
print("  Label map:", LABEL_PATH)
print("  Normalization stats:", NORM_PATH)

model.eval()
y_true, y_pred = [], []
infer_times = []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(DEVICE)
        t0 = time.time()
        logits = model(Xb)
        t1 = time.time()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(yb.numpy().tolist())
        infer_times.append((t1 - t0) / Xb.size(0))

avg_infer_time = float(np.mean(infer_times) * 1000) if infer_times else 0.0
metrics = evaluate_predictions(y_true, y_pred, id2label, "CNN")

print(f"\nAverage single-window inference time: {avg_infer_time:.3f} ms")

process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1024 / 1024
print(f"Current Python process memory usage: {mem_mb:.2f} MB")

cuda_mem_mb = 0.0
if DEVICE == "cuda":
    cuda_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"CUDA memory usage: {cuda_mem_mb:.2f} MB")


def _c_array_float(name, arr):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    lines = [f"const float {name}[{arr.size}] = {{"]
    for i in range(0, arr.size, 12):
        chunk = arr[i:i + 12]
        s = ", ".join([f"{v:.8e}f" for v in chunk])
        lines.append("  " + s + ("," if i + 12 < arr.size else ""))
    lines.append("};\n")
    return "\n".join(lines)


def export_model_to_h(h_path: str, model_cpu: nn.Module, mean_np: np.ndarray, std_np: np.ndarray, id2label_map: dict):
    model_cpu.eval()
    conv1 = model_cpu.net[0]
    conv2 = model_cpu.net[2]
    fc = model_cpu.fc

    conv1_w = conv1.weight.detach().numpy().astype(np.float32)
    conv1_b = conv1.bias.detach().numpy().astype(np.float32)
    conv2_w = conv2.weight.detach().numpy().astype(np.float32)
    conv2_b = conv2.bias.detach().numpy().astype(np.float32)
    fc_w = fc.weight.detach().numpy().astype(np.float32)
    fc_b = fc.bias.detach().numpy().astype(np.float32)

    k = fc_b.shape[0]

    lines = []
    lines.append("#pragma once")
    lines.append("// Auto-generated. Put this file next to your Arduino .ino\n")
    lines.append(f"#define CNN_IN_CH   {C_IN}")
    lines.append(f"#define CNN_T       {WINDOW_LEN}")
    lines.append("#define CNN_C1_OUT  16")
    lines.append("#define CNN_C2_OUT  32")
    lines.append(f"#define CNN_K       {k}\n")
    lines.append(_c_array_float("NORM_MEAN", mean_np))
    lines.append(_c_array_float("NORM_STD", std_np))
    lines.append(_c_array_float("CONV1_W", conv1_w))
    lines.append(_c_array_float("CONV1_B", conv1_b))
    lines.append(_c_array_float("CONV2_W", conv2_w))
    lines.append(_c_array_float("CONV2_B", conv2_b))
    lines.append(_c_array_float("FC_W", fc_w))
    lines.append(_c_array_float("FC_B", fc_b))
    lines.append("const char* LABELS[CNN_K] = {")
    for i in range(k):
        lab = str(id2label_map[i]).replace('"', '\\"')
        comma = "," if i < k - 1 else ""
        lines.append(f'  "{lab}"{comma}')
    lines.append("};\n")
    lines.append("// Channel order in input window:")
    lines.append("// " + ", ".join(EXT_COLS) + "\n")

    with open(h_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nExported Arduino header:")
    print("  ", h_path)


model_cpu = SimpleCNN(num_classes=len(labels), c_in=C_IN).cpu()
model_cpu.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
export_model_to_h(H_PATH, model_cpu, mean, std, id2label)


def estimate_cnn_flash_bytes(c_in, num_classes):
    b_float = 4
    conv1_w = 16 * c_in * 3 * b_float
    conv1_b = 16 * b_float
    conv2_w = 32 * 16 * 3 * b_float
    conv2_b = 32 * b_float
    fc_w = num_classes * 32 * b_float
    fc_b = num_classes * b_float
    norm = (c_in * b_float) * 2
    total = conv1_w + conv1_b + conv2_w + conv2_b + fc_w + fc_b + norm
    return {
        "conv1_w": conv1_w,
        "conv1_b": conv1_b,
        "conv2_w": conv2_w,
        "conv2_b": conv2_b,
        "fc_w": fc_w,
        "fc_b": fc_b,
        "norm_mean_std": norm,
        "total": total,
    }


est = estimate_cnn_flash_bytes(c_in=C_IN, num_classes=len(labels))
print("\n=== Arduino Flash Estimate (float=4B) ===")
for k, v in est.items():
    print(f"{k:15s}: {v:8d} bytes  ({v / 1024:.2f} KB)")

model_disk_size = get_file_size(MODEL_PATH)
print(f"Model file size on disk: {model_disk_size} bytes ({format_size(model_disk_size)})")

payload = {
    "model_name": "CNN",
    "generated_at": now_str(),
    "params": params,
    "key_functions": key_functions,
    "notes": notes,
    "train_time_sec": float(train_time),
    "avg_infer_time_ms": avg_infer_time,
    "memory_rss_mb": float(mem_mb),
    "cuda_memory_mb": float(cuda_mem_mb),
    "model_disk_size_bytes": int(model_disk_size),
    "estimated_deploy_bytes": int(est["total"]),
    "n_classes": len(labels),
    "n_train_windows": int(len(y_tr)),
    "n_test_windows": int(len(y_te)),
    "parameter_count": int(param_count),
    "loss_curve_path": LOSS_FIG_PATH,
    "loss_history": loss_history,
    "metrics": metrics,
}
save_metrics_json(METRICS_PATH, payload)
print("\nMetrics summary saved:")
print("  ", METRICS_PATH)
