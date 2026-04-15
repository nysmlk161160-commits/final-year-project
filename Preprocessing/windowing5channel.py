import pandas as pd
import numpy as np
import os
from collections import Counter
from glob import glob

INPUT_DIR = r"D:\gesture_project\cnn_version_test\五通道最终版本\labeled"
OUT_DIR   = r"D:\gesture_project\cnn_version_test\五通道最终版本\windows_static_csv_6ch"
os.makedirs(OUT_DIR, exist_ok=True)

FS = 50
WINDOW_SEC = 1.0
STRIDE_SEC = 0.25

WINDOW_LEN = int(FS * WINDOW_SEC)
STRIDE_LEN = int(FS * STRIDE_SEC)
STABLE_RATIO = 0.8

INPUT_CSV_LIST = sorted(glob(os.path.join(INPUT_DIR, "s*.csv")))

if len(INPUT_CSV_LIST) == 0:
    raise ValueError(f"No CSV files starting with 's' were found in {INPUT_DIR}")

signal_cols = ["a0", "a1", "a2", "a3", "a4", "roll", "pitch"]
label_col = "label"

total_windows = 0
kept_windows  = 0
win_id = 0

for INPUT_CSV in INPUT_CSV_LIST:
    print(f"Processing: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["time_ms"] + signal_cols + [label_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"File {INPUT_CSV} is missing column: {c}")

    num_samples = len(df)

    for start in range(0, num_samples - WINDOW_LEN + 1, STRIDE_LEN):
        total_windows += 1
        end = start + WINDOW_LEN
        window = df.iloc[start:end]

        labels = window[label_col].values
        major_label, count = Counter(labels).most_common(1)[0]
        ratio = count / WINDOW_LEN

        if ratio < STABLE_RATIO:
            continue

        out_df = window[["time_ms"] + signal_cols].copy()
        out_df["window_label"] = major_label
        out_df["window_id"] = win_id

        out_path = os.path.join(
            OUT_DIR,
            f"win_{win_id:04d}_label_{major_label}.csv"
        )
        out_df.to_csv(out_path, index=False)

        win_id += 1
        kept_windows += 1

print("Window extraction completed (7 raw channels: 5 flex + roll + pitch)")
print(f"Total candidate windows: {total_windows}")
print(f"Kept stable windows: {kept_windows}")
print(f"Output directory: {OUT_DIR}")
print(f"window_len={WINDOW_LEN} samples, stride={STRIDE_LEN} samples")
