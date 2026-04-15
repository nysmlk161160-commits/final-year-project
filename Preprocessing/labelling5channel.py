import pandas as pd
import os

INPUT_CSV  = r"D:\gesture_project\cnn_version_test\五通道最终版本\手势库五通道\s00000_01.csv"
OUTPUT_CSV = r"D:\gesture_project\cnn_version_test\五通道最终版本\labeled\s00000_01_label.csv"

threshold = {
    "a0": 500,
    "a1": 500,
    "a2": 500,
    "a3": 500,
    "a4": 500,
    "roll":  0,
    "pitch": 0,
}

enable_channel = {
    "a0": True,
    "a1": True,
    "a2": True,
    "a3": True,
    "a4": True,
    "roll":  True,
    "pitch": True,
}

ACTIVE_LABEL = "0000001"
IDLE_LABEL   = "0000001"

df = pd.read_csv(INPUT_CSV)

required_cols = ["time_ms", "a0", "a1", "a2", "a3", "a4", "roll", "pitch"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

def apply_threshold(row):
    for ch in ["a0", "a1", "a2", "a3", "a4"]:
        if enable_channel.get(ch, False):
            if row[ch] >= threshold[ch]:
                return IDLE_LABEL

    for ch in ["roll", "pitch"]:
        if enable_channel.get(ch, False):
            if abs(float(row[ch])) >= threshold[ch]:
                return IDLE_LABEL

    return ACTIVE_LABEL

df["label"] = df.apply(apply_threshold, axis=1)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print("Done. Output file generated:")
print(OUTPUT_CSV)
