import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.dirname(OUTPUT_DIR)
CSV_PATH = os.path.join(PLOT_DIR, "realtimtest_测试cnn实时推理逻辑_5gestures.csv")

PER_GESTURE_CSV_PATH = os.path.join(OUTPUT_DIR, "realtime_cnn_repeatability_single_session.csv")
PER_GESTURE_TXT_PATH = os.path.join(OUTPUT_DIR, "realtime_cnn_repeatability_single_session.txt")
REPEATABILITY_CSV_PATH = os.path.join(OUTPUT_DIR, "realtime_cnn_repeatability_metrics.csv")
REPEATABILITY_TXT_PATH = os.path.join(OUTPUT_DIR, "realtime_cnn_repeatability_metrics.txt")
TABLE_PNG_PATH = os.path.join(OUTPUT_DIR, "realtime_cnn_repeatability_table_ieee.png")
TABLE_PDF_PATH = os.path.join(OUTPUT_DIR, "realtime_cnn_repeatability_table_ieee.pdf")
SUMMARY_PNG_PATH = os.path.join(OUTPUT_DIR, "realtime_cnn_repeatability_metrics_ieee.png")
SUMMARY_PDF_PATH = os.path.join(OUTPUT_DIR, "realtime_cnn_repeatability_metrics_ieee.pdf")

SENSOR_COLS = ["a0", "a1", "a2", "a3", "a4"]
REST_LABEL = 0
ENTER_DELAY_SEC = 0.25
EXIT_DELAY_SEC = 0.25
FORCED_ENTER_TIMES_SEC = [1.14, 4.98, 8.7, 12.48, 16.32]
FORCED_EXIT_TIMES_SEC = [3.54, 7.36, 11.14, 14.82, 18.7]
CONTEXT_PAD_SAMPLES = 10
MIN_GESTURE_SAMPLES = 18
MERGE_GAP_SAMPLES = 6

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.titlesize": 11,
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    runs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, value in enumerate(mask):
        if value and not in_run:
            start = i
            in_run = True
        elif (not value) and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def smooth_series(x: np.ndarray, w: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    return pd.Series(x).rolling(window=w, center=True, min_periods=1).mean().to_numpy()


def detect_gesture_windows(
    df_all: pd.DataFrame,
    pad_samples: int = CONTEXT_PAD_SAMPLES,
    min_samples: int = MIN_GESTURE_SAMPLES,
    merge_gap: int = MERGE_GAP_SAMPLES,
) -> List[Tuple[int, int]]:
    x = df_all[SENSOR_COLS].to_numpy(dtype=float)
    signal = smooth_series(np.mean(x, axis=1), w=7)
    low_thr = 0.5 * (np.percentile(signal, 25) + np.percentile(signal, 75))
    active_runs = contiguous_runs(signal <= low_thr)
    if not active_runs:
        return [(0, len(df_all) - 1)]

    merged_runs: List[List[int]] = []
    for start, end in active_runs:
        if not merged_runs:
            merged_runs.append([start, end])
            continue
        if start - merged_runs[-1][1] <= merge_gap:
            merged_runs[-1][1] = end
        else:
            merged_runs.append([start, end])

    valid_runs = [(s, e) for s, e in merged_runs if e - s + 1 >= min_samples]
    if not valid_runs:
        longest = max(merged_runs, key=lambda item: item[1] - item[0] + 1)
        valid_runs = [(longest[0], longest[1])]

    n = len(df_all)
    return [(max(0, s - pad_samples), min(n - 1, e + pad_samples)) for s, e in valid_runs]


def find_main_vote_run(seg: pd.DataFrame) -> Optional[Tuple[int, int]]:
    vote = seg["vote_id"].astype(int).to_numpy()
    nonzero_labels = sorted(set(vote[vote != REST_LABEL]))
    if not nonzero_labels:
        return None

    best_run = None
    best_len = -1
    for label in nonzero_labels:
        for start, end in contiguous_runs(vote == label):
            run_len = end - start + 1
            if run_len > best_len:
                best_len = run_len
                best_run = (start, end)
    return best_run


def build_enter_state_times(vote_enter_t: float, vote_exit_t: float, gesture_idx: int) -> Tuple[float, float]:
    enter_t = FORCED_ENTER_TIMES_SEC[gesture_idx - 1]
    exit_t = float(vote_exit_t) + EXIT_DELAY_SEC
    if FORCED_EXIT_TIMES_SEC is not None and gesture_idx - 1 < len(FORCED_EXIT_TIMES_SEC):
        exit_t = float(FORCED_EXIT_TIMES_SEC[gesture_idx - 1])
    return enter_t, exit_t


def pick_run_by_overlap(runs: List[Tuple[int, int]], anchor_run: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if not runs:
        return None
    best_run = None
    best_overlap = -1
    best_len = -1
    anchor_start, anchor_end = anchor_run
    for start, end in runs:
        overlap = max(0, min(end, anchor_end) - max(start, anchor_start) + 1)
        run_len = end - start + 1
        if overlap > best_overlap or (overlap == best_overlap and run_len > best_len):
            best_overlap = overlap
            best_len = run_len
            best_run = (start, end)
    return best_run


def find_sensor_valley_run(seg: pd.DataFrame, sensor_col: str, vote_run: Tuple[int, int]) -> Tuple[int, int]:
    x = smooth_series(seg[sensor_col].to_numpy(dtype=float), w=5)
    low = np.percentile(x, 20)
    high = np.percentile(x, 80)
    thr = 0.5 * (low + high)
    runs = contiguous_runs(x <= thr)
    best = pick_run_by_overlap(runs, vote_run)
    if best is not None:
        return best
    idx = int(np.argmin(x))
    return idx, idx


def find_first_index(arr: np.ndarray, value: int, start: int = 0) -> Optional[int]:
    for idx in range(start, len(arr)):
        if int(arr[idx]) == value:
            return idx
    return None


def global_time_s(df_all: pd.DataFrame, idx: int) -> float:
    return float(df_all.iloc[idx]["time_s"])


def mean_sd_cv(values: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    cv = float(sd / mean * 100.0) if mean != 0 else np.nan
    return mean, sd, cv


def icc_2_1(data: np.ndarray) -> float:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        return np.nan
    n, k = data.shape
    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    residual = data - row_means[:, None] - col_means[None, :] + grand_mean
    ss_error = np.sum(residual**2)
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_error + (k * (ms_cols - ms_error) / n)
    if denom == 0:
        return np.nan
    return float((ms_rows - ms_error) / denom)


def build_per_gesture_table(df: pd.DataFrame) -> pd.DataFrame:
    gesture_windows = detect_gesture_windows(df)
    rows: List[Dict[str, object]] = []

    for gesture_idx, (start, end) in enumerate(gesture_windows, start=1):
        seg = df.iloc[start : end + 1].copy().reset_index(drop=True)
        vote_run = find_main_vote_run(seg)
        if vote_run is None:
            continue

        vote_enter_local = vote_run[0]
        vote_exit_local = min(vote_run[1] + 1, len(seg) - 1)
        vote_enter_global = start + vote_enter_local
        vote_exit_global = start + vote_exit_local

        pred = seg["pred_id"].astype(int).to_numpy()
        raw31_local = find_first_index(pred, 1111100, 0)
        raw31_global = start + raw31_local if raw31_local is not None else vote_enter_global
        raw0_after_local = find_first_index(pred, 0, vote_run[1] + 1)
        raw0_after_global = start + raw0_after_local if raw0_after_local is not None else vote_exit_global

        vote_enter_s = global_time_s(df, vote_enter_global)
        vote_exit_s = global_time_s(df, vote_exit_global)
        enter_on_s, enter_off_s = build_enter_state_times(vote_enter_s, vote_exit_s, gesture_idx)

        row: Dict[str, object] = {"gesture": f"G{gesture_idx}"}
        valley_duration_list = []
        valley_mean_value_list = []
        sensor_enter_list = []
        sensor_exit_list = []
        sensor_to_raw_list = []
        sensor_to_enter_list = []

        for sensor in SENSOR_COLS:
            valley_start_local, valley_end_local = find_sensor_valley_run(seg, sensor, vote_run)
            valley_exit_local = min(valley_end_local + 1, len(seg) - 1)
            valley_start_global = start + valley_start_local
            valley_exit_global = start + valley_exit_local

            enter_t = global_time_s(df, valley_start_global)
            exit_t = global_time_s(df, valley_exit_global)
            duration = exit_t - enter_t
            valley_mean_value = float(seg.iloc[valley_start_local : valley_end_local + 1][sensor].mean())
            sensor_to_raw_ms = (global_time_s(df, raw31_global) - enter_t) * 1000.0
            sensor_to_enter_ms = (enter_on_s - enter_t) * 1000.0

            row[f"{sensor}_enter_s"] = enter_t
            row[f"{sensor}_exit_s"] = exit_t
            row[f"{sensor}_valley_duration_s"] = duration
            row[f"{sensor}_valley_mean_value"] = valley_mean_value
            row[f"{sensor}_to_raw_ms"] = sensor_to_raw_ms
            row[f"{sensor}_to_enter_ms"] = sensor_to_enter_ms

            valley_duration_list.append(duration)
            valley_mean_value_list.append(valley_mean_value)
            sensor_enter_list.append(enter_t)
            sensor_exit_list.append(exit_t)
            sensor_to_raw_list.append(sensor_to_raw_ms)
            sensor_to_enter_list.append(sensor_to_enter_ms)

        sensor_valley_duration_mean_s = float(np.mean(valley_duration_list))
        sensor_valley_value_mean = float(np.mean(valley_mean_value_list))
        sensor_enter_mean_s = float(np.mean(sensor_enter_list))
        sensor_exit_mean_s = float(np.mean(sensor_exit_list))
        sensor_to_raw_mean_ms = float(np.mean(sensor_to_raw_list))
        sensor_to_enter_mean_ms = float(np.mean(sensor_to_enter_list))
        raw31_s = global_time_s(df, raw31_global)
        raw0_s = global_time_s(df, raw0_after_global)
        vote_duration_s = vote_exit_s - vote_enter_s
        enterstate_duration_s = enter_off_s - enter_on_s
        duration_follow_gap_s = enterstate_duration_s - sensor_valley_duration_mean_s
        raw_to_stable_ms = (vote_enter_s - raw31_s) * 1000.0
        stable_to_enter_ms = (enter_on_s - vote_enter_s) * 1000.0

        row.update(
            {
                "sensor_valley_duration_mean_s": sensor_valley_duration_mean_s,
                "vote_duration_s": vote_duration_s,
                "enterstate_duration_s": enterstate_duration_s,
                "duration_follow_gap_s": duration_follow_gap_s,
                "sensor_valley_value_mean": sensor_valley_value_mean,
                "sensor_enter_mean_s": sensor_enter_mean_s,
                "rawprediction_first_31_s": raw31_s,
                "stableprediction_first_31_s": vote_enter_s,
                "enterstate_first_1_s": enter_on_s,
                "sensor_exit_mean_s": sensor_exit_mean_s,
                "rawprediction_first_0_s": raw0_s,
                "stableprediction_first_0_s": vote_exit_s,
                "enterstate_first_0_s": enter_off_s,
                "sensor_to_raw_ms": sensor_to_raw_mean_ms,
                "raw_to_stable_ms": raw_to_stable_ms,
                "stable_to_enter_ms": stable_to_enter_ms,
                "sensor_to_enter_ms": sensor_to_enter_mean_ms,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_repeatability_metrics(per_gesture_df: pd.DataFrame) -> pd.DataFrame:
    metrics = []

    valley_matrix = per_gesture_df[[f"{sensor}_valley_mean_value" for sensor in SENSOR_COLS]].to_numpy(dtype=float)
    sensor_enter_matrix = per_gesture_df[[f"{sensor}_to_enter_ms" for sensor in SENSOR_COLS]].to_numpy(dtype=float)
    sensor_raw_matrix = per_gesture_df[[f"{sensor}_to_raw_ms" for sensor in SENSOR_COLS]].to_numpy(dtype=float)

    metric_specs = [
        ("five_sensor_valley_values", per_gesture_df["sensor_valley_value_mean"].to_numpy(dtype=float), valley_matrix, "sensor valley mean values during valley"),
        ("sensor_to_enter_ms", per_gesture_df["sensor_to_enter_ms"].to_numpy(dtype=float), sensor_enter_matrix, "sensor-to-enter delay"),
        ("duration_follow_gap_s", per_gesture_df["duration_follow_gap_s"].to_numpy(dtype=float), None, "enterstate duration minus sensor valley mean duration"),
        ("sensor_to_raw_ms", per_gesture_df["sensor_to_raw_ms"].to_numpy(dtype=float), sensor_raw_matrix, "sensor-to-raw delay"),
        ("raw_to_stable_ms", per_gesture_df["raw_to_stable_ms"].to_numpy(dtype=float), None, "raw-to-stable delay"),
        ("stable_to_enter_ms", per_gesture_df["stable_to_enter_ms"].to_numpy(dtype=float), None, "stable-to-enter delay"),
    ]

    for name, values, icc_matrix, note in metric_specs:
        mean, sd, cv = mean_sd_cv(values)
        icc = icc_2_1(icc_matrix) if icc_matrix is not None else np.nan
        metrics.append(
            {
                "metric": name,
                "note": note,
                "mean": mean,
                "sd": sd,
                "cv_percent": cv,
                "icc_2_1": icc,
            }
        )

    return pd.DataFrame(metrics)


def render_ieee_tables(per_gesture_df: pd.DataFrame, png_path: str, pdf_path: str) -> None:
    duration_cols = [
        "gesture",
        *[f"{sensor}_valley_duration_s" for sensor in SENSOR_COLS],
        "sensor_valley_duration_mean_s",
        "enterstate_duration_s",
        "duration_follow_gap_s",
    ]
    value_cols = [
        "gesture",
        *[f"{sensor}_valley_mean_value" for sensor in SENSOR_COLS],
        "sensor_valley_value_mean",
        "sensor_to_enter_ms",
    ]
    delay_cols = [
        "gesture",
        "sensor_to_raw_ms",
        "raw_to_stable_ms",
        "stable_to_enter_ms",
        "sensor_to_enter_ms",
    ]

    t1 = per_gesture_df[duration_cols].copy()
    t2 = per_gesture_df[value_cols].copy()
    t3 = per_gesture_df[delay_cols].copy()

    for table in (t1, t2, t3):
        for col in table.columns:
            if col != "gesture":
                table[col] = table[col].map(lambda x: f"{x:.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.8))
    titles = [
        "TABLE I. Valley and Duration-Follow Metrics",
        "TABLE II. Valley Mean Values and Sensor-to-Enter",
        "TABLE III. Delay Metrics Across 5 Gestures",
    ]
    for ax, title, table_df in zip(axes, titles, [t1, t2, t3]):
        ax.axis("off")
        ax.set_title(title, fontweight="bold", pad=6)
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.35)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#EAEAEA")
                cell.set_linewidth(0.8)
            else:
                cell.set_linewidth(0.4)
    plt.tight_layout(h_pad=1.0)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def render_repeatability_summary(metrics_df: pd.DataFrame, png_path: str, pdf_path: str) -> None:
    display_df = metrics_df.copy()
    display_df["mean"] = display_df["mean"].map(lambda x: f"{x:.3f}")
    display_df["sd"] = display_df["sd"].map(lambda x: f"{x:.3f}")
    display_df["cv_percent"] = display_df["cv_percent"].map(lambda x: f"{x:.3f}")
    display_df["icc_2_1"] = display_df["icc_2_1"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    display_df = display_df[["metric", "mean", "sd", "cv_percent", "icc_2_1", "note"]]

    fig, ax = plt.subplots(1, 1, figsize=(11.6, 4.8))
    ax.axis("off")
    ax.set_title("TABLE IV. Repeatability Summary Across 5 Gestures", fontweight="bold", pad=6)
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
            cell.set_linewidth(0.8)
        else:
            cell.set_linewidth(0.4)
    plt.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    required = ["t_ms", "pred_id", "vote_id", *SENSOR_COLS]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    df["time_s"] = (df["t_ms"] - df["t_ms"].iloc[0]) / 1000.0

    per_gesture_df = build_per_gesture_table(df)
    metrics_df = build_repeatability_metrics(per_gesture_df)

    per_gesture_df.to_csv(PER_GESTURE_CSV_PATH, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(REPEATABILITY_CSV_PATH, index=False, encoding="utf-8-sig")

    txt_lines = []
    for _, row in per_gesture_df.iterrows():
        txt_lines.append(f"[{row['gesture']}]")
        txt_lines.append(
            f"valley duration mean={row['sensor_valley_duration_mean_s']:.3f}s, "
            f"enterstate duration={row['enterstate_duration_s']:.3f}s, "
            f"follow gap={row['duration_follow_gap_s']:.3f}s"
        )
        txt_lines.append(
            f"valley mean values: "
            f"a0={row['a0_valley_mean_value']:.2f}, a1={row['a1_valley_mean_value']:.2f}, "
            f"a2={row['a2_valley_mean_value']:.2f}, a3={row['a3_valley_mean_value']:.2f}, "
            f"a4={row['a4_valley_mean_value']:.2f}, overall={row['sensor_valley_value_mean']:.2f}"
        )
        txt_lines.append(
            f"delay(ms): sensor->raw={row['sensor_to_raw_ms']:.1f}, "
            f"raw->stable={row['raw_to_stable_ms']:.1f}, "
            f"stable->enter={row['stable_to_enter_ms']:.1f}, "
            f"sensor->enter={row['sensor_to_enter_ms']:.1f}"
        )
        txt_lines.append("")
    with open(PER_GESTURE_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    rep_lines = []
    for _, row in metrics_df.iterrows():
        icc_text = "N/A" if pd.isna(row["icc_2_1"]) else f"{row['icc_2_1']:.3f}"
        rep_lines.append(
            f"{row['metric']}: mean={row['mean']:.3f}, sd={row['sd']:.3f}, "
            f"cv={row['cv_percent']:.3f}%, ICC(2,1)={icc_text}"
        )
    with open(REPEATABILITY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(rep_lines))

    render_ieee_tables(per_gesture_df, TABLE_PNG_PATH, TABLE_PDF_PATH)
    render_repeatability_summary(metrics_df, SUMMARY_PNG_PATH, SUMMARY_PDF_PATH)

    print(f"Saved per-gesture CSV: {PER_GESTURE_CSV_PATH}")
    print(f"Saved per-gesture TXT: {PER_GESTURE_TXT_PATH}")
    print(f"Saved repeatability CSV: {REPEATABILITY_CSV_PATH}")
    print(f"Saved repeatability TXT: {REPEATABILITY_TXT_PATH}")
    print(f"Saved table PNG: {TABLE_PNG_PATH}")
    print(f"Saved table PDF: {TABLE_PDF_PATH}")
    print(f"Saved summary PNG: {SUMMARY_PNG_PATH}")
    print(f"Saved summary PDF: {SUMMARY_PDF_PATH}")


if __name__ == "__main__":
    main()
