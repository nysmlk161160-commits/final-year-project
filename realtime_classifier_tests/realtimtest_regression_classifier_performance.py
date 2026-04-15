import csv
import os
import pickle
import time
import traceback

import numpy as np
import serial


DEFAULT_PORT = "COM5"
DEFAULT_BAUD = 500000
DEFAULT_TIMEOUT = 0.5
DEFAULT_USE_ONLINE_FLEX_DIFF = True
DEFAULT_WINDOW_LEN = 20
DEFAULT_INFER_EVERY_N_SAMPLES = 3
DEFAULT_SMOOTH_VOTE_N = 10
DEFAULT_PRINT_EVERY_SEC = 0.02
SAVE_CSV = False


def parse_line_7ch(line: str):
    line = line.strip()
    if not line or line.startswith("time_ms") or line.startswith("PROF"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 8:
        return None
    try:
        t_ms = int(float(parts[0]))
        x7 = np.array(
            [
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
            ],
            dtype=np.float32,
        )
        return t_ms, x7
    except Exception:
        return None


def majority_vote(ids):
    if not ids:
        return None
    vals, counts = np.unique(np.array(ids, dtype=np.int32), return_counts=True)
    return int(vals[np.argmax(counts)])


def main():
    port = DEFAULT_PORT
    baud = DEFAULT_BAUD
    timeout = DEFAULT_TIMEOUT

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, "models")

    model_path = os.path.join(model_dir, "regression_window_model.pkl")
    label_path = os.path.join(model_dir, "regression_label_map.npy")
    norm_path = os.path.join(model_dir, "regression_norm_stats.npz")
    save_csv_path = os.path.join(script_dir, "realtime_regression_classifier_log.csv")

    use_online_flex_diff = DEFAULT_USE_ONLINE_FLEX_DIFF
    window_len = DEFAULT_WINDOW_LEN
    infer_every_n_samples = DEFAULT_INFER_EVERY_N_SAMPLES
    smooth_vote_n = DEFAULT_SMOOTH_VOTE_N
    print_every_sec = DEFAULT_PRINT_EVERY_SEC

    print("=== Realtime Logistic Regression Classifier Performance Test ===")
    print(f"Port: {port}")
    print(f"Baud: {baud}")
    print(f"Timeout: {timeout}")
    print(f"Model path: {model_path}")
    print(f"Label path: {label_path}")
    print(f"Normalization path: {norm_path}")
    print(f"Use online flex diff: {use_online_flex_diff}")
    print(f"Window length: {window_len}")
    print(f"Infer every n samples: {infer_every_n_samples}")
    print(f"Voting length: {smooth_vote_n}")
    print(f"Print period: {print_every_sec}")
    print(f"Save CSV: {SAVE_CSV}")
    if SAVE_CSV:
        print(f"CSV path: {save_csv_path}")
    print("")

    for required_path in (model_path, label_path, norm_path):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required file not found: {required_path}")

    with open(model_path, "rb") as handle:
        model = pickle.load(handle)

    label2id = np.load(label_path, allow_pickle=True).item()
    id2label = {v: k for k, v in label2id.items()}

    norm = np.load(norm_path)
    mean = norm["mean"].astype(np.float32)
    std = norm["std"].astype(np.float32)
    c_in = int(mean.shape[0])

    if c_in == 7 and use_online_flex_diff:
        raise ValueError("The normalization statistics expect 7 input channels. Set USE_ONLINE_FLEX_DIFF to False.")
    if c_in == 12 and not use_online_flex_diff:
        raise ValueError("The normalization statistics expect 12 input channels. Set USE_ONLINE_FLEX_DIFF to True.")
    if c_in not in (7, 12):
        raise ValueError(f"Unsupported input channel count: {c_in}. Expected 7 or 12.")

    print(f"Classes: {len(label2id)}")
    print(f"Input channels: {c_in}")
    print(f"id2label: {id2label}")
    print("")

    ring = np.zeros((window_len, c_in), dtype=np.float32)
    ring_count = 0
    ring_head = 0
    pred_hist = []
    sample_count = 0
    last_emit_time = 0.0
    prev_a = None

    if c_in == 7:
        header = ["t_ms", "ring_count", "window_len", "pred_id", "pred_label", "vote_id", "vote_label", "a0", "a1", "a2", "a3", "a4", "roll", "pitch"]
    else:
        header = ["t_ms", "ring_count", "window_len", "pred_id", "pred_label", "vote_id", "vote_label", "a0", "a1", "a2", "a3", "a4", "roll", "pitch", "da0", "da1", "da2", "da3", "da4"]

    def infer_current_window():
        nonlocal ring_count, ring_head
        if ring_count < window_len:
            return None
        idx = ring_head
        win = np.empty((window_len, c_in), dtype=np.float32)
        for t in range(window_len):
            win[t] = ring[idx]
            idx = (idx + 1) % window_len
        win = (win - mean) / std
        x_flat = win.reshape(1, -1)
        pred = model.predict(x_flat)
        return int(pred[0])

    csv_file = None
    csv_writer = None

    try:
        if SAVE_CSV:
            csv_file = open(save_csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)
            csv_file.flush()

        print(",".join(header))

        try:
            ser = serial.Serial(port, baud, timeout=timeout)
        except Exception as exc:
            raise RuntimeError(f"Failed to open serial port: port={port}, baud={baud}, error={exc}")

        time.sleep(2.0)
        ser.reset_input_buffer()

        print("Serial opened. Start reading...")
        print("Press Ctrl+C to stop.")
        print("")

        while True:
            raw = ser.readline()
            if not raw:
                continue
            parsed = parse_line_7ch(raw.decode("utf-8", errors="ignore"))
            if parsed is None:
                continue

            t_ms, x7 = parsed
            if c_in == 7:
                x = x7
            else:
                a = x7[:5].astype(np.float32)
                da = np.zeros((5,), dtype=np.float32) if prev_a is None else a - prev_a
                prev_a = a
                x = np.concatenate([x7, da], axis=0)

            ring[ring_head] = x
            ring_head = (ring_head + 1) % window_len
            ring_count = min(window_len, ring_count + 1)
            sample_count += 1

            if ring_count >= window_len and (sample_count % infer_every_n_samples == 0):
                pred_id = infer_current_window()
                if pred_id is not None:
                    pred_hist.append(pred_id)
                    if len(pred_hist) > smooth_vote_n:
                        pred_hist.pop(0)

            now = time.time()
            if now - last_emit_time < print_every_sec:
                continue
            last_emit_time = now

            if ring_count < window_len:
                pred_id_out = pred_label_out = vote_id_out = vote_label_out = "WARMUP"
            else:
                if pred_hist:
                    current_pred_id = pred_hist[-1]
                    pred_id_out = str(current_pred_id)
                    pred_label_out = str(id2label.get(current_pred_id, str(current_pred_id)))
                else:
                    pred_id_out = "NA"
                    pred_label_out = "NA"
                voted_id = majority_vote(pred_hist)
                if voted_id is None:
                    vote_id_out = "NA"
                    vote_label_out = "NA"
                else:
                    vote_id_out = str(voted_id)
                    vote_label_out = str(id2label.get(voted_id, str(voted_id)))

            if c_in == 7:
                row = [int(t_ms), int(ring_count), int(window_len), pred_id_out, pred_label_out, vote_id_out, vote_label_out, f"{x7[0]:.0f}", f"{x7[1]:.0f}", f"{x7[2]:.0f}", f"{x7[3]:.0f}", f"{x7[4]:.0f}", f"{x7[5]:.2f}", f"{x7[6]:.2f}"]
            else:
                row = [int(t_ms), int(ring_count), int(window_len), pred_id_out, pred_label_out, vote_id_out, vote_label_out, f"{x7[0]:.0f}", f"{x7[1]:.0f}", f"{x7[2]:.0f}", f"{x7[3]:.0f}", f"{x7[4]:.0f}", f"{x7[5]:.2f}", f"{x7[6]:.2f}", f"{x[7]:.0f}", f"{x[8]:.0f}", f"{x[9]:.0f}", f"{x[10]:.0f}", f"{x[11]:.0f}"]

            print(",".join(map(str, row)))
            if csv_writer is not None:
                csv_writer.writerow(row)
                csv_file.flush()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Stopping...")
    except Exception as exc:
        print("\nERROR:", str(exc))
        print(traceback.format_exc())
    finally:
        try:
            if "ser" in locals() and ser is not None and ser.is_open:
                ser.close()
                print("Serial closed.")
        except Exception:
            pass
        try:
            if csv_file is not None:
                csv_file.close()
                print("CSV file closed.")
        except Exception:
            pass
        print("Program exited.")


if __name__ == "__main__":
    main()
