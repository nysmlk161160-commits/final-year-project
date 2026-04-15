import asyncio
import time
import traceback
import numpy as np
import torch
from app_defaults import BATCH_SIZE
from runtime_model import SimpleCNN
from runtime_detectors import PrimitiveDetector, PipelineDetector, TaskDetector
from runtime_rules import (
    normalize_pipeline_entry, pipeline_sequence_names_from_entry, normalize_ble_rule_entry, normalize_mouse_rule_entry,
    collect_ble_rule_hits, collect_label_hits, collect_mouse_rule_hits
)
from runtime_transport_udp import UDPBridge, parse_batch_binary, majority_vote
from runtime_mouse import MouseController

def emit_runtime_events(gui_queue, t_ms, primitive_msgs, ble_msgs, task_msgs, pipeline_msgs=None, mouse_msgs=None):
    ts_text = f"[{int(t_ms)} ms]"

    for m in (primitive_msgs or []):
        et = "PRIMITIVE"
        if str(m).startswith("ENTER"):
            et = "ENTER"
        elif str(m).startswith("EXIT"):
            et = "EXIT"
        gui_queue.put(("event", {
            "ts": ts_text,
            "type": et,
            "detail": str(m),
        }))

    for m in (pipeline_msgs or []):
        gui_queue.put(("event", {
            "ts": ts_text,
            "type": "PIPELINE",
            "detail": str(m),
        }))

    for m in (ble_msgs or []):
        gui_queue.put(("event", {
            "ts": ts_text,
            "type": "UDP",
            "detail": str(m),
        }))

    for m in (mouse_msgs or []):
        gui_queue.put(("event", {
            "ts": ts_text,
            "type": "MOUSE",
            "detail": str(m),
        }))

    for m in (task_msgs or []):
        et = "TASK"
        sm = str(m)
        if sm.startswith("TASK START"):
            et = "TASK_START"
        elif sm.startswith("TASK SUCCESS"):
            et = "TASK_SUCCESS"
        elif sm.startswith("TASK FAIL"):
            et = "TASK_FAIL"
        elif sm.startswith("TASK END"):
            et = "TASK_END"
        elif sm.startswith("TASK PROGRESS"):
            et = "TASK_PROGRESS"

        gui_queue.put(("event", {
            "ts": ts_text,
            "type": et,
            "detail": sm,
        }))


async def run_realtime_worker_async(config, gui_queue, stop_event):
    bridge = None
    try:
        current_task_run = None
        task_log_index = 0

        def record_task_log(message, t_ms=None, kind="runtime"):
            nonlocal current_task_run, task_log_index
            if current_task_run is None:
                return
            task_log_index += 1
            current_task_run["log_rows"].append({
                "index": task_log_index,
                "t_ms": "" if t_ms is None else int(t_ms),
                "kind": str(kind),
                "message": str(message),
            })

        def runtime_log(message, t_ms=None, kind="runtime"):
            gui_queue.put(("log", str(message)))
            record_task_log(message, t_ms=t_ms, kind=kind)

        def record_task_messages(messages, t_ms, kind):
            for item in (messages or []):
                record_task_log(item, t_ms=t_ms, kind=kind)

        def start_task_run(t_ms):
            nonlocal current_task_run, task_log_index
            task_log_index = 0
            current_task_run = {
                "run_id": f"{config['TASK_NAME']}_{int(t_ms)}_{int(time.time() * 1000)}",
                "task_name": str(config["TASK_NAME"]),
                "target_type": str(config["TASK_TARGET_TYPE"]),
                "target_name": str(config["TASK_TARGET_NAME"]),
                "label_mode": str(config["TASK_LABEL_MODE"]),
                "required_count": int(config["TASK_REQUIRED_COUNT"]),
                "observed_count": 0,
                "result": "running",
                "result_reason": "",
                "success": False,
                "started_t_ms": int(t_ms),
                "ended_t_ms": None,
                "duration_ms": None,
                "raw_rows": [],
                "log_rows": [],
            }

        def append_task_raw_row(t_ms, x7, phase, pred="NA", vote="NA",
                                primitive_msgs=None, pipeline_msgs=None, ble_msgs=None, task_msgs=None, mouse_msgs=None):
            if current_task_run is None:
                return
            current_task_run["raw_rows"].append({
                "t_ms": int(t_ms),
                "phase": str(phase),
                "pred": str(pred),
                "vote": str(vote),
                "a0": float(x7[0]),
                "a1": float(x7[1]),
                "a2": float(x7[2]),
                "a3": float(x7[3]),
                "a4": float(x7[4]),
                "roll": float(x7[5]),
                "pitch": float(x7[6]),
                "primitive_msgs": " | ".join(str(x) for x in (primitive_msgs or [])),
                "pipeline_msgs": " | ".join(str(x) for x in (pipeline_msgs or [])),
                "udp_msgs": " | ".join(str(x) for x in (ble_msgs or [])),
                "task_msgs": " | ".join(str(x) for x in (task_msgs or [])),
                "mouse_msgs": " | ".join(str(x) for x in (mouse_msgs or [])),
            })

        def finalize_task_run(t_ms, success, reason, task_msgs):
            nonlocal current_task_run
            if current_task_run is None:
                return
            current_task_run["observed_count"] = int(task_detector.count)
            current_task_run["success"] = bool(success)
            current_task_run["result"] = "success" if success else "failed"
            current_task_run["result_reason"] = str(reason)
            current_task_run["ended_t_ms"] = int(t_ms)
            current_task_run["duration_ms"] = int(t_ms) - int(current_task_run["started_t_ms"])
            current_task_run = None

        def maybe_finalize_task_from_messages(t_ms, task_msgs):
            success = any(str(m).startswith("TASK SUCCESS") for m in (task_msgs or []))
            failed = any(str(m).startswith("TASK FAIL") for m in (task_msgs or []))
            if success:
                finalize_task_run(t_ms, True, "success", task_msgs)
            elif failed:
                reason = "failed"
                for item in (task_msgs or []):
                    text = str(item)
                    if text.startswith("TASK FAIL") and "reason=" in text:
                        reason = text.split("reason=", 1)[1].split()[0]
                        break
                finalize_task_run(t_ms, False, reason, task_msgs)

        runtime_log("=== Realtime PC Inference (Wi-Fi UDP batch stream from Arduino1) ===")
        runtime_log(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        runtime_log(f"MODEL: {config['MODEL_PATH']}")
        runtime_log(f"LABEL: {config['LABEL_PATH']}")
        runtime_log(f"NORM : {config['NORM_PATH']}")
        runtime_log(f"HOST_IP (Python bind): {config['HOST_IP']}")
        runtime_log(f"SENSOR_PORT (Arduino1 -> Python): {config['SENSOR_PORT']}")
        runtime_log(f"ARDUINO2_IP (Python -> Arduino2): {config['ARDUINO2_IP']}")
        runtime_log(f"CMD_PORT (Python -> Arduino2): {config['CMD_PORT']}")
        runtime_log(f"Binary batch format: header='<HHI', frame='<I5Hhh', batch_bytes={BATCH_SIZE}")
        runtime_log("Communication logic: Arduino1 sends 5-sample UDP batch, Python replays 5 frames one by one into the original processing pipeline.")
        runtime_log("Parameter changes only take effect after stopping and restarting.")

        label2id = np.load(config["LABEL_PATH"], allow_pickle=True).item()
        id2label = {v: k for k, v in label2id.items()}
        K = len(label2id)

        norm = np.load(config["NORM_PATH"])
        mean = norm["mean"].astype(np.float32)
        std = norm["std"].astype(np.float32)
        C_IN = int(mean.shape[0])

        use_diff = bool(config["USE_ONLINE_FLEX_DIFF"])
        if C_IN == 7 and use_diff:
            raise ValueError("norm_stats has 7 channels, but USE_ONLINE_FLEX_DIFF=True. Please set it to False.")
        if C_IN == 12 and not use_diff:
            raise ValueError("norm_stats has 12 channels, but USE_ONLINE_FLEX_DIFF=False. Please set it to True.")
        if C_IN not in (7, 12):
            raise ValueError(f"Unsupported input channel count C_IN={C_IN}, expected 7 or 12")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleCNN(K, C_IN).to(device)
        model.load_state_dict(torch.load(config["MODEL_PATH"], map_location=device))
        model.eval()

        primitive_detector = PrimitiveDetector(
            rest_label=config["REST_LABEL"],
            label_stable_ms=config["LABEL_STABLE_MS"],
            quick_tap_max_ms=config["QUICK_TAP_MAX_MS"],
            long_hold_min_ms=config["LONG_HOLD_MIN_MS"],
            double_tap_gap_ms=config["DOUBLE_TAP_GAP_MS"],
            enter_confirm_ms=config["ENTER_CONFIRM_MS"],
        )
        mouse_control_enabled = bool(config.get("MOUSE_CONTROL_ENABLED", False))
        runtime_pipeline_enabled = not mouse_control_enabled
        runtime_udp_enabled = not mouse_control_enabled
        runtime_task_enabled = bool(config["TASK_ENABLED"]) and (not mouse_control_enabled)

        runtime_pipeline = {
            "name": config["PIPELINE_NAME"],
            "signal_name": f"sig_{config['PIPELINE_NAME']}",
            "sequence": [{"name": x, "type": "either"} for x in config["PIPELINE_SEQUENCE"]],
            "step_gap_ms": config["PIPELINE_STEP_GAP_MS"],
            "cooldown_ms": config["PIPELINE_COOLDOWN_MS"],
            "enabled": bool(config["PIPELINE_ENABLED"]) and runtime_pipeline_enabled,
        }

        pipeline_map = {}
        for p in config.get("PIPELINE_LIBRARY", []):
            q = normalize_pipeline_entry(p)
            if q:
                pipeline_map[q["name"]] = q
        q = normalize_pipeline_entry(runtime_pipeline)
        if q:
            pipeline_map[q["name"]] = q

        pipeline_detectors = [
            PipelineDetector(
                enabled=bool(p["enabled"]) and runtime_pipeline_enabled,
                pipeline_name=p["name"],
                sequence=pipeline_sequence_names_from_entry(p),
                step_gap_ms=p["step_gap_ms"],
                cooldown_ms=p["cooldown_ms"],
            )
            for p in sorted(pipeline_map.values(), key=lambda z: z["name"])
        ]

        ble_rules = []
        for r in config.get("BLE_RULE_LIBRARY", []):
            q = normalize_ble_rule_entry(r)
            if q:
                ble_rules.append(q)
        mouse_rules = []
        for r in config.get("MOUSE_RULE_LIBRARY", []):
            q = normalize_mouse_rule_entry(r)
            if q:
                mouse_rules.append(q)
        mouse_controller = MouseController(gui_queue=gui_queue, enabled=mouse_control_enabled, interval_sec=0.025)
        mouse_controller.start()

        gui_queue.put(("log", f"Loaded pipeline detectors: {[x.pipeline_name for x in pipeline_detectors]}"))
        gui_queue.put(("log", f"Loaded UDP signal rules: {[x['rule_name'] for x in ble_rules]}"))
        gui_queue.put(("log", f"Loaded mouse rules: {[x['rule_name'] for x in mouse_rules]}"))
        if mouse_control_enabled:
            gui_queue.put(("log", "Mouse Control Mode Enabled: pipeline/task/udp disabled for this run"))
            if not mouse_controller.available:
                gui_queue.put(("log", "[MouseController] pyautogui unavailable. Install dependency: pip install pyautogui"))

        task_detector = TaskDetector(
            enabled=runtime_task_enabled,
            task_name=config["TASK_NAME"],
            target_type=config["TASK_TARGET_TYPE"],
            target_name=config["TASK_TARGET_NAME"],
            label_mode=config["TASK_LABEL_MODE"],
            duration_sec=config["TASK_DURATION_SEC"],
            required_count=config["TASK_REQUIRED_COUNT"],
        )
        task_started = False

        bridge = UDPBridge(config, gui_queue)
        gui_queue.put(("status", "Connecting"))
        await bridge.connect()
        gui_queue.put(("status", "Running"))
        runtime_log("UDP receiver opened. Start reading binary 5-sample batch packets...")

        window_len = int(config["WINDOW_LEN"])
        ring = np.zeros((window_len, C_IN), dtype=np.float32)
        ring_count = 0
        ring_head = 0
        pred_hist = []
        sample_count = 0
        prev_a = None

        last_gui_emit_time = 0.0
        last_frame_debug_log_time = 0.0
        last_batch_debug_log_time = 0.0
        prev_frame_t_ms = None
        prev_batch_id = None

        def infer_current_window():
            nonlocal ring_head, ring_count
            if ring_count < window_len:
                return None
            idx = ring_head
            win = np.empty((window_len, C_IN), dtype=np.float32)
            for t in range(window_len):
                win[t] = ring[idx]
                idx = (idx + 1) % window_len
            win = (win - mean) / std
            x_t = torch.from_numpy(win.T[None, ...]).to(device)
            with torch.no_grad():
                return int(torch.argmax(model(x_t)[0]).item())

        async def process_one_frame(t_ms, x7):
            nonlocal ring, ring_count, ring_head, pred_hist, sample_count, prev_a
            nonlocal last_gui_emit_time, task_started
            nonlocal prev_frame_t_ms, last_frame_debug_log_time

            now_sec = time.time()
            dt_board = None if prev_frame_t_ms is None else (t_ms - prev_frame_t_ms)
            prev_frame_t_ms = t_ms
            if now_sec - last_frame_debug_log_time >= 0.5:
                last_frame_debug_log_time = now_sec
                runtime_log(f"[FRAME_RATE] t_ms={t_ms} dt_board={dt_board if dt_board is not None else 'NA'}ms", t_ms=t_ms, kind="frame")

            if not task_started and runtime_task_enabled:
                start_msgs = task_detector.start(t_ms)
                if start_msgs:
                    start_task_run(t_ms)
                for m in start_msgs:
                    runtime_log(m, t_ms=t_ms, kind="task")
                if start_msgs:
                    emit_runtime_events(
                        gui_queue=gui_queue,
                        t_ms=t_ms,
                        primitive_msgs=[],
                        ble_msgs=[],
                        task_msgs=start_msgs,
                        pipeline_msgs=[],
                        mouse_msgs=[],
                    )
                gui_queue.put(("task_state", task_detector.get_status_text(t_ms)))
                task_started = True

            if C_IN == 7:
                x = x7
            else:
                a = x7[:5].astype(np.float32)
                da = np.zeros((5,), dtype=np.float32) if prev_a is None else (a - prev_a)
                prev_a = a
                x = np.concatenate([x7, da], axis=0)

            ring[ring_head] = x
            ring_head = (ring_head + 1) % window_len
            ring_count = min(window_len, ring_count + 1)
            sample_count += 1

            if ring_count < window_len:
                append_task_raw_row(t_ms, x7, phase="warmup")
                now_gui = time.time()
                if now_gui - last_gui_emit_time >= float(config["PRINT_EVERY_SEC"]):
                    last_gui_emit_time = now_gui
                    gui_queue.put(("warmup", {"t_ms": t_ms, "ring_count": ring_count, "window_len": window_len, "x7": x7.tolist()}))
                return

            if sample_count % int(config["INFER_EVERY_N_SAMPLES"]) != 0:
                if runtime_task_enabled and task_started:
                    task_msgs = task_detector.tick(t_ms)
                    record_task_messages(task_msgs, t_ms=t_ms, kind="task")
                    append_task_raw_row(t_ms, x7, phase="no_infer", task_msgs=task_msgs)
                    maybe_finalize_task_from_messages(t_ms, task_msgs)

                    if task_msgs:
                        emit_runtime_events(
                            gui_queue=gui_queue,
                            t_ms=t_ms,
                            primitive_msgs=[],
                            ble_msgs=[],
                            task_msgs=task_msgs,
                            pipeline_msgs=[],
                            mouse_msgs=[],
                        )

                        now_gui = time.time()
                        if now_gui - last_gui_emit_time >= float(config["PRINT_EVERY_SEC"]):
                            last_gui_emit_time = now_gui
                            gui_queue.put(("data", {
                                "timestamp": f"[{t_ms} ms]",
                                "pred": "NA",
                                "vote": "NA",
                                "a0": f"{x7[0]:.0f}",
                                "a1": f"{x7[1]:.0f}",
                                "a2": f"{x7[2]:.0f}",
                                "a3": f"{x7[3]:.0f}",
                                "a4": f"{x7[4]:.0f}",
                                "roll": f"{x7[5]:.2f}",
                                "pitch": f"{x7[6]:.2f}",
                                "raw_line": (
                                    f"[{t_ms:>6} ms] pred=NA       vote=NA       "
                                    f"a0={x7[0]:.0f} a1={x7[1]:.0f} a2={x7[2]:.0f} a3={x7[3]:.0f} a4={x7[4]:.0f} "
                                    f"roll={x7[5]:.2f} pitch={x7[6]:.2f} | " + " | ".join(task_msgs)
                                ),
                                "primitive_msgs": [],
                                "pipeline_msgs": [],
                                "ble_msgs": [],
                                "task_msgs": [],
                                "task_state": task_detector.get_status_text(t_ms),
                            }))
                return

            pred_id = infer_current_window()
            if pred_id is None:
                return

            pred_hist.append(pred_id)
            if len(pred_hist) > int(config["SMOOTH_VOTE_N"]):
                pred_hist.pop(0)

            instant = id2label.get(pred_hist[-1], str(pred_hist[-1])) if pred_hist else "NA"
            voted_id = majority_vote(pred_hist)
            voted = id2label.get(voted_id, str(voted_id)) if voted_id is not None else "NA"

            primitive_msgs, structured_events = [], []
            pipeline_msgs, pipeline_complete_events = [], []
            ble_msgs, task_msgs, mouse_msgs = [], [], []

            if voted != "NA":
                primitive_msgs, structured_events = primitive_detector.update(voted, int(t_ms))

                if runtime_pipeline_enabled:
                    for evt in structured_events:
                        for pdet in pipeline_detectors:
                            m, done = pdet.update_event(evt)
                            pipeline_msgs += m
                            pipeline_complete_events += done

                if runtime_udp_enabled:
                    ble_hits = collect_ble_rule_hits(structured_events, primitive_msgs, ble_rules, t_ms)
                    for hit in ble_hits:
                        ble_msgs.append(
                            f"UDP SIGNAL HIT [{hit['rule_name']}] gesture={hit['gesture_name']}[{hit['gesture_type']}] -> {hit['signal_name']}"
                        )
                        ok, ret = await bridge.send_signal(hit["signal_name"])
                        ble_msgs.append(
                            f"UDP SEND {hit['signal_name']} -> {ret}" if ok else f"UDP SEND FAIL {hit['signal_name']} -> {ret}"
                        )

                if runtime_task_enabled and task_started:
                    if config["TASK_TARGET_TYPE"] == "label":
                        label_hits = collect_label_hits(
                            structured_events, primitive_msgs, config["TASK_TARGET_NAME"], config["TASK_LABEL_MODE"], t_ms
                        )
                        for hit_type, hit_name, hit_t in label_hits:
                            task_msgs += task_detector.process_hit(hit_type, hit_name, hit_t)

                    if config["TASK_TARGET_TYPE"] == "pipeline":
                        for pevt in pipeline_complete_events:
                            task_msgs += task_detector.process_hit("pipeline", str(pevt.get("pipeline_name", "")), int(pevt.get("t_ms", t_ms)))

                    task_msgs += task_detector.tick(t_ms)
                if mouse_control_enabled:
                    mouse_hits = collect_mouse_rule_hits(structured_events, primitive_msgs, mouse_rules, t_ms)
                    for hit in mouse_hits:
                        new_msgs = mouse_controller.handle_hit(hit)
                        mouse_msgs += new_msgs
                        for m in new_msgs:
                            runtime_log(m, t_ms=t_ms, kind="mouse")

            record_task_messages(primitive_msgs, t_ms=t_ms, kind="primitive")
            record_task_messages(pipeline_msgs, t_ms=t_ms, kind="pipeline")
            record_task_messages(ble_msgs, t_ms=t_ms, kind="udp")
            record_task_messages(task_msgs, t_ms=t_ms, kind="task")
            record_task_messages(mouse_msgs, t_ms=t_ms, kind="mouse")
            append_task_raw_row(
                t_ms, x7, phase="infer", pred=instant, vote=voted,
                primitive_msgs=primitive_msgs, pipeline_msgs=pipeline_msgs,
                ble_msgs=ble_msgs, task_msgs=task_msgs, mouse_msgs=mouse_msgs,
            )
            maybe_finalize_task_from_messages(t_ms, task_msgs)

            if primitive_msgs or pipeline_msgs or ble_msgs or task_msgs or mouse_msgs:
                emit_runtime_events(
                    gui_queue=gui_queue,
                    t_ms=t_ms,
                    primitive_msgs=primitive_msgs,
                    ble_msgs=ble_msgs,
                    task_msgs=task_msgs,
                    pipeline_msgs=pipeline_msgs,
                    mouse_msgs=mouse_msgs,
                )

            all_msgs = primitive_msgs + pipeline_msgs + ble_msgs + task_msgs + mouse_msgs
            msg = (
                f"[{t_ms:>6} ms] pred={instant:<8} vote={voted:<8} "
                f"a0={x7[0]:.0f} a1={x7[1]:.0f} a2={x7[2]:.0f} a3={x7[3]:.0f} a4={x7[4]:.0f} "
                f"roll={x7[5]:.2f} pitch={x7[6]:.2f}"
            )
            if all_msgs:
                msg += " | " + " | ".join(all_msgs)

            now_gui = time.time()
            if now_gui - last_gui_emit_time >= float(config["PRINT_EVERY_SEC"]):
                last_gui_emit_time = now_gui
                gui_queue.put(("data", {
                    "timestamp": f"[{t_ms} ms]",
                    "pred": str(instant),
                    "vote": str(voted),
                    "a0": f"{x7[0]:.0f}",
                    "a1": f"{x7[1]:.0f}",
                    "a2": f"{x7[2]:.0f}",
                    "a3": f"{x7[3]:.0f}",
                    "a4": f"{x7[4]:.0f}",
                    "roll": f"{x7[5]:.2f}",
                    "pitch": f"{x7[6]:.2f}",
                    "raw_line": msg,
                    "primitive_msgs": [],
                    "pipeline_msgs": [],
                    "ble_msgs": [],
                    "task_msgs": [],
                    "task_state": task_detector.get_status_text(t_ms) if runtime_task_enabled else ("Task Disabled (Mouse Control Mode)" if mouse_control_enabled else "Task Disabled"),
                }))

        while not stop_event.is_set():
            packet, addr = await bridge.get_packet(timeout=0.2)
            if packet is None:
                continue

            parsed = parse_batch_binary(packet)

            if parsed is None:
                if packet:
                    now = time.time()
                    if now - bridge.last_nondata_log_time > 1.0:
                        bridge.last_nondata_log_time = now
                        runtime_log(f"[UDP][NON-DATA] from={addr} len={len(packet)} bytes={packet[:32]!r}")
                continue

            batch_id, frames = parsed

            now_sec = time.time()
            if now_sec - last_batch_debug_log_time >= 0.5:
                gap_text = "NA" if prev_batch_id is None else str(batch_id - prev_batch_id)
                runtime_log(f"[UDP][BATCH] from={addr} batch_id={batch_id} frame_count={len(frames)} batch_gap={gap_text}")
                last_batch_debug_log_time = now_sec
            prev_batch_id = batch_id

            for t_ms, x7 in frames:
                if stop_event.is_set():
                    break
                await process_one_frame(t_ms, x7)

        gui_queue.put(("status", "Stopped"))
        runtime_log("Worker stopped normally.")

    except Exception as e:
        gui_queue.put(("status", "Error"))
        gui_queue.put(("log", "ERROR: " + str(e)))
        gui_queue.put(("log", traceback.format_exc()))
    finally:
        try:
            if "mouse_controller" in locals() and mouse_controller is not None:
                mouse_controller.stop()
        except Exception:
            pass
        if bridge is not None:
            try:
                await bridge.disconnect()
            except Exception:
                pass


def run_realtime_worker(config, gui_queue, stop_event):
    asyncio.run(run_realtime_worker_async(config, gui_queue, stop_event))
