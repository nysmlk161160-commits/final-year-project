import copy
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from app_defaults import *
from app_state import save_json_state, load_json_state
from runtime_rules import parse_pipeline_sequence_text, normalize_pipeline_entry, normalize_ble_rule_entry, normalize_mouse_rule_entry
from runtime_worker import run_realtime_worker
from ui_builders import (
    PipelineEditor, PipelineCard, BLERuleEditor, BLERuleCard, MouseRuleEditor, MouseRuleCard,
)

class GestureMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Realtime Gesture Monitor")
        self.root.geometry("1450x900")
        self.root.minsize(1200, 760)

        self.gui_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = None
        self.is_running = False

        self.settings_window = None
        self.log_buffer = []
        self.gesture_library_window = None
        self.gesture_library_photo = None

        self._state_loading = False
        self._save_after_id = None

        self.build_style()
        self.build_variables()
        self.load_state_if_exists()
        self.build_layout()
        self.bind_persistent_traces()

        self.root.protocol("WM_DELETE_WINDOW", self.on_root_close)
        self.root.after(50, self.process_gui_queue)

    def build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Title.TLabel", font=("Arial", 18, "bold"))
        style.configure("BigValue.TLabel", font=("Consolas", 16, "bold"))
        style.configure("Treeview", font=("Consolas", 10), rowheight=24)
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))

    def load_gesture_names_from_label_map(self):
        try:
            label2id = np.load(self.label_path_var.get().strip(), allow_pickle=True).item()
            return [str(k) for k, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
        except Exception as e:
            print(f"[WARN] Failed to load label map: {e}")
            return ["0", "1", "2", "3"]

    def get_default_pipeline_library(self):
        return [{
            "name": DEFAULT_PIPELINE_NAME,
            "signal_name": f"sig_{DEFAULT_PIPELINE_NAME}",
            "sequence": [{"name": x, "type": "either"} for x in parse_pipeline_sequence_text(DEFAULT_PIPELINE_SEQUENCE)],
            "step_gap_ms": DEFAULT_PIPELINE_STEP_GAP_MS,
            "cooldown_ms": DEFAULT_PIPELINE_COOLDOWN_MS,
            "enabled": DEFAULT_PIPELINE_ENABLED,
        }]

    def get_default_ble_rule_library(self):
        label0 = self.gesture_names[0] if self.gesture_names else "0"
        return [{
            "rule_name": "rule_signal0",
            "signal_name": "signal0",
            "gesture": {"name": label0, "type": "either"},
            "enabled": True,
        }]

    def get_default_mouse_rule_library(self):
        label0 = self.gesture_names[0] if self.gesture_names else "0"
        return [{
            "rule_name": "mouse_rule_click",
            "mouse_action": "click",
            "gesture": {"name": label0, "type": "tap"},
            "move_speed": DEFAULT_MOUSE_MOVE_SPEED,
            "enabled": True,
        }]

    def build_variables(self):
        self.model_path_var = tk.StringVar(value=DEFAULT_MODEL_PATH)
        self.label_path_var = tk.StringVar(value=DEFAULT_LABEL_PATH)
        self.norm_path_var = tk.StringVar(value=DEFAULT_NORM_PATH)

        self.window_len_var = tk.StringVar(value=str(DEFAULT_WINDOW_LEN))
        self.infer_every_var = tk.StringVar(value=str(DEFAULT_INFER_EVERY_N_SAMPLES))
        self.smooth_vote_var = tk.StringVar(value=str(DEFAULT_SMOOTH_VOTE_N))
        self.print_every_var = tk.StringVar(value=str(DEFAULT_PRINT_EVERY_SEC))
        self.use_diff_var = tk.BooleanVar(value=DEFAULT_USE_ONLINE_FLEX_DIFF)

        self.rest_label_var = tk.StringVar(value=DEFAULT_REST_LABEL)
        self.label_stable_var = tk.StringVar(value=str(DEFAULT_LABEL_STABLE_MS))
        self.quick_tap_var = tk.StringVar(value=str(DEFAULT_QUICK_TAP_MAX_MS))
        self.long_hold_var = tk.StringVar(value=str(DEFAULT_LONG_HOLD_MIN_MS))
        self.double_tap_gap_var = tk.StringVar(value=str(DEFAULT_DOUBLE_TAP_GAP_MS))
        self.enter_confirm_var = tk.StringVar(value=str(DEFAULT_ENTER_CONFIRM_MS))

        self.pipeline_enabled_var = tk.BooleanVar(value=DEFAULT_PIPELINE_ENABLED)
        self.pipeline_name_var = tk.StringVar(value=DEFAULT_PIPELINE_NAME)
        self.pipeline_sequence_var = tk.StringVar(value=DEFAULT_PIPELINE_SEQUENCE)
        self.pipeline_step_gap_var = tk.StringVar(value=str(DEFAULT_PIPELINE_STEP_GAP_MS))
        self.pipeline_cooldown_var = tk.StringVar(value=str(DEFAULT_PIPELINE_COOLDOWN_MS))

        self.task_enabled_var = tk.BooleanVar(value=DEFAULT_TASK_ENABLED)
        self.task_name_var = tk.StringVar(value=DEFAULT_TASK_NAME)
        self.task_target_type_var = tk.StringVar(value=DEFAULT_TASK_TARGET_TYPE)
        self.task_target_name_var = tk.StringVar(value=DEFAULT_TASK_TARGET_NAME)
        self.task_label_mode_var = tk.StringVar(value=DEFAULT_TASK_LABEL_MODE)
        self.task_duration_var = tk.StringVar(value=str(DEFAULT_TASK_DURATION_SEC))
        self.task_required_count_var = tk.StringVar(value=str(DEFAULT_TASK_REQUIRED_COUNT))

        self.host_ip_var = tk.StringVar(value=DEFAULT_HOST_IP)
        self.sensor_port_var = tk.StringVar(value=str(DEFAULT_SENSOR_PORT))
        self.arduino2_ip_var = tk.StringVar(value=DEFAULT_ARDUINO2_IP)
        self.cmd_port_var = tk.StringVar(value=str(DEFAULT_CMD_PORT))
        self.udp_send_enabled_var = tk.BooleanVar(value=DEFAULT_UDP_SEND_ENABLED)
        self.mouse_control_enabled_var = tk.BooleanVar(value=DEFAULT_MOUSE_CONTROL_ENABLED)

        self.gesture_library_image_path_var = tk.StringVar(value=DEFAULT_GESTURE_LIBRARY_IMAGE_PATH)

        self.current_mode_var = tk.StringVar(value="Idle")
        self.status_var = tk.StringVar(value="Ready")
        self.current_task_var = tk.StringVar(value="Task Disabled")

        self.gesture_names = self.load_gesture_names_from_label_map()

        self.pipeline_library = self.get_default_pipeline_library()
        self.ble_rule_library = self.get_default_ble_rule_library()
        self.mouse_rule_library = self.get_default_mouse_rule_library()

    def bind_persistent_traces(self):
        vars_to_watch = [
            self.model_path_var, self.label_path_var, self.norm_path_var,
            self.window_len_var, self.infer_every_var, self.smooth_vote_var, self.print_every_var,
            self.use_diff_var,
            self.rest_label_var, self.label_stable_var, self.quick_tap_var, self.long_hold_var,
            self.double_tap_gap_var, self.enter_confirm_var,
            self.pipeline_enabled_var, self.pipeline_name_var, self.pipeline_sequence_var,
            self.pipeline_step_gap_var, self.pipeline_cooldown_var,
            self.task_enabled_var, self.task_name_var, self.task_target_type_var,
            self.task_target_name_var, self.task_label_mode_var, self.task_duration_var,
            self.task_required_count_var,
            self.host_ip_var, self.sensor_port_var, self.arduino2_ip_var, self.cmd_port_var,
            self.udp_send_enabled_var,
            self.mouse_control_enabled_var,
            self.gesture_library_image_path_var,
        ]
        for v in vars_to_watch:
            v.trace_add("write", self.on_persistent_value_changed)

    def on_persistent_value_changed(self, *args):
        if self._state_loading:
            return
        self.schedule_save_state()

    def schedule_save_state(self):
        if self._state_loading:
            return
        try:
            if self._save_after_id is not None:
                self.root.after_cancel(self._save_after_id)
        except Exception:
            pass
        self._save_after_id = self.root.after(200, self.save_state)

    def safe_bool(self, x, default=False):
        try:
            return bool(x)
        except Exception:
            return default

    def get_state_dict(self):
        return {
            "vars": {
                "MODEL_PATH": self.model_path_var.get(),
                "LABEL_PATH": self.label_path_var.get(),
                "NORM_PATH": self.norm_path_var.get(),
                "WINDOW_LEN": self.window_len_var.get(),
                "INFER_EVERY_N_SAMPLES": self.infer_every_var.get(),
                "SMOOTH_VOTE_N": self.smooth_vote_var.get(),
                "PRINT_EVERY_SEC": self.print_every_var.get(),
                "USE_ONLINE_FLEX_DIFF": bool(self.use_diff_var.get()),

                "REST_LABEL": self.rest_label_var.get(),
                "LABEL_STABLE_MS": self.label_stable_var.get(),
                "QUICK_TAP_MAX_MS": self.quick_tap_var.get(),
                "LONG_HOLD_MIN_MS": self.long_hold_var.get(),
                "DOUBLE_TAP_GAP_MS": self.double_tap_gap_var.get(),
                "ENTER_CONFIRM_MS": self.enter_confirm_var.get(),

                "PIPELINE_ENABLED": bool(self.pipeline_enabled_var.get()),
                "PIPELINE_NAME": self.pipeline_name_var.get(),
                "PIPELINE_SEQUENCE": self.pipeline_sequence_var.get(),
                "PIPELINE_STEP_GAP_MS": self.pipeline_step_gap_var.get(),
                "PIPELINE_COOLDOWN_MS": self.pipeline_cooldown_var.get(),

                "TASK_ENABLED": bool(self.task_enabled_var.get()),
                "TASK_NAME": self.task_name_var.get(),
                "TASK_TARGET_TYPE": self.task_target_type_var.get(),
                "TASK_TARGET_NAME": self.task_target_name_var.get(),
                "TASK_LABEL_MODE": self.task_label_mode_var.get(),
                "TASK_DURATION_SEC": self.task_duration_var.get(),
                "TASK_REQUIRED_COUNT": self.task_required_count_var.get(),

                "HOST_IP": self.host_ip_var.get(),
                "SENSOR_PORT": self.sensor_port_var.get(),
                "ARDUINO2_IP": self.arduino2_ip_var.get(),
                "CMD_PORT": self.cmd_port_var.get(),
                "UDP_SEND_ENABLED": bool(self.udp_send_enabled_var.get()),
                "MOUSE_CONTROL_ENABLED": bool(self.mouse_control_enabled_var.get()),

                "GESTURE_LIBRARY_IMAGE_PATH": self.gesture_library_image_path_var.get(),
            },
            "pipeline_library": copy.deepcopy(self.pipeline_library),
            "ble_rule_library": copy.deepcopy(self.ble_rule_library),
            "mouse_rule_library": copy.deepcopy(self.mouse_rule_library),
        }

    def save_state(self):
        if self._state_loading:
            return
        self._save_after_id = None
        try:
            save_json_state(APP_STATE_PATH, self.get_state_dict())
        except Exception as e:
            print(f"[WARN] save_state failed: {e}")

    def load_state_if_exists(self):
        try:
            state = load_json_state(APP_STATE_PATH)
            if state is None:
                return
        except Exception as e:
            print(f"[WARN] load_state failed: {e}")
            return

        self._state_loading = True
        try:
            vars_map = state.get("vars", {})

            self.model_path_var.set(str(vars_map.get("MODEL_PATH", DEFAULT_MODEL_PATH)))
            self.label_path_var.set(str(vars_map.get("LABEL_PATH", DEFAULT_LABEL_PATH)))
            self.norm_path_var.set(str(vars_map.get("NORM_PATH", DEFAULT_NORM_PATH)))

            self.window_len_var.set(str(vars_map.get("WINDOW_LEN", DEFAULT_WINDOW_LEN)))
            self.infer_every_var.set(str(vars_map.get("INFER_EVERY_N_SAMPLES", DEFAULT_INFER_EVERY_N_SAMPLES)))
            self.smooth_vote_var.set(str(vars_map.get("SMOOTH_VOTE_N", DEFAULT_SMOOTH_VOTE_N)))
            self.print_every_var.set(str(vars_map.get("PRINT_EVERY_SEC", DEFAULT_PRINT_EVERY_SEC)))
            self.use_diff_var.set(bool(vars_map.get("USE_ONLINE_FLEX_DIFF", DEFAULT_USE_ONLINE_FLEX_DIFF)))

            self.rest_label_var.set(str(vars_map.get("REST_LABEL", DEFAULT_REST_LABEL)))
            self.label_stable_var.set(str(vars_map.get("LABEL_STABLE_MS", DEFAULT_LABEL_STABLE_MS)))
            self.quick_tap_var.set(str(vars_map.get("QUICK_TAP_MAX_MS", DEFAULT_QUICK_TAP_MAX_MS)))
            self.long_hold_var.set(str(vars_map.get("LONG_HOLD_MIN_MS", DEFAULT_LONG_HOLD_MIN_MS)))
            self.double_tap_gap_var.set(str(vars_map.get("DOUBLE_TAP_GAP_MS", DEFAULT_DOUBLE_TAP_GAP_MS)))
            self.enter_confirm_var.set(str(vars_map.get("ENTER_CONFIRM_MS", DEFAULT_ENTER_CONFIRM_MS)))

            self.pipeline_enabled_var.set(bool(vars_map.get("PIPELINE_ENABLED", DEFAULT_PIPELINE_ENABLED)))
            self.pipeline_name_var.set(str(vars_map.get("PIPELINE_NAME", DEFAULT_PIPELINE_NAME)))
            self.pipeline_sequence_var.set(str(vars_map.get("PIPELINE_SEQUENCE", DEFAULT_PIPELINE_SEQUENCE)))
            self.pipeline_step_gap_var.set(str(vars_map.get("PIPELINE_STEP_GAP_MS", DEFAULT_PIPELINE_STEP_GAP_MS)))
            self.pipeline_cooldown_var.set(str(vars_map.get("PIPELINE_COOLDOWN_MS", DEFAULT_PIPELINE_COOLDOWN_MS)))

            self.task_enabled_var.set(bool(vars_map.get("TASK_ENABLED", DEFAULT_TASK_ENABLED)))
            self.task_name_var.set(str(vars_map.get("TASK_NAME", DEFAULT_TASK_NAME)))
            self.task_target_type_var.set(str(vars_map.get("TASK_TARGET_TYPE", DEFAULT_TASK_TARGET_TYPE)))
            self.task_target_name_var.set(str(vars_map.get("TASK_TARGET_NAME", DEFAULT_TASK_TARGET_NAME)))
            self.task_label_mode_var.set(str(vars_map.get("TASK_LABEL_MODE", DEFAULT_TASK_LABEL_MODE)))
            self.task_duration_var.set(str(vars_map.get("TASK_DURATION_SEC", DEFAULT_TASK_DURATION_SEC)))
            self.task_required_count_var.set(str(vars_map.get("TASK_REQUIRED_COUNT", DEFAULT_TASK_REQUIRED_COUNT)))

            self.host_ip_var.set(str(vars_map.get("HOST_IP", DEFAULT_HOST_IP)))
            self.sensor_port_var.set(str(vars_map.get("SENSOR_PORT", DEFAULT_SENSOR_PORT)))
            self.arduino2_ip_var.set(str(vars_map.get("ARDUINO2_IP", DEFAULT_ARDUINO2_IP)))
            self.cmd_port_var.set(str(vars_map.get("CMD_PORT", DEFAULT_CMD_PORT)))
            self.udp_send_enabled_var.set(bool(vars_map.get("UDP_SEND_ENABLED", DEFAULT_UDP_SEND_ENABLED)))
            self.mouse_control_enabled_var.set(bool(vars_map.get("MOUSE_CONTROL_ENABLED", DEFAULT_MOUSE_CONTROL_ENABLED)))

            self.gesture_library_image_path_var.set(str(vars_map.get("GESTURE_LIBRARY_IMAGE_PATH", DEFAULT_GESTURE_LIBRARY_IMAGE_PATH)))

            plib = state.get("pipeline_library", None)
            if isinstance(plib, list) and len(plib) > 0:
                clean = []
                for p in plib:
                    q = normalize_pipeline_entry(p)
                    if q:
                        clean.append(q)
                if clean:
                    self.pipeline_library = clean

            rlib = state.get("ble_rule_library", None)
            if isinstance(rlib, list) and len(rlib) > 0:
                clean = []
                for r in rlib:
                    q = normalize_ble_rule_entry(r)
                    if q:
                        clean.append(q)
                if clean:
                    self.ble_rule_library = clean

            mlib = state.get("mouse_rule_library", None)
            if isinstance(mlib, list) and len(mlib) > 0:
                clean = []
                for r in mlib:
                    q = normalize_mouse_rule_entry(r)
                    if q:
                        clean.append(q)
                if clean:
                    self.mouse_rule_library = clean

            self.gesture_names = self.load_gesture_names_from_label_map()

            if not self.ble_rule_library:
                self.ble_rule_library = self.get_default_ble_rule_library()
            if not self.mouse_rule_library:
                self.mouse_rule_library = self.get_default_mouse_rule_library()

        finally:
            self._state_loading = False

    def add_config_row(self, parent, row, label_text, var):
        ttk.Label(parent, text=label_text, font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(parent, textvariable=var, font=("Consolas", 11)).grid(row=row, column=1, sticky="ew", pady=6)

    def create_metric_card(self, parent, row, col, title, var):
        frame = ttk.Frame(parent, padding=10, relief="ridge")
        frame.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)
        ttk.Label(frame, text=title, font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(frame, textvariable=var, style="BigValue.TLabel").pack(anchor="w", pady=(8, 0))

    def build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)

        header = ttk.Frame(self.root, padding=(14, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Realtime Gesture Monitor", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        btn = ttk.Frame(header)
        btn.grid(row=0, column=1, sticky="e")
        ttk.Button(btn, text="Start Monitoring", command=self.start_monitor).grid(row=0, column=0, padx=4)
        ttk.Button(btn, text="Stop Monitoring", command=self.stop_monitor).grid(row=0, column=1, padx=4)
        ttk.Button(btn, text="Clear Logs", command=self.clear_logs).grid(row=0, column=2, padx=4)
        ttk.Button(btn, text="Settings / Managers", command=self.open_settings_window).grid(row=0, column=3, padx=4)
        ttk.Button(btn, text="Gesture Label Library", command=self.open_gesture_library_window).grid(row=0, column=4, padx=4)

        self.build_overview_panel()
        self.build_log_panel()

    def build_overview_panel(self):
        overview = ttk.LabelFrame(self.root, text="Realtime Data Overview", padding=12)
        overview.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        overview.columnconfigure(0, weight=1)

        rawf = ttk.Frame(overview)
        rawf.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        rawf.columnconfigure(1, weight=1)

        ttk.Label(rawf, text="Raw line:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.raw_line_label = tk.Label(
            rawf,
            text="Waiting for data...",
            anchor="w",
            justify="left",
            font=("Consolas", 12),
            bg="#111111",
            fg="#00FF7F",
            padx=10,
            pady=10
        )
        self.raw_line_label.grid(row=0, column=1, sticky="ew")

        metrics = ttk.Frame(overview)
        metrics.grid(row=1, column=0, sticky="ew")
        for i in range(3):
            metrics.columnconfigure(i, weight=1)

        self.create_metric_card(metrics, 0, 0, "Mode", self.current_mode_var)
        self.create_metric_card(metrics, 0, 1, "Status", self.status_var)
        self.create_metric_card(metrics, 0, 2, "Task", self.current_task_var)

            

    def build_log_panel(self):
        event_frame = ttk.LabelFrame(self.root, text="Event Log", padding=12)
        event_frame.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))
        event_frame.columnconfigure(0, weight=1)
        event_frame.rowconfigure(0, weight=1)

        self.event_tree = ttk.Treeview(event_frame, columns=("time", "type", "detail"), show="headings", height=14)
        for col, text, width, anchor in [
            ("time", "Time", 140, "center"),
            ("type", "Type", 160, "center"),
            ("detail", "Detail", 980, "w"),
        ]:
            self.event_tree.heading(col, text=text)
            self.event_tree.column(col, width=width, anchor=anchor)
        self.event_tree.grid(row=0, column=0, sticky="nsew")

        s = ttk.Scrollbar(event_frame, orient="vertical", command=self.event_tree.yview)
        s.grid(row=0, column=1, sticky="ns")
        self.event_tree.configure(yscrollcommand=s.set)

    def open_gesture_library_window(self):
        if self.gesture_library_window is not None and self.gesture_library_window.winfo_exists():
            self.gesture_library_window.lift()
            self.gesture_library_window.focus_force()
            return

        self.gesture_library_window = tk.Toplevel(self.root)
        self.gesture_library_window.title("Gesture-Label Reference Library")
        self.gesture_library_window.geometry("980x760")
        self.gesture_library_window.minsize(760, 560)

        root = ttk.Frame(self.gesture_library_window, padding=16)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        ttk.Label(
            root,
            text="Gesture-Label Reference Library",
            font=("Arial", 16, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        top = ttk.Frame(root)
        top.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="IMAGE_PATH", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(top, textvariable=self.gesture_library_image_path_var, font=("Consolas", 11)).grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Refresh", command=self.refresh_gesture_library_window).grid(row=0, column=2, padx=(8, 0))

        self.gesture_library_display = ttk.Frame(root)
        self.gesture_library_display.grid(row=2, column=0, sticky="nsew")
        self.gesture_library_display.columnconfigure(0, weight=1)
        self.gesture_library_display.rowconfigure(0, weight=1)

        self.refresh_gesture_library_window()

    def refresh_gesture_library_window(self):
        if not hasattr(self, "gesture_library_display") or not self.gesture_library_display.winfo_exists():
            return

        for c in self.gesture_library_display.winfo_children():
            c.destroy()

        img_path = self.gesture_library_image_path_var.get().strip()

        if not img_path:
            ttk.Label(
                self.gesture_library_display,
                text="No image inserted yet.",
                font=("Arial", 14, "italic"),
                anchor="center",
                justify="center"
            ).grid(row=0, column=0, sticky="nsew")
            return

        if not os.path.exists(img_path):
            ttk.Label(
                self.gesture_library_display,
                text=f"Image not found:\n{img_path}",
                font=("Arial", 12, "italic"),
                anchor="center",
                justify="center"
            ).grid(row=0, column=0, sticky="nsew")
            return

        try:
            self.gesture_library_photo = tk.PhotoImage(file=img_path)
            lbl = tk.Label(self.gesture_library_display, image=self.gesture_library_photo, bg="white")
            lbl.grid(row=0, column=0, sticky="nsew")
        except Exception as e:
            ttk.Label(
                self.gesture_library_display,
                text=f"Failed to load image.\nPlease use a Tkinter-supported format such as PNG/GIF.\n\n{e}",
                font=("Arial", 11, "italic"),
                anchor="center",
                justify="center"
            ).grid(row=0, column=0, sticky="nsew")

    def open_settings_window(self):
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            self.settings_window.focus_force()
            return

        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Settings / Managers")
        self.settings_window.geometry("1550x930")
        self.settings_window.minsize(1300, 820)

        root = ttk.Frame(self.settings_window, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.settings_notebook = ttk.Notebook(root)
        self.settings_notebook.grid(row=0, column=0, sticky="nsew")

        tab_config = ttk.Frame(self.settings_notebook)
        tab_pipeline = ttk.Frame(self.settings_notebook)
        tab_task = ttk.Frame(self.settings_notebook)
        tab_udp = ttk.Frame(self.settings_notebook)
        tab_mouse = ttk.Frame(self.settings_notebook)

        self.settings_notebook.add(tab_config, text="Parameters")
        self.settings_notebook.add(tab_pipeline, text="Pipeline")
        self.settings_notebook.add(tab_task, text="Task")
        self.settings_notebook.add(tab_udp, text="Wi-Fi / UDP")
        self.settings_notebook.add(tab_mouse, text="Mouse Control")

        self.build_config_panel(tab_config)
        self.build_pipeline_manager_panel(tab_pipeline)
        self.build_task_config_panel(tab_task)
        self.build_ble_panel(tab_udp)
        self.build_mouse_panel(tab_mouse)

        self.populate_initial_content()

    def make_tab_topbar(self, parent, reset_callback):
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(0, 8))
        ttk.Button(top, text="Restore This Page To Default", command=reset_callback).pack(side="left")
        return top

    def build_config_panel(self, parent):
        wrapper = ttk.Frame(parent, padding=8)
        wrapper.pack(fill="both", expand=True)

        self.make_tab_topbar(wrapper, self.reset_parameters_page_to_default)

        panel = ttk.Frame(wrapper)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(4, weight=1)

        basic = ttk.LabelFrame(panel, text="Basic Inference Parameters", padding=10)
        basic.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        basic.columnconfigure(1, weight=1)
        self.add_config_row(basic, 0, "WINDOW_LEN", self.window_len_var)
        self.add_config_row(basic, 1, "INFER_EVERY_N_SAMPLES", self.infer_every_var)
        self.add_config_row(basic, 2, "SMOOTH_VOTE_N", self.smooth_vote_var)
        self.add_config_row(basic, 3, "PRINT_EVERY_SEC", self.print_every_var)

        paths = ttk.LabelFrame(panel, text="Model / Path Parameters", padding=10)
        paths.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        paths.columnconfigure(1, weight=1)
        self.add_config_row(paths, 0, "MODEL_PATH", self.model_path_var)
        self.add_config_row(paths, 1, "LABEL_PATH", self.label_path_var)
        self.add_config_row(paths, 2, "NORM_PATH", self.norm_path_var)
        ttk.Checkbutton(paths, text="USE_ONLINE_FLEX_DIFF", variable=self.use_diff_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        prim = ttk.LabelFrame(panel, text="Primitive Detector Parameters", padding=10)
        prim.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        prim.columnconfigure(1, weight=1)
        self.add_config_row(prim, 0, "REST_LABEL", self.rest_label_var)
        self.add_config_row(prim, 1, "LABEL_STABLE_MS", self.label_stable_var)
        self.add_config_row(prim, 2, "ENTER_CONFIRM_MS", self.enter_confirm_var)
        self.add_config_row(prim, 3, "QUICK_TAP_MAX_MS", self.quick_tap_var)
        self.add_config_row(prim, 4, "LONG_HOLD_MIN_MS", self.long_hold_var)
        self.add_config_row(prim, 5, "DOUBLE_TAP_GAP_MS", self.double_tap_gap_var)

        pipe = ttk.LabelFrame(panel, text="Runtime Pipeline Parameters", padding=10)
        pipe.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        pipe.columnconfigure(1, weight=1)
        self.add_config_row(pipe, 0, "PIPELINE_NAME", self.pipeline_name_var)
        self.add_config_row(pipe, 1, "PIPELINE_SEQUENCE", self.pipeline_sequence_var)
        self.add_config_row(pipe, 2, "STEP_GAP_MS", self.pipeline_step_gap_var)
        self.add_config_row(pipe, 3, "COOLDOWN_MS", self.pipeline_cooldown_var)
        ttk.Checkbutton(pipe, text="PIPELINE_ENABLED", variable=self.pipeline_enabled_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        note = ttk.LabelFrame(panel, text="Runtime Notes / Log", padding=10)
        note.grid(row=4, column=0, sticky="nsew")
        note.columnconfigure(0, weight=1)
        note.rowconfigure(0, weight=1)
        self.note_text = tk.Text(note, height=18, wrap="word", font=("Consolas", 10))
        self.note_text.grid(row=0, column=0, sticky="nsew")
        s = ttk.Scrollbar(note, orient="vertical", command=self.note_text.yview)
        s.grid(row=0, column=1, sticky="ns")
        self.note_text.configure(yscrollcommand=s.set)

    def build_pipeline_manager_panel(self, parent):
        wrapper = ttk.Frame(parent, padding=8)
        wrapper.pack(fill="both", expand=True)

        self.make_tab_topbar(wrapper, self.reset_pipeline_page_to_default)

        panel = ttk.Frame(wrapper)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        topbar = ttk.Frame(panel)
        topbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(topbar, text="New Pipeline", command=self.new_pipeline).pack(side="left", padx=(0, 6))
        ttk.Button(topbar, text="Refresh Pipelines", command=self.refresh_pipeline_list).pack(side="left", padx=6)
        ttk.Button(topbar, text="Reload Labels", command=self.reload_gesture_names).pack(side="left", padx=6)
        ttk.Button(topbar, text="Load From Runtime Params", command=self.capture_runtime_params_to_library).pack(side="left", padx=6)

        upper = ttk.LabelFrame(panel, text="Existing Pipelines", padding=8)
        upper.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        upper.columnconfigure(0, weight=1)
        upper.rowconfigure(0, weight=1)

        self.pipeline_canvas = tk.Canvas(upper, bg="white", highlightthickness=1, highlightbackground="#d0d0d0")
        self.pipeline_canvas.grid(row=0, column=0, sticky="nsew")
        ttk.Scrollbar(upper, orient="vertical", command=self.pipeline_canvas.yview).grid(row=0, column=1, sticky="ns")
        self.pipeline_list_inner = tk.Frame(self.pipeline_canvas, bg="white")
        self.pipeline_window = self.pipeline_canvas.create_window((0, 0), window=self.pipeline_list_inner, anchor="nw")
        self.pipeline_list_inner.bind("<Configure>", lambda e: self.pipeline_canvas.configure(scrollregion=self.pipeline_canvas.bbox("all")))
        self.pipeline_canvas.bind("<Configure>", lambda e: self.pipeline_canvas.itemconfig(self.pipeline_window, width=e.width))

        lower = ttk.LabelFrame(panel, text="Notes", padding=8)
        lower.grid(row=2, column=0, sticky="ew")
        self.pipeline_note = tk.Text(lower, height=12, wrap="word", font=("Consolas", 10))
        self.pipeline_note.pack(fill="both", expand=True)
        self.pipeline_note.insert("1.0",
            "1) New Pipeline: open the visual builder.\n"
            "2) gesture blocks are loaded automatically from label_map.npy.\n"
            "3) runtime still loads all enabled pipelines in the pipeline library.\n"
            "4) when task target is pipeline, it is also selected from here.\n"
        )
        self.pipeline_note.config(state="disabled")
        self.refresh_pipeline_list()

    def build_task_config_panel(self, parent):
        wrapper = ttk.Frame(parent, padding=8)
        wrapper.pack(fill="both", expand=True)

        self.make_tab_topbar(wrapper, self.reset_task_page_to_default)

        panel = ttk.Frame(wrapper)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)

        form = ttk.LabelFrame(panel, text="Task Parameters", padding=10)
        form.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        form.columnconfigure(1, weight=1)
        self.add_config_row(form, 0, "TASK_NAME", self.task_name_var)

        ttk.Label(form, text="TARGET_TYPE", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        self.task_target_type_box = ttk.Combobox(form, textvariable=self.task_target_type_var, values=TASK_TARGET_TYPE_OPTIONS, state="readonly", font=("Consolas", 11))
        self.task_target_type_box.grid(row=1, column=1, sticky="ew", pady=6)
        self.task_target_type_box.bind("<<ComboboxSelected>>", lambda e: self.update_task_target_options())

        ttk.Label(form, text="TARGET_NAME", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", padx=(0, 10), pady=6)
        self.task_target_name_box = ttk.Combobox(form, textvariable=self.task_target_name_var, values=[], state="readonly", font=("Consolas", 11))
        self.task_target_name_box.grid(row=2, column=1, sticky="ew", pady=6)

        ttk.Label(form, text="LABEL_MODE", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="w", padx=(0, 10), pady=6)
        self.task_label_mode_box = ttk.Combobox(form, textvariable=self.task_label_mode_var, values=TASK_LABEL_MODE_OPTIONS, state="readonly", font=("Consolas", 11))
        self.task_label_mode_box.grid(row=3, column=1, sticky="ew", pady=6)

        self.add_config_row(form, 4, "DURATION_SEC", self.task_duration_var)
        self.add_config_row(form, 5, "REQUIRED_COUNT", self.task_required_count_var)
        ttk.Checkbutton(form, text="TASK_ENABLED", variable=self.task_enabled_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        btn = ttk.Frame(panel)
        btn.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(btn, text="Refresh Targets", command=self.update_task_target_options).pack(side="left")
        ttk.Button(btn, text="Use Current Pipeline", command=self.fill_task_with_current_runtime_pipeline).pack(side="left", padx=(8, 0))

        note = ttk.LabelFrame(panel, text="Task Notes", padding=8)
        note.grid(row=2, column=0, sticky="nsew")
        note.columnconfigure(0, weight=1)
        note.rowconfigure(0, weight=1)
        self.task_note_text = tk.Text(note, height=18, wrap="word", font=("Consolas", 10))
        self.task_note_text.grid(row=0, column=0, sticky="nsew")
        s = ttk.Scrollbar(note, orient="vertical", command=self.task_note_text.yview)
        s.grid(row=0, column=1, sticky="ns")
        self.task_note_text.configure(yscrollcommand=s.set)
        self.task_note_text.insert("1.0",
            "1) If TASK_ENABLED=True, task mode starts automatically after monitoring begins.\n"
            "2) The target can be a label or a pipeline.\n"
            "3) Label targets can use either / tap / long_hold.\n"
            "4) Logs show TASK START / PROGRESS / SUCCESS / FAIL / END.\n"
        )
        self.update_task_target_options()

    def build_ble_panel(self, parent):
        wrapper = ttk.Frame(parent, padding=8)
        wrapper.pack(fill="both", expand=True)

        self.make_tab_topbar(wrapper, self.reset_udp_page_to_default)

        panel = ttk.Frame(wrapper)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        cfg = ttk.LabelFrame(panel, text="Wi-Fi / UDP Parameters", padding=10)
        cfg.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        cfg.columnconfigure(1, weight=1)

        self.add_config_row(cfg, 0, "HOST_IP (Python bind)", self.host_ip_var)
        self.add_config_row(cfg, 1, "SENSOR_PORT", self.sensor_port_var)
        self.add_config_row(cfg, 2, "ARDUINO2_IP", self.arduino2_ip_var)
        self.add_config_row(cfg, 3, "CMD_PORT", self.cmd_port_var)
        ttk.Checkbutton(cfg, text="UDP_SEND_ENABLED", variable=self.udp_send_enabled_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        row_img = ttk.Frame(cfg)
        row_img.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        row_img.columnconfigure(1, weight=1)
        ttk.Label(row_img, text="GESTURE_LIBRARY_IMAGE_PATH", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(row_img, textvariable=self.gesture_library_image_path_var, font=("Consolas", 11)).grid(row=0, column=1, sticky="ew")

        mgr = ttk.LabelFrame(panel, text="UDP Signal Rules", padding=8)
        mgr.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        mgr.columnconfigure(0, weight=1)
        mgr.rowconfigure(1, weight=1)

        topbar = ttk.Frame(mgr)
        topbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(topbar, text="New Rule", command=self.new_ble_rule).pack(side="left", padx=(0, 6))
        ttk.Button(topbar, text="Refresh Rules", command=self.refresh_ble_rule_list).pack(side="left", padx=6)
        ttk.Button(topbar, text="Reload Labels", command=self.reload_gesture_names).pack(side="left", padx=6)
        ttk.Button(topbar, text="Use Arduino2 IP = 10.133.138.6", command=self.fill_arduino2_ip_default).pack(side="left", padx=6)

        self.ble_canvas = tk.Canvas(mgr, bg="white", highlightthickness=1, highlightbackground="#d0d0d0")
        self.ble_canvas.grid(row=1, column=0, sticky="nsew")
        ttk.Scrollbar(mgr, orient="vertical", command=self.ble_canvas.yview).grid(row=1, column=1, sticky="ns")
        self.ble_list_inner = tk.Frame(self.ble_canvas, bg="white")
        self.ble_window = self.ble_canvas.create_window((0, 0), window=self.ble_list_inner, anchor="nw")
        self.ble_list_inner.bind("<Configure>", lambda e: self.ble_canvas.configure(scrollregion=self.ble_canvas.bbox("all")))
        self.ble_canvas.bind("<Configure>", lambda e: self.ble_canvas.itemconfig(self.ble_window, width=e.width))

        note = ttk.LabelFrame(panel, text="Wi-Fi / UDP Notes", padding=8)
        note.grid(row=2, column=0, sticky="ew")
        self.ble_note_text = tk.Text(note, height=14, wrap="word", font=("Consolas", 10))
        self.ble_note_text.pack(fill="both", expand=True)
        self.ble_note_text.insert("1.0",
            "Wi-Fi / UDP communication logic:\n"
            "1) Arduino1 sends one UDP binary batch every 100 ms.\n"
            "2) Each batch contains 5 samples collected at 20 ms interval.\n"
            "3) Python receives one 98-byte packet and replays the 5 frames one by one.\n"
            "4) Original CNN / smoothing / primitive / pipeline / task logic still runs frame by frame.\n"
            "5) When an enabled rule is hit, Python sends signal0~signal5 to Arduino2 via UDP.\n"
            "6) Current Arduino2 IP can be set to 10.133.138.6.\n"
        )
        self.refresh_ble_rule_list()

    def build_mouse_panel(self, parent):
        wrapper = ttk.Frame(parent, padding=8)
        wrapper.pack(fill="both", expand=True)

        self.make_tab_topbar(wrapper, self.reset_mouse_page_to_default)

        panel = ttk.Frame(wrapper)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        cfg = ttk.LabelFrame(panel, text="Mouse Control Parameters", padding=10)
        cfg.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Checkbutton(cfg, text="MOUSE_CONTROL_ENABLED", variable=self.mouse_control_enabled_var).grid(row=0, column=0, sticky="w")
        tk.Label(cfg, text="When enabled for a run: pipeline/task/udp execution is gated off.", bg="#f7f7f7", fg="#555555",
                 anchor="w", justify="left", font=("Arial", 10)).grid(row=1, column=0, sticky="w", pady=(6, 0))

        mgr = ttk.LabelFrame(panel, text="Mouse Rules", padding=8)
        mgr.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        mgr.columnconfigure(0, weight=1)
        mgr.rowconfigure(1, weight=1)

        topbar = ttk.Frame(mgr)
        topbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(topbar, text="New Mouse Rule", command=self.new_mouse_rule).pack(side="left", padx=(0, 6))
        ttk.Button(topbar, text="Refresh Rules", command=self.refresh_mouse_rule_list).pack(side="left", padx=6)
        ttk.Button(topbar, text="Reload Labels", command=self.reload_gesture_names).pack(side="left", padx=6)

        self.mouse_canvas = tk.Canvas(mgr, bg="white", highlightthickness=1, highlightbackground="#d0d0d0")
        self.mouse_canvas.grid(row=1, column=0, sticky="nsew")
        ttk.Scrollbar(mgr, orient="vertical", command=self.mouse_canvas.yview).grid(row=1, column=1, sticky="ns")
        self.mouse_list_inner = tk.Frame(self.mouse_canvas, bg="white")
        self.mouse_window = self.mouse_canvas.create_window((0, 0), window=self.mouse_list_inner, anchor="nw")
        self.mouse_list_inner.bind("<Configure>", lambda e: self.mouse_canvas.configure(scrollregion=self.mouse_canvas.bbox("all")))
        self.mouse_canvas.bind("<Configure>", lambda e: self.mouse_canvas.itemconfig(self.mouse_window, width=e.width))

        note = ttk.LabelFrame(panel, text="Mouse Control Notes", padding=8)
        note.grid(row=2, column=0, sticky="ew")
        self.mouse_note_text = tk.Text(note, height=12, wrap="word", font=("Consolas", 10))
        self.mouse_note_text.pack(fill="both", expand=True)
        self.mouse_note_text.insert("1.0",
            "Mouse rule behavior:\n"
            "1) move_left/right/up/down: start on ENTER(gesture), stop on EXIT(gesture).\n"
            "2) recenter/click: one-shot action on completed gesture.\n"
            "3) either uses EXIT(label), tap uses PRIM QuickTap(label), long_hold uses PRIM LongHold(label).\n"
            "4) Enable MOUSE_CONTROL_ENABLED to gate off pipeline/task/udp for that run.\n"
        )
        self.refresh_mouse_rule_list()

    def populate_initial_content(self):
        if hasattr(self, "note_text") and self.note_text.winfo_exists():
            self.note_text.delete("1.0", "end")
            self.note_text.insert("1.0",
                "This version uses Wi-Fi / UDP communication:\n\n"
                "1) Arduino1 -> Python: UDP binary batch stream.\n"
                "2) Each batch contains 5 samples.\n"
                "3) Python replays the 5 samples one by one into the original pipeline.\n"
                "4) Python -> Arduino2: UDP signal command send.\n"
                "5) Primitive / Pipeline / Task / UDP rule detection all still run on replayed frames.\n"
                "6) PRINT_EVERY_SEC only throttles GUI display, not recognition logic.\n"
                "7) After parameter changes, stop and restart for them to take effect.\n"
                "8) If MOUSE_CONTROL_ENABLED=True, pipeline/task/udp execution is gated off for that run.\n"
                "\n"
            )
            for line in self.log_buffer[-200:]:
                self.note_text.insert("end", line + "\n")
            self.note_text.see("end")

    def append_log(self, text):
        self.log_buffer.append(text)
        if len(self.log_buffer) > 500:
            self.log_buffer = self.log_buffer[-500:]

        if hasattr(self, "note_text") and self.note_text.winfo_exists():
            self.note_text.insert("end", text + "\n")
            self.note_text.see("end")
            line_count = int(self.note_text.index("end-1c").split(".")[0])
            if line_count > 350:
                self.note_text.delete("1.0", "30.0")

    def append_event(self, evt_type, detail, ts):
        self.event_tree.insert("", "end", values=(ts, evt_type, detail))
        items = self.event_tree.get_children()
        if len(items) > 350:
            self.event_tree.delete(items[0])

    def clear_logs(self):
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        self.log_buffer.clear()
        if hasattr(self, "note_text") and self.note_text.winfo_exists():
            self.note_text.delete("1.0", "end")
            self.populate_initial_content()
        if hasattr(self, "task_note_text") and self.task_note_text.winfo_exists():
            self.task_note_text.delete("1.0", "end")
            self.task_note_text.insert("1.0", "Task note cleared.\n")
        if hasattr(self, "ble_note_text") and self.ble_note_text.winfo_exists():
            self.ble_note_text.delete("1.0", "end")
            self.ble_note_text.insert("1.0", "UDP note cleared.\n")
        if hasattr(self, "mouse_note_text") and self.mouse_note_text.winfo_exists():
            self.mouse_note_text.delete("1.0", "end")
            self.mouse_note_text.insert("1.0", "Mouse note cleared.\n")
        if self.mouse_control_enabled_var.get():
            self.current_task_var.set("Task Disabled (Mouse Control Mode)")
        else:
            self.current_task_var.set("Task Disabled" if not self.task_enabled_var.get() else "Task Pending")

    def show_current_config(self):
        summary = (
            f"MODEL_PATH = {self.model_path_var.get()}\n"
            f"LABEL_PATH = {self.label_path_var.get()}\n"
            f"NORM_PATH = {self.norm_path_var.get()}\n"
            f"WINDOW_LEN = {self.window_len_var.get()}\n"
            f"INFER_EVERY_N_SAMPLES = {self.infer_every_var.get()}\n"
            f"SMOOTH_VOTE_N = {self.smooth_vote_var.get()}\n"
            f"PRINT_EVERY_SEC = {self.print_every_var.get()}\n"
            f"USE_ONLINE_FLEX_DIFF = {self.use_diff_var.get()}\n"
            f"REST_LABEL = {self.rest_label_var.get()}\n"
            f"LABEL_STABLE_MS = {self.label_stable_var.get()}\n"
            f"ENTER_CONFIRM_MS = {self.enter_confirm_var.get()}\n"
            f"QUICK_TAP_MAX_MS = {self.quick_tap_var.get()}\n"
            f"LONG_HOLD_MIN_MS = {self.long_hold_var.get()}\n"
            f"DOUBLE_TAP_GAP_MS = {self.double_tap_gap_var.get()}\n"
            f"PIPELINE_ENABLED = {self.pipeline_enabled_var.get()}\n"
            f"PIPELINE_NAME = {self.pipeline_name_var.get()}\n"
            f"PIPELINE_SEQUENCE = {self.pipeline_sequence_var.get()}\n"
            f"PIPELINE_STEP_GAP_MS = {self.pipeline_step_gap_var.get()}\n"
            f"PIPELINE_COOLDOWN_MS = {self.pipeline_cooldown_var.get()}\n"
            f"TASK_ENABLED = {self.task_enabled_var.get()}\n"
            f"TASK_NAME = {self.task_name_var.get()}\n"
            f"TASK_TARGET_TYPE = {self.task_target_type_var.get()}\n"
            f"TASK_TARGET_NAME = {self.task_target_name_var.get()}\n"
            f"TASK_LABEL_MODE = {self.task_label_mode_var.get()}\n"
            f"TASK_DURATION_SEC = {self.task_duration_var.get()}\n"
            f"TASK_REQUIRED_COUNT = {self.task_required_count_var.get()}\n"
            f"HOST_IP = {self.host_ip_var.get()}\n"
            f"SENSOR_PORT = {self.sensor_port_var.get()}\n"
            f"ARDUINO2_IP = {self.arduino2_ip_var.get()}\n"
            f"CMD_PORT = {self.cmd_port_var.get()}\n"
            f"UDP_SEND_ENABLED = {self.udp_send_enabled_var.get()}\n"
            f"MOUSE_CONTROL_ENABLED = {self.mouse_control_enabled_var.get()}\n"
            f"GESTURE_LIBRARY_IMAGE_PATH = {self.gesture_library_image_path_var.get()}\n"
            f"gesture_names = {self.gesture_names}\n"
            f"pipeline_library = {[p['name'] for p in self.pipeline_library]}\n"
            f"udp_rule_library = {[r['rule_name'] for r in self.ble_rule_library]}\n"
            f"mouse_rule_library = {[r['rule_name'] for r in self.mouse_rule_library]}\n"
            f"Binary batch format = header '<HHI', frame '<I5Hhh', batch_bytes={BATCH_SIZE}\n"
            f"Arduino2 default IP = 10.133.138.6\n"
            f"Note: parameter changes only take effect after stopping and restarting.\n"
        )
        self.append_log(summary)
        self.open_settings_window()

    def reset_parameters_page_to_default(self):
        if not messagebox.askyesno("Confirm", "Restore Parameters page to defaults?"):
            return
        self._state_loading = True
        try:
            self.window_len_var.set(str(DEFAULT_WINDOW_LEN))
            self.infer_every_var.set(str(DEFAULT_INFER_EVERY_N_SAMPLES))
            self.smooth_vote_var.set(str(DEFAULT_SMOOTH_VOTE_N))
            self.print_every_var.set(str(DEFAULT_PRINT_EVERY_SEC))
            self.use_diff_var.set(DEFAULT_USE_ONLINE_FLEX_DIFF)

            self.model_path_var.set(DEFAULT_MODEL_PATH)
            self.label_path_var.set(DEFAULT_LABEL_PATH)
            self.norm_path_var.set(DEFAULT_NORM_PATH)

            self.rest_label_var.set(DEFAULT_REST_LABEL)
            self.label_stable_var.set(str(DEFAULT_LABEL_STABLE_MS))
            self.enter_confirm_var.set(str(DEFAULT_ENTER_CONFIRM_MS))
            self.quick_tap_var.set(str(DEFAULT_QUICK_TAP_MAX_MS))
            self.long_hold_var.set(str(DEFAULT_LONG_HOLD_MIN_MS))
            self.double_tap_gap_var.set(str(DEFAULT_DOUBLE_TAP_GAP_MS))

            self.pipeline_name_var.set(DEFAULT_PIPELINE_NAME)
            self.pipeline_sequence_var.set(DEFAULT_PIPELINE_SEQUENCE)
            self.pipeline_step_gap_var.set(str(DEFAULT_PIPELINE_STEP_GAP_MS))
            self.pipeline_cooldown_var.set(str(DEFAULT_PIPELINE_COOLDOWN_MS))
            self.pipeline_enabled_var.set(DEFAULT_PIPELINE_ENABLED)
        finally:
            self._state_loading = False
        self.save_state()
        self.append_log("[Defaults] Parameters page restored to defaults.")

    def reset_pipeline_page_to_default(self):
        if not messagebox.askyesno("Confirm", "Restore Pipeline page to defaults?"):
            return
        self._state_loading = True
        try:
            self.pipeline_library = self.get_default_pipeline_library()
            self.pipeline_name_var.set(DEFAULT_PIPELINE_NAME)
            self.pipeline_sequence_var.set(DEFAULT_PIPELINE_SEQUENCE)
            self.pipeline_step_gap_var.set(str(DEFAULT_PIPELINE_STEP_GAP_MS))
            self.pipeline_cooldown_var.set(str(DEFAULT_PIPELINE_COOLDOWN_MS))
            self.pipeline_enabled_var.set(DEFAULT_PIPELINE_ENABLED)
        finally:
            self._state_loading = False
        self.refresh_pipeline_list()
        self.update_task_target_options()
        self.save_state()
        self.append_log("[Defaults] Pipeline page restored to defaults.")

    def reset_task_page_to_default(self):
        if not messagebox.askyesno("Confirm", "Restore Task page to defaults?"):
            return
        self._state_loading = True
        try:
            self.task_enabled_var.set(DEFAULT_TASK_ENABLED)
            self.task_name_var.set(DEFAULT_TASK_NAME)
            self.task_target_type_var.set(DEFAULT_TASK_TARGET_TYPE)
            self.task_target_name_var.set(DEFAULT_TASK_TARGET_NAME)
            self.task_label_mode_var.set(DEFAULT_TASK_LABEL_MODE)
            self.task_duration_var.set(str(DEFAULT_TASK_DURATION_SEC))
            self.task_required_count_var.set(str(DEFAULT_TASK_REQUIRED_COUNT))
        finally:
            self._state_loading = False
        self.update_task_target_options()
        self.save_state()
        self.append_log("[Defaults] Task page restored to defaults.")

    def reset_udp_page_to_default(self):
        if not messagebox.askyesno("Confirm", "Restore Wi-Fi / UDP page to defaults?"):
            return
        self._state_loading = True
        try:
            self.host_ip_var.set(DEFAULT_HOST_IP)
            self.sensor_port_var.set(str(DEFAULT_SENSOR_PORT))
            self.arduino2_ip_var.set(DEFAULT_ARDUINO2_IP)
            self.cmd_port_var.set(str(DEFAULT_CMD_PORT))
            self.udp_send_enabled_var.set(DEFAULT_UDP_SEND_ENABLED)
            self.gesture_library_image_path_var.set(DEFAULT_GESTURE_LIBRARY_IMAGE_PATH)
            self.ble_rule_library = self.get_default_ble_rule_library()
        finally:
            self._state_loading = False
        self.refresh_ble_rule_list()
        self.save_state()
        self.append_log("[Defaults] Wi-Fi / UDP page restored to defaults.")

    def reset_mouse_page_to_default(self):
        if not messagebox.askyesno("Confirm", "Restore Mouse Control page to defaults?"):
            return
        self._state_loading = True
        try:
            self.mouse_control_enabled_var.set(DEFAULT_MOUSE_CONTROL_ENABLED)
            self.mouse_rule_library = self.get_default_mouse_rule_library()
        finally:
            self._state_loading = False
        self.refresh_mouse_rule_list()
        self.save_state()
        self.append_log("[Defaults] Mouse Control page restored to defaults.")

    def get_runtime_pipeline_entry(self):
        name = self.pipeline_name_var.get().strip() or "pipeline_runtime"
        return {
            "name": name,
            "signal_name": f"sig_{name}",
            "sequence": [{"name": x, "type": "either"} for x in parse_pipeline_sequence_text(self.pipeline_sequence_var.get())],
            "step_gap_ms": int(self.pipeline_step_gap_var.get() or "1000"),
            "cooldown_ms": int(self.pipeline_cooldown_var.get() or "1200"),
            "enabled": bool(self.pipeline_enabled_var.get()),
        }

    def get_runtime_pipeline_library_snapshot(self):
        merged = {}
        for p in self.pipeline_library:
            q = normalize_pipeline_entry(copy.deepcopy(p))
            if q:
                merged[q["name"]] = q
        q = normalize_pipeline_entry(self.get_runtime_pipeline_entry())
        if q:
            merged[q["name"]] = q
        return list(merged.values())

    def new_pipeline(self):
        PipelineEditor(self.root, self.gesture_names, existing=None, on_ok=self.create_pipeline)

    def create_pipeline(self, data):
        names = [p["name"] for p in self.pipeline_library]
        if data["name"] in names and not messagebox.askyesno("Duplicate Name", f"Pipeline '{data['name']}' already exists. Overwrite it?"):
            return
        self.pipeline_library = [p for p in self.pipeline_library if p["name"] != data["name"]] + [data]
        self.refresh_pipeline_list()
        self.save_state()
        self.append_log(f"[PipelineManager] Created pipeline: {data['name']}")

    def edit_pipeline(self, data):
        PipelineEditor(self.root, self.gesture_names, existing=copy.deepcopy(data),
                       on_ok=lambda new_data, old_name=data["name"]: self.update_pipeline(old_name, new_data))

    def update_pipeline(self, old_name, new_data):
        updated = False
        for i, p in enumerate(self.pipeline_library):
            if p["name"] == old_name:
                self.pipeline_library[i] = new_data
                updated = True
                break
        if not updated:
            self.pipeline_library.append(new_data)
        self.refresh_pipeline_list()
        self.save_state()
        self.append_log(f"[PipelineManager] Updated pipeline: {new_data['name']}")

    def delete_pipeline(self, data):
        if not messagebox.askyesno("Delete Confirmation", f"Delete pipeline '{data['name']}'?"):
            return
        self.pipeline_library = [p for p in self.pipeline_library if p["name"] != data["name"]]
        self.refresh_pipeline_list()
        self.save_state()
        self.append_log(f"[PipelineManager] Deleted pipeline: {data['name']}")

    def apply_pipeline_to_runtime(self, data):
        seq_text = ",".join([x["name"] for x in data.get("sequence", [])])
        self.pipeline_name_var.set(data.get("name", "pipeline1"))
        self.pipeline_sequence_var.set(seq_text)
        self.pipeline_step_gap_var.set(str(data.get("step_gap_ms", 1000)))
        self.pipeline_cooldown_var.set(str(data.get("cooldown_ms", 1200)))
        self.pipeline_enabled_var.set(bool(data.get("enabled", True)))
        self.update_task_target_options()
        self.save_state()
        self.append_log(f"[PipelineManager] Applied to runtime params: name={data.get('name')} sequence={seq_text}")

    def capture_runtime_params_to_library(self, silent=False):
        data = self.get_runtime_pipeline_entry()
        if data["name"] in [p["name"] for p in self.pipeline_library] and not silent:
            if not messagebox.askyesno("Already Exists", f"Pipeline '{data['name']}' already exists. Overwrite it?"):
                return
        self.pipeline_library = [p for p in self.pipeline_library if p["name"] != data["name"]] + [data]
        self.refresh_pipeline_list()
        self.save_state()
        self.append_log(f"[PipelineManager] Captured runtime params into library: {data['name']}")

    def refresh_pipeline_list(self):
        if not hasattr(self, "pipeline_list_inner") or not self.pipeline_list_inner.winfo_exists():
            return
        for c in self.pipeline_list_inner.winfo_children():
            c.destroy()
        if not self.pipeline_library:
            tk.Label(self.pipeline_list_inner, text="No pipeline created yet.", bg="white", fg="#666666",
                     font=("Arial", 11, "italic"), pady=16).pack(fill="x")
        else:
            for p in self.pipeline_library:
                PipelineCard(self.pipeline_list_inner, p, self.edit_pipeline, self.delete_pipeline,
                             self.apply_pipeline_to_runtime).pack(fill="x", padx=8, pady=8)
        self.update_task_target_options()

    def fill_arduino2_ip_default(self):
        self.arduino2_ip_var.set("10.133.138.6")
        self.save_state()
        self.append_log("[UDP] ARDUINO2_IP set to 10.133.138.6")

    def new_ble_rule(self):
        BLERuleEditor(self.root, self.gesture_names, existing=None, on_ok=self.create_ble_rule)

    def create_ble_rule(self, data):
        name = data["rule_name"]
        if name in [r["rule_name"] for r in self.ble_rule_library]:
            if not messagebox.askyesno("Duplicate Name", f"UDP rule '{name}' already exists. Overwrite it?"):
                return
        self.ble_rule_library = [r for r in self.ble_rule_library if r["rule_name"] != name] + [data]
        self.refresh_ble_rule_list()
        self.save_state()
        self.append_log(f"[UDP] Created rule: {name}")

    def edit_ble_rule(self, data):
        BLERuleEditor(self.root, self.gesture_names, existing=copy.deepcopy(data),
                      on_ok=lambda new_data, old_name=data["rule_name"]: self.update_ble_rule(old_name, new_data))

    def update_ble_rule(self, old_name, new_data):
        updated = False
        for i, r in enumerate(self.ble_rule_library):
            if r["rule_name"] == old_name:
                self.ble_rule_library[i] = new_data
                updated = True
                break
        if not updated:
            self.ble_rule_library.append(new_data)
        self.refresh_ble_rule_list()
        self.save_state()
        self.append_log(f"[UDP] Updated rule: {new_data['rule_name']}")

    def delete_ble_rule(self, data):
        if not messagebox.askyesno("Delete Confirmation", f"Delete UDP rule '{data['rule_name']}'?"):
            return
        self.ble_rule_library = [r for r in self.ble_rule_library if r["rule_name"] != data["rule_name"]]
        self.refresh_ble_rule_list()
        self.save_state()
        self.append_log(f"[UDP] Deleted rule: {data['rule_name']}")

    def refresh_ble_rule_list(self):
        if not hasattr(self, "ble_list_inner") or not self.ble_list_inner.winfo_exists():
            return
        for c in self.ble_list_inner.winfo_children():
            c.destroy()
        if not self.ble_rule_library:
            tk.Label(self.ble_list_inner, text="No UDP signal rule yet.", bg="white", fg="#666666",
                     font=("Arial", 11, "italic"), pady=16).pack(fill="x")
        else:
            for r in self.ble_rule_library:
                BLERuleCard(self.ble_list_inner, r, self.edit_ble_rule, self.delete_ble_rule).pack(fill="x", padx=8, pady=8)

    def new_mouse_rule(self):
        MouseRuleEditor(self.root, self.gesture_names, existing=None, on_ok=self.create_mouse_rule)

    def create_mouse_rule(self, data):
        name = data["rule_name"]
        if name in [r["rule_name"] for r in self.mouse_rule_library]:
            if not messagebox.askyesno("Duplicate Name", f"Mouse rule '{name}' already exists. Overwrite it?"):
                return
        self.mouse_rule_library = [r for r in self.mouse_rule_library if r["rule_name"] != name] + [data]
        self.refresh_mouse_rule_list()
        self.save_state()
        self.append_log(f"[Mouse] Created rule: {name}")

    def edit_mouse_rule(self, data):
        MouseRuleEditor(self.root, self.gesture_names, existing=copy.deepcopy(data),
                        on_ok=lambda new_data, old_name=data["rule_name"]: self.update_mouse_rule(old_name, new_data))

    def update_mouse_rule(self, old_name, new_data):
        updated = False
        for i, r in enumerate(self.mouse_rule_library):
            if r["rule_name"] == old_name:
                self.mouse_rule_library[i] = new_data
                updated = True
                break
        if not updated:
            self.mouse_rule_library.append(new_data)
        self.refresh_mouse_rule_list()
        self.save_state()
        self.append_log(f"[Mouse] Updated rule: {new_data['rule_name']}")

    def delete_mouse_rule(self, data):
        if not messagebox.askyesno("Delete Confirmation", f"Delete mouse rule '{data['rule_name']}'?"):
            return
        self.mouse_rule_library = [r for r in self.mouse_rule_library if r["rule_name"] != data["rule_name"]]
        self.refresh_mouse_rule_list()
        self.save_state()
        self.append_log(f"[Mouse] Deleted rule: {data['rule_name']}")

    def refresh_mouse_rule_list(self):
        if not hasattr(self, "mouse_list_inner") or not self.mouse_list_inner.winfo_exists():
            return
        for c in self.mouse_list_inner.winfo_children():
            c.destroy()
        if not self.mouse_rule_library:
            tk.Label(self.mouse_list_inner, text="No mouse rule yet.", bg="white", fg="#666666",
                     font=("Arial", 11, "italic"), pady=16).pack(fill="x")
        else:
            for r in self.mouse_rule_library:
                MouseRuleCard(self.mouse_list_inner, r, self.edit_mouse_rule, self.delete_mouse_rule).pack(fill="x", padx=8, pady=8)

    def reload_gesture_names(self):
        self.gesture_names = self.load_gesture_names_from_label_map()
        self.update_task_target_options()
        self.save_state()
        self.append_log(f"[Labels] Reloaded: {self.gesture_names}")
        messagebox.showinfo("Reloaded", f"label_map.npy reloaded.\n\nCurrent labels:\n{self.gesture_names}")

    def get_available_pipeline_names_for_task(self):
        names = []
        for p in self.get_runtime_pipeline_library_snapshot():
            name = str(p.get("name", "")).strip()
            if name:
                names.append(name)
        return sorted(list(dict.fromkeys(names)))

    def update_task_target_options(self):
        if not hasattr(self, "task_target_name_box") or not self.task_target_name_box.winfo_exists():
            return
        target_type = self.task_target_type_var.get().strip() or "label"
        if target_type == "pipeline":
            options = self.get_available_pipeline_names_for_task()
            self.task_label_mode_box.configure(state="disabled")
        else:
            options = [str(x) for x in self.gesture_names]
            self.task_label_mode_box.configure(state="readonly")
        self.task_target_name_box["values"] = options
        cur = self.task_target_name_var.get().strip()
        self.task_target_name_var.set(cur if cur in options else (options[0] if options else ""))

    def fill_task_with_current_runtime_pipeline(self):
        runtime_name = self.pipeline_name_var.get().strip()
        if not runtime_name:
            return messagebox.showinfo("Hint", "There is no valid PIPELINE_NAME in the current runtime parameter area.")
        self.task_target_type_var.set("pipeline")
        self.capture_runtime_params_to_library(silent=True)
        self.update_task_target_options()
        if runtime_name in list(self.task_target_name_box["values"]):
            self.task_target_name_var.set(runtime_name)
            self.append_log(f"[TaskConfig] Set target to current runtime pipeline: {runtime_name}")

    def build_config_dict(self):
        target_type = self.task_target_type_var.get().strip()
        target_name = self.task_target_name_var.get().strip()
        mouse_mode = bool(self.mouse_control_enabled_var.get())

        if (not mouse_mode) and target_type == "pipeline":
            names = self.get_available_pipeline_names_for_task()
            if target_name and names and target_name not in names:
                raise ValueError(f"TASK_TARGET_NAME='{target_name}' is not in the pipeline library.")

        if (not mouse_mode) and target_type == "label":
            labels = [str(x) for x in self.gesture_names]
            if target_name and labels and target_name not in labels:
                raise ValueError(f"TASK_TARGET_NAME='{target_name}' is not in the label library.")
            if self.task_label_mode_var.get().strip() not in TASK_LABEL_MODE_OPTIONS:
                raise ValueError("TASK_LABEL_MODE must be either / tap / long_hold.")

        return {
            "MODEL_PATH": self.model_path_var.get().strip(),
            "LABEL_PATH": self.label_path_var.get().strip(),
            "NORM_PATH": self.norm_path_var.get().strip(),
            "USE_ONLINE_FLEX_DIFF": bool(self.use_diff_var.get()),
            "WINDOW_LEN": int(self.window_len_var.get()),
            "INFER_EVERY_N_SAMPLES": int(self.infer_every_var.get()),
            "SMOOTH_VOTE_N": int(self.smooth_vote_var.get()),
            "PRINT_EVERY_SEC": float(self.print_every_var.get()),
            "REST_LABEL": self.rest_label_var.get().strip(),
            "LABEL_STABLE_MS": int(self.label_stable_var.get()),
            "ENTER_CONFIRM_MS": int(self.enter_confirm_var.get()),
            "QUICK_TAP_MAX_MS": int(self.quick_tap_var.get()),
            "LONG_HOLD_MIN_MS": int(self.long_hold_var.get()),
            "DOUBLE_TAP_GAP_MS": int(self.double_tap_gap_var.get()),
            "PIPELINE_ENABLED": bool(self.pipeline_enabled_var.get()),
            "PIPELINE_NAME": self.pipeline_name_var.get().strip(),
            "PIPELINE_SEQUENCE": parse_pipeline_sequence_text(self.pipeline_sequence_var.get()),
            "PIPELINE_STEP_GAP_MS": int(self.pipeline_step_gap_var.get()),
            "PIPELINE_COOLDOWN_MS": int(self.pipeline_cooldown_var.get()),
            "PIPELINE_LIBRARY": self.get_runtime_pipeline_library_snapshot(),
            "TASK_ENABLED": bool(self.task_enabled_var.get()),
            "TASK_NAME": self.task_name_var.get().strip(),
            "TASK_TARGET_TYPE": target_type,
            "TASK_TARGET_NAME": target_name,
            "TASK_LABEL_MODE": self.task_label_mode_var.get().strip(),
            "TASK_DURATION_SEC": float(self.task_duration_var.get()),
            "TASK_REQUIRED_COUNT": int(self.task_required_count_var.get()),
            "HOST_IP": self.host_ip_var.get().strip(),
            "SENSOR_PORT": int(self.sensor_port_var.get()),
            "ARDUINO2_IP": self.arduino2_ip_var.get().strip(),
            "CMD_PORT": int(self.cmd_port_var.get()),
            "UDP_SEND_ENABLED": bool(self.udp_send_enabled_var.get()),
            "BLE_RULE_LIBRARY": copy.deepcopy(self.ble_rule_library),
            "MOUSE_CONTROL_ENABLED": bool(self.mouse_control_enabled_var.get()),
            "MOUSE_RULE_LIBRARY": copy.deepcopy(self.mouse_rule_library),
        }

    def start_monitor(self):
        if self.is_running:
            return messagebox.showinfo("Hint", "Monitoring is already running.")
        try:
            config = self.build_config_dict()
            if (not config["MOUSE_CONTROL_ENABLED"]) and config["PIPELINE_ENABLED"] and len(config["PIPELINE_SEQUENCE"]) == 0:
                raise ValueError("PIPELINE_SEQUENCE cannot be empty.")
            if (not config["MOUSE_CONTROL_ENABLED"]) and config["TASK_ENABLED"]:
                if config["TASK_TARGET_TYPE"] not in TASK_TARGET_TYPE_OPTIONS:
                    raise ValueError("TASK_TARGET_TYPE must be label or pipeline.")
                if not config["TASK_TARGET_NAME"]:
                    raise ValueError("TASK_TARGET_NAME cannot be empty.")
                if config["TASK_DURATION_SEC"] <= 0:
                    raise ValueError("TASK_DURATION_SEC must be > 0.")
                if config["TASK_REQUIRED_COUNT"] <= 0:
                    raise ValueError("TASK_REQUIRED_COUNT must be > 0.")
            if not config["HOST_IP"]:
                raise ValueError("HOST_IP cannot be empty.")
            if config["SENSOR_PORT"] <= 0:
                raise ValueError("SENSOR_PORT must be > 0.")
            if config["CMD_PORT"] <= 0:
                raise ValueError("CMD_PORT must be > 0.")
        except Exception as e:
            return messagebox.showerror("Parameter Error", f"Failed to parse parameters:\n{e}")

        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=run_realtime_worker, args=(config, self.gui_queue, self.stop_event), daemon=True)
        self.worker_thread.start()
        self.is_running = True
        self.status_var.set("Starting...")
        if self.mouse_control_enabled_var.get():
            self.current_mode_var.set("Mouse Control Mode")
            self.current_task_var.set("Task Disabled (Mouse Control Mode)")
            self.append_log("Mouse Control Mode Enabled: pipeline/task/udp disabled for this run")
        else:
            self.current_mode_var.set("Connecting")
            self.current_task_var.set("Task Pending" if self.task_enabled_var.get() else "Task Disabled")
        self.append_log("Starting worker thread...")
        self.append_log("Note: current parameters are loaded only once for this run. Changes during runtime do not apply immediately.")

    def stop_monitor(self):
        if not self.is_running:
            return
        self.append_log("Stopping worker thread...")
        self.status_var.set("Stopping...")
        self.current_mode_var.set("Stopping")
        if self.stop_event is not None:
            self.stop_event.set()
        self.is_running = False

    def process_gui_queue(self):
        latest_data = None

        try:
            while True:
                kind, payload = self.gui_queue.get_nowait()

                if kind == "data":
                    latest_data = payload
                    continue

                if kind == "log":
                    self.append_log(str(payload))

                elif kind == "status":
                    self.status_var.set(str(payload))
                    if payload == "Running":
                        self.current_mode_var.set("Mouse Control Mode" if self.mouse_control_enabled_var.get() else "Running")
                    elif payload == "Stopped":
                        self.current_mode_var.set("Idle")
                        self.is_running = False
                    elif payload == "Error":
                        self.current_mode_var.set("Error")
                        self.is_running = False
                    elif payload == "Connecting":
                        self.current_mode_var.set("Connecting")

                elif kind == "task_state":
                    self.current_task_var.set(str(payload))

                elif kind == "warmup":
                    t_ms, rc, wl, x7 = payload["t_ms"], payload["ring_count"], payload["window_len"], payload["x7"]
                    self.current_mode_var.set("Warmup")
                    self.raw_line_label.config(
                        text=f"[{t_ms:>6} ms] WARMUP ({rc}/{wl}) pred=WARMUP vote=WARMUP "
                             f"a0={x7[0]:.0f} a1={x7[1]:.0f} a2={x7[2]:.0f} a3={x7[3]:.0f} a4={x7[4]:.0f} "
                             f"roll={x7[5]:.2f} pitch={x7[6]:.2f}"
                    )

                elif kind == "event":
                    evt_type = str(payload.get("type", "EVENT"))
                    detail = str(payload.get("detail", ""))
                    ts = str(payload.get("ts", ""))

                    if evt_type == "TASK_START":
                        self.current_mode_var.set("Task Mode")
                    elif evt_type == "TASK_SUCCESS":
                        self.current_mode_var.set("Task Success")
                    elif evt_type == "TASK_FAIL":
                        self.current_mode_var.set("Task Failed")

                    self.append_event(evt_type, detail, ts)

        except queue.Empty:
            pass

        if latest_data is not None:
            payload = latest_data
            self.raw_line_label.config(text=payload["raw_line"])
            self.current_task_var.set(payload.get("task_state", self.current_task_var.get()))

        self.root.after(50, self.process_gui_queue)

    def on_root_close(self):
        try:
            self.save_state()
        except Exception:
            pass
        try:
            if self.stop_event is not None:
                self.stop_event.set()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = GestureMonitorApp(root)

    if not app.arduino2_ip_var.get().strip():
        app.arduino2_ip_var.set("10.133.138.6")

    root.mainloop()
