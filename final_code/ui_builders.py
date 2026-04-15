import tkinter as tk
from tkinter import ttk, messagebox
from app_defaults import GESTURE_TYPE_OPTIONS, SIGNAL_OPTIONS, MOUSE_ACTION_OPTIONS, DEFAULT_MOUSE_ACTION, DEFAULT_MOUSE_MOVE_SPEED
from runtime_rules import normalize_mouse_rule_entry
from runtime_transport_udp import signal_name_to_digit

class GesturePaletteBlock(tk.Frame):
    def __init__(self, master, gesture_name, default_type="either", drag_callback=None, **kwargs):
        super().__init__(master, bd=1, relief="solid", bg="#d9ecff", **kwargs)
        self.gesture_name = str(gesture_name)
        self.drag_callback = drag_callback
        self.drag_win = None
        self.type_var = tk.StringVar(value=default_type if default_type in GESTURE_TYPE_OPTIONS else "either")

        self.name_label = tk.Label(
            self, text=self.gesture_name, bg="#d9ecff", fg="#111111",
            font=("Arial", 10, "bold"), cursor="hand2", padx=10, pady=4
        )
        self.name_label.pack(fill="x")

        self.type_box = ttk.Combobox(self, textvariable=self.type_var, values=GESTURE_TYPE_OPTIONS, state="readonly", width=10)
        self.type_box.pack(fill="x", padx=6, pady=(0, 6))

        for ev in ("<Button-1>", "<B1-Motion>", "<ButtonRelease-1>"):
            getattr(self.name_label, "bind")(ev, getattr(self, {"<Button-1>": "on_press", "<B1-Motion>": "on_motion", "<ButtonRelease-1>": "on_release"}[ev]))

    def get_block_data(self):
        return {"name": self.gesture_name, "type": self.type_var.get().strip() or "either"}

    def on_press(self, event):
        if self.drag_callback:
            self.drag_callback("start", self.get_block_data(), event)
        self.drag_win = tk.Toplevel(self)
        self.drag_win.overrideredirect(True)
        self.drag_win.attributes("-topmost", True)
        tk.Label(
            self.drag_win,
            text=f"{self.gesture_name} [{self.type_var.get()}]",
            bg="#b7dbff", fg="black", bd=1, relief="solid", padx=14, pady=8,
            font=("Arial", 10, "bold"),
        ).pack()
        self.drag_win.geometry(f"+{self.winfo_pointerx()+8}+{self.winfo_pointery()+8}")

    def on_motion(self, event):
        if self.drag_win:
            self.drag_win.geometry(f"+{self.winfo_pointerx()+8}+{self.winfo_pointery()+8}")

    def on_release(self, event):
        if self.drag_callback:
            self.drag_callback("drop", self.get_block_data(), event)
        if self.drag_win:
            self.drag_win.destroy()
            self.drag_win = None


class SequenceSlot(tk.Frame):
    def __init__(self, master, index, remove_callback=None, placeholder_text="Drop Gesture Here", **kwargs):
        super().__init__(master, bd=0, bg="white", **kwargs)
        self.index = index
        self.remove_callback = remove_callback
        self.value = None
        self.placeholder_text = placeholder_text

        self.placeholder = tk.Label(
            self, text=self.placeholder_text, bg="#f0f0f0", fg="#9a9a9a",
            bd=1, relief="solid", width=18, height=4, font=("Arial", 10, "italic")
        )
        self.placeholder.pack(fill="both", expand=True)
        self.value_frame = tk.Frame(self, bg="white")

    def is_inside_screen_xy(self, x_root, y_root):
        x1, y1 = self.winfo_rootx(), self.winfo_rooty()
        x2, y2 = x1 + self.winfo_width(), y1 + self.winfo_height()
        return x1 <= x_root <= x2 and y1 <= y_root <= y2

    def set_value(self, block_data):
        self.value = dict(block_data)
        self.placeholder.pack_forget()
        for c in self.value_frame.winfo_children():
            c.destroy()

        card = tk.Frame(self.value_frame, bg="#cce8cc", bd=1, relief="solid")
        card.pack(fill="both", expand=True)
        txt = f"{self.value['name']} [{self.value['type']}]"
        tk.Label(card, text=txt, bg="#cce8cc", fg="#111111", font=("Arial", 10, "bold"),
                 justify="center", padx=18, pady=16).pack(side="left", fill="both", expand=True)
        tk.Button(card, text="×", width=2, bg="#f7d6d6", relief="flat", cursor="hand2",
                  command=self.clear_value).pack(side="right", padx=4, pady=4)
        self.value_frame.pack(fill="both", expand=True)

    def clear_value(self):
        self.value = None
        self.value_frame.pack_forget()
        self.placeholder.pack(fill="both", expand=True)
        if self.remove_callback:
            self.remove_callback(self.index)

class PipelineEditor(tk.Toplevel):
    def __init__(self, master, gesture_names, existing=None, on_ok=None):
        super().__init__(master)
        self.title("Pipeline Builder")
        self.geometry("1220x760")
        self.minsize(1120, 720)
        self.configure(bg="#f7f7f7")
        self.gesture_names, self.existing, self.on_ok = gesture_names[:], existing, on_ok
        self.sequence_slots, self.sequence_values = [], []

        self.transient(master)
        self.grab_set()
        self._build_ui()
        if existing:
            self._load_existing(existing)
        else:
            self._ensure_trailing_empty_slot()
        self._refresh_preview()

    def _build_ui(self):
        root = tk.Frame(self, bg="#f7f7f7")
        root.pack(fill="both", expand=True, padx=16, pady=16)
        tk.Label(root, text="Pipeline Setup / Builder", bg="#f7f7f7", fg="#111111",
                 font=("Arial", 16, "bold"), anchor="w").pack(fill="x", pady=(0, 12))

        content = tk.Frame(root, bg="#f7f7f7")
        content.pack(fill="both", expand=True)
        left = tk.Frame(content, bg="#f7f7f7")
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right = tk.Frame(content, bg="#f7f7f7", width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        setup_frame = ttk.LabelFrame(left, text="Setup / Available Gestures", padding=10)
        setup_frame.pack(fill="x", pady=(0, 12))
        palette_frame = tk.Frame(setup_frame, bg="#f7f7f7")
        palette_frame.pack(fill="x")
        for i, g in enumerate(self.gesture_names):
            GesturePaletteBlock(palette_frame, g, "either", self._handle_drag_event).grid(row=i // 3, column=i % 3, padx=8, pady=8, sticky="nw")

        builder_frame = ttk.LabelFrame(left, text="Pipeline Sequence Builder", padding=10)
        builder_frame.pack(fill="both", expand=True)
        tk.Label(builder_frame, text="Drag a gesture block into the placeholder below. After inserting one block, the next arrow and empty slot will be created automatically.",
                 bg="#f7f7f7", fg="#555555", anchor="w", justify="left", font=("Arial", 10)).pack(fill="x", pady=(0, 10))

        self.canvas = tk.Canvas(builder_frame, bg="white", highlightthickness=1, highlightbackground="#d0d0d0")
        self.canvas.pack(side="left", fill="both", expand=True)
        ttk.Scrollbar(builder_frame, orient="vertical", command=self.canvas.yview).pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=lambda *args: None)
        self.seq_inner = tk.Frame(self.canvas, bg="white")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.seq_inner, anchor="nw")
        self.seq_inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

        param_frame = ttk.LabelFrame(right, text="Pipeline Parameters", padding=10)
        param_frame.pack(fill="x")
        self.var_name = tk.StringVar(value="pipeline_new")
        self.var_signal = tk.StringVar(value="sig_pipeline_new")
        self.var_step_gap = tk.StringVar(value="1000")
        self.var_cooldown = tk.StringVar(value="1200")
        self.var_enabled = tk.BooleanVar(value=True)
        self._add_form_row(param_frame, "PIPELINE_NAME", self.var_name)
        self._add_form_row(param_frame, "SIGNAL_NAME", self.var_signal)
        self._add_form_row(param_frame, "STEP_GAP_MS", self.var_step_gap)
        self._add_form_row(param_frame, "COOLDOWN_MS", self.var_cooldown)
        ttk.Checkbutton(param_frame, text="PIPELINE_ENABLED", variable=self.var_enabled).pack(anchor="w", pady=(8, 0))

        preview_frame = ttk.LabelFrame(right, text="Preview", padding=10)
        preview_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.preview_text = tk.Text(preview_frame, height=18, wrap="word", font=("Consolas", 10))
        self.preview_text.pack(fill="both", expand=True)

        btn_frame = tk.Frame(right, bg="#f7f7f7")
        btn_frame.pack(fill="x", pady=(12, 0))
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(btn_frame, text="OK", command=self._on_ok).pack(side="right")

        for v in [self.var_name, self.var_signal, self.var_step_gap, self.var_cooldown]:
            v.trace_add("write", lambda *args: self._refresh_preview())
        self.var_enabled.trace_add("write", lambda *args: self._refresh_preview())

    def _add_form_row(self, parent, label_text, var):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label_text, width=16).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

    def _load_existing(self, data):
        self.var_name.set(data.get("name", "pipeline_new"))
        self.var_signal.set(data.get("signal_name", "sig_pipeline_new"))
        self.var_step_gap.set(str(data.get("step_gap_ms", 1000)))
        self.var_cooldown.set(str(data.get("cooldown_ms", 1200)))
        self.var_enabled.set(bool(data.get("enabled", True)))
        for block in data.get("sequence", []):
            self._append_slot_with_value(block)
        self._ensure_trailing_empty_slot()

    def _handle_drag_event(self, kind, block_data, event):
        if kind != "drop":
            return
        x_root, y_root = self.winfo_pointerx(), self.winfo_pointery()
        for i, slot in enumerate(self.sequence_slots):
            if slot.is_inside_screen_xy(x_root, y_root):
                slot.set_value(block_data)
                self.sequence_values[i] = dict(block_data)
                self._ensure_trailing_empty_slot()
                self._trim_extra_empty_slots()
                self._refresh_preview()
                break

    def _append_slot_with_value(self, block_data=None):
        idx = len(self.sequence_slots)
        row = tk.Frame(self.seq_inner, bg="white")
        row.pack(anchor="w", pady=16, padx=18)
        slot = SequenceSlot(row, idx, self._on_slot_remove)
        slot.pack(side="left")
        if block_data is not None:
            slot.set_value(block_data)
        self.sequence_slots.append(slot)
        self.sequence_values.append(block_data)
        arrow = tk.Label(row, text="→", bg="white", fg="#888888", font=("Arial", 22, "bold"), padx=16)
        arrow.pack(side="left")
        row.arrow_widget = arrow
        self._update_arrow_visibility()

    def _ensure_trailing_empty_slot(self):
        if not self.sequence_values or self.sequence_values[-1] is not None:
            self._append_slot_with_value(None)

    def _trim_extra_empty_slots(self):
        while len(self.sequence_values) >= 2 and self.sequence_values[-1] is None and self.sequence_values[-2] is None:
            self.sequence_slots.pop().master.destroy()
            self.sequence_values.pop()
        self._update_arrow_visibility()

    def _on_slot_remove(self, index):
        if 0 <= index < len(self.sequence_values):
            self.sequence_values[index] = None
            vals = [v for v in self.sequence_values if v is not None]
            for slot in self.sequence_slots[:]:
                slot.master.destroy()
            self.sequence_slots.clear()
            self.sequence_values.clear()
            for v in vals:
                self._append_slot_with_value(v)
            self._ensure_trailing_empty_slot()
            self._trim_extra_empty_slots()
            self._refresh_preview()

    def _update_arrow_visibility(self):
        for i, slot in enumerate(self.sequence_slots):
            row = slot.master
            if i == len(self.sequence_slots) - 1:
                row.arrow_widget.pack_forget()
            elif not row.arrow_widget.winfo_ismapped():
                row.arrow_widget.pack(side="left")

    def _refresh_preview(self):
        seq = [v for v in self.sequence_values if v is not None]
        lines = [
            f"PIPELINE_NAME: {self.var_name.get()}",
            f"SIGNAL_NAME: {self.var_signal.get()}",
            f"STEP_GAP_MS: {self.var_step_gap.get()}",
            f"COOLDOWN_MS: {self.var_cooldown.get()}",
            f"ENABLED: {self.var_enabled.get()}",
            "",
            "SEQUENCE:",
            (" " + " -> ".join([f"{x['name']}[{x['type']}]" for x in seq])) if seq else " [empty]"
        ]
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", "\n".join(lines))

    def _collect_pipeline_data(self):
        return {
            "name": self.var_name.get().strip(),
            "signal_name": self.var_signal.get().strip(),
            "sequence": [dict(v) for v in self.sequence_values if v is not None],
            "step_gap_ms": int(self.var_step_gap.get().strip() or "0"),
            "cooldown_ms": int(self.var_cooldown.get().strip() or "0"),
            "enabled": bool(self.var_enabled.get()),
        }

    def _on_ok(self):
        try:
            data = self._collect_pipeline_data()
        except ValueError:
            messagebox.showerror("Error", "STEP_GAP_MS / COOLDOWN_MS must be integers.")
            return
        if not data["name"]:
            return messagebox.showerror("Error", "PIPELINE_NAME cannot be empty.")
        if not data["signal_name"]:
            return messagebox.showerror("Error", "SIGNAL_NAME cannot be empty.")
        if not data["sequence"]:
            return messagebox.showerror("Error", "Sequence cannot be empty.")
        if self.on_ok:
            self.on_ok(data)
        self.destroy()


class PipelineCard(tk.Frame):
    def __init__(self, master, data, edit_callback=None, delete_callback=None, apply_callback=None, **kwargs):
        super().__init__(master, bd=1, relief="solid", bg="white", **kwargs)
        self.data = data
        self.edit_callback = edit_callback
        self.delete_callback = delete_callback
        self.apply_callback = apply_callback
        self.expanded = False

        header = tk.Frame(self, bg="white")
        header.pack(fill="x", padx=8, pady=8)
        tk.Label(header, text=f"{data['name']} [{'Enabled' if data.get('enabled') else 'Disabled'}]",
                 bg="white", fg="#111111", font=("Arial", 11, "bold"), anchor="w").pack(side="left", fill="x", expand=True)
        ttk.Button(header, text="Apply", width=8, command=lambda: self.apply_callback and self.apply_callback(self.data)).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Delete", width=8, command=lambda: self.delete_callback and self.delete_callback(self.data)).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Edit", width=8, command=lambda: self.edit_callback and self.edit_callback(self.data)).pack(side="right", padx=(6, 0))
        self.toggle_btn = ttk.Button(header, text="Expand", width=8, command=self.toggle)
        self.toggle_btn.pack(side="right", padx=(6, 0))
        self.body = tk.Frame(self, bg="#fafafa")

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            for c in self.body.winfo_children():
                c.destroy()
            seq_text = " -> ".join([f"{x['name']}[{x.get('type', 'either')}]" for x in self.data.get("sequence", [])])
            for r in [
                f"SIGNAL_NAME: {self.data.get('signal_name', '')}",
                f"STEP_GAP_MS: {self.data.get('step_gap_ms', '')}",
                f"COOLDOWN_MS: {self.data.get('cooldown_ms', '')}",
                f"SEQUENCE: {seq_text}",
            ]:
                tk.Label(self.body, text=r, bg="#fafafa", fg="#333333", anchor="w",
                         justify="left", font=("Consolas", 10)).pack(fill="x", padx=8, pady=3)
            self.body.pack(fill="x", padx=8, pady=(0, 8))
            self.toggle_btn.config(text="Collapse")
        else:
            self.body.pack_forget()
            self.toggle_btn.config(text="Expand")

class BLERuleEditor(tk.Toplevel):
    def __init__(self, master, gesture_names, existing=None, on_ok=None):
        super().__init__(master)
        self.title("UDP Signal Rule Builder")
        self.geometry("1120x720")
        self.minsize(980, 660)
        self.configure(bg="#f7f7f7")
        self.gesture_names, self.existing, self.on_ok = gesture_names[:], existing, on_ok
        self.slot_value = None

        self.transient(master)
        self.grab_set()
        self._build_ui()
        if existing:
            self._load_existing(existing)
        self._refresh_preview()

    def _build_ui(self):
        root = tk.Frame(self, bg="#f7f7f7")
        root.pack(fill="both", expand=True, padx=16, pady=16)
        tk.Label(root, text="UDP Signal Rule Builder", bg="#f7f7f7", fg="#111111",
                 font=("Arial", 16, "bold"), anchor="w").pack(fill="x", pady=(0, 12))

        content = tk.Frame(root, bg="#f7f7f7")
        content.pack(fill="both", expand=True)
        left = tk.Frame(content, bg="#f7f7f7")
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right = tk.Frame(content, bg="#f7f7f7", width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        setup_frame = ttk.LabelFrame(left, text="Available Gesture Blocks", padding=10)
        setup_frame.pack(fill="x", pady=(0, 12))
        palette_frame = tk.Frame(setup_frame, bg="#f7f7f7")
        palette_frame.pack(fill="x")
        for i, g in enumerate(self.gesture_names):
            GesturePaletteBlock(palette_frame, g, "either", self._handle_drag_event).grid(row=i // 3, column=i % 3, padx=8, pady=8, sticky="nw")

        builder = ttk.LabelFrame(left, text="Target Gesture", padding=10)
        builder.pack(fill="both", expand=True)
        tk.Label(builder, text="Drag one gesture block into the area below. This rule means: when that complete gesture is detected, send the corresponding UDP signal command.",
                 bg="#f7f7f7", fg="#555555", anchor="w", justify="left", font=("Arial", 10)).pack(fill="x", pady=(0, 10))
        self.slot = SequenceSlot(builder, 0, remove_callback=lambda idx: self._clear_slot(), placeholder_text="Drop ONE Gesture Here")
        self.slot.pack(anchor="nw", padx=20, pady=20)

        param_frame = ttk.LabelFrame(right, text="UDP Rule Parameters", padding=10)
        param_frame.pack(fill="x")
        self.var_rule_name = tk.StringVar(value="rule1")
        self.var_signal_name = tk.StringVar(value="signal0")
        self.var_enabled = tk.BooleanVar(value=True)
        self._add_row(param_frame, "RULE_NAME", self.var_rule_name)
        row = ttk.Frame(param_frame)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="SIGNAL_NAME", width=16).pack(side="left")
        ttk.Combobox(row, textvariable=self.var_signal_name, values=SIGNAL_OPTIONS, state="readonly").pack(side="left", fill="x", expand=True)
        ttk.Checkbutton(param_frame, text="RULE_ENABLED", variable=self.var_enabled).pack(anchor="w", pady=(8, 0))

        preview_frame = ttk.LabelFrame(right, text="Preview", padding=10)
        preview_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.preview_text = tk.Text(preview_frame, height=18, wrap="word", font=("Consolas", 10))
        self.preview_text.pack(fill="both", expand=True)

        btn_frame = tk.Frame(right, bg="#f7f7f7")
        btn_frame.pack(fill="x", pady=(12, 0))
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(btn_frame, text="OK", command=self._on_ok).pack(side="right")

        self.var_rule_name.trace_add("write", lambda *args: self._refresh_preview())
        self.var_signal_name.trace_add("write", lambda *args: self._refresh_preview())
        self.var_enabled.trace_add("write", lambda *args: self._refresh_preview())

    def _add_row(self, parent, label_text, var):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label_text, width=16).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

    def _load_existing(self, data):
        self.var_rule_name.set(data.get("rule_name", "rule1"))
        self.var_signal_name.set(data.get("signal_name", "signal0"))
        self.var_enabled.set(bool(data.get("enabled", True)))
        g = data.get("gesture")
        if g:
            self.slot_value = dict(g)
            self.slot.set_value(self.slot_value)

    def _handle_drag_event(self, kind, block_data, event):
        if kind != "drop":
            return
        x_root, y_root = self.winfo_pointerx(), self.winfo_pointery()
        if self.slot.is_inside_screen_xy(x_root, y_root):
            self.slot_value = dict(block_data)
            self.slot.set_value(self.slot_value)
            self._refresh_preview()

    def _clear_slot(self):
        self.slot_value = None
        self._refresh_preview()

    def _refresh_preview(self):
        g = self.slot_value
        lines = [
            f"RULE_NAME: {self.var_rule_name.get()}",
            f"SIGNAL_NAME: {self.var_signal_name.get()}",
            f"ENABLED: {self.var_enabled.get()}",
            "",
            "GESTURE:",
            f" {g['name']}[{g['type']}]" if g else " [empty]",
            "",
            "SEND:",
            f" {self.var_signal_name.get()} -> {signal_name_to_digit(self.var_signal_name.get())}"
        ]
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", "\n".join(lines))

    def _collect(self):
        return {
            "rule_name": self.var_rule_name.get().strip(),
            "signal_name": self.var_signal_name.get().strip(),
            "gesture": dict(self.slot_value) if self.slot_value else None,
            "enabled": bool(self.var_enabled.get()),
        }

    def _on_ok(self):
        data = self._collect()
        if not data["rule_name"]:
            return messagebox.showerror("Error", "RULE_NAME cannot be empty.")
        if not data["signal_name"]:
            return messagebox.showerror("Error", "SIGNAL_NAME cannot be empty.")
        if data["gesture"] is None:
            return messagebox.showerror("Error", "You must drag in one gesture block.")
        if self.on_ok:
            self.on_ok(data)
        self.destroy()


class BLERuleCard(tk.Frame):
    def __init__(self, master, data, edit_callback=None, delete_callback=None, **kwargs):
        super().__init__(master, bd=1, relief="solid", bg="white", **kwargs)
        self.data = data
        self.edit_callback = edit_callback
        self.delete_callback = delete_callback
        self.expanded = False

        header = tk.Frame(self, bg="white")
        header.pack(fill="x", padx=8, pady=8)
        title = f"{data['rule_name']} [{data['signal_name']}] [{'Enabled' if data.get('enabled') else 'Disabled'}]"
        tk.Label(header, text=title, bg="white", fg="#111111", font=("Arial", 11, "bold"), anchor="w").pack(side="left", fill="x", expand=True)
        ttk.Button(header, text="Delete", width=8, command=lambda: self.delete_callback and self.delete_callback(self.data)).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Edit", width=8, command=lambda: self.edit_callback and self.edit_callback(self.data)).pack(side="right", padx=(6, 0))
        self.toggle_btn = ttk.Button(header, text="Expand", width=8, command=self.toggle)
        self.toggle_btn.pack(side="right", padx=(6, 0))
        self.body = tk.Frame(self, bg="#fafafa")

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            for c in self.body.winfo_children():
                c.destroy()
            g = self.data.get("gesture", {})
            for r in [
                f"GESTURE: {g.get('name', '')}[{g.get('type', 'either')}]",
                f"SIGNAL: {self.data.get('signal_name', '')} -> {signal_name_to_digit(self.data.get('signal_name', ''))}",
            ]:
                tk.Label(self.body, text=r, bg="#fafafa", fg="#333333", anchor="w",
                         justify="left", font=("Consolas", 10)).pack(fill="x", padx=8, pady=3)
            self.body.pack(fill="x", padx=8, pady=(0, 8))
            self.toggle_btn.config(text="Collapse")
        else:
            self.body.pack_forget()
            self.toggle_btn.config(text="Expand")

class MouseRuleEditor(tk.Toplevel):
    def __init__(self, master, gesture_names, existing=None, on_ok=None):
        super().__init__(master)
        self.title("Mouse Rule Builder")
        self.geometry("1120x720")
        self.minsize(980, 660)
        self.configure(bg="#f7f7f7")
        self.gesture_names, self.existing, self.on_ok = gesture_names[:], existing, on_ok
        self.slot_value = None

        self.transient(master)
        self.grab_set()
        self._build_ui()
        if existing:
            self._load_existing(existing)
        self._refresh_preview()

    def _build_ui(self):
        root = tk.Frame(self, bg="#f7f7f7")
        root.pack(fill="both", expand=True, padx=16, pady=16)
        tk.Label(root, text="Mouse Rule Builder", bg="#f7f7f7", fg="#111111",
                 font=("Arial", 16, "bold"), anchor="w").pack(fill="x", pady=(0, 12))

        content = tk.Frame(root, bg="#f7f7f7")
        content.pack(fill="both", expand=True)
        left = tk.Frame(content, bg="#f7f7f7")
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right = tk.Frame(content, bg="#f7f7f7", width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        setup_frame = ttk.LabelFrame(left, text="Available Gesture Blocks", padding=10)
        setup_frame.pack(fill="x", pady=(0, 12))
        palette_frame = tk.Frame(setup_frame, bg="#f7f7f7")
        palette_frame.pack(fill="x")
        for i, g in enumerate(self.gesture_names):
            GesturePaletteBlock(palette_frame, g, "either", self._handle_drag_event).grid(row=i // 3, column=i % 3, padx=8, pady=8, sticky="nw")

        builder = ttk.LabelFrame(left, text="Target Gesture", padding=10)
        builder.pack(fill="both", expand=True)
        tk.Label(builder, text="Drag one gesture block into the area below. This rule maps one completed gesture to one mouse action.",
                 bg="#f7f7f7", fg="#555555", anchor="w", justify="left", font=("Arial", 10)).pack(fill="x", pady=(0, 10))
        self.slot = SequenceSlot(builder, 0, remove_callback=lambda idx: self._clear_slot(), placeholder_text="Drop ONE Gesture Here")
        self.slot.pack(anchor="nw", padx=20, pady=20)

        param_frame = ttk.LabelFrame(right, text="Mouse Rule Parameters", padding=10)
        param_frame.pack(fill="x")
        self.var_rule_name = tk.StringVar(value="mouse_rule1")
        self.var_mouse_action = tk.StringVar(value=DEFAULT_MOUSE_ACTION)
        self.var_move_speed = tk.StringVar(value=str(DEFAULT_MOUSE_MOVE_SPEED))
        self.var_enabled = tk.BooleanVar(value=True)
        self._add_row(param_frame, "RULE_NAME", self.var_rule_name)

        row_action = ttk.Frame(param_frame)
        row_action.pack(fill="x", pady=4)
        ttk.Label(row_action, text="MOUSE_ACTION", width=16).pack(side="left")
        ttk.Combobox(row_action, textvariable=self.var_mouse_action, values=MOUSE_ACTION_OPTIONS, state="readonly").pack(side="left", fill="x", expand=True)

        self._add_row(param_frame, "MOVE_SPEED", self.var_move_speed)
        ttk.Checkbutton(param_frame, text="ENABLED", variable=self.var_enabled).pack(anchor="w", pady=(8, 0))

        preview_frame = ttk.LabelFrame(right, text="Preview", padding=10)
        preview_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.preview_text = tk.Text(preview_frame, height=18, wrap="word", font=("Consolas", 10))
        self.preview_text.pack(fill="both", expand=True)

        btn_frame = tk.Frame(right, bg="#f7f7f7")
        btn_frame.pack(fill="x", pady=(12, 0))
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(btn_frame, text="OK", command=self._on_ok).pack(side="right")

        self.var_rule_name.trace_add("write", lambda *args: self._refresh_preview())
        self.var_mouse_action.trace_add("write", lambda *args: self._refresh_preview())
        self.var_move_speed.trace_add("write", lambda *args: self._refresh_preview())
        self.var_enabled.trace_add("write", lambda *args: self._refresh_preview())

    def _add_row(self, parent, label_text, var):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label_text, width=16).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

    def _load_existing(self, data):
        self.var_rule_name.set(data.get("rule_name", "mouse_rule1"))
        self.var_mouse_action.set(data.get("mouse_action", DEFAULT_MOUSE_ACTION))
        self.var_move_speed.set(str(data.get("move_speed", DEFAULT_MOUSE_MOVE_SPEED)))
        self.var_enabled.set(bool(data.get("enabled", True)))
        g = data.get("gesture")
        if g:
            self.slot_value = dict(g)
            self.slot.set_value(self.slot_value)

    def _handle_drag_event(self, kind, block_data, event):
        if kind != "drop":
            return
        x_root, y_root = self.winfo_pointerx(), self.winfo_pointery()
        if self.slot.is_inside_screen_xy(x_root, y_root):
            self.slot_value = dict(block_data)
            self.slot.set_value(self.slot_value)
            self._refresh_preview()

    def _clear_slot(self):
        self.slot_value = None
        self._refresh_preview()

    def _refresh_preview(self):
        g = self.slot_value
        lines = [
            f"RULE_NAME: {self.var_rule_name.get()}",
            f"MOUSE_ACTION: {self.var_mouse_action.get()}",
            f"MOVE_SPEED: {self.var_move_speed.get()}",
            f"ENABLED: {self.var_enabled.get()}",
            "",
            "GESTURE:",
            f" {g['name']}[{g['type']}]" if g else " [empty]",
            "",
            "Runtime behavior:",
            " move_*: start on ENTER(...) and stop on EXIT(...)",
            " recenter/click: trigger once when gesture action is completed",
        ]
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", "\n".join(lines))

    def _collect(self):
        return {
            "rule_name": self.var_rule_name.get().strip(),
            "mouse_action": self.var_mouse_action.get().strip(),
            "gesture": dict(self.slot_value) if self.slot_value else None,
            "move_speed": int(self.var_move_speed.get().strip() or str(DEFAULT_MOUSE_MOVE_SPEED)),
            "enabled": bool(self.var_enabled.get()),
        }

    def _on_ok(self):
        try:
            data = normalize_mouse_rule_entry(self._collect())
        except Exception:
            return messagebox.showerror("Error", "MOVE_SPEED must be an integer.")
        if data is None:
            return messagebox.showerror("Error", "Please set RULE_NAME, MOUSE_ACTION and one gesture block.")
        if self.on_ok:
            self.on_ok(data)
        self.destroy()


class MouseRuleCard(tk.Frame):
    def __init__(self, master, data, edit_callback=None, delete_callback=None, **kwargs):
        super().__init__(master, bd=1, relief="solid", bg="white", **kwargs)
        self.data = data
        self.edit_callback = edit_callback
        self.delete_callback = delete_callback
        self.expanded = False

        header = tk.Frame(self, bg="white")
        header.pack(fill="x", padx=8, pady=8)
        title = f"{data['rule_name']} [{data['mouse_action']}] [{'Enabled' if data.get('enabled') else 'Disabled'}]"
        tk.Label(header, text=title, bg="white", fg="#111111", font=("Arial", 11, "bold"), anchor="w").pack(side="left", fill="x", expand=True)
        ttk.Button(header, text="Delete", width=8, command=lambda: self.delete_callback and self.delete_callback(self.data)).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Edit", width=8, command=lambda: self.edit_callback and self.edit_callback(self.data)).pack(side="right", padx=(6, 0))
        self.toggle_btn = ttk.Button(header, text="Expand", width=8, command=self.toggle)
        self.toggle_btn.pack(side="right", padx=(6, 0))
        self.body = tk.Frame(self, bg="#fafafa")

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            for c in self.body.winfo_children():
                c.destroy()
            g = self.data.get("gesture", {})
            for r in [
                f"GESTURE: {g.get('name', '')}[{g.get('type', 'either')}]",
                f"MOUSE_ACTION: {self.data.get('mouse_action', '')}",
                f"MOVE_SPEED: {self.data.get('move_speed', DEFAULT_MOUSE_MOVE_SPEED)}",
            ]:
                tk.Label(self.body, text=r, bg="#fafafa", fg="#333333", anchor="w",
                         justify="left", font=("Consolas", 10)).pack(fill="x", padx=8, pady=3)
            self.body.pack(fill="x", padx=8, pady=(0, 8))
            self.toggle_btn.config(text="Collapse")
        else:
            self.body.pack_forget()
            self.toggle_btn.config(text="Expand")

