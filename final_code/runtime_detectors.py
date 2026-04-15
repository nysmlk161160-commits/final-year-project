from app_defaults import (
    DEFAULT_REST_LABEL,
    DEFAULT_LABEL_STABLE_MS,
    DEFAULT_QUICK_TAP_MAX_MS,
    DEFAULT_LONG_HOLD_MIN_MS,
    DEFAULT_DOUBLE_TAP_GAP_MS,
    DEFAULT_ENTER_CONFIRM_MS,
)

class PrimitiveDetector:
    def __init__(
        self,
        rest_label=DEFAULT_REST_LABEL,
        label_stable_ms=DEFAULT_LABEL_STABLE_MS,
        quick_tap_max_ms=DEFAULT_QUICK_TAP_MAX_MS,
        long_hold_min_ms=DEFAULT_LONG_HOLD_MIN_MS,
        double_tap_gap_ms=DEFAULT_DOUBLE_TAP_GAP_MS,
        enter_confirm_ms=DEFAULT_ENTER_CONFIRM_MS,
    ):
        self.rest_label = str(rest_label)
        self.label_stable_ms = int(label_stable_ms)
        self.quick_tap_max_ms = int(quick_tap_max_ms)
        self.long_hold_min_ms = int(long_hold_min_ms)
        self.double_tap_gap_ms = int(double_tap_gap_ms)
        self.enter_confirm_ms = int(enter_confirm_ms)

        self.confirmed_label = self.rest_label
        self.confirmed_since_ms = None
        self.candidate_label = None
        self.candidate_since_ms = None
        self.active_label = None
        self.active_start_ms = None
        self.last_quick_tap_end_ms = {}
        self.pending_enter_label = None
        self.pending_enter_start_ms = None

    def _emit_enter(self, label, start_ms):
        return [f"ENTER({label})"], [{"type": "ENTER", "label": str(label), "t_ms": int(start_ms)}]

    def _finalize_action(self, label, end_t_ms, direct_switch=False):
        msgs, events = [], []
        if self.active_label == label and self.active_start_ms is not None:
            dur = end_t_ms - self.active_start_ms
            msgs.append(f"EXIT({label}) dur={dur}ms" + (" (direct_switch)" if direct_switch else ""))
            events.append({"type": "EXIT", "label": str(label), "t_ms": int(end_t_ms), "dur_ms": int(dur)})

            if dur <= self.quick_tap_max_ms:
                msgs.append(f"PRIM QuickTap({label}) dur={dur}ms")
                if label in self.last_quick_tap_end_ms:
                    gap = end_t_ms - self.last_quick_tap_end_ms[label]
                    if gap <= self.double_tap_gap_ms:
                        msgs.append(f"PRIM DoubleQuickTap({label}) gap={gap}ms")
                self.last_quick_tap_end_ms[label] = end_t_ms

            if dur >= self.long_hold_min_ms:
                msgs.append(f"PRIM LongHold({label}) dur={dur}ms")

        return msgs, events

    def update(self, voted_label, t_ms):
        msgs, events = [], []
        voted_label = str(voted_label)

        if self.confirmed_since_ms is None:
            self.confirmed_label = voted_label
            self.confirmed_since_ms = t_ms
            self.candidate_label = voted_label
            self.candidate_since_ms = t_ms
            if voted_label != self.rest_label:
                self.pending_enter_label = voted_label
                self.pending_enter_start_ms = t_ms
            return msgs, events

        if self.pending_enter_label is not None:
            if voted_label == self.pending_enter_label:
                if (t_ms - self.pending_enter_start_ms) >= self.enter_confirm_ms:
                    self.active_label = self.pending_enter_label
                    self.active_start_ms = self.pending_enter_start_ms
                    m, e = self._emit_enter(self.active_label, self.active_start_ms)
                    msgs += m
                    events += e
                    self.pending_enter_label = None
                    self.pending_enter_start_ms = None
            else:
                self.pending_enter_label = None
                self.pending_enter_start_ms = None

        if voted_label == self.confirmed_label:
            self.candidate_label = voted_label
            self.candidate_since_ms = t_ms
            return msgs, events

        if voted_label != self.candidate_label:
            self.candidate_label = voted_label
            self.candidate_since_ms = t_ms
            return msgs, events

        if (t_ms - self.candidate_since_ms) < self.label_stable_ms:
            return msgs, events

        old_label, new_label = self.confirmed_label, voted_label
        self.confirmed_label = new_label
        self.confirmed_since_ms = self.candidate_since_ms

        if old_label == self.rest_label and new_label != self.rest_label:
            self.pending_enter_label = new_label
            self.pending_enter_start_ms = self.candidate_since_ms
            return msgs, events

        if old_label != self.rest_label and new_label == self.rest_label:
            if self.pending_enter_label == old_label:
                self.pending_enter_label = None
                self.pending_enter_start_ms = None
            m, e = self._finalize_action(old_label, t_ms, direct_switch=False)
            msgs += m
            events += e
            self.active_label = None
            self.active_start_ms = None
            self.pending_enter_label = None
            self.pending_enter_start_ms = None
            return msgs, events

        if old_label != self.rest_label and new_label != self.rest_label:
            m, e = self._finalize_action(old_label, t_ms, direct_switch=True)
            msgs += m
            events += e
            self.active_label = None
            self.active_start_ms = None
            self.pending_enter_label = new_label
            self.pending_enter_start_ms = self.candidate_since_ms
            return msgs, events

        return msgs, events

class PipelineDetector:
    def __init__(self, enabled=True, pipeline_name="pipeline1", sequence=None, step_gap_ms=1000, cooldown_ms=1200):
        self.enabled = bool(enabled)
        self.pipeline_name = str(pipeline_name).strip() or "pipeline1"
        self.sequence = [str(x).strip() for x in (sequence or []) if str(x).strip()]
        self.step_gap_ms = int(step_gap_ms)
        self.cooldown_ms = int(cooldown_ms)
        self.last_detect_time_ms = None
        self.reset()

    def reset(self):
        self.matched_count = 0
        self.waiting_for_enter = False
        self.last_exit_t_ms = None

    def update_event(self, evt):
        msgs, done = [], []
        if not self.enabled or not self.sequence:
            return msgs, done

        evt_type, evt_label, evt_t_ms = str(evt.get("type", "")), str(evt.get("label", "")), int(evt.get("t_ms", 0))

        if self.last_detect_time_ms is not None and (evt_t_ms - self.last_detect_time_ms) < self.cooldown_ms:
            return msgs, done

        if self.matched_count == 0:
            if evt_type == "EXIT" and evt_label == self.sequence[0]:
                self.matched_count = 1
                self.waiting_for_enter = True
                self.last_exit_t_ms = evt_t_ms
                msgs.append(f"PIPE step 1/{len(self.sequence)} completed by EXIT({evt_label}) ({self.pipeline_name})")
                if len(self.sequence) == 1:
                    seq_txt = "->".join(self.sequence)
                    msgs.append(f"PRIM Pipeline({self.pipeline_name}) [{seq_txt}]")
                    done.append({"type": "PIPELINE_COMPLETE", "pipeline_name": self.pipeline_name, "sequence_text": seq_txt, "t_ms": evt_t_ms})
                    self.last_detect_time_ms = evt_t_ms
                    self.reset()
            return msgs, done

        if self.waiting_for_enter:
            expected = self.sequence[self.matched_count]
            if evt_type == "ENTER":
                gap = None if self.last_exit_t_ms is None else evt_t_ms - self.last_exit_t_ms
                if gap is None or gap > self.step_gap_ms:
                    self.reset()
                    return msgs, done
                if evt_label == expected:
                    self.matched_count += 1
                    self.waiting_for_enter = False
                    msgs.append(f"PIPE step {self.matched_count}/{len(self.sequence)} started by ENTER({evt_label}) gap={gap}ms ({self.pipeline_name})")
                    if self.matched_count >= len(self.sequence):
                        seq_txt = "->".join(self.sequence)
                        msgs.append(f"PRIM Pipeline({self.pipeline_name}) [{seq_txt}]")
                        done.append({"type": "PIPELINE_COMPLETE", "pipeline_name": self.pipeline_name, "sequence_text": seq_txt, "t_ms": evt_t_ms})
                        self.last_detect_time_ms = evt_t_ms
                        self.reset()
                else:
                    self.reset()
            return msgs, done

        current = self.sequence[self.matched_count - 1]
        if evt_type == "EXIT" and evt_label == current:
            self.waiting_for_enter = True
            self.last_exit_t_ms = evt_t_ms
            msgs.append(f"PIPE step {self.matched_count}/{len(self.sequence)} completed by EXIT({evt_label}) ({self.pipeline_name})")
        return msgs, done

class TaskDetector:
    def __init__(self, enabled=False, task_name="task1", target_type="label", target_name="1",
                 label_mode="either", duration_sec=10.0, required_count=5):
        self.enabled = bool(enabled)
        self.task_name = str(task_name).strip() or "task1"
        self.target_type = str(target_type).strip() or "label"
        self.target_name = str(target_name).strip()
        self.label_mode = str(label_mode).strip() or "either"
        self.duration_sec = float(duration_sec)
        self.required_count = max(1, int(required_count))
        self.started = False
        self.finished = False
        self.success = False
        self.start_t_ms = None
        self.deadline_t_ms = None
        self.count = 0

    def start(self, t_ms):
        if not self.enabled or self.started:
            return []
        self.started, self.finished, self.success = True, False, False
        self.count = 0
        self.start_t_ms = int(t_ms)
        self.deadline_t_ms = int(t_ms + self.duration_sec * 1000.0)
        tail = f"{self.target_name}[{self.label_mode}]" if self.target_type == "label" else self.target_name
        return [f"TASK START [{self.task_name}] target={self.target_type}:{tail} need={self.required_count} within={self.duration_sec:.2f}s"]

    def _finish_success(self, t_ms):
        self.finished, self.success = True, True
        elapsed = 0.0 if self.start_t_ms is None else max(0.0, (int(t_ms) - self.start_t_ms) / 1000.0)
        return [f"TASK SUCCESS [{self.task_name}] {self.count}/{self.required_count} elapsed={elapsed:.2f}s",
                f"TASK END [{self.task_name}] success"]

    def _finish_fail(self, t_ms, reason="timeout"):
        self.finished, self.success = True, False
        elapsed = 0.0 if self.start_t_ms is None else max(0.0, (int(t_ms) - self.start_t_ms) / 1000.0)
        return [f"TASK FAIL [{self.task_name}] reason={reason} progress={self.count}/{self.required_count} elapsed={elapsed:.2f}s",
                f"TASK END [{self.task_name}] failed"]

    def process_hit(self, hit_type, hit_name, t_ms):
        if not self.enabled or not self.started or self.finished:
            return []
        t_ms = int(t_ms)
        if self.deadline_t_ms is not None and t_ms > self.deadline_t_ms:
            return self._finish_fail(t_ms, "timeout")
        if str(hit_type) == self.target_type and str(hit_name) == self.target_name:
            self.count += 1
            msgs = [f"TASK PROGRESS [{self.task_name}] {self.count}/{self.required_count} by {self.target_type}({self.target_name})"]
            if self.count >= self.required_count:
                msgs += self._finish_success(t_ms)
            return msgs
        return []

    def tick(self, t_ms):
        if not self.enabled or not self.started or self.finished:
            return []
        if self.deadline_t_ms is not None and int(t_ms) > self.deadline_t_ms:
            return self._finish_fail(t_ms, "timeout")
        return []

    def get_status_text(self, now_t_ms=None):
        if not self.enabled:
            return "Task Disabled"
        if not self.started:
            return "Task Pending"
        if self.finished:
            return f"Success {self.count}/{self.required_count}" if self.success else f"Failed {self.count}/{self.required_count}"
        if self.start_t_ms is None or self.deadline_t_ms is None:
            return f"Running {self.count}/{self.required_count}"
        remain_ms = max(0, self.deadline_t_ms - int(now_t_ms if now_t_ms is not None else self.start_t_ms))
        return f"Running {self.count}/{self.required_count} | left {remain_ms/1000.0:.2f}s"
