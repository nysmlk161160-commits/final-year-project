import threading
import time
from app_defaults import DEFAULT_MOUSE_MOVE_SPEED
try:
    import pyautogui
except Exception:
    pyautogui = None

class MouseController:
    def __init__(self, gui_queue, enabled=False, interval_sec=0.025):
        self.gui_queue = gui_queue
        self.enabled = bool(enabled)
        self.interval_sec = max(0.01, float(interval_sec))
        self.available = pyautogui is not None
        self._active_rules = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def _delta_from_action(self, action, speed):
        speed = int(speed)
        if action == "move_left":
            return -speed, 0
        if action == "move_right":
            return speed, 0
        if action == "move_up":
            return 0, -speed
        if action == "move_down":
            return 0, speed
        return 0, 0

    def start(self):
        if not self.enabled or not self.available:
            return
        try:
            pyautogui.FAILSAFE = False
        except Exception:
            pass
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.8)
        with self._lock:
            self._active_rules.clear()

    def _run_loop(self):
        while not self._stop_event.is_set():
            dx_total, dy_total = 0, 0
            with self._lock:
                for item in self._active_rules.values():
                    dx_total += int(item["dx"])
                    dy_total += int(item["dy"])
            if dx_total != 0 or dy_total != 0:
                try:
                    pyautogui.moveRel(dx_total, dy_total, duration=0)
                except Exception as e:
                    self.gui_queue.put(("log", f"[MouseController][ERROR] moveRel failed: {e}"))
            time.sleep(self.interval_sec)

    def handle_hit(self, hit):
        msgs = []
        if not self.enabled:
            return msgs
        if not self.available:
            return ["[MouseController] pyautogui is unavailable. Install with: pip install pyautogui"]

        kind = str(hit.get("kind", ""))
        rule_name = str(hit.get("rule_name", ""))
        action = str(hit.get("mouse_action", ""))
        gesture_name = str(hit.get("gesture_name", ""))
        speed = int(hit.get("move_speed", DEFAULT_MOUSE_MOVE_SPEED))

        if kind == "move_start":
            dx, dy = self._delta_from_action(action, speed)
            with self._lock:
                existed = rule_name in self._active_rules
                self._active_rules[rule_name] = {"dx": dx, "dy": dy}
            if not existed:
                msgs.append(f"MOUSE START [{rule_name}] gesture={gesture_name} -> {action} speed={speed}")

        elif kind == "move_stop":
            with self._lock:
                existed = rule_name in self._active_rules
                self._active_rules.pop(rule_name, None)
            if existed:
                msgs.append(f"MOUSE STOP [{rule_name}] gesture={gesture_name} -> {action}")

        elif kind == "oneshot":
            try:
                if action == "recenter":
                    w, h = pyautogui.size()
                    pyautogui.moveTo(int(w // 2), int(h // 2), duration=0)
                    msgs.append(f"MOUSE ONESHOT [{rule_name}] gesture={gesture_name} -> recenter")
                elif action == "click":
                    pyautogui.click(button="left")
                    msgs.append(f"MOUSE ONESHOT [{rule_name}] gesture={gesture_name} -> click")
            except Exception as e:
                msgs.append(f"[MouseController][ERROR] oneshot failed ({action}): {e}")

        return msgs


