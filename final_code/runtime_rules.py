from app_defaults import (
    DEFAULT_PIPELINE_NAME,
    DEFAULT_PIPELINE_STEP_GAP_MS,
    DEFAULT_PIPELINE_COOLDOWN_MS,
    GESTURE_TYPE_OPTIONS,
    DEFAULT_MOUSE_ACTION,
    MOUSE_ACTION_OPTIONS,
    DEFAULT_MOUSE_MOVE_SPEED,
)

def parse_pipeline_sequence_text(text):
    if text is None:
        return []
    s = str(text).strip().replace("，", ",").replace("->", ",").replace("=>", ",")
    return [p.strip() for p in s.split(",") if p.strip()]


def normalize_pipeline_entry(p):
    if p is None:
        return None
    name = str(p.get("name", "")).strip()
    if not name:
        return None
    seq_names = []
    for x in p.get("sequence", []):
        n = str(x.get("name", "")).strip() if isinstance(x, dict) else str(x).strip()
        if n:
            seq_names.append(n)
    return {
        "name": name,
        "signal_name": str(p.get("signal_name", f"sig_{name}")).strip() or f"sig_{name}",
        "sequence": [{"name": s, "type": "either"} for s in seq_names],
        "step_gap_ms": int(p.get("step_gap_ms", 1000)),
        "cooldown_ms": int(p.get("cooldown_ms", 1200)),
        "enabled": bool(p.get("enabled", True)),
    }


def pipeline_sequence_names_from_entry(p):
    if p is None:
        return []
    out = []
    for x in p.get("sequence", []):
        n = str(x.get("name", "")).strip() if isinstance(x, dict) else str(x).strip()
        if n:
            out.append(n)
    return out


def normalize_ble_rule_entry(r):
    if r is None:
        return None
    rule_name = str(r.get("rule_name", "")).strip()
    signal_name = str(r.get("signal_name", "")).strip()
    gesture = r.get("gesture", None)
    if not rule_name or not signal_name or not gesture:
        return None
    g_name = str(gesture.get("name", "")).strip()
    g_type = str(gesture.get("type", "either")).strip() or "either"
    if not g_name:
        return None
    return {
        "rule_name": rule_name,
        "signal_name": signal_name,
        "gesture": {"name": g_name, "type": g_type},
        "enabled": bool(r.get("enabled", True)),
    }


def signal_name_to_digit(signal_name):
    s = str(signal_name).strip().lower()
    if s.startswith("signal") and s[6:].isdigit():
        n = int(s[6:])
        if 0 <= n <= 5:
            return str(n)
    return None


def signal_name_to_command_text(signal_name):
    s = str(signal_name).strip()
    digit = signal_name_to_digit(s)
    if digit is not None:
        return s
    return None


def parse_line_7ch(line):
    line = str(line).strip()
    if not line or line.startswith("time_ms") or line.startswith("PROF"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 8:
        return None
    try:
        t_ms = int(float(parts[0]))
        vals = [float(x) for x in parts[1:8]]
        return t_ms, np.array(vals, dtype=np.float32)
    except Exception:
        return None


def parse_batch_binary(data):
    if data is None:
        return None
    if len(data) != BATCH_SIZE:
        return None

    try:
        magic, version, batch_id = HEADER_STRUCT.unpack_from(data, 0)
    except Exception:
        return None

    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        return None

    frames = []
    offset = HEADER_SIZE

    for _ in range(BATCH_FRAME_COUNT):
        try:
            t_ms, a0, a1, a2, a3, a4, roll_x100, pitch_x100 = FRAME_STRUCT.unpack_from(data, offset)
        except Exception:
            return None

        roll = roll_x100 / 100.0
        pitch = pitch_x100 / 100.0
        x7 = np.array([a0, a1, a2, a3, a4, roll, pitch], dtype=np.float32)
        frames.append((int(t_ms), x7))
        offset += FRAME_SIZE

    return int(batch_id), frames


def majority_vote(ids):
    if not ids:
        return None
    vals, counts = np.unique(np.array(ids, dtype=np.int32), return_counts=True)
    return int(vals[np.argmax(counts)])


def collect_label_hits(structured_events, primitive_msgs, target_label, label_mode, fallback_t_ms):
    hits = []
    target_label = str(target_label)
    label_mode = str(label_mode).strip() or "either"
    if label_mode == "either":
        for evt in structured_events:
            if str(evt.get("type")) == "EXIT" and str(evt.get("label")) == target_label:
                hits.append(("label", target_label, int(evt.get("t_ms", fallback_t_ms))))
    elif label_mode == "tap":
        prefix = f"PRIM QuickTap({target_label})"
        for m in primitive_msgs:
            if str(m).startswith(prefix):
                hits.append(("label", target_label, int(fallback_t_ms)))
    elif label_mode == "long_hold":
        prefix = f"PRIM LongHold({target_label})"
        for m in primitive_msgs:
            if str(m).startswith(prefix):
                hits.append(("label", target_label, int(fallback_t_ms)))
    return hits


def collect_ble_rule_hits(structured_events, primitive_msgs, rules, fallback_t_ms):
    hits = []
    for r in rules:
        if not r.get("enabled", True):
            continue
        g = r["gesture"]
        label_hits = collect_label_hits(structured_events, primitive_msgs, g["name"], g["type"], fallback_t_ms)
        for _, _, hit_t in label_hits:
            hits.append({
                "rule_name": r["rule_name"],
                "signal_name": r["signal_name"],
                "gesture_name": g["name"],
                "gesture_type": g["type"],
                "t_ms": hit_t,
            })
    return hits


def normalize_mouse_rule_entry(r):
    if r is None:
        return None
    rule_name = str(r.get("rule_name", "")).strip()
    mouse_action = str(r.get("mouse_action", DEFAULT_MOUSE_ACTION)).strip() or DEFAULT_MOUSE_ACTION
    gesture = r.get("gesture", None)
    if not rule_name or not gesture or mouse_action not in MOUSE_ACTION_OPTIONS:
        return None
    g_name = str(gesture.get("name", "")).strip()
    g_type = str(gesture.get("type", "either")).strip() or "either"
    if not g_name or g_type not in GESTURE_TYPE_OPTIONS:
        return None
    move_speed = int(r.get("move_speed", DEFAULT_MOUSE_MOVE_SPEED))
    if move_speed <= 0:
        move_speed = DEFAULT_MOUSE_MOVE_SPEED
    return {
        "rule_name": rule_name,
        "mouse_action": mouse_action,
        "gesture": {"name": g_name, "type": g_type},
        "move_speed": move_speed,
        "enabled": bool(r.get("enabled", True)),
    }


def collect_mouse_rule_hits(structured_events, primitive_msgs, rules, fallback_t_ms):
    hits = []
    for r in rules:
        if not r.get("enabled", True):
            continue
        g = r["gesture"]
        action = r["mouse_action"]
        if action in ("move_left", "move_right", "move_up", "move_down"):
            for evt in structured_events:
                evt_type = str(evt.get("type", ""))
                evt_label = str(evt.get("label", ""))
                if evt_label != g["name"]:
                    continue
                if evt_type == "ENTER":
                    hits.append({
                        "kind": "move_start",
                        "rule_name": r["rule_name"],
                        "mouse_action": action,
                        "gesture_name": g["name"],
                        "gesture_type": g["type"],
                        "move_speed": int(r.get("move_speed", DEFAULT_MOUSE_MOVE_SPEED)),
                        "t_ms": int(evt.get("t_ms", fallback_t_ms)),
                    })
                elif evt_type == "EXIT":
                    hits.append({
                        "kind": "move_stop",
                        "rule_name": r["rule_name"],
                        "mouse_action": action,
                        "gesture_name": g["name"],
                        "gesture_type": g["type"],
                        "move_speed": int(r.get("move_speed", DEFAULT_MOUSE_MOVE_SPEED)),
                        "t_ms": int(evt.get("t_ms", fallback_t_ms)),
                    })
        else:
            label_hits = collect_label_hits(structured_events, primitive_msgs, g["name"], g["type"], fallback_t_ms)
            for _, _, hit_t in label_hits:
                hits.append({
                    "kind": "oneshot",
                    "rule_name": r["rule_name"],
                    "mouse_action": action,
                    "gesture_name": g["name"],
                    "gesture_type": g["type"],
                    "move_speed": int(r.get("move_speed", DEFAULT_MOUSE_MOVE_SPEED)),
                    "t_ms": int(hit_t),
                })
    return hits


