import asyncio
import socket
import struct
import numpy as np
from app_defaults import (
    HEADER_STRUCT, FRAME_STRUCT, FRAME_SIZE, HEADER_SIZE, BATCH_SIZE,
    BATCH_FRAME_COUNT, PACKET_MAGIC, PACKET_VERSION, DEFAULT_CMD_PORT
)

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
    vals, cnts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
    return int(vals[np.argmax(cnts)])


class UDPBridge:
    def __init__(self, cfg, gui_queue):
        self.cfg = cfg
        self.gui_queue = gui_queue

        self.sensor_sock = None
        self.cmd_sock = None

        self.last_nondata_log_time = 0.0

    def log(self, msg):
        self.gui_queue.put(("log", msg))

    async def connect(self):
        host_ip = self.cfg["HOST_IP"]
        sensor_port = int(self.cfg["SENSOR_PORT"])

        self.sensor_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sensor_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sensor_sock.bind((host_ip, sensor_port))
        self.sensor_sock.setblocking(False)

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.setblocking(False)

        self.log(f"[UDP] sensor receiver bound at {host_ip}:{sensor_port}")
        self.log(f"[UDP] expecting binary batch packet: header='<HHI', frame='<I5Hhh', batch_bytes={BATCH_SIZE}")

    async def disconnect(self):
        try:
            if self.sensor_sock is not None:
                self.sensor_sock.close()
        except Exception:
            pass

        try:
            if self.cmd_sock is not None:
                self.cmd_sock.close()
        except Exception:
            pass

        self.sensor_sock = None
        self.cmd_sock = None
        self.log("[UDP] Disconnected")

    async def get_packet(self, timeout=0.2):
        if self.sensor_sock is None:
            return None, None
        try:
            loop = asyncio.get_running_loop()
            data, addr = await asyncio.wait_for(loop.sock_recvfrom(self.sensor_sock, 4096), timeout=timeout)
            return data, addr
        except asyncio.TimeoutError:
            return None, None
        except Exception as e:
            self.log(f"[UDP][RECV][ERROR] {e}")
            return None, None

    async def send_signal(self, signal_name):
        if not self.cfg.get("UDP_SEND_ENABLED", True):
            return False, "UDP_SEND_DISABLED"

        arduino2_ip = str(self.cfg.get("ARDUINO2_IP", "")).strip()
        cmd_port = int(self.cfg.get("CMD_PORT", DEFAULT_CMD_PORT))

        if not arduino2_ip:
            return False, "ARDUINO2_IP is empty"

        cmd_text = signal_name_to_command_text(signal_name)
        if cmd_text is None:
            return False, f"Invalid signal name: {signal_name}"

        if self.cmd_sock is None:
            return False, "cmd_sock not ready"

        try:
            payload = cmd_text.encode("utf-8")
            self.cmd_sock.sendto(payload, (arduino2_ip, cmd_port))
            return True, cmd_text
        except Exception as e:
            return False, str(e)


