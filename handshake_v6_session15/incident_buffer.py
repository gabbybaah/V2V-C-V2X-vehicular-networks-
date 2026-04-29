# =============================================================================
# handshake_v5/incident_buffer.py  —  Circular 30-second road-state buffer
#
# Every tick, sim_road pushes the full road snapshot here.
# On a significant event (HARD_BRAKE chain ≥ 3, ROGUE_VIOLATION,
# CONVOY_OVERTAKE, EMERG_CORRIDOR), the buffer writes the last 30 seconds
# of state to logs/incidents/<event>_<timestamp>.json
#
# Useful for:
#   - Post-run replay of dramatic events
#   - ML debugging: see exactly what the model saw before a bad prediction
#   - Human review of CVC violations
# =============================================================================
import os, json, time, threading, logging
from collections import deque

log = logging.getLogger("v5.incident_buffer")

INCIDENTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "incidents")

# At 10 ticks/sec, 30s = 300 ticks
MAX_TICKS  = 300
# Significant event types that trigger a write
TRIGGER_EVENTS = frozenset({
    "HARD_BRAKE", "BRAKE_CHAIN", "CONVOY_OVERTAKE",
    "ROGUE_CROSS", "SPLIT_BRAIN_START", "BREAKDOWN",
    "PREEMPT", "CORRIDOR",
})
# Minimum seconds between incident writes (avoid flood on one event)
WRITE_COOLDOWN_S = 15.0


class IncidentBuffer:
    """
    Thread-safe circular buffer of full road snapshots.
    Call push(snapshot, tick, elapsed_s) every sim tick.
    Call check_events(events) with new token events to auto-trigger writes.
    """

    def __init__(self, max_ticks: int = MAX_TICKS):
        self._buf      = deque(maxlen=max_ticks)
        self._lock     = threading.Lock()
        self._last_write = 0.0
        self._write_count = 0

    def push(self, snapshot: list, tick: int, elapsed_s: float):
        """Append current road state. Called every sim tick."""
        frame = {
            "tick":      tick,
            "elapsed_s": round(elapsed_s, 2),
            "ts":        time.time(),
            "cars": [
                {
                    "label":   c.get("label",""),
                    "car_type":c.get("car_type",""),
                    "lane":    c.get("lane",0),
                    "pos":     round(c.get("road_pos_m",0), 1),
                    "spd":     round(c.get("speed_kmh",0), 1),
                    "state":   c.get("state",""),
                    "profile": c.get("profile",""),
                    "lat":     round(c.get("lateral_offset_m",0), 3),
                    "ml_conf": round(c.get("ml_conf",0), 3),
                }
                for c in snapshot
            ],
        }
        with self._lock:
            self._buf.append(frame)

    def check_events(self, events: list):
        """
        Called with new token events each tick.
        If a trigger event is found, write incident if cooldown passed.
        """
        if not events:
            return
        now = time.time()
        if now - self._last_write < WRITE_COOLDOWN_S:
            return
        for ev in events:
            ev_type = ev.get("event","")
            if ev_type in TRIGGER_EVENTS:
                self._last_write = now
                t = threading.Thread(
                    target=self._write_incident,
                    args=(ev_type, ev.get("detail",""), ev.get("ts", now)),
                    daemon=True,
                )
                t.start()
                break

    def _write_incident(self, event_type: str, detail: str, event_ts: float):
        with self._lock:
            frames = list(self._buf)

        if not frames:
            return

        os.makedirs(INCIDENTS_DIR, exist_ok=True)
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(event_ts))
        safe_type = event_type.replace("/","_").replace(" ","_")[:20]
        fname     = f"{safe_type}_{ts_str}.json"
        fpath     = os.path.join(INCIDENTS_DIR, fname)

        incident = {
            "event_type": event_type,
            "detail":     detail,
            "event_ts":   event_ts,
            "written_ts": time.time(),
            "n_frames":   len(frames),
            "duration_s": round(frames[-1]["elapsed_s"] - frames[0]["elapsed_s"], 1)
                          if len(frames) >= 2 else 0,
            "frames":     frames,
        }

        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(incident, f, separators=(",",":"))
            self._write_count += 1
            log.info(f"Incident saved: {fname} ({len(frames)} frames)")
            print(f"  📼 Incident replay saved: {fname}")
        except Exception as e:
            log.warning(f"Could not write incident: {e}")

    def size(self) -> int:
        with self._lock:
            return len(self._buf)

    def write_count(self) -> int:
        return self._write_count

    def clear(self):
        with self._lock:
            self._buf.clear()
