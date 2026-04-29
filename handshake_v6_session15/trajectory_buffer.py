# =============================================================================
# handshake_v5/trajectory_buffer.py  —  Rolling observation history per car
#
# Each SmartNPC and PlayerCar holds one TrajectoryBuffer.
# Every tick: buffer.update(snapshot) appends current state of all visible cars.
# To query: history = buffer.get(car_id) → ndarray (40,5) or None
#
# Features stored per tick per car:
#   [0] road_pos_m          (normalised: /ROAD_LENGTH_M)
#   [1] speed_ms            (normalised: /SPEED_LIMIT_MS*2)
#   [2] lane                (0,1,2  /  2.0  → 0..1)
#   [3] lateral_offset_m    (normalised: /LANE_WIDTH_M + 0.5 → 0..1)
#   [4] state_encoded       (integer 0-9 /9.0 → 0..1)
# =============================================================================

import numpy as np
from collections import deque
from config import ML_HISTORY_TICKS, ML_FEATURES, ML_MIN_HISTORY_TICKS
from road_geometry import RG

# Map RState strings to integers
_STATE_ENC = {
    "DRIVING":0, "OVERTAKING":1, "LANE_CHANGE":2, "BRAKING":3,
    "YIELDING":4, "BROKEN_DOWN":5, "PLATOONING":6, "SCHOOL_ZONE":7,
    "SHOULDER":8, "DONE":9,
}
_SPEED_NORM = 13.4 * 2.0   # normalise against 2× speed limit


def _encode_snapshot_entry(entry: dict) -> np.ndarray:
    """Convert one car's snapshot dict → normalised feature vector (5,)."""
    pos     = entry.get("road_pos_m", 0.0)  / RG.ROAD_LENGTH_M
    spd     = entry.get("speed_kmh", 0.0) / 3.6 / _SPEED_NORM
    lane    = entry.get("lane", 0) / 2.0
    lat     = (entry.get("lateral_offset_m", 0.0) / RG.LANE_WIDTH_M) + 0.5
    state   = _STATE_ENC.get(entry.get("state", "DRIVING"), 0) / 9.0
    return np.array([
        np.clip(pos,  0.0, 1.0),
        np.clip(spd,  0.0, 1.0),
        np.clip(lane, 0.0, 1.0),
        np.clip(lat,  0.0, 1.0),
        np.clip(state,0.0, 1.0),
    ], dtype=np.float32)


class TrajectoryBuffer:
    """
    Maintains last HISTORY_TICKS observations for every car seen in the
    snapshot. Called by SmartNPC.tick() before any driving logic runs.
    """

    def __init__(self, history_len: int = ML_HISTORY_TICKS):
        self._history_len = history_len
        # car_id → deque of ndarray(5,)
        self._buffers: dict[str, deque] = {}
        # car_id → tick count (how many ticks we have seen this car)
        self._counts:  dict[str, int]   = {}

    def update(self, snapshot: list):
        """
        Ingest one tick's snapshot list.
        snapshot: list of dicts with car_id, road_pos_m, speed_kmh, lane,
                  lateral_offset_m, state.
        """
        seen = set()
        for entry in snapshot:
            cid = entry.get("car_id")
            if not cid:
                continue
            seen.add(cid)
            if cid not in self._buffers:
                self._buffers[cid] = deque(maxlen=self._history_len)
                self._counts[cid]  = 0
            self._buffers[cid].append(_encode_snapshot_entry(entry))
            self._counts[cid] += 1

        # Remove cars no longer in snapshot (drove away / done)
        gone = set(self._buffers.keys()) - seen
        for cid in gone:
            del self._buffers[cid]
            self._counts.pop(cid, None)

    def get(self, car_id: str) -> np.ndarray | None:
        """
        Return history array shape (HISTORY_TICKS, FEATURES) for car_id,
        or None if we don't have ML_MIN_HISTORY_TICKS ticks yet.
        Pads with zeros at the front if we have some but not full history.
        """
        buf = self._buffers.get(car_id)
        if buf is None:
            return None
        count = self._counts.get(car_id, 0)
        if count < ML_MIN_HISTORY_TICKS:
            return None
        arr = np.array(buf, dtype=np.float32)          # (n, 5), n ≤ history_len
        if len(arr) < self._history_len:
            pad = np.zeros((self._history_len - len(arr), ML_FEATURES), dtype=np.float32)
            arr = np.vstack([pad, arr])
        return arr   # shape (40, 5)

    def has_history(self, car_id: str) -> bool:
        return self._counts.get(car_id, 0) >= ML_MIN_HISTORY_TICKS

    def car_count(self) -> int:
        return len(self._buffers)

    def clear(self):
        self._buffers.clear()
        self._counts.clear()
