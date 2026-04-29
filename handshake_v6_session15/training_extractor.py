# =============================================================================
# handshake_v5/training_extractor.py  —  JSON run log → .npz training data
#
# Called automatically from sim_logger.finish() in a background thread.
# Reads the JSON event log, reconstructs per-car trajectories from
# snapshot data stored every 5s, and produces sliding-window examples.
#
# Output shape per run:
#   X : (N, 40, 5)   — input sequences (normalised features)
#   y : (N, 4)        — targets: [norm_pos, norm_speed, action_idx, is_moving]
#   meta : (N, 1)     — profile index (0-5) for each example
#
# Saved to: logs/training/<sim_type>_<run_id>.npz
# =============================================================================

import os, json, logging, threading
import numpy as np

log = logging.getLogger("v5.training_extractor")

HISTORY_LEN = 40   # ticks — must match ML_HISTORY_TICKS
PREDICT_LEN = 20   # ticks ahead — 2 seconds at 10 ticks/sec
SNAP_TICK_INTERVAL = 5.0   # sim_logger snapshots every 5s → ~50 ticks between snaps
# We reconstruct tick-level from 5s snapshots by linear interpolation
# This gives approximate but sufficient training signal.

TRAINING_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "training")

_SPEED_NORM = 13.4 * 2.0
from road_geometry import RG

_PROFILE_IDX = {
    "COMMUTER":0,"TAILGATER":1,"HESITANT":2,
    "LATE_BRAKER":3,"AGGR_OT":4,"WANDERER":5,
}

_STATE_ENC = {
    "DRIVING":0,"OVERTAKING":1,"LANE_CHANGE":2,"BRAKING":3,
    "YIELDING":4,"BROKEN_DOWN":5,"PLATOONING":6,"SCHOOL_ZONE":7,
    "SHOULDER":8,"DONE":9,
}

_ACTION_MAP = {
    "continue":0,"soft_brake":1,"hard_brake":2,
    "lane_left":3,"lane_right":4,"stop":5,
}


def _norm_features(snap_entry: dict) -> np.ndarray:
    pos   = snap_entry.get("road_pos_m",0.0) / RG.ROAD_LENGTH_M
    spd   = snap_entry.get("speed_kmh",0.0) / 3.6 / _SPEED_NORM
    lane  = snap_entry.get("lane",0) / 2.0
    lat   = (snap_entry.get("lateral_offset_m",0.0) / RG.LANE_WIDTH_M) + 0.5
    state = _STATE_ENC.get(snap_entry.get("state","DRIVING"),0) / 9.0
    return np.array([
        np.clip(pos,0,1), np.clip(spd,0,1), np.clip(lane,0,1),
        np.clip(lat,0,1), np.clip(state,0,1)
    ], dtype=np.float32)


def _infer_action(prev: dict, curr: dict) -> int:
    """Infer what action happened between two snapshots."""
    state_now = curr.get("state","DRIVING")
    state_pre = prev.get("state","DRIVING")
    spd_now  = curr.get("speed_kmh",0.0) / 3.6
    spd_pre  = prev.get("speed_kmh",0.0) / 3.6
    lane_now = curr.get("lane",0)
    lane_pre = prev.get("lane",0)
    if state_now == "DONE" or spd_now < 0.3:
        return 5  # stop
    if lane_now < lane_pre:
        return 4  # lane_right (lower index = right in our setup)
    if lane_now > lane_pre:
        return 3  # lane_left
    delta_spd = spd_now - spd_pre
    if delta_spd < -2.5:
        return 2  # hard_brake
    if delta_spd < -0.5:
        return 1  # soft_brake
    return 0       # continue


def _infer_action_from_ticks(prev: dict, curr: dict) -> int:
    """Infer action from compact tick dicts (pos/spd/lane/lat/st keys)."""
    st_now  = curr.get("st", "DRIVING")
    spd_now = curr.get("spd", 0.0) / 3.6
    spd_pre = prev.get("spd", 0.0) / 3.6
    lane_now = curr.get("lane", 0)
    lane_pre = prev.get("lane", 0)
    if st_now == "DONE" or spd_now < 0.3:
        return 5
    if lane_now < lane_pre:
        return 4
    if lane_now > lane_pre:
        return 3
    delta = spd_now - spd_pre
    if delta < -2.5:
        return 2
    if delta < -0.5:
        return 1
    return 0


def process(json_path: str, async_mode: bool = True):
    """
    Extract training data from a JSON run log.
    If async_mode=True, runs in a background thread (non-blocking).
    """
    if async_mode:
        t = threading.Thread(target=_process_sync, args=(json_path,), daemon=True)
        t.start()
    else:
        _process_sync(json_path)


def _process_sync(json_path: str):
    try:
        _do_extract(json_path)
    except Exception as e:
        log.warning(f"Training extractor failed: {e}")


def _do_extract(json_path: str):
    if not os.path.exists(json_path):
        log.warning(f"JSON log not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    run_id   = data.get("meta", {}).get("run_id", "unknown")
    sim_type = data.get("meta", {}).get("sim_type", "road")

    if sim_type != "road":
        return

    # V5: use car_trajectories dict written by sim_logger
    car_trajectories = data.get("car_trajectories", {})
    if not car_trajectories:
        log.info(f"No car_trajectories in {run_id} — run may be too short or old format")
        return

    X_list    = []
    y_list    = []
    meta_list = []

    for label, car_data in car_trajectories.items():
        profile  = car_data.get("profile", "COMMUTER")
        prof_idx = _PROFILE_IDX.get(profile, 0)
        ticks    = car_data.get("ticks", [])

        if len(ticks) < HISTORY_LEN + PREDICT_LEN + 2:
            continue  # not enough data

        # Build normalised feature vectors
        feats = []
        for t in ticks:
            pos   = t.get("pos", 0.0) / RG.ROAD_LENGTH_M
            spd   = t.get("spd", 0.0) / 3.6 / _SPEED_NORM
            lane  = t.get("lane", 0) / 2.0
            lat   = (t.get("lat", 0.0) / RG.LANE_WIDTH_M) + 0.5
            state = _STATE_ENC.get(t.get("st", "DRIVING"), 0) / 9.0
            feats.append(np.array([
                np.clip(pos,  0.0, 1.0),
                np.clip(spd,  0.0, 1.0),
                np.clip(lane, 0.0, 1.0),
                np.clip(lat,  0.0, 1.0),
                np.clip(state,0.0, 1.0),
            ], dtype=np.float32))

        n    = len(feats)
        step = max(1, PREDICT_LEN // 2)

        for i in range(0, n - HISTORY_LEN - PREDICT_LEN, step):
            x_window   = np.array(feats[i : i + HISTORY_LEN], dtype=np.float32)
            target_idx = i + HISTORY_LEN + PREDICT_LEN - 1
            prev_t     = ticks[i + HISTORY_LEN - 1]
            target_t   = ticks[target_idx]

            tgt_pos = float(target_t.get("pos", 0.0)) / RG.ROAD_LENGTH_M
            tgt_spd = float(target_t.get("spd", 0.0)) / 3.6 / _SPEED_NORM
            action_idx = _infer_action_from_ticks(prev_t, target_t)
            is_moving  = 0.0 if target_t.get("st", "DRIVING") == "DONE" else 1.0

            y_vec = np.array([
                np.clip(tgt_pos, 0, 1),
                np.clip(tgt_spd, 0, 1),
                float(action_idx),
                is_moving,
            ], dtype=np.float32)

            X_list.append(x_window)
            y_list.append(y_vec)
            meta_list.append(np.array([float(prof_idx)], dtype=np.float32))

    if not X_list:
        log.info(f"No training examples extracted from {run_id} (cars too short)")
        return

    X    = np.stack(X_list)
    y    = np.stack(y_list)
    meta = np.stack(meta_list)

    os.makedirs(TRAINING_DIR, exist_ok=True)
    out_path = os.path.join(TRAINING_DIR, f"{sim_type}_{run_id}.npz")
    np.savez_compressed(out_path, X=X, y=y, meta=meta)
    log.info(f"Saved {len(X)} training examples → {out_path}")
    print(f"  🎓 Training data: {len(X)} examples → {os.path.basename(out_path)}")
