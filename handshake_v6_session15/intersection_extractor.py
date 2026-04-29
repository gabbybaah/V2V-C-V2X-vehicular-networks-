# =============================================================================
# handshake_v6/intersection_extractor.py  —  Intersection logs → NPZ training data
#
# M1: reads intersection JSON log files, builds sliding-window feature arrays
# per arm per run, saves to logs/training/intersection_<run_id>.npz
#
# Feature vector per tick (6 features):
#   0  queued_norm       queued_count / MAX_QUEUE
#   1  on_road_norm      on_road_count / 4
#   2  crossed_pct       done / total_cars (arm)
#   3  throughput_norm   throughput_rate / 60.0
#   4  is_green          1.0 if arm is green else 0.0
#   5  near_miss_cum     cumulative near_miss / 10 (capped)
#
# Sequence length: 30 ticks  (3 seconds at 10Hz)
#
# Labels per example (4 values):
#   0  ticks_to_clear_norm   ticks until arm queue empty / 300
#   1  near_miss_occurred    1.0 if a near_miss happens in next 30 ticks
#   2  rogue_present         1.0 if rogue car logged in run
#   3  congestion_pct        queued_norm at end of window
#
# Called from main_intersection.py after each run.
# =============================================================================
import os, json, glob, logging, threading
import numpy as np

log = logging.getLogger("v6.int_extractor")

HISTORY_LEN    = 30
MAX_QUEUE      = 40.0   # RUSH_HOUR_CARS max
TICKS_NORM     = 300.0  # normalise ticks_to_clear
TRAINING_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "logs", "training")
LOG_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "logs")


def _find_intersection_logs():
    return sorted(glob.glob(os.path.join(LOG_DIR, "intersection_*.json")))


def _arm_snapshots_from_log(run_data: dict) -> dict:
    """
    Returns {arm_name: [list of feature vecs over time]} from run snapshot list.
    """
    snapshots = run_data.get("snapshots", [])
    arms_data = {}

    # Count total near-misses across the run
    events = run_data.get("events", [])
    near_miss_times = [e.get("ts", 0) for e in events
                       if "NEAR_MISS" in e.get("event", "")]
    rogue_flag = 1.0 if any("ROGUE" in e.get("event", "") for e in events) else 0.0

    start_ts = run_data.get("start_ts", 0.0)
    nm_cumulative = 0

    for snap in snapshots:
        arms = snap.get("arms", {})
        snap_ts = snap.get("_log_ts", 0.0)
        # count near-misses up to this snapshot
        nm_cumulative = sum(1 for t in near_miss_times if t <= snap_ts)

        for arm_name, arm_data in arms.items():
            if arm_name not in arms_data:
                arms_data[arm_name] = {"ticks": [], "rogue": rogue_flag}

            total = arm_data.get("total", max(arm_data.get("queued", 0) +
                                               arm_data.get("on_road", 0) +
                                               arm_data.get("done", 0), 1))
            feat = np.array([
                min(arm_data.get("queued", 0) / MAX_QUEUE, 1.0),
                min(arm_data.get("on_road", 0) / 4.0, 1.0),
                arm_data.get("done", 0) / max(total, 1),
                min(arm_data.get("throughput_rate", 0) / 60.0, 1.0),
                1.0 if arm_name in snap.get("light_green", []) else 0.0,
                min(nm_cumulative / 10.0, 1.0),
            ], dtype=np.float32)
            arms_data[arm_name]["ticks"].append(feat)

    return arms_data


def _build_examples_for_arm(ticks: list, rogue: float):
    """Sliding window over ticks list → (X, y) arrays."""
    n = len(ticks)
    if n < HISTORY_LEN + 5:
        return None, None

    X_list, y_list = [], []
    for i in range(n - HISTORY_LEN - 1):
        window = ticks[i: i + HISTORY_LEN]
        future = ticks[i + HISTORY_LEN: min(i + HISTORY_LEN + 30, n)]

        X = np.stack(window)  # (30, 6)

        # Label 0: ticks until queued drops to 0 (from window end)
        queued_now = window[-1][0]  # normalized queued at window end
        ticks_to_clear = 0.0
        for j, f in enumerate(future):
            if f[0] <= 0.05:  # <5% = cleared
                ticks_to_clear = j / TICKS_NORM
                break
        else:
            ticks_to_clear = len(future) / TICKS_NORM  # not cleared in window

        # Label 1: near-miss in next 30 ticks
        near_miss_occurred = 1.0 if any(f[5] > window[-1][5] for f in future) else 0.0

        # Label 2: rogue present in run
        # Label 3: congestion at end of future window
        congestion_end = future[-1][0] if future else queued_now

        y = np.array([ticks_to_clear, near_miss_occurred, rogue, congestion_end],
                     dtype=np.float32)
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        return None, None
    return np.stack(X_list), np.stack(y_list)


def extract_run(log_path: str) -> tuple:
    """Extract training data from one run log. Returns (X, y, n_examples)."""
    try:
        with open(log_path, "r") as f:
            run_data = json.load(f)
    except Exception as e:
        log.warning(f"Could not read {log_path}: {e}")
        return None, None, 0

    if run_data.get("sim_type") != "intersection":
        return None, None, 0

    arms_data = _arm_snapshots_from_log(run_data)

    all_X, all_y = [], []
    for arm_name, arm_info in arms_data.items():
        ticks  = arm_info["ticks"]
        rogue  = arm_info["rogue"]
        X, y   = _build_examples_for_arm(ticks, rogue)
        if X is not None:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return None, None, 0

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    return X_all, y_all, len(X_all)


def run(run_id: str = None, async_mode: bool = True):
    """Entry point from main_intersection.py after each run."""
    if async_mode:
        t = threading.Thread(target=_run_sync, args=(run_id,), daemon=True)
        t.start()
    else:
        _run_sync(run_id)


def _run_sync(run_id: str = None):
    try:
        os.makedirs(TRAINING_DIR, exist_ok=True)

        if run_id:
            log_files = [os.path.join(LOG_DIR, f"intersection_{run_id}.json")]
            log_files = [f for f in log_files if os.path.exists(f)]
        else:
            log_files = _find_intersection_logs()

        if not log_files:
            log.info("No intersection logs to extract")
            return

        # Only process new logs (not already extracted)
        existing = set(os.path.splitext(os.path.basename(f))[0]
                       for f in glob.glob(os.path.join(TRAINING_DIR, "intersection_*.npz")))

        new_count = 0
        for lf in log_files:
            base = os.path.splitext(os.path.basename(lf))[0]
            if base in existing:
                continue
            X, y, n = extract_run(lf)
            if X is not None and n > 0:
                out = os.path.join(TRAINING_DIR, f"{base}.npz")
                np.savez_compressed(out, X=X, y=y)
                log.info(f"Extracted {n} examples from {base} → {out}")
                new_count += 1

        if new_count:
            log.info(f"M1: Extracted {new_count} intersection run(s)")
            # Trigger model training
            from intersection_trainer import train_intersection_models
            train_intersection_models(async_mode=True)
    except Exception as e:
        log.warning(f"intersection_extractor error: {e}")


def load_all_npz():
    """Load all intersection NPZ files → (X, y) for training."""
    files = sorted(glob.glob(os.path.join(TRAINING_DIR, "intersection_*.npz")))
    if not files:
        return None, None, 0

    all_X, all_y = [], []
    for f in files:
        try:
            d = np.load(f)
            all_X.append(d["X"])
            all_y.append(d["y"])
        except Exception:
            pass

    if not all_X:
        return None, None, 0

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y, len(files)
