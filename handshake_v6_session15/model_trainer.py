# =============================================================================
# handshake_v5/model_trainer.py  —  Incremental GRU training from .npz files
#
# Loads all logs/training/*.npz, concatenates, trains/fine-tunes the predictor.
# Run 0 (no .npz yet)   → skips, prints cold-start notice
# Run 1                  → trains from scratch  lr=1e-3
# Run 2+                 → fine-tunes           lr=2e-4 (don't forget earlier runs)
#
# Atomic write: model written to .tmp then renamed — safe if sim is running.
# Called from main_road.py in a background thread after each run ends.
# =============================================================================

import os, glob, logging, time, threading
import numpy as np

log = logging.getLogger("v5.model_trainer")

TRAINING_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "training")

MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "predictor.pt")

BATCH_SIZE = 32
EPOCHS_COLD = 8
EPOCHS_FINE = 3
LR_COLD     = 1e-3
LR_FINE     = 2e-4

_ACTION_NAMES = ["continue","soft_brake","hard_brake","lane_left","lane_right","stop"]


# ── Load all training data ────────────────────────────────────────────────────

def _load_all_npz():
    """Returns (X, y, meta) concatenated from all .npz files, or None."""
    files = sorted(glob.glob(os.path.join(TRAINING_DIR, "*.npz")))
    if not files:
        return None, None, None, 0

    X_all, y_all, meta_all = [], [], []
    for fpath in files:
        try:
            d = np.load(fpath)
            X_all.append(d["X"])
            y_all.append(d["y"])
            if "meta" in d:
                meta_all.append(d["meta"])
        except Exception as e:
            log.warning(f"Could not load {fpath}: {e}")

    if not X_all:
        return None, None, None, 0

    X    = np.concatenate(X_all, axis=0).astype(np.float32)
    y    = np.concatenate(y_all, axis=0).astype(np.float32)
    meta = np.concatenate(meta_all, axis=0) if meta_all else None
    return X, y, meta, len(files)


# ── One training step ─────────────────────────────────────────────────────────

def _train_batch(predictor, X_batch, y_batch, lr):
    """Train on one batch. Returns mean loss."""
    losses = []
    for i in range(len(X_batch)):
        history = X_batch[i]          # (40, 5)
        targets = {
            "pos":       float(y_batch[i, 0]),
            "speed":     float(y_batch[i, 1]),
            "action_idx":int(y_batch[i, 2]),
            "is_moving": float(y_batch[i, 3]),
        }
        outputs = predictor.forward_train(history)
        loss    = predictor.backward_train(history, targets, outputs, lr=lr)
        losses.append(loss)
    return float(np.mean(losses))


# ── Main training function ────────────────────────────────────────────────────

def train_from_logs(async_mode: bool = True, callback=None):
    """
    Entry point called from main_road.py after each run.
    async_mode=True → runs in background thread, non-blocking.
    callback(msg) → called with status string when done.
    """
    if async_mode:
        t = threading.Thread(target=_train_sync, args=(callback,), daemon=True)
        t.start()
    else:
        _train_sync(callback)


def _train_sync(callback=None):
    try:
        msg = _do_train()
        if callback: callback(msg)
    except Exception as e:
        log.warning(f"Training failed: {e}")
        if callback: callback(f"Training error: {e}")


def _do_train():
    X, y, meta, n_files = _load_all_npz()

    if X is None or len(X) == 0:
        return "No training data yet — cold start remains active"

    from ml_predictor import Predictor
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Decide: cold train or fine-tune
    is_cold = not os.path.exists(MODEL_PATH)
    predictor = Predictor(MODEL_PATH)

    epochs = EPOCHS_COLD if is_cold else EPOCHS_FINE
    lr     = LR_COLD     if is_cold else LR_FINE
    mode   = "training" if is_cold else "fine-tuning"

    n_examples = len(X)
    t0 = time.time()

    # Shuffle once
    idx = np.random.permutation(n_examples)
    X, y = X[idx], y[idx]

    # Action distribution for logging
    action_counts = np.bincount(y[:, 2].astype(int), minlength=6)

    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_losses = []
        for start in range(0, n_examples, BATCH_SIZE):
            end     = min(start + BATCH_SIZE, n_examples)
            Xb, yb  = X[start:end], y[start:end]
            loss    = _train_batch(predictor, Xb, yb, lr)
            epoch_losses.append(loss)
        mean_loss = float(np.mean(epoch_losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
        log.info(f"Epoch {epoch+1}/{epochs}  loss={mean_loss:.4f}")

    elapsed = time.time() - t0

    # Atomic save
    tmp_path = MODEL_PATH + ".tmp"
    predictor.save(tmp_path)
    # save() writes tmp_path already, rename to final
    if os.path.exists(tmp_path):
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        os.rename(tmp_path, MODEL_PATH)
    elif os.path.exists(tmp_path + ".npz"):
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        os.rename(tmp_path + ".npz", MODEL_PATH)

    # Build action distribution string
    dist = "  ".join(f"{_ACTION_NAMES[i]}={action_counts[i]}"
                     for i in range(6) if action_counts[i] > 0)

    params = predictor.param_count()
    msg = (f"🧠 Model {mode} complete — "
           f"{n_examples} examples  {n_files} runs  "
           f"loss={best_loss:.4f}  {elapsed:.1f}s  "
           f"params={params:,}\n"
           f"   Actions: {dist}")
    print(f"\n  {msg}")
    return msg


# ── Quick eval (print confusion on held-out 10%) ─────────────────────────────

def quick_eval():
    """Run after training to show action prediction accuracy."""
    X, y, meta, n_files = _load_all_npz()
    if X is None or len(X) < 20:
        print("  Not enough data for eval")
        return

    from ml_predictor import Predictor
    if not os.path.exists(MODEL_PATH):
        print("  No model to evaluate")
        return

    predictor = Predictor(MODEL_PATH)
    n         = len(X)
    n_eval    = max(10, n // 10)
    idx_eval  = np.random.choice(n, n_eval, replace=False)

    correct = 0
    pos_err = []
    for i in idx_eval:
        pred_pos, pred_spd, act_probs, conf = predictor.predict(X[i])
        true_action = int(y[i, 2])
        pred_action = int(np.argmax(act_probs))
        if pred_action == true_action:
            correct += 1
        from road_geometry import RG
        true_pos = float(y[i, 0]) * RG.ROAD_LENGTH_M
        pos_err.append(abs(pred_pos - true_pos))

    acc     = correct / n_eval * 100
    mae_pos = float(np.mean(pos_err))
    print(f"  📊 Quick eval ({n_eval} samples): action_acc={acc:.1f}%  pos_MAE={mae_pos:.1f}m")
    return acc, mae_pos
