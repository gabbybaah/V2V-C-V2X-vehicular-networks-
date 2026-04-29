# =============================================================================
# handshake_v6/intersection_trainer.py  —  Train intersection ML models
#
# M1 + M2:  Trains IntersectionPredictor (queue clearance GRU) from NPZ data.
# M3:       Trains AnomalyScorer (autoencoder) on *normal* behaviour only.
#
# Called automatically from intersection_extractor.run() after each run.
# Atomic model writes — safe if sim is running concurrently.
# =============================================================================
import os, glob, logging, time, threading
import numpy as np

log = logging.getLogger("v6.int_trainer")

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "logs", "training")
MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
INT_MODEL_PATH     = os.path.join(MODEL_DIR, "int_predictor.npz")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_scorer.npz")

BATCH      = 32
EPOCHS_COLD = 10
EPOCHS_FINE = 4
LR_COLD     = 8e-4
LR_FINE     = 2e-4


# ── Adam state ────────────────────────────────────────────────────────────────

class _Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self._m, self._v, self._t = {}, {}, 0

    def step(self, params: dict, grads: dict) -> dict:
        self._t += 1
        updated = {}
        for k, g in grads.items():
            if k not in self._m:
                self._m[k] = np.zeros_like(g)
                self._v[k] = np.zeros_like(g)
            self._m[k] = self.b1 * self._m[k] + (1 - self.b1) * g
            self._v[k] = self.b2 * self._v[k] + (1 - self.b2) * g * g
            mh = self._m[k] / (1 - self.b1 ** self._t)
            vh = self._v[k] / (1 - self.b2 ** self._t)
            updated[k] = params[k] - self.lr * mh / (np.sqrt(vh) + self.eps)
        return updated


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
def _tanh(x):    return np.tanh(np.clip(x, -15, 15))
def _relu(x):    return np.maximum(0, x)


# ── Minimal GRU forward for training (numpy, batch) ──────────────────────────

def _gru_forward_batch(x_batch, params, h_size):
    """
    x_batch: (B, T, F)
    params: Wr, Ur, br, Wz, Uz, bz, Wn, Un, bn
    Returns: final hidden (B, h_size)
    """
    B, T, F = x_batch.shape
    h = np.zeros((B, h_size), dtype=np.float32)
    for t in range(T):
        x = x_batch[:, t, :]                                     # (B, F)
        r = _sigmoid(x @ params['Wr'].T + h @ params['Ur'].T + params['br'])
        z = _sigmoid(x @ params['Wz'].T + h @ params['Uz'].T + params['bz'])
        n = _tanh(   x @ params['Wn'].T + (r * h) @ params['Un'].T + params['bn'])
        h = (1 - z) * n + z * h
    return h


def _gru_params_init(in_size, h_size, rng):
    s = np.sqrt(2.0 / (in_size + h_size))
    p = {}
    for name, (rows, cols) in [('Wr',(h_size,in_size)), ('Ur',(h_size,h_size)),
                                 ('Wz',(h_size,in_size)), ('Uz',(h_size,h_size)),
                                 ('Wn',(h_size,in_size)), ('Un',(h_size,h_size))]:
        p[name] = rng.normal(0, s, (rows, cols)).astype(np.float32)
    for name in ('br','bz','bn'):
        p[name] = np.zeros(h_size, dtype=np.float32)
    return p


# ── Queue Clearance Predictor training (M2) ──────────────────────────────────

def _train_clearance_predictor(X, y, is_cold: bool):
    """
    X: (N, 30, 6)  y: (N, 4)
    y[:,0] = ticks_to_clear_norm  ← primary regression target
    y[:,1] = near_miss_occurred   ← binary
    y[:,3] = congestion_end       ← regression
    """
    rng  = np.random.default_rng(0)
    H1, H2 = 48, 24

    gru1 = _gru_params_init(6,  H1, rng)
    gru2 = _gru_params_init(H1, H2, rng)
    # Output head: [clearance_norm, near_miss_prob, congestion]
    s = np.sqrt(2.0 / (H2 + 3))
    Wo = rng.normal(0, s, (3, H2)).astype(np.float32)
    bo = np.zeros(3, dtype=np.float32)

    # Load existing weights if fine-tuning
    if not is_cold and os.path.exists(INT_MODEL_PATH):
        try:
            d = dict(np.load(INT_MODEL_PATH, allow_pickle=True))
            for k in gru1: gru1[k] = d[f"g1_{k}"]
            for k in gru2: gru2[k] = d[f"g2_{k}"]
            Wo, bo = d["Wo"], d["bo"]
        except Exception as e:
            log.warning(f"Could not load int model for fine-tune: {e}")

    epochs = EPOCHS_COLD if is_cold else EPOCHS_FINE
    lr     = LR_COLD     if is_cold else LR_FINE

    opt_g1 = _Adam(lr); opt_g2 = _Adam(lr); opt_out = _Adam(lr)
    N = len(X)
    best_loss = float('inf')
    best_params = None

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        Xs, ys = X[idx], y[idx]
        losses = []
        for s0 in range(0, N, BATCH):
            Xb = Xs[s0:s0+BATCH]                 # (B, 30, 6)
            yb = ys[s0:s0+BATCH]                  # (B, 4)
            B  = len(Xb)

            # Forward
            h1  = _gru_forward_batch(Xb, gru1, H1)          # (B, H1)
            h1e = h1[:, None, :]                             # (B, 1, H1)
            h2  = _gru_forward_batch(h1e, gru2, H2)          # (B, H2)
            out = h2 @ Wo.T + bo                             # (B, 3)

            # Predictions
            pred_clear = _sigmoid(out[:, 0])                 # (B,)
            pred_nm    = _sigmoid(out[:, 1])
            pred_cong  = _sigmoid(out[:, 2])

            tgt_clear  = yb[:, 0]
            tgt_nm     = yb[:, 1]
            tgt_cong   = yb[:, 3]

            # MSE + BCE
            loss_c = np.mean((pred_clear - tgt_clear) ** 2)
            loss_n = -np.mean(tgt_nm * np.log(pred_nm + 1e-7) +
                              (1 - tgt_nm) * np.log(1 - pred_nm + 1e-7))
            loss_g = np.mean((pred_cong - tgt_cong) ** 2)
            loss   = loss_c + 0.5 * loss_n + 0.3 * loss_g
            losses.append(float(loss))

            # Backward (output layer only — frozen GRU numerical grads too slow)
            # Gradient w.r.t. out (pre-activation)
            d_clear = 2 * (pred_clear - tgt_clear) * pred_clear * (1 - pred_clear) / B
            d_nm    = 0.5 * ((pred_nm - tgt_nm) / (pred_nm * (1 - pred_nm) + 1e-7) *
                              pred_nm * (1 - pred_nm)) / B
            d_cong  = 0.6 * (pred_cong - tgt_cong) * pred_cong * (1 - pred_cong) / B
            d_out = np.stack([d_clear, d_nm, d_cong], axis=1)   # (B, 3)

            g_Wo = d_out.T @ h2                                  # (3, H2)
            g_bo = d_out.sum(axis=0)                             # (3,)

            grads_out = {'Wo': g_Wo, 'bo': g_bo}
            new_out   = opt_out.step({'Wo': Wo, 'bo': bo}, grads_out)
            Wo, bo    = new_out['Wo'], new_out['bo']

        epoch_loss = float(np.mean(losses))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_params = {'gru1': {k: v.copy() for k, v in gru1.items()},
                           'gru2': {k: v.copy() for k, v in gru2.items()},
                           'Wo': Wo.copy(), 'bo': bo.copy()}
        log.info(f"  Clearance epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}")

    return best_params, best_loss


# ── Anomaly Scorer training (M3) ─────────────────────────────────────────────

def _train_anomaly_scorer(X, y):
    """
    Train an autoencoder on *normal* examples only (near_miss=0, rogue=0).
    X: (N, 30, 6)  y: (N, 4)
    Normal = y[:,1]==0 (no near miss) and y[:,2]==0 (no rogue).
    At inference: reconstruction error → anomaly score.
    """
    # Only train on normal examples
    normal_mask = (y[:, 1] < 0.5) & (y[:, 2] < 0.5)
    Xn = X[normal_mask]
    if len(Xn) < 20:
        log.info("  Not enough normal examples for anomaly scorer — skipping")
        return None

    # Flatten per-example: (N, 30*6=180) then encode → 32 → decode → 180
    N     = len(Xn)
    Xflat = Xn.reshape(N, -1).astype(np.float32)   # (N, 180)
    D_in, D_h = 180, 32

    rng  = np.random.default_rng(1)
    s    = np.sqrt(2.0 / (D_in + D_h))
    We   = rng.normal(0, s, (D_h,  D_in)).astype(np.float32)
    be   = np.zeros(D_h,  dtype=np.float32)
    Wd   = rng.normal(0, s, (D_in, D_h)).astype(np.float32)
    bd   = np.zeros(D_in, dtype=np.float32)

    if os.path.exists(ANOMALY_MODEL_PATH):
        try:
            d  = dict(np.load(ANOMALY_MODEL_PATH))
            We, be, Wd, bd = d['We'], d['be'], d['Wd'], d['bd']
        except Exception:
            pass

    opt = _Adam(LR_FINE)
    for epoch in range(6):
        idx = np.random.permutation(N)
        Xs  = Xflat[idx]
        losses = []
        for s0 in range(0, N, BATCH):
            Xb  = Xs[s0:s0+BATCH]               # (B, 180)
            B   = len(Xb)
            enc = _relu(Xb @ We.T + be)          # (B, 32)
            dec = _sigmoid(enc @ Wd.T + bd)      # (B, 180)
            loss = np.mean((dec - Xb) ** 2)
            losses.append(float(loss))

            # Backward
            d_dec = 2 * (dec - Xb) * dec * (1 - dec) / B   # (B, 180)
            g_Wd  = d_dec.T @ enc
            g_bd  = d_dec.sum(0)
            d_enc = (d_dec @ Wd) * (enc > 0)                # relu grad
            g_We  = d_enc.T @ Xb
            g_be  = d_enc.sum(0)

            up = opt.step({'We':We,'be':be,'Wd':Wd,'bd':bd},
                          {'We':g_We,'be':g_be,'Wd':g_Wd,'bd':g_bd})
            We, be, Wd, bd = up['We'], up['be'], up['Wd'], up['bd']

        log.info(f"  Anomaly epoch {epoch+1}/6  recon_loss={np.mean(losses):.5f}")

    # Compute reconstruction error stats on normal data (for normalisation)
    enc  = _relu(Xflat @ We.T + be)
    dec  = _sigmoid(enc @ Wd.T + bd)
    errs = np.mean((dec - Xflat) ** 2, axis=1)
    return {'We': We, 'be': be, 'Wd': Wd, 'bd': bd,
            'err_mean': float(errs.mean()), 'err_std': float(errs.std() + 1e-8)}


# ── Entry point ───────────────────────────────────────────────────────────────

def train_intersection_models(async_mode: bool = True, callback=None):
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
        log.warning(f"Intersection training failed: {e}", exc_info=True)
        if callback: callback(f"Training error: {e}")


def _do_train():
    from intersection_extractor import load_all_npz
    X, y, n_files = load_all_npz()
    if X is None or len(X) < 30:
        return f"M1: Not enough intersection data yet ({0 if X is None else len(X)} examples)"

    os.makedirs(MODEL_DIR, exist_ok=True)
    is_cold = not os.path.exists(INT_MODEL_PATH)
    t0 = time.time()

    # M2: train clearance predictor
    log.info(f"M2: Training clearance predictor ({len(X)} examples, {'cold' if is_cold else 'fine-tune'})")
    params, best_loss = _train_clearance_predictor(X, y, is_cold)

    # Save clearance predictor atomically
    tmp = INT_MODEL_PATH + ".tmp.npz"
    save_d = {}
    for k, v in params['gru1'].items(): save_d[f"g1_{k}"] = v
    for k, v in params['gru2'].items(): save_d[f"g2_{k}"] = v
    save_d['Wo'] = params['Wo']
    save_d['bo'] = params['bo']
    np.savez_compressed(tmp, **save_d)
    if os.path.exists(INT_MODEL_PATH): os.remove(INT_MODEL_PATH)
    os.rename(tmp, INT_MODEL_PATH)

    # M3: train anomaly scorer
    log.info("M3: Training anomaly scorer on normal examples")
    anom_params = _train_anomaly_scorer(X, y)
    if anom_params:
        tmp2 = ANOMALY_MODEL_PATH + ".tmp.npz"
        np.savez_compressed(tmp2, **anom_params)
        if os.path.exists(ANOMALY_MODEL_PATH): os.remove(ANOMALY_MODEL_PATH)
        os.rename(tmp2, ANOMALY_MODEL_PATH)

    elapsed = time.time() - t0
    msg = (f"🧠 Intersection ML trained — {len(X)} examples  {n_files} runs  "
           f"loss={best_loss:.4f}  {elapsed:.1f}s  "
           f"anomaly={'✓' if anom_params else 'skip (need more normal data)'}")
    log.info(msg)
    print(f"\n  {msg}")
    return msg
