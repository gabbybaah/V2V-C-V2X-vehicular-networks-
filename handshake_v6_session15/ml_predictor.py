# =============================================================================
# handshake_v5/ml_predictor.py  —  Pure-numpy GRU behavioural predictor
#
# Architecture:
#   Input   : (40, 5)  — 40-tick history, 5 normalised features
#   GRU-1   : 64 hidden units
#   GRU-2   : 32 hidden units
#   Head A  : pred_pos (1), pred_speed (1)   ← continuous regression
#   Head B  : action probs (6)               ← softmax classification
#   Head C  : confidence (1)                 ← sigmoid
#
# When no weights file exists → confidence = 0.0 for every prediction.
# Smart cars fall back to current position (V4 behaviour). Safe cold start.
#
# Weights stored as numpy .npz compressed.
# =============================================================================

import os, numpy as np, logging
from config import (ML_GRU_HIDDEN, ML_GRU_HIDDEN2, ML_FEATURES,
                    ML_ACTIONS, ML_MODEL_PATH)

log = logging.getLogger("v5.ml_predictor")

# ── Activations ───────────────────────────────────────────────────────────────
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
def _tanh(x):    return np.tanh(np.clip(x, -15, 15))
def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
def _relu(x):    return np.maximum(0, x)


# ── Single GRU layer (numpy) ──────────────────────────────────────────────────
class _GRULayer:
    """Single GRU layer. forward() runs sequence, returns final hidden state."""

    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator):
        s = np.sqrt(2.0 / (input_size + hidden_size))
        # Reset gate
        self.Wr = rng.normal(0, s, (hidden_size, input_size)).astype(np.float32)
        self.Ur = rng.normal(0, s, (hidden_size, hidden_size)).astype(np.float32)
        self.br = np.zeros(hidden_size, dtype=np.float32)
        # Update gate
        self.Wz = rng.normal(0, s, (hidden_size, input_size)).astype(np.float32)
        self.Uz = rng.normal(0, s, (hidden_size, hidden_size)).astype(np.float32)
        self.bz = np.zeros(hidden_size, dtype=np.float32)
        # Candidate hidden state
        self.Wn = rng.normal(0, s, (hidden_size, input_size)).astype(np.float32)
        self.Un = rng.normal(0, s, (hidden_size, hidden_size)).astype(np.float32)
        self.bn = np.zeros(hidden_size, dtype=np.float32)
        self.hidden_size = hidden_size

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """
        x_seq : (T, input_size)
        returns: final hidden state (hidden_size,)
        """
        h = np.zeros(self.hidden_size, dtype=np.float32)
        for t in range(len(x_seq)):
            x = x_seq[t]
            r = _sigmoid(self.Wr @ x + self.Ur @ h + self.br)
            z = _sigmoid(self.Wz @ x + self.Uz @ h + self.bz)
            n = _tanh(   self.Wn @ x + r * (self.Un @ h) + self.bn)
            h = (1.0 - z) * n + z * h
        return h

    def forward_with_cache(self, x_seq: np.ndarray):
        """
        Full forward pass storing intermediate values for BPTT.
        Returns: (final_h, list_of_cache_dicts)
        """
        h = np.zeros(self.hidden_size, dtype=np.float32)
        cache = []
        for t in range(len(x_seq)):
            x   = x_seq[t]
            pre_r = self.Wr @ x + self.Ur @ h + self.br
            pre_z = self.Wz @ x + self.Uz @ h + self.bz
            r = _sigmoid(pre_r)
            z = _sigmoid(pre_z)
            pre_n = self.Wn @ x + r * (self.Un @ h) + self.bn
            n = _tanh(pre_n)
            h_new = (1.0 - z) * n + z * h
            cache.append({"x":x, "h":h, "r":r, "z":z, "n":n,
                          "pre_r":pre_r, "pre_z":pre_z, "pre_n":pre_n, "h_new":h_new})
            h = h_new
        return h, cache

    def backward(self, dh_final: np.ndarray, cache: list):
        """
        BPTT through full sequence.
        Returns: (grads_dict, dh_0, dx_0)
          dh_0  — gradient w.r.t. initial hidden state
          dx_0  — gradient w.r.t. input at t=0 (used when this layer's input = prev layer's output)
        """
        H = self.hidden_size
        dWr = np.zeros_like(self.Wr); dUr = np.zeros_like(self.Ur); dbr = np.zeros_like(self.br)
        dWz = np.zeros_like(self.Wz); dUz = np.zeros_like(self.Uz); dbz = np.zeros_like(self.bz)
        dWn = np.zeros_like(self.Wn); dUn = np.zeros_like(self.Un); dbn = np.zeros_like(self.bn)

        dh    = dh_final.copy()
        dx_0  = np.zeros(self.Wr.shape[1], dtype=np.float32)

        for t_idx, step in enumerate(reversed(cache)):
            t = len(cache) - 1 - t_idx
            x   = step["x"];  h   = step["h"]
            r   = step["r"];  z   = step["z"]
            n   = step["n"]
            pre_r = step["pre_r"]; pre_z = step["pre_z"]; pre_n = step["pre_n"]

            dz   = (h - n) * dh
            dn   = (1.0 - z) * dh
            dh_p = z * dh

            dpre_n = dn * (1.0 - n*n)
            dWn += np.outer(dpre_n, x)
            dbn += dpre_n
            dr   = (self.Un @ h) * dpre_n
            dh_p += self.Un.T @ (r * dpre_n)
            dx    = self.Wn.T @ dpre_n   # gradient w.r.t. input x from n gate

            dpre_r = dr * r * (1.0 - r)
            dWr += np.outer(dpre_r, x)
            dUr += np.outer(dpre_r, h)
            dbr += dpre_r
            dh_p += self.Ur.T @ dpre_r
            dx   += self.Wr.T @ dpre_r   # from r gate

            dpre_z = dz * z * (1.0 - z)
            dWz += np.outer(dpre_z, x)
            dUz += np.outer(dpre_z, h)
            dbz += dpre_z
            dh_p += self.Uz.T @ dpre_z
            dx   += self.Wz.T @ dpre_z   # from z gate

            if t == 0:
                dx_0 = dx   # gradient w.r.t. t=0 input (= h1 from GRU1)

            dh = dh_p

        grads = {"Wr":dWr,"Ur":dUr,"br":dbr,
                 "Wz":dWz,"Uz":dUz,"bz":dbz,
                 "Wn":dWn,"Un":dUn,"bn":dbn}
        return grads, dh, dx_0

    def apply_grads(self, grads: dict, lr: float):
        for k, v in grads.items():
            getattr(self, k)[...] -= lr * np.clip(v, -5.0, 5.0)

    def params(self) -> dict:
        return {"Wr":self.Wr,"Ur":self.Ur,"br":self.br,
                "Wz":self.Wz,"Uz":self.Uz,"bz":self.bz,
                "Wn":self.Wn,"Un":self.Un,"bn":self.bn}

    def load_params(self, d: dict):
        for k in ("Wr","Ur","br","Wz","Uz","bz","Wn","Un","bn"):
            if k in d: getattr(self, k)[...] = d[k]


# ── Output heads ──────────────────────────────────────────────────────────────
class _OutputHeads:
    """Three output heads from final hidden state (h2, size=32)."""

    def __init__(self, hidden2: int, n_actions: int, rng: np.random.Generator):
        s = np.sqrt(2.0 / hidden2)
        # Head A: position + speed regression (shared layer then split)
        self.W_reg  = rng.normal(0, s, (16, hidden2)).astype(np.float32)
        self.b_reg  = np.zeros(16, dtype=np.float32)
        self.W_pos  = rng.normal(0, s, (1, 16)).astype(np.float32)
        self.b_pos  = np.zeros(1, dtype=np.float32)
        self.W_spd  = rng.normal(0, s, (1, 16)).astype(np.float32)
        self.b_spd  = np.zeros(1, dtype=np.float32)
        # Head B: action classification
        self.W_act  = rng.normal(0, s, (n_actions, hidden2)).astype(np.float32)
        self.b_act  = np.zeros(n_actions, dtype=np.float32)
        # Head C: confidence
        self.W_conf = rng.normal(0, s, (1, hidden2)).astype(np.float32)
        self.b_conf = np.zeros(1, dtype=np.float32)

    def forward(self, h2: np.ndarray):
        reg_mid = _relu(self.W_reg @ h2 + self.b_reg)
        pred_pos = float((self.W_pos @ reg_mid + self.b_pos)[0])
        pred_spd = float((self.W_spd @ reg_mid + self.b_spd)[0])
        act_logits = self.W_act @ h2 + self.b_act
        act_probs  = _softmax(act_logits)
        conf = float(_sigmoid(self.W_conf @ h2 + self.b_conf)[0])
        return pred_pos, pred_spd, act_probs, conf, (reg_mid, act_logits)

    def backward(self, h2, reg_mid, act_logits, act_probs,
                 d_pred_pos, d_pred_spd, d_act_probs, d_conf):
        """Returns (grads, dh2)."""
        dh2 = np.zeros_like(h2)
        grads = {}

        # Head A backward
        d_pos_scalar = np.array([d_pred_pos], dtype=np.float32)
        d_spd_scalar = np.array([d_pred_spd], dtype=np.float32)
        grads["W_pos"] = np.outer(d_pos_scalar, reg_mid)
        grads["b_pos"] = d_pos_scalar
        grads["W_spd"] = np.outer(d_spd_scalar, reg_mid)
        grads["b_spd"] = d_spd_scalar
        d_reg_mid = self.W_pos.T @ d_pos_scalar + self.W_spd.T @ d_spd_scalar
        d_reg_mid *= (reg_mid > 0).astype(np.float32)  # relu backward
        grads["W_reg"] = np.outer(d_reg_mid, h2)
        grads["b_reg"] = d_reg_mid
        dh2 += self.W_reg.T @ d_reg_mid

        # Head B backward (softmax cross-entropy gradient)
        grads["W_act"] = np.outer(d_act_probs, h2)
        grads["b_act"] = d_act_probs.copy()
        dh2 += self.W_act.T @ d_act_probs

        # Head C backward
        d_conf_s = np.array([d_conf], dtype=np.float32)
        grads["W_conf"] = np.outer(d_conf_s, h2)
        grads["b_conf"] = d_conf_s
        dh2 += self.W_conf.T @ d_conf_s

        return grads, dh2

    def apply_grads(self, grads: dict, lr: float):
        for k, v in grads.items():
            getattr(self, k)[...] -= lr * np.clip(v, -5.0, 5.0)

    def params(self) -> dict:
        return {k: getattr(self, k)
                for k in ("W_reg","b_reg","W_pos","b_pos","W_spd","b_spd",
                          "W_act","b_act","W_conf","b_conf")}

    def load_params(self, d: dict):
        for k in ("W_reg","b_reg","W_pos","b_pos","W_spd","b_spd",
                  "W_act","b_act","W_conf","b_conf"):
            if k in d: getattr(self, k)[...] = d[k]


# ── Adam optimiser ────────────────────────────────────────────────────────────
class _Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.t = 0; self.m = {}; self.v = {}

    def step(self, param_name: str, param: np.ndarray, grad: np.ndarray):
        self.t += 1
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
        g = np.clip(grad, -5.0, 5.0)
        self.m[param_name] = self.beta1*self.m[param_name] + (1-self.beta1)*g
        self.v[param_name] = self.beta2*self.v[param_name] + (1-self.beta2)*(g*g)
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Main Predictor class ──────────────────────────────────────────────────────

# Action index map
ACTION_NAMES  = ["continue","soft_brake","hard_brake","lane_left","lane_right","stop"]
ACTION_CONTINUE   = 0
ACTION_SOFT_BRAKE = 1
ACTION_HARD_BRAKE = 2
ACTION_LANE_LEFT  = 3
ACTION_LANE_RIGHT = 4
ACTION_STOP       = 5


class Predictor:
    """
    Singleton-style predictor. Import and call predict(history) from SmartNPC.
    Returns (pred_pos_m, pred_speed_ms, action_probs, confidence).
    confidence=0.0 when no weights file — smart cars fall back to current pos.
    """

    def __init__(self, model_path: str = None):
        self._path   = model_path or ML_MODEL_PATH
        self._loaded = False
        self._rng    = np.random.default_rng(42)
        self._build()
        self._try_load()

    def _build(self):
        """Initialise architecture with random weights."""
        rng = self._rng
        self.gru1  = _GRULayer(ML_FEATURES,   ML_GRU_HIDDEN,  rng)
        self.gru2  = _GRULayer(ML_GRU_HIDDEN,  ML_GRU_HIDDEN2, rng)
        self.heads = _OutputHeads(ML_GRU_HIDDEN2, ML_ACTIONS, rng)

    def _try_load(self):
        """Load weights if model file exists, else cold start (conf=0)."""
        if not os.path.exists(self._path):
            log.info(f"No model at {self._path} — cold start (confidence=0)")
            return
        try:
            data = np.load(self._path, allow_pickle=True)
            d = dict(data)
            self.gru1.load_params({k[5:]: d[k] for k in d if k.startswith("gru1_")})
            self.gru2.load_params({k[5:]: d[k] for k in d if k.startswith("gru2_")})
            self.heads.load_params({k[6:]: d[k] for k in d if k.startswith("heads_")})
            self._loaded = True
            log.info(f"Model loaded from {self._path}")
        except Exception as e:
            log.warning(f"Could not load model: {e} — cold start")

    @property
    def loaded(self) -> bool:
        return self._loaded

    def predict(self, history: np.ndarray | None):
        """
        history : ndarray (40, 5) normalised or None.
        Returns : (pred_pos_m, pred_speed_ms, action_probs_6, confidence)
        When cold start or history=None → confidence=0.0, others meaningless.
        """
        if history is None or not self._loaded:
            return 0.0, 0.0, np.ones(ML_ACTIONS)/ML_ACTIONS, 0.0

        try:
            h1 = self.gru1.forward(history)          # (64,)
            h2 = self.gru2.forward(h1.reshape(1,-1)) # GRU2 sees single-step from h1
            pred_pos_n, pred_spd_n, act_probs, conf, _ = self.heads.forward(h2)

            from road_geometry import RG
            SPEED_NORM = 13.4 * 2.0
            pred_pos_m  = float(np.clip(pred_pos_n,  0, 1)) * RG.ROAD_LENGTH_M
            pred_spd_ms = float(np.clip(pred_spd_n,  0, 1)) * SPEED_NORM
            return pred_pos_m, pred_spd_ms, act_probs, conf
        except Exception as e:
            log.debug(f"Predict error: {e}")
            return 0.0, 0.0, np.ones(ML_ACTIONS)/ML_ACTIONS, 0.0

    def forward_train(self, history: np.ndarray):
        """Full forward with cache for BPTT. Returns outputs + caches."""
        h1, cache1 = self.gru1.forward_with_cache(history)
        h2, cache2 = self.gru2.forward_with_cache(h1.reshape(1, -1))
        pred_pos_n, pred_spd_n, act_probs, conf, (reg_mid, act_logits) = self.heads.forward(h2)
        return (pred_pos_n, pred_spd_n, act_probs, conf,
                h1, h2, cache1, cache2, reg_mid, act_logits)

    def backward_train(self, history, targets, outputs, lr=1e-3):
        """
        One training step. targets: dict with keys pos, speed, action_idx, is_moving.
        Returns scalar loss.
        """
        (pred_pos_n, pred_spd_n, act_probs, conf,
         h1, h2, cache1, cache2, reg_mid, act_logits) = outputs

        # ── Compute losses ────────────────────────────────────────────────────
        # Regression: MSE on normalised pos and speed
        d_pred_pos = 2.0 * (pred_pos_n - targets["pos"])
        d_pred_spd = 2.0 * (pred_spd_n - targets["speed"])
        pos_loss   = (pred_pos_n - targets["pos"])**2
        spd_loss   = (pred_spd_n - targets["speed"])**2

        # Classification: cross-entropy on action
        act_idx    = int(targets["action_idx"])
        ce_grad    = act_probs.copy()
        ce_grad[act_idx] -= 1.0   # ∂CE/∂logits = p - one_hot
        act_loss   = -np.log(act_probs[act_idx] + 1e-9)

        # Confidence: binary CE — label=1 if still moving (is_moving)
        is_moving  = float(targets.get("is_moving", 1.0))
        conf_loss  = -(is_moving*np.log(conf+1e-9) + (1-is_moving)*np.log(1-conf+1e-9))
        d_conf     = conf - is_moving   # ∂BCE/∂sigmoid_input

        total_loss = pos_loss + spd_loss + act_loss + 0.5*conf_loss

        # ── Backprop through heads ────────────────────────────────────────────
        hg, dh2 = self.heads.backward(
            h2, reg_mid, act_logits, act_probs,
            d_pred_pos, d_pred_spd, ce_grad, d_conf)
        for k, v in hg.items():
            self.heads.__dict__[k] -= lr * np.clip(v, -5.0, 5.0)

        # ── Backprop through GRU2 (single step — input was h1) ────────────────
        g2, _, dx_from_gru2 = self.gru2.backward(dh2, cache2)
        for k, v in g2.items():
            setattr(self.gru2, k, getattr(self.gru2, k) - lr * np.clip(v, -5.0, 5.0))

        # dx_from_gru2 is grad w.r.t. h1 (the output of GRU1 that fed into GRU2)
        # Pass that as dh_final into GRU1 backward
        g1, _, _ = self.gru1.backward(dx_from_gru2, cache1)
        for k, v in g1.items():
            setattr(self.gru1, k, getattr(self.gru1, k) - lr * np.clip(v, -5.0, 5.0))

        return float(total_loss)

    def save(self, path: str = None):
        """Atomic save: write to .tmp then rename."""
        path = path or self._path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        d = {}
        for k, v in self.gru1.params().items():  d[f"gru1_{k}"] = v
        for k, v in self.gru2.params().items():  d[f"gru2_{k}"] = v
        for k, v in self.heads.params().items(): d[f"heads_{k}"] = v
        np.savez_compressed(tmp, **d)
        if os.path.exists(path): os.remove(path)
        os.rename(tmp + ".npz" if not tmp.endswith(".npz") else tmp,
                  path if path.endswith(".npz") else path)
        self._loaded = True
        log.info(f"Model saved to {path}")

    def param_count(self) -> int:
        total = 0
        for p in self.gru1.params().values():  total += p.size
        for p in self.gru2.params().values():  total += p.size
        for p in self.heads.params().values(): total += p.size
        return total


# ── Alias for backward compatibility ─────────────────────────────────────────
GRUPredictor = Predictor   # Predictor uses a GRU internally; both names work

# ── Module-level singleton ────────────────────────────────────────────────────
_predictor_instance: Predictor | None = None

def get_predictor() -> Predictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance


# =============================================================================
# V6 M2: IntersectionPredictor  — queue clearance GRU
# V6 M3: AnomalyScorer          — autoencoder for rogue-car detection
# =============================================================================

_INT_MODEL_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "model", "int_predictor.npz")
_ANOMALY_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "model", "anomaly_scorer.npz")


class IntersectionPredictor:
    """
    M2: Predicts (ticks_to_clear_norm, near_miss_prob, congestion_end) for one arm.
    Input: (30, 6) feature sequence — built by ArmQueue._build_ml_features().
    Cold start: returns confidence=0.0, all predictions=0.5 (no action taken).
    """

    def __init__(self, model_path: str = None):
        self._path   = model_path or _INT_MODEL_PATH
        self._loaded = False
        self._H1, self._H2 = 48, 24
        self._gru1 = self._gru2 = self._Wo = self._bo = None
        self._try_load()

    def _try_load(self):
        if not os.path.exists(self._path):
            return
        try:
            d = dict(np.load(self._path, allow_pickle=True))
            self._gru1 = {k[3:]: d[k] for k in d if k.startswith("g1_")}
            self._gru2 = {k[3:]: d[k] for k in d if k.startswith("g2_")}
            self._Wo   = d["Wo"]
            self._bo   = d["bo"]
            self._loaded = True
            log.info("IntersectionPredictor loaded")
        except Exception as e:
            log.warning(f"IntersectionPredictor load error: {e}")

    def reload(self):
        self._loaded = False
        self._try_load()

    @staticmethod
    def _gru_step(x_seq, params, h_size):
        h = np.zeros(h_size, dtype=np.float32)
        for t in range(len(x_seq)):
            x = x_seq[t]
            r = _sigmoid(x @ params['Wr'].T + h @ params['Ur'].T + params['br'])
            z = _sigmoid(x @ params['Wz'].T + h @ params['Uz'].T + params['bz'])
            n = _tanh(   x @ params['Wn'].T + (r * h) @ params['Un'].T + params['bn'])
            h = (1 - z) * n + z * h
        return h

    def predict(self, features: np.ndarray):
        """
        features: (30, 6) float32
        Returns (ticks_to_clear_norm, near_miss_prob, congestion_prob, confidence)
        confidence=0.0 on cold start.
        """
        if not self._loaded or features is None:
            return 0.5, 0.5, 0.5, 0.0
        try:
            h1  = self._gru_step(features, self._gru1, self._H1)
            h1e = h1.reshape(1, -1)
            h2  = self._gru_step(h1e, self._gru2, self._H2)
            out = h2 @ self._Wo.T + self._bo
            clear = float(_sigmoid(out[0]))
            nm    = float(_sigmoid(out[1]))
            cong  = float(_sigmoid(out[2]))
            # Confidence: higher when predictions are decisive (away from 0.5)
            conf  = float(min(abs(clear - 0.5) * 2 + abs(nm - 0.5), 1.0))
            return clear, nm, cong, conf
        except Exception as e:
            log.warning(f"IntersectionPredictor.predict error: {e}")
            return 0.5, 0.5, 0.5, 0.0

    @property
    def loaded(self):
        return self._loaded


class AnomalyScorer:
    """
    M3: Scores a car's feature window for anomalous behaviour (0.0=normal, 1.0=rogue).
    Uses a numpy autoencoder: high reconstruction error = anomalous.
    Input: (30, 6) feature sequence (same as IntersectionPredictor).
    Cold start: score=0.0 always.
    """
    ROGUE_THRESHOLD = 0.7   # score above this → emit ROGUE_ALERT

    def __init__(self, model_path: str = None):
        self._path   = model_path or _ANOMALY_MODEL_PATH
        self._loaded = False
        self._We = self._be = self._Wd = self._bd = None
        self._err_mean = 0.01
        self._err_std  = 0.01
        self._try_load()

    def _try_load(self):
        if not os.path.exists(self._path):
            return
        try:
            d = dict(np.load(self._path, allow_pickle=True))
            self._We       = d['We']
            self._be       = d['be']
            self._Wd       = d['Wd']
            self._bd       = d['bd']
            self._err_mean = float(d.get('err_mean', 0.01))
            self._err_std  = float(d.get('err_std',  0.01))
            self._loaded   = True
            log.info("AnomalyScorer loaded")
        except Exception as e:
            log.warning(f"AnomalyScorer load error: {e}")

    def reload(self):
        self._loaded = False
        self._try_load()

    def score(self, features: np.ndarray) -> float:
        """
        features: (30, 6) float32
        Returns anomaly score 0.0–1.0. >0.7 suggests rogue behaviour.
        """
        if not self._loaded or features is None:
            return 0.0
        try:
            flat = features.reshape(-1).astype(np.float32)     # (180,)
            enc  = np.maximum(0, flat @ self._We.T + self._be) # relu (32,)
            dec  = _sigmoid(enc @ self._Wd.T + self._bd)       # (180,)
            err  = float(np.mean((dec - flat) ** 2))
            # Normalise to 0-1: z-score clamped
            z    = (err - self._err_mean) / (self._err_std * 3 + 1e-8)
            return float(np.clip(_sigmoid(z * 2), 0.0, 1.0))
        except Exception as e:
            log.warning(f"AnomalyScorer.score error: {e}")
            return 0.0

    @property
    def loaded(self):
        return self._loaded

    @property
    def threshold(self):
        return self.ROGUE_THRESHOLD


# ── Module-level singletons (M2 + M3) ────────────────────────────────────────
_int_predictor_instance: IntersectionPredictor | None = None
_anomaly_scorer_instance: AnomalyScorer | None        = None

def get_intersection_predictor() -> IntersectionPredictor:
    global _int_predictor_instance
    if _int_predictor_instance is None:
        _int_predictor_instance = IntersectionPredictor()
    return _int_predictor_instance

def get_anomaly_scorer() -> AnomalyScorer:
    global _anomaly_scorer_instance
    if _anomaly_scorer_instance is None:
        _anomaly_scorer_instance = AnomalyScorer()
    return _anomaly_scorer_instance
