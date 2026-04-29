# =============================================================================
# handshake_v4/tokens.py  —  Handshake Token Manager
#
# Features added:
#   #9  — Token conflict detection + resolution
#   #10 — Token lost fallback timer (TOKEN_TIMEOUT_S)
#   #11 — Emergency mid-negotiation cancel + reissue with Slot 0
# =============================================================================
import time, logging
import numpy as np
from config import (Msg, CarType, NEG_WINDOW_S, PLATOON_MAX,
                    TRUST_HIGH, TOKEN_TIMEOUT_S, PLATOON_GAP_S)

log = logging.getLogger("v4.tokens")


class TokenManager:
    def __init__(self, arm: str, radio, platoon_limit: int = PLATOON_MAX):
        self.arm           = arm
        self.radio         = radio
        self.platoon_limit = platoon_limit
        self._requests     = {}      # car_id → request dict
        self._token_issued = False
        self._issue_time   = 0.0
        self._slots        = {}      # car_id → slot_number
        self._cancelled    = False   # set True if emergency arrives mid-neg
        self.events        = []
        self._window_end   = time.time() + NEG_WINDOW_S
        # M2: ML feature buffer for IntersectionPredictor
        self._ml_feat_buf  = []      # rolling feature vecs
        self.ml_clearance  = 0.5    # last predicted ticks_to_clear_norm
        self.ml_confidence = 0.0    # last prediction confidence
        self._effective_gap = PLATOON_GAP_S  # may be tightened/widened by M2

    # ── Step 1: collect NEG_REQUESTs ─────────────────────────────────────────

    def receive_neg_request(self, msg: dict):
        car_id   = msg.get("from")
        label    = msg.get("label", car_id[:6] if car_id else "?")
        car_type = msg.get("car_type", CarType.SMART)
        q_pos    = msg.get("queue_pos", 99)
        trust    = msg.get("trust", TRUST_HIGH)

        if not car_id:
            return

        # Feature #9: detect conflict — two cars claiming same queue_pos
        existing = [r for r in self._requests.values()
                    if r.get("queue_pos") == q_pos
                    and r.get("car_id") != car_id
                    and r.get("car_type") != CarType.EMERGENCY]
        if existing and car_type != CarType.EMERGENCY:
            conflict_label = existing[0].get("label", "?")
            self._log("SLOT_CONFLICT",
                f"⚠ CONFLICT: {label} and {conflict_label} both claim pos {q_pos} — "
                f"resolving by timestamp (first-arrived wins)")
            # First-arrived keeps its position; new one gets bumped to end
            q_pos = max(r["queue_pos"] for r in self._requests.values()) + 1
            self._log("CONFLICT_RESOLVED",
                f"✅ {label} reassigned to pos {q_pos}")

        if car_id not in self._requests:
            self._requests[car_id] = {
                "car_id":    car_id,
                "label":     label,
                "car_type":  car_type,
                "queue_pos": q_pos,
                "trust":     trust,
                "ts":        time.time(),
            }
            self._log("NEG_RECV",
                f"{label} NEG_REQUEST (pos:{q_pos} type:{car_type})")

        # Feature #11: emergency arrives mid-negotiation — cancel and reissue
        if car_type == CarType.EMERGENCY and self._token_issued:
            self._handle_emergency_mid_negotiation(car_id, label)

    def _handle_emergency_mid_negotiation(self, emerg_id: str, emerg_label: str):
        """Cancel outstanding token and reissue with emergency as Slot 0."""
        self._cancelled = True
        # Broadcast TOKEN_CANCEL
        self.radio.send({
            "type":   Msg.TOKEN_CANCEL,
            "from":   f"TKM-{self.arm}",
            "arm":    self.arm,
            "reason": f"Emergency {emerg_label} preemption",
            "ts":     time.time(),
        })
        self._log("TOKEN_CANCEL",
            f"🚨 EMERGENCY {emerg_label} arrived mid-negotiation — "
            f"cancelling token, reissuing with emergency Slot 0  CVC 21806")
        # Reset and immediately reissue
        self._token_issued = False
        self._slots        = {}
        self.issue_token()

    # ── Step 2: window state ──────────────────────────────────────────────────

    def window_open(self) -> bool:
        return time.time() < self._window_end

    # ── Step 3+4: assign slots and broadcast PASSAGE_TOKEN ───────────────────

    def issue_token(self, legacy_cars: list = None):
        if self._token_issued and not self._cancelled:
            return
        self._token_issued = True
        self._cancelled    = False
        self._issue_time   = time.time()

        all_cars = list(self._requests.values())
        if legacy_cars:
            for (cid, lbl, qpos) in legacy_cars:
                if cid not in self._requests:
                    all_cars.append({
                        "car_id":    cid,
                        "label":     lbl,
                        "car_type":  CarType.LEGACY,
                        "queue_pos": qpos,
                        "trust":     0,
                    })

        emerg  = [c for c in all_cars if c["car_type"] == CarType.EMERGENCY]
        others = sorted(
            [c for c in all_cars if c["car_type"] != CarType.EMERGENCY],
            key=lambda c: (c["queue_pos"], c["ts"])
        )
        ordered = (emerg + others)[:self.platoon_limit]

        # M2: consult IntersectionPredictor — adjust slot gap if confident
        self._effective_gap = self._ml_adjusted_gap()

        slot_list = []
        for slot, car in enumerate(ordered):
            self._slots[car["car_id"]] = slot
            slot_list.append({
                "car_id":   car["car_id"],
                "label":    car["label"],
                "car_type": car["car_type"],
                "slot":     slot,
            })

        token = {
            "type":      Msg.PASSAGE_TOKEN,
            "from":      f"TKM-{self.arm}",
            "arm":       self.arm,
            "slots":     slot_list,
            "issued_at": self._issue_time,
            "ts":        time.time(),
        }
        self.radio.send(token)

        # Feature #10: also broadcast the timeout so cars know fallback
        self.radio.send({
            "type":    "TOKEN_TIMEOUT_INFO",
            "from":    f"TKM-{self.arm}",
            "arm":     self.arm,
            "timeout": TOKEN_TIMEOUT_S,
            "ts":      time.time(),
        })

        names = ", ".join(f"{e['label']}=S{e['slot']}" for e in slot_list)
        self._log("TOKEN_ISSUED",
            f"PASSAGE_TOKEN [{names}]  ({len(slot_list)} cars)")
        log.info(f"[TKM {self.arm}] Token issued: {names}")
        return token

    def push_ml_features(self, feat: np.ndarray):
        """Called by ArmQueue each tick to feed the predictor."""
        self._ml_feat_buf.append(feat)
        if len(self._ml_feat_buf) > 30:
            self._ml_feat_buf.pop(0)
        # Re-run prediction periodically
        if len(self._ml_feat_buf) == 30 and len(self._ml_feat_buf) % 10 == 0:
            self._run_prediction()

    def _run_prediction(self):
        try:
            from ml_predictor import get_intersection_predictor
            pred = get_intersection_predictor()
            if not pred.loaded:
                return
            seq = np.stack(self._ml_feat_buf)   # (30, 6)
            clear, nm, cong, conf = pred.predict(seq)
            self.ml_clearance  = clear
            self.ml_confidence = conf
            if conf >= 0.6:
                self._log("ML_PREDICT",
                    f"★ Clearance {clear:.2f}  near-miss risk {nm:.2f}  conf {conf:.2f}")
        except Exception:
            pass

    def _ml_adjusted_gap(self) -> float:
        """M2: tighten or widen gap based on ML clearance prediction."""
        if self.ml_confidence < 0.6:
            return PLATOON_GAP_S
        # Low clearance time predicted → tighten gap to move cars faster
        # High congestion risk → widen gap for safety
        factor = 1.0
        if self.ml_clearance < 0.3:      # queue will clear fast → be bold
            factor = 0.85
        elif self.ml_clearance > 0.7:    # long queue expected → be cautious
            factor = 1.15
        adjusted = PLATOON_GAP_S * factor
        return float(np.clip(adjusted, PLATOON_GAP_S * 0.7, PLATOON_GAP_S * 1.5))

    # ── Step 5: car acknowledges ──────────────────────────────────────────────

    def receive_token_ack(self, car_id: str, label: str, slot: int):
        self._log("TOKEN_ACK", f"{label} acknowledged Slot #{slot}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_slot(self, car_id: str) -> int:
        return self._slots.get(car_id, -1)

    def _log(self, event: str, detail: str):
        self.events.append({
            "ts": time.time(), "arm": self.arm,
            "event": event, "detail": detail,
        })
        log.debug(f"[TKM {self.arm}] {event}: {detail}")

    @property
    def request_count(self) -> int:
        return len(self._requests)

    @property
    def token_issued(self) -> bool:
        return self._token_issued
