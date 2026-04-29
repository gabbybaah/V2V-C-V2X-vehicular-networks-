# =============================================================================
# handshake_v4/car_intersection.py  —  Car behaviour in intersection queue
#
# Each car has:
#   queue_pos    : position in queue (0 = front)
#   queue_dist_m : distance to stop line in metres (positive = further back)
#   state        : IState enum
#   slot         : assigned crossing slot (-1 = not assigned)
#   events       : timestamped event log for dashboard
#
# SmartCar  — has radio, sends NEG_REQUEST, receives PASSAGE_TOKEN
# LegacyCar — no radio, moves when physically safe and platoon is green
# EmergencyCar — sends EMERG_PREEMPT, gets Slot 0, ignores red
# =============================================================================
import time, threading, logging
import random as _random
import numpy as np
from config import (CarType, IState, Msg, CAR_LENGTH_M, CAR_GAP_STOPPED_M,
                    CROSS_SPEED_MS, EMERGENCY_MS, MIN_GAP_M, INTER_BOX_M,
                    PLATOON_GAP_S, TRUST_HIGH, SIM_TICK_S, QUEUE_CREEP_MS,
                    MANEUVER_DIST_MULT, MANEUVER_ICONS, MANEUVER_WEIGHTS,
                    INT_STARTUP_BASE, INT_STARTUP_PER_POS)
from law import Law

log = logging.getLogger("v4.car_int")


def _make_id(label: str) -> str:
    import hashlib
    return hashlib.sha256(f"{label}{time.time()}".encode()).hexdigest()[:10]


# ─────────────────────────────────────────────────────────────────────────────
# Base car
# ─────────────────────────────────────────────────────────────────────────────

class BaseCar:
    def __init__(self, label: str, car_type: str, arm: str, queue_pos: int, lane: int = 0):
        self.label      = label
        self.car_type   = car_type
        self.arm        = arm
        self.queue_pos  = queue_pos
        self.lane       = lane
        self.car_id     = _make_id(label)
        self.queue_dist_m = queue_pos * (CAR_LENGTH_M + CAR_GAP_STOPPED_M)
        self.speed_ms   = 0.0
        self.state      = IState.QUEUED
        self.slot       = -1           # assigned by TokenManager
        self._move_after = None        # time.time() when authorised to depart
        self.crossed    = False
        self.trust      = {}           # car_id → trust score
        self.neighbours = {}           # car_id → last beacon
        self.events     = []
        self.msgs_sent  = 0
        self.msgs_recv  = 0
        self.law_log    = []
        self._lock      = threading.Lock()
        # V6 I2: maneuver type
        self.maneuver     = _random.choices(['STRAIGHT','TURN_RIGHT','TURN_LEFT'],
                                            weights=MANEUVER_WEIGHTS)[0]
        self._cross_dist_m = INTER_BOX_M * MANEUVER_DIST_MULT[self.maneuver]
        # V6 I1: startup delay at stop line
        _base = INT_STARTUP_BASE.get(car_type, 0.8)
        self._startup_delay_s  = _base + queue_pos * INT_STARTUP_PER_POS
        self._startup_elapsed  = 0.0
        self._startup_done     = False

    @property
    def speed_kmh(self) -> float:
        return self.speed_ms * 3.6

    def log_event(self, event: str, detail: str = ""):
        self.events.append({"ts": time.time(), "event": event,
                            "detail": detail, "label": self.label})
        log.debug(f"[{self.label}] {event}: {detail}")

    def log_law(self, cvc: str, reason: str):
        self.law_log.append({"cvc": cvc, "reason": reason, "ts": time.time()})

    def status(self) -> dict:
        return {
            "label":      self.label,
            "car_type":   self.car_type,
            "arm":        self.arm,
            "lane":       self.lane,
            "queue_pos":  self.queue_pos,
            "dist_m":     round(self.queue_dist_m, 1),
            "speed_kmh":  round(self.speed_kmh, 1),
            "state":      self.state,
            "slot":       self.slot,
            "crossed":    self.crossed,
            "msgs_sent":  self.msgs_sent,
            "msgs_recv":  self.msgs_recv,
            "neighbours": len(self.neighbours),
            "maneuver":   self.maneuver,
            "cross_dist_m": round(self._cross_dist_m, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SmartCar
# ─────────────────────────────────────────────────────────────────────────────

class SmartCar(BaseCar):
    def __init__(self, label: str, arm: str, queue_pos: int, lane: int = 0):
        super().__init__(label, CarType.SMART, arm, queue_pos, lane)
        self._neg_sent     = False    # have we sent NEG_REQUEST this phase?
        self._token_recvd  = False    # have we received our PASSAGE_TOKEN?
        self.yielding_for  = None     # car_id of emergency we're yielding for
        # M3: anomaly detection
        self._ml_feat_buf  = []       # rolling (30,6) feature buffer
        self._last_anomaly_score = 0.0
        self._rogue_alert_sent   = False

    # ── Outgoing messages ────────────────────────────────────────────────────

    def send_beacon(self, radio):
        radio.send({
            "type":      Msg.BEACON,
            "from":      self.car_id,
            "label":     self.label,
            "car_type":  self.car_type,
            "arm":       self.arm,
            "lane":      self.lane,
            "queue_pos": self.queue_pos,
            "dist_m":    round(self.queue_dist_m, 1),
            "speed_kmh": round(self.speed_kmh, 1),
            "state":     self.state,
            "slot":      self.slot,
            "ts":        time.time(),
        })
        self.msgs_sent += 1

    def send_neg_request(self, radio):
        if self._neg_sent:
            return
        self._neg_sent = True
        radio.send({
            "type":      Msg.NEG_REQUEST,
            "from":      self.car_id,
            "label":     self.label,
            "car_type":  self.car_type,
            "arm":       self.arm,
            "queue_pos": self.queue_pos,
            "trust":     TRUST_HIGH,
            "ts":        time.time(),
        })
        self.msgs_sent += 1
        self.state = IState.NEG
        self.log_event("NEG_REQUEST_SENT", f"Requested crossing slot (pos:{self.queue_pos})")

    # ── Incoming messages ────────────────────────────────────────────────────

    def receive(self, msg: dict):
        self.msgs_recv += 1
        mtype   = msg.get("type")
        from_id = msg.get("from")

        if from_id and from_id not in self.trust:
            self.trust[from_id] = TRUST_HIGH

        if mtype == Msg.BEACON:
            self.neighbours[from_id] = msg
            # Detect emergency in same arm behind us
            if (msg.get("car_type") == CarType.EMERGENCY
                    and msg.get("arm") == self.arm
                    and msg.get("queue_pos", 99) > self.queue_pos):
                self.log_event("EMERGENCY_BEHIND",
                               f"CVC 21806: {msg.get('label','?')} at pos "
                               f"{msg.get('queue_pos','?')} — preparing to yield")

        elif mtype == Msg.PASSAGE_TOKEN:
            self._handle_token(msg)

        elif mtype == Msg.EMERG_PREEMPT:
            self._handle_preempt(msg)

        elif mtype == Msg.HARD_BRAKE:
            self.log_event("BRAKE_WARN_RECV",
                           f"Chain warning from {msg.get('label','?')}")
            self.speed_ms = max(0.0, self.speed_ms - 2.0)

    def _handle_token(self, msg: dict):
        if self._token_recvd:
            return
        slots = msg.get("slots", [])
        my_entry = next((s for s in slots if s.get("car_id") == self.car_id), None)
        if my_entry is None:
            return
        self._token_recvd = True
        self.slot = my_entry["slot"]
        self.state = IState.TOKEN_OK
        self.log_event("TOKEN_RECEIVED", f"PASSAGE_TOKEN received — Slot #{self.slot}")
        self.log_law("CVC 21450", f"Slot #{self.slot} assigned — may proceed when slot opens")
        # Broadcast TOKEN_ACK so TokenManager records it
        if hasattr(self, "_radio_ref") and self._radio_ref:
            self._radio_ref.send({
                "type":  Msg.TOKEN_ACK,
                "from":  self.car_id,
                "label": self.label,
                "arm":   self.arm,
                "slot":  self.slot,
            })
            self.msgs_sent += 1

    def _handle_preempt(self, msg: dict):
        from_id    = msg.get("from")
        from_label = msg.get("label", "?")
        if self.state not in (IState.CROSSING, IState.DONE):
            # Only log the first time we yield for this specific emergency car
            already_logged = getattr(self, "_yielded_for", set())
            if from_id not in already_logged:
                already_logged.add(from_id)
                self._yielded_for = already_logged
                self.log_event("YIELDING", f"CVC 21806: yielding for {from_label}")
                self.log_law("CVC 21806", f"Emergency preemption from {from_label}")
            self.yielding_for  = from_id
            self._yield_start  = time.time()
            self.state         = IState.YIELDING
            self.speed_ms      = 0.0
            self._move_after   = None

    # ── Queue tick ────────────────────────────────────────────────────────────

    def _update_ml_features(self, green: bool, gap_ahead_m: float):
        """Build rolling (30,6) feature buffer for M3 anomaly scoring."""
        from config import INTER_BOX_M
        feat = np.array([
            min(self.queue_dist_m / max(INTER_BOX_M * 3, 1.0), 1.0),  # dist norm
            min(self.speed_kmh / 60.0, 1.0),                            # speed norm
            1.0 if green else 0.0,                                       # green
            1.0 if self.slot >= 0 else 0.0,                              # has token
            min(gap_ahead_m / 20.0, 1.0),                               # gap norm
            1.0 if self.state == IState.MOVING else 0.0,                # moving
        ], dtype=np.float32)
        self._ml_feat_buf.append(feat)
        if len(self._ml_feat_buf) > 30:
            self._ml_feat_buf.pop(0)

    def _check_anomaly(self, radio):
        """M3: score behaviour; emit ROGUE_ALERT if anomalous and no token."""
        if len(self._ml_feat_buf) < 30:
            return
        try:
            from ml_predictor import get_anomaly_scorer
            scorer = get_anomaly_scorer()
            if not scorer.loaded:
                return
            seq   = np.stack(self._ml_feat_buf)       # (30, 6)
            score = scorer.score(seq)
            self._last_anomaly_score = score
            # Alert: high score AND car is moving without a valid token
            if (score >= scorer.threshold
                    and self.slot < 0
                    and self.state == IState.MOVING
                    and not self._rogue_alert_sent):
                self._rogue_alert_sent = True
                radio.send({
                    "type":   "ROGUE_ALERT",
                    "from":   self.car_id,
                    "label":  self.label,
                    "arm":    self.arm,
                    "score":  round(score, 3),
                    "ts":     time.time(),
                })
                self.log_event("ROGUE_ALERT",
                    f"⚠ Anomaly score {score:.2f} — car moving without token")
            elif score < scorer.threshold * 0.7:
                self._rogue_alert_sent = False   # reset if score drops
        except Exception:
            pass

    def tick_queue(self, dt: float, green: bool, gap_ahead_m: float,
                   in_platoon: bool, radio):
        self._radio_ref = radio   # store for TOKEN_ACK in receive()
        self.send_beacon(radio)
        # M3: update feature buffer and check anomaly every tick
        self._update_ml_features(green, gap_ahead_m)
        self._check_anomaly(radio)

        # ── Yielding state ───────────────────────────────────────────────
        if self.state == IState.YIELDING:
            if self.yielding_for:
                # Resume when emergency beacon no longer near, OR after timeout
                still_near = any(
                    b.get("from") == self.yielding_for
                    and b.get("state") not in ("DONE", "CROSSING")
                    for b in self.neighbours.values()
                )
                timed_out = (
                    hasattr(self, "_yield_start")
                    and time.time() - self._yield_start > 18.0
                )
                if not still_near or timed_out:
                    self.state        = IState.QUEUED
                    self.yielding_for = None
                    self._neg_sent    = False
                    self.log_event("YIELD_DONE", "Emergency cleared — resuming queue")
            return

        # ── Red / not in platoon — send NEG_REQUEST early so TKM has it ─
        if green and not self._neg_sent and in_platoon:
            self.send_neg_request(radio)

        if not green or not in_platoon:
            self.speed_ms    = 0.0
            self.state       = IState.QUEUED
            self._move_after = None
            return

        # ── Have token? Stagger departure by slot ────────────────────────
        if self.slot >= 0:
            if self._move_after is None:
                self._move_after = time.time() + self.slot * PLATOON_GAP_S
            if time.time() < self._move_after:
                self.speed_ms = 0.0
                return

        # ── Move forward ─────────────────────────────────────────────────
        self.state = IState.MOVING
        safe, cvc, reason = Law.safe_following(self.speed_ms, gap_ahead_m)
        if not safe:
            self.speed_ms = max(0.0, self.speed_ms - 3.0 * dt)
        else:
            self.speed_ms = min(CROSS_SPEED_MS, self.speed_ms + 4.0 * dt)
        self.queue_dist_m = max(0.0, self.queue_dist_m - self.speed_ms * dt)
        self.log_law(cvc, reason)

    def reset_for_next_green(self):
        """Called when light goes red — reset negotiation state."""
        self._neg_sent    = False
        self._token_recvd = False
        self._move_after  = None
        if self.state not in (IState.CROSSING, IState.DONE):
            self.state = IState.QUEUED
            self.slot  = -1


# ─────────────────────────────────────────────────────────────────────────────
# LegacyCar
# ─────────────────────────────────────────────────────────────────────────────

class LegacyCar(BaseCar):
    def __init__(self, label: str, arm: str, queue_pos: int, lane: int = 0):
        super().__init__(label, CarType.LEGACY, arm, queue_pos, lane)

    def tick_queue(self, dt: float, green: bool, gap_ahead_m: float,
                   in_platoon: bool, radio=None):
        if not green or not in_platoon:
            self.speed_ms = 0.0
            self.state    = IState.QUEUED
            return

        # Legacy car: just follow the car ahead (no slot stagger)
        if self.slot >= 0 and self._move_after is not None:
            if time.time() < self._move_after:
                self.speed_ms = 0.0
                return

        self.state = IState.MOVING
        if gap_ahead_m < MIN_GAP_M:
            self.speed_ms = max(0.0, self.speed_ms - 3.0 * dt)
        else:
            self.speed_ms = min(QUEUE_CREEP_MS, self.speed_ms + 2.0 * dt)
        self.queue_dist_m = max(0.0, self.queue_dist_m - self.speed_ms * dt)

    def reset_for_next_green(self):
        self._move_after = None
        if self.state not in (IState.CROSSING, IState.DONE):
            self.state = IState.QUEUED
            self.slot  = -1


# ─────────────────────────────────────────────────────────────────────────────
# EmergencyCar
# ─────────────────────────────────────────────────────────────────────────────

class EmergencyCar(SmartCar):
    def __init__(self, label: str, arm: str, queue_pos: int, lane: int = 0):
        super().__init__(label, arm, queue_pos, lane)
        self.car_type        = CarType.EMERGENCY
        self._preempt_sent   = False
        # Override startup — emergencies get through immediately (0 delay)
        self._startup_delay_s = 0.0
        self._startup_elapsed = 0.0
        self._startup_done    = True

    def send_preempt(self, radio, light=None):
        if self._preempt_sent:
            return
        self._preempt_sent = True
        radio.send({
            "type":      Msg.EMERG_PREEMPT,
            "from":      self.car_id,
            "label":     self.label,
            "car_type":  CarType.EMERGENCY,
            "arm":       self.arm,
            "queue_pos": self.queue_pos,
            "ts":        time.time(),
        })
        self.msgs_sent += 1
        self.log_event("PREEMPT_BROADCAST",
                       f"CVC 21806: emergency preemption from queue pos {self.queue_pos}")
        self.log_law("CVC 21806", "Emergency vehicle preemption broadcast")
        # Preempt the traffic light
        if light is not None:
            light.preempt(self.arm)

    def send_neg_request(self, radio):
        """Emergency always requests Slot 0."""
        if self._neg_sent:
            return
        self._neg_sent = True
        radio.send({
            "type":      Msg.NEG_REQUEST,
            "from":      self.car_id,
            "label":     self.label,
            "car_type":  CarType.EMERGENCY,
            "arm":       self.arm,
            "queue_pos": self.queue_pos,
            "trust":     192,
            "ts":        time.time(),
        })
        self.msgs_sent += 1
        self.state = IState.NEG
        self.log_event("NEG_REQUEST_SENT",
                       f"Emergency requested Slot 0 (pos:{self.queue_pos})")

    def tick_queue(self, dt: float, green: bool, gap_ahead_m: float,
                   in_platoon: bool, radio, light=None):
        self.send_beacon(radio)
        self.send_preempt(radio, light)
        self.send_neg_request(radio)

        # Emergency ignores red (CVC 21055) — only yields to immediate physical gap
        self.state = IState.MOVING
        safe_gap   = MIN_GAP_M * 0.6
        if gap_ahead_m < safe_gap:
            self.speed_ms = max(0.0, self.speed_ms - 4.0 * dt)
        else:
            self.speed_ms = min(EMERGENCY_MS, self.speed_ms + 5.0 * dt)
        self.queue_dist_m = max(0.0, self.queue_dist_m - self.speed_ms * dt)
        self.log_law("CVC 21055", "Emergency vehicle — signal exemption")

    def reset_for_next_green(self):
        pass   # Emergency cars don't reset — they keep going


# ─────────────────────────────────────────────────────────────────────────────
# RogueCar — Feature #8: ignores token, cuts queue
# ─────────────────────────────────────────────────────────────────────────────

class RogueCar(SmartCar):
    """
    Has V2X radio and receives the token, but deliberately ignores its slot.
    Crosses out of order — logged as a protocol violation.
    Demonstrates WHY the handshake protocol needs enforcement.
    """

    def __init__(self, label: str, arm: str, queue_pos: int, lane: int = 0):
        super().__init__(label, arm, queue_pos, lane)
        self.car_type    = CarType.ROGUE
        self._rogue_triggered = False
        # Recalculate startup with ROGUE base (was computed as SMART)
        self._startup_delay_s = INT_STARTUP_BASE.get(CarType.ROGUE, 0.3) + self.queue_pos * INT_STARTUP_PER_POS
        self._startup_done    = (self._startup_delay_s <= 0.0)
        self._startup_elapsed = 0.0

    def tick_queue(self, dt: float, green: bool, gap_ahead_m: float,
                   in_platoon: bool, radio):
        self._radio_ref = radio
        self.send_beacon(radio)

        if self.state == IState.YIELDING:
            return

        if green and not self._neg_sent and in_platoon:
            self.send_neg_request(radio)

        if not green or not in_platoon:
            self.speed_ms = 0.0
            self.state    = IState.QUEUED
            return

        # Feature #8: Rogue car has a token but ignores it
        # Instead of waiting for its slot, it cuts in immediately after any gap
        if self._token_recvd and self.slot > 0 and not self._rogue_triggered:
            if gap_ahead_m > MIN_GAP_M:
                self._rogue_triggered = True
                self.state = IState.ROGUE_GO
                # Broadcast violation
                radio.send({
                    "type":      Msg.ROGUE_CROSS,
                    "from":      self.car_id,
                    "label":     self.label,
                    "arm":       self.arm,
                    "slot":      self.slot,
                    "queue_pos": self.queue_pos,
                    "ts":        time.time(),
                })
                self.msgs_sent += 1
                self.log_event("ROGUE_VIOLATION",
                    f"⛔ {self.label} IGNORING Slot #{self.slot} — cutting queue!  PROTOCOL VIOLATION")
                self.log_event("ROGUE_VIOLATION",
                    f"Without Handshake enforcement, {self.label} crosses out of order — CVC 21453")

        # Move forward (rogue ignores slot timing)
        if self.state in (IState.ROGUE_GO, IState.MOVING):
            self.state = IState.ROGUE_GO if self._rogue_triggered else IState.MOVING
            safe, cvc, reason = Law.safe_following(self.speed_ms, gap_ahead_m)
            if not safe:
                self.speed_ms = max(0.0, self.speed_ms - 3.0 * dt)
            else:
                self.speed_ms = min(CROSS_SPEED_MS * 1.2, self.speed_ms + 4.0 * dt)
            self.queue_dist_m = max(0.0, self.queue_dist_m - self.speed_ms * dt)
            return

        # Has token and in slot — normal behaviour
        if self.slot >= 0:
            if self._move_after is None:
                self._move_after = time.time() + self.slot * PLATOON_GAP_S
            if time.time() < self._move_after:
                self.speed_ms = 0.0
                return

        self.state = IState.MOVING
        safe, cvc, reason = Law.safe_following(self.speed_ms, gap_ahead_m)
        if not safe:
            self.speed_ms = max(0.0, self.speed_ms - 3.0 * dt)
        else:
            self.speed_ms = min(CROSS_SPEED_MS, self.speed_ms + 4.0 * dt)
        self.queue_dist_m = max(0.0, self.queue_dist_m - self.speed_ms * dt)


# ─────────────────────────────────────────────────────────────────────────────
# Token fallback mixin — Feature #10
# Applied to SmartCar: if TOKEN_TIMEOUT_S passes with no token, depart anyway
# ─────────────────────────────────────────────────────────────────────────────

# Patch SmartCar.tick_queue to add fallback timer
_original_smart_tick = SmartCar.tick_queue

def _smart_tick_with_fallback(self, dt, green, gap_ahead_m, in_platoon, radio):
    from config import TOKEN_TIMEOUT_S, IState
    # Start the fallback timer when green starts and we sent a request
    if green and self._neg_sent and not self._token_recvd:
        if not hasattr(self, '_fallback_timer_start'):
            self._fallback_timer_start = time.time()
        elif time.time() - self._fallback_timer_start > TOKEN_TIMEOUT_S:
            # Feature #10: TOKEN LOST — fallback to physical gap departure
            if not hasattr(self, '_fallback_logged'):
                self._fallback_logged = True
                self.state = IState.TOKEN_OK  # force move-allowed
                self.slot  = 99              # fallback slot marker
                self.log_event("TOKEN_FALLBACK",
                    f"⏱ {self.label} TOKEN TIMEOUT ({TOKEN_TIMEOUT_S}s) — "
                    f"fallback to gap-based departure  CVC 21703")
                if hasattr(self, 'token_events'):
                    self.token_events.append({
                        "ts": time.time(), "event": "TOKEN_FALLBACK",
                        "detail": f"⏱ {self.label} no token after {TOKEN_TIMEOUT_S}s — "
                                  f"fallback departure  CVC 21703",
                        "label": self.label,
                    })
    elif green and self._token_recvd:
        # Reset timer on next green
        if hasattr(self, '_fallback_timer_start'):
            del self._fallback_timer_start
        if hasattr(self, '_fallback_logged'):
            del self._fallback_logged
    _original_smart_tick(self, dt, green, gap_ahead_m, in_platoon, radio)

SmartCar.tick_queue = _smart_tick_with_fallback
