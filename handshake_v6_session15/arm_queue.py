# =============================================================================
# handshake_v4/arm_queue.py  —  One arm of the intersection
#
# Features added:
#   #12 — Ambulance corridor: all arms yield simultaneously (coordinated)
#   #13 — Pedestrian crossing interrupt
#   #14 — Right-of-way CVC 21800 (uncontrolled intersection mode)
#   #15 — Throughput stats (cars/min, peak throughput, cycle efficiency)
# =============================================================================
import time, logging, random
from config import (CarType, IState, Msg, CAR_LENGTH_M, CAR_GAP_STOPPED_M,
                    OUTBOUND_HOLD_TICKS,
                    INTER_BOX_M, CROSS_SPEED_MS, ROAD_LENGTH_M, PLATOON_MAX,
                    NEG_WINDOW_S, DEPART_SPEED_MS, SIM_TICK_S, TRUST_HIGH)
from car_intersection import SmartCar, LegacyCar, EmergencyCar, RogueCar, BaseCar
from tokens import TokenManager
from law import Law

log = logging.getLogger("v4.armq")

# Which arm a car exits to based on entry arm + maneuver
EXIT_ARM = {
    "North": {"STRAIGHT": "South", "TURN_RIGHT": "East",  "TURN_LEFT": "West"},
    "South": {"STRAIGHT": "North", "TURN_RIGHT": "West",  "TURN_LEFT": "East"},
    "East":  {"STRAIGHT": "West",  "TURN_RIGHT": "South", "TURN_LEFT": "North"},
    "West":  {"STRAIGHT": "East",  "TURN_RIGHT": "North", "TURN_LEFT": "South"},
}


class ArmQueue:
    def __init__(self, arm: str, cars: list, radio, light=None,
                 uncontrolled: bool = False):
        self.arm         = arm
        self.radio       = radio
        self.light       = light
        self.uncontrolled = uncontrolled  # Feature #14: CVC 21800 mode

        self._queue  = list(cars)
        self._road   = []
        self._done   = []
        self._all    = list(cars)

        self._green       = False
        self._platoon_on  = False
        self._platoon_cnt = 0
        self._platoon_lim = PLATOON_MAX

        self._tkm: TokenManager | None = None
        self._total_crossed = 0
        self._outbound     = []   # V6: [(car_snapshot, ticks_remaining)]
        self._emergency_seen = False

        # Feature #15: throughput history
        self._throughput_history: list[tuple] = []  # (timestamp, cumulative_done)
        self._last_tput_check = time.time()
        self._cycle_start     = time.time()
        self._cycle_cleared   = 0
        self.peak_throughput  = 0.0   # cars/min peak
        self.total_cycles     = 0
        self.avg_cycle_cars   = 0.0

        # Feature #12: ambulance corridor tracking
        self._corridor_active = False

        # Feature #13: pedestrian hold
        self._ped_hold = False

        self.token_events: list = []

    # ── Phase control ─────────────────────────────────────────────────────────

    def _tick_outbound(self):
        """V6: decay outbound car visibility counter."""
        self._outbound = [(s, t-1) for s,t in self._outbound if t > 1]

    def on_green(self):
        if self._ped_hold:
            return   # Pedestrian trumps green
        self._green       = True
        self._platoon_on  = True
        self._platoon_cnt = 0
        self._cycle_start = time.time()
        self._cycle_cleared = 0

        self._tkm = TokenManager(self.arm, self.radio, self._platoon_lim)

        legacy = [
            (c.car_id, c.label, c.queue_pos)
            for c in self._queue[:self._platoon_lim]
            if c.car_type == CarType.LEGACY
        ]
        for (cid, lbl, qpos) in legacy:
            self._tkm.receive_neg_request({
                "from": cid, "label": lbl,
                "car_type": CarType.LEGACY,
                "queue_pos": qpos, "trust": 0,
            })
        log.info(f"[{self.arm}] GREEN — {min(self._platoon_lim, len(self._queue))} eligible")

    def on_red(self):
        self._green      = False
        self._platoon_on = False

        if self._tkm:
            self.token_events.extend(self._tkm.events)
            self._tkm = None

        for car in self._queue:
            car.reset_for_next_green()
            car.speed_ms = 0.0

        self.total_cycles += 1
        if self._cycle_cleared > 0:
            dur = max(0.1, time.time() - self._cycle_start)
            rate = self._cycle_cleared / dur * 60
            if rate > self.peak_throughput:
                self.peak_throughput = rate
            n = self.total_cycles
            self.avg_cycle_cars = ((n-1)*self.avg_cycle_cars + self._cycle_cleared) / n

        log.info(f"[{self.arm}] RED — {len(self._queue)} still queued")

    # ── Feature #12: Ambulance corridor ──────────────────────────────────────

    def on_ambulance_corridor(self, emerg_label: str):
        """Force all cars to yield — called when emergency detected."""
        self._corridor_active = True
        self._green = False
        self._platoon_on = False
        for car in self._queue:
            if car.state not in (IState.CROSSING, IState.DONE):
                car.state    = IState.YIELDING
                car.speed_ms = 0.0
                car.log_event("CORRIDOR_YIELD",
                    f"🚨 AMBULANCE CORRIDOR — {emerg_label} has clear path  CVC 21806")
        self.token_events.append({
            "ts":     time.time(),
            "arm":    self.arm,
            "event":  "AMBULANCE_CORRIDOR",
            "detail": f"🚨 ALL ARMS YIELD — corridor open for {emerg_label}  CVC 21806",
        })
        log.info(f"[{self.arm}] AMBULANCE CORRIDOR active for {emerg_label}")

    def on_corridor_clear(self):
        self._corridor_active = False
        for car in self._queue:
            if car.state == IState.YIELDING:
                car.state     = IState.QUEUED
                car.speed_ms  = 0.0
                if isinstance(car, SmartCar):
                    car._neg_sent    = False
                    car._token_recvd = False

    # ── Feature #13: Pedestrian interrupt ────────────────────────────────────

    def on_pedestrian(self):
        self._ped_hold  = True
        self._green     = False
        self._platoon_on = False
        for car in self._queue:
            if car.state not in (IState.CROSSING, IState.DONE):
                car.speed_ms = 0.0
                if car.state == IState.MOVING:
                    car.state = IState.QUEUED
        self.token_events.append({
            "ts":     time.time(),
            "arm":    self.arm,
            "event":  "PEDESTRIAN",
            "detail": f"🚶 {self.arm} PEDESTRIAN CROSSING — all vehicles yield  CVC 21950",
        })

    def on_pedestrian_clear(self):
        self._ped_hold = False

    # ── Feature #14: Right-of-way CVC 21800 (uncontrolled mode) ──────────────

    def _right_of_way_check(self, other_queues: dict) -> bool:
        """
        CVC 21800 — at uncontrolled intersection, vehicle on right has priority.
        Returns True if this arm has right of way.
        """
        if not self.uncontrolled:
            return True  # normal light mode — light decides
        for other_arm, other_q in other_queues.items():
            if other_arm == self.arm:
                continue
            if (other_q._queue and other_q._queue[0].state != IState.DONE):
                has_row = Law.right_of_way_cvc21800(self.arm, other_arm)
                if has_row:
                    # We yield to this arm
                    if self._queue:
                        self._queue[0].log_event("CVC_21800_YIELD",
                            f"CVC 21800: yielding to {other_arm} (right of way)")
                    return False
        return True

    # ── Main tick ─────────────────────────────────────────────────────────────

    def tick(self, dt: float, other_queues: dict = None):
        self._drain_messages()
        if not self._ped_hold and not self._corridor_active:
            self._maybe_issue_token()
        self._tick_queue(dt)
        self._advance_crossers()
        self._tick_road(dt)
        self._tick_outbound()
        self._update_throughput()
        self._push_ml_features()   # M2: feed predictor
        self._harvest_car_events() # collect CROSSED, TOKEN_ACK etc. from individual cars

    def _harvest_car_events(self):
        """
        Sweep every car's events list and promote new ones to the arm queue's
        token_events so they appear in the dashboard. Clears each car after reading.
        """
        for car in list(self._queue) + list(self._road) + list(self._done):
            if car.events:
                for ev in car.events:
                    # Tag with arm so dashboard can show which arm it came from
                    ev.setdefault("arm", self.arm)
                self.token_events.extend(car.events)
                car.events.clear()
        # Keep arm-level list bounded
        if len(self.token_events) > 400:
            self.token_events = self.token_events[-300:]

    def _push_ml_features(self):
        """M2: build 6-feature snapshot and push to TokenManager."""
        if self._tkm is None:
            return
        try:
            import numpy as np
            from config import INTER_BOX_M
            s     = self.status()
            total = max(s.get("total", 1), 1)
            feat  = np.array([
                min(s.get("queued",   0) / 40.0,  1.0),
                min(s.get("on_road",  0) / 4.0,   1.0),
                s.get("done", 0) / total,
                min(s.get("throughput_rate", 0) / 60.0, 1.0),
                1.0 if s.get("light") == "GREEN" else 0.0,
                min(self._tkm.ml_confidence, 1.0),
            ], dtype=np.float32)
            self._tkm.push_ml_features(feat)
        except Exception:
            pass

    def _update_throughput(self):
        """Feature #15: track throughput over time."""
        now = time.time()
        if now - self._last_tput_check >= 5.0:
            self._last_tput_check = now
            self._throughput_history.append((now, self._total_crossed))
            if len(self._throughput_history) > 60:
                self._throughput_history = self._throughput_history[-30:]

    def get_throughput_rate(self) -> float:
        """Cars/min over the last 30 seconds."""
        if len(self._throughput_history) < 2:
            return 0.0
        t0, c0 = self._throughput_history[0]
        t1, c1 = self._throughput_history[-1]
        dur = max(0.1, t1 - t0)
        return (c1 - c0) / dur * 60

    def _drain_messages(self):
        for msg in self.radio.drain():
            mtype   = msg.get("type")
            from_id = msg.get("from")

            if mtype == Msg.NEG_REQUEST and self._tkm:
                if msg.get("arm") == self.arm:
                    self._tkm.receive_neg_request(msg)

            if mtype == Msg.TOKEN_ACK:
                label_ack = msg.get("label","?")
                slot_ack  = msg.get("slot",-1)
                # Log directly so it's visible even if tkm was cleared by red phase
                self.token_events.append({
                    "ts": time.time(), "arm": self.arm,
                    "event": "TOKEN_ACK",
                    "detail": f"✓ {label_ack} acknowledged Slot #{slot_ack} — cleared to cross",
                })
                if self._tkm:
                    self._tkm.receive_token_ack(
                        msg.get("from"), label_ack, slot_ack)

            # Feature #11: handle TOKEN_CANCEL
            if mtype == Msg.TOKEN_CANCEL:
                if msg.get("arm") == self.arm:
                    for car in self._queue:
                        if isinstance(car, SmartCar):
                            car._token_recvd = False
                            car._neg_sent    = False
                            car.slot         = -1
                            car.state        = IState.QUEUED
                    self.token_events.append({
                        "ts": time.time(), "arm": self.arm,
                        "event": "TOKEN_CANCEL",
                        "detail": f"🚨 Token cancelled — {msg.get('reason','')}",
                    })

            # Feature #8: log rogue crossing violation
            if mtype == Msg.ROGUE_CROSS:
                self.token_events.append({
                    "ts": time.time(), "arm": self.arm,
                    "event": "ROGUE_VIOLATION",
                    "detail": f"⛔ {msg.get('label','?')} ROGUE CROSS — "
                              f"ignored Slot #{msg.get('slot','?')}  VIOLATION",
                })

            for car in self._queue:
                if isinstance(car, (SmartCar, RogueCar)) and msg.get("from") != car.car_id:
                    car.receive(msg)
            for car in self._road:
                if isinstance(car, (SmartCar, RogueCar)) and msg.get("from") != car.car_id:
                    car.receive(msg)

    def _maybe_issue_token(self):
        if self._tkm and not self._tkm.token_issued and not self._tkm.window_open():
            # The global shared Radio bus may have consumed this arm's NEG_REQUEST
            # messages before our own drain() ran (another arm drained them first).
            # Directly register any SmartCar in our queue that sent a request
            # but whose message was lost to the race — ensures TKM always sees them.
            for car in self._queue:
                if (isinstance(car, SmartCar)
                        and car._neg_sent
                        and car.car_id not in self._tkm._requests):
                    self._tkm.receive_neg_request({
                        "from":      car.car_id,
                        "label":     car.label,
                        "car_type":  car.car_type,
                        "arm":       self.arm,
                        "queue_pos": car.queue_pos,
                        "trust":     TRUST_HIGH,
                        "ts":        time.time(),
                    })
            token = self._tkm.issue_token()
            self.token_events.extend(self._tkm.events)
            self._tkm.events.clear()
            # Deliver PASSAGE_TOKEN directly to this arm's SmartCars.
            # Sending via radio would have the same race: another arm's drain()
            # would consume it first.
            if token:
                for car in self._queue:
                    if isinstance(car, SmartCar) and car.car_id != token.get("from"):
                        car.receive(token)

    def _tick_queue(self, dt: float):
        for i, car in enumerate(self._queue):
            gap = 9999.0 if i == 0 else max(
                0.5,
                car.queue_dist_m - self._queue[i-1].queue_dist_m - CAR_LENGTH_M
            )
            in_platoon = (self._platoon_on
                          and self._platoon_cnt < self._platoon_lim
                          and i < self._platoon_lim)

            if isinstance(car, EmergencyCar):
                car.tick_queue(dt, self._green, gap, in_platoon,
                               self.radio, self.light)
                self._emergency_seen = True
            elif isinstance(car, RogueCar):
                car.tick_queue(dt, self._green, gap, in_platoon, self.radio)
            elif isinstance(car, SmartCar):
                car.tick_queue(dt, self._green, gap, in_platoon, self.radio)
            else:
                car.tick_queue(dt, self._green, gap, in_platoon)

    def _advance_crossers(self):
        to_cross = []
        for car in self._queue:
            if car.queue_dist_m > 0.1:
                continue
            if car.speed_ms <= 0.0:
                continue
            if self._ped_hold and car.car_type != CarType.EMERGENCY:
                continue   # Feature #13: pedestrian holds all but emergency
            if self._green or car.car_type == CarType.EMERGENCY:
                to_cross.append(car)

        for car in to_cross:
            car.queue_dist_m = 0.0
            car.state        = IState.CROSSING
            car.road_pos_m   = 0.0
            car.crossed      = True
            self._total_crossed  += 1
            # Assign destination arm based on maneuver
            maneuver = getattr(car, 'maneuver', 'STRAIGHT')
            dest_arm = EXIT_ARM.get(self.arm, {}).get(maneuver, "South")
            car._dest_arm = dest_arm
            # V6 D4: push snapshot to outbound buffer of the DESTINATION arm
            self._outbound.append(({"label":car.label,"car_type":car.car_type,
                "maneuver":maneuver, "dest_arm": dest_arm,
                "entry_arm": self.arm}, OUTBOUND_HOLD_TICKS * 3))
            self._platoon_cnt    += 1
            self._cycle_cleared  += 1
            self._queue.remove(car)
            self._road.append(car)
            for j, c in enumerate(self._queue):
                c.queue_pos = j
            rogue_flag = " ⛔ROGUE" if getattr(car, '_rogue_triggered', False) else ""
            car.log_event("CROSSED",
                f"CVC 21450 — entered box → exiting {dest_arm}{rogue_flag}")
            log.info(f"[{self.arm}] {car.label} crossed → {dest_arm} (total: {self._total_crossed}){rogue_flag}")

    def _tick_road(self, dt: float):
        road_snapshot = [
            {"car_id": c.car_id, "label": c.label, "lane": c.lane,
             "road_pos_m": getattr(c, "road_pos_m", 0.0),
             "speed_kmh": c.speed_kmh}
            for c in self._road
        ]
        to_done = []
        for car in self._road:
            if car.state == IState.CROSSING:
                car.speed_ms   = CROSS_SPEED_MS
                car.road_pos_m = getattr(car, "road_pos_m", 0.0) + car.speed_ms * dt
                if car.road_pos_m >= INTER_BOX_M:
                    car.state    = IState.DONE
                    car.speed_ms = DEPART_SPEED_MS
                    car.road_pos_m = INTER_BOX_M
            if car.state == IState.DONE:
                to_done.append(car)

        for car in to_done:
            if car in self._road:
                self._road.remove(car)
            self._done.append(car)

    # ── Counts ────────────────────────────────────────────────────────────────

    def queued_count(self)  -> int: return len(self._queue)
    def road_count(self)    -> int: return len(self._road)
    def done_count(self)    -> int: return len(self._done)
    def total_count(self)   -> int: return len(self._all)

    def status(self) -> dict:
        tput = self.get_throughput_rate()
        return {
            "arm":             self.arm,
            "queued":          self.queued_count(),
            "on_road":         self.road_count(),
            "done":            self.done_count(),
            "total":           self.total_count(),
            "total_crossed":   self._total_crossed,
            "green":           self._green,
            "emergency":       self._emergency_seen,
            "token_active":    self._tkm is not None,
            "corridor_active": self._corridor_active,
            "ped_hold":        self._ped_hold,
            "throughput_rate": round(tput, 1),
            "peak_throughput": round(self.peak_throughput, 1),
            "avg_cycle_cars":  round(self.avg_cycle_cars, 1),
            "ml_confidence":   round(self._tkm.ml_confidence if self._tkm else 0.0, 2),
            "ml_clearance":    round(self._tkm.ml_clearance  if self._tkm else 0.5, 2),
            "queue_cars": [
                {"label": c.label, "type": c.car_type,
                 "state": c.state, "slot": c.slot,
                 "dist": round(c.queue_dist_m, 0),
                 "cross_dist_m": round(c._cross_dist_m, 1),
                 "maneuver": getattr(c, "maneuver", "STRAIGHT"),
                 "token_holder": c.slot >= 0 and c.state in ("TOKEN_OK","MOVING","CROSSING")}
                for c in self._queue[:20]
            ],
            "road_cars": [
                {"label": c.label, "type": c.car_type, "state": c.state,
                 "progress": min(1.0, round(getattr(c,"road_pos_m",0.0)/INTER_BOX_M, 2)),
                 "dest_arm": getattr(c, "_dest_arm", "South"),
                 "maneuver": getattr(c, "maneuver", "STRAIGHT"),
                 "token_holder": c.slot >= 0 and c.state in ("TOKEN_OK","MOVING","CROSSING")}
                for c in self._road[:8]
            ],
            "outbound_cars": [
                {"label": s["label"], "type": s.get("car_type","SMART"),
                 "maneuver": s.get("maneuver","STRAIGHT"),
                 "dest_arm": s.get("dest_arm","South"),
                 "entry_arm": s.get("entry_arm", self.arm)}
                for s,_ in self._outbound[:6]
            ],
        }

    def get_token_events(self) -> list:
        evts = list(self.token_events)
        if self._tkm:
            evts.extend(self._tkm.events)
        return evts
