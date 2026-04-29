# =============================================================================
# handshake_v4/sim_intersection.py  —  Intersection Simulation Coordinator
#
# Features added:
#   #5  — Rush hour mode (--rush-hour, 40 cars/arm)
#   #8  — Rogue car (--rogue, one car per arm ignores its token slot)
#   #12 — Ambulance corridor (all arms yield when emergency detected)
#   #13 — Pedestrian crossing (random interrupts each cycle, or --pedestrian)
#   #14 — Uncontrolled intersection (--uncontrolled, CVC 21800 right-of-way)
#   #15 — Throughput stats across all arms
# =============================================================================
import time, random, threading, logging
from config import (ARMS, CARS_PER_ARM, RUSH_HOUR_CARS, FLEET, CarType,
                    Phase, SIM_TICK_S, MACHINE_ARMS, PEDESTRIAN_PROB,
                    PEDESTRIAN_HOLD_S, INT_NEAR_MISS_M)
from radio import Radio
from traffic_light import TrafficLight
from car_intersection import SmartCar, LegacyCar, EmergencyCar, RogueCar
from arm_queue import ArmQueue

log = logging.getLogger("v4.sim_int")


def build_fleet(arm: str, n: int, seed: int = 0,
                include_rogue: bool = False) -> list:
    random.seed(seed)
    cars  = []
    e_pos = max(1, n // 3)
    for i in range(n):
        lbl = f"{arm[0]}{i+1:02d}"
        if i == e_pos:
            cars.append(EmergencyCar(lbl + "🚨", arm, i, lane=i % 2))
        elif include_rogue and i == n // 2:
            # Feature #8: one rogue car mid-queue
            cars.append(RogueCar(lbl + "⛔R", arm, i, lane=i % 2))
        else:
            r = random.random()
            if r < FLEET["smart"]:
                cars.append(SmartCar(lbl + "S", arm, i, lane=i % 2))
            elif r < FLEET["smart"] + FLEET["legacy"]:
                cars.append(LegacyCar(lbl + "L", arm, i, lane=i % 2))
            else:
                cars.append(EmergencyCar(lbl + "🚨", arm, i, lane=i % 2))
    return cars


class IntersectionSim:
    def __init__(self, my_arms: list = None, cars_per_arm: int = CARS_PER_ARM,
                 loss: float = 0.0, host_light: bool = True,
                 rush_hour: bool = False, include_rogue: bool = False,
                 uncontrolled: bool = False, pedestrian_enabled: bool = True,
                 spat_delay: float = 0.0):
        self.my_arms      = my_arms or ARMS
        self.cars_per_arm = RUSH_HOUR_CARS if rush_hour else cars_per_arm
        self.loss         = loss
        self.host_light   = host_light
        self.rush_hour    = rush_hour
        self.uncontrolled = uncontrolled
        self._pedestrian_enabled = pedestrian_enabled
        self._running     = False
        self._thread      = None
        self.start_time   = 0.0
        self.tick_n       = 0
        self.events       = []

        # Feature #12: ambulance corridor state
        self._corridor_active = False
        self._corridor_arm    = None
        self._corridor_end    = 0.0

        # Feature #13: pedestrian timing
        self._ped_active  = False
        self._ped_end     = 0.0
        self._last_ped_check = 0.0

        # Feature #15: throughput history for graph
        self.throughput_history: list = []

        # V6 I4: intersection near-miss tracking
        self._box_near_miss_count = 0
        self._crossing_cars       = {}  # arm -> car currently crossing  # [(ts, total_done)]

        # Feature #22: SPaT delay queue
        self.spat_delay         = spat_delay
        self._spat_delay_queue: list = []   # (release_at, spat_dict)
        self._spat_lock         = threading.Lock()
        self._spat_stale_events: list = []

        if host_light:
            self.light = TrafficLight("INT-01")
            self.light.on_phase_change(self._on_phase_change)
        else:
            self.light = None

        self.queues: dict[str, ArmQueue] = {}
        for arm in self.my_arms:
            radio = Radio(f"ARM-{arm}", loss=loss)
            fleet = build_fleet(arm, self.cars_per_arm,
                                seed=42 + ARMS.index(arm) * 100,
                                include_rogue=include_rogue)
            self.queues[arm] = ArmQueue(arm, fleet, radio,
                                        light=self.light,
                                        uncontrolled=uncontrolled)

    # ── Phase callback ─────────────────────────────────────────────────────────

    def _on_phase_change(self, phase: str, green_arms: list):
        if isinstance(green_arms, str):
            green_arms = [green_arms]
        green_arms = green_arms or []

        # Feature #13: check for pedestrian trigger on each new green phase
        if phase in (Phase.NS_GREEN, Phase.EW_GREEN) and self._pedestrian_enabled:
            if (not self._ped_active
                    and random.random() < PEDESTRIAN_PROB
                    and time.time() - self._last_ped_check > 30.0):
                self._trigger_pedestrian()
                return

        for arm, q in self.queues.items():
            if arm in green_arms:
                q.on_green()
            else:
                q.on_red()

        self.events.append({
            "ts": time.time(), "event": "LIGHT_CHANGE",
            "label": "LIGHT", "phase": phase, "green": green_arms,
        })

    # ── Feature #13: Pedestrian crossing ────────────────────────────────────

    def _trigger_pedestrian(self):
        self._ped_active     = True
        self._ped_end        = time.time() + PEDESTRIAN_HOLD_S
        self._last_ped_check = time.time()

        if self.light and self.host_light:
            self.light.pedestrian_crossing()

        for q in self.queues.values():
            q.on_pedestrian()

        self.events.append({
            "ts": time.time(), "event": "PEDESTRIAN",
            "label": "CROSSING",
            "detail": f"🚶 PEDESTRIAN CROSSING — all arms hold {PEDESTRIAN_HOLD_S}s  CVC 21950",
        })
        log.info("[IntersectionSim] Pedestrian crossing triggered")

        def _clear():
            time.sleep(PEDESTRIAN_HOLD_S + 0.5)
            self._ped_active = False
            for q in self.queues.values():
                q.on_pedestrian_clear()
        threading.Thread(target=_clear, daemon=True).start()

    def trigger_pedestrian(self):
        """Public — called from main with --pedestrian flag."""
        self._trigger_pedestrian()

    # ── Feature #12: Ambulance corridor ──────────────────────────────────────

    def trigger_ambulance_corridor(self, arm: str = "North"):
        """
        Force all four arms to yield simultaneously — simulates an ambulance
        needing a clear path through the full intersection.
        """
        self._corridor_active = True
        self._corridor_arm    = arm
        self._corridor_end    = time.time() + 15.0

        if self.light and self.host_light:
            self.light.preempt(arm)

        for q in self.queues.values():
            q.on_ambulance_corridor(f"AMBULANCE-{arm}")

        self.events.append({
            "ts": time.time(), "event": "AMBULANCE_CORRIDOR",
            "label": "EMERG",
            "detail": f"🚨 AMBULANCE CORRIDOR OPEN — {arm} arm  CVC 21806",
        })
        log.info(f"[IntersectionSim] Ambulance corridor opened for {arm}")

        def _clear():
            time.sleep(16.0)
            self._corridor_active = False
            for q in self.queues.values():
                q.on_corridor_clear()
        threading.Thread(target=_clear, daemon=True).start()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        self.start_time = time.time()
        if self.host_light:
            self.light.start()
            self._on_phase_change(Phase.NS_GREEN, ["North", "South"])
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="sim-intersection")
        self._thread.start()

        # Feature #12: schedule ambulance corridor demo at T+40s if rush hour
        if self.rush_hour:
            def _schedule_corridor():
                time.sleep(40.0)
                if self._running:
                    self.trigger_ambulance_corridor("North")
            threading.Thread(target=_schedule_corridor, daemon=True).start()

        log.info(f"[IntersectionSim] started — rush={self.rush_hour} "
                 f"uncontrolled={self.uncontrolled} arms={self.my_arms} "
                 f"cars/arm={self.cars_per_arm}")

    def stop(self):
        self._running = False
        if self.host_light and self.light:
            self.light.stop()

    def _loop(self):
        dt = SIM_TICK_S
        while self._running:
            t0 = time.time()
            self.tick_n += 1
            for arm, q in self.queues.items():
                if getattr(q, "_dead", False):
                    continue   # Feature #21: skip dead arm
                q.tick(dt, other_queues=self.queues if self.uncontrolled else None)

            # Feature #25: right-of-way negotiation check (every 10 ticks)
            if self.tick_n % 10 == 0:
                self._check_row_negotiation()

            # Feature #15: throughput snapshot every 5s
            if self.tick_n % 50 == 0:
                done = sum(q.done_count() for q in self.queues.values())
                self.throughput_history.append((time.time(), done))
                if len(self.throughput_history) > 60:
                    self.throughput_history = self.throughput_history[-30:]

            # Feature #22: broadcast SPaT and apply delay if set
            if self.light and self.tick_n % 4 == 0:   # every ~400ms
                spat = self.light.get_spat()
                if self.spat_delay > 0:
                    release_at = time.time() + self.spat_delay
                    with self._spat_lock:
                        self._spat_delay_queue.append((release_at, spat))
                    # Flush expired SPaT packets to arm callbacks
                    now = time.time()
                    with self._spat_lock:
                        ready, remaining = [], []
                        for (ra, sp) in self._spat_delay_queue:
                            (ready if now >= ra else remaining).append((ra, sp))
                        self._spat_delay_queue = remaining
                    for (_, sp) in ready:
                        age = now - sp.get("ts", now)
                        if age >= self.spat_delay * 0.8:
                            stale_evt = {
                                "ts":     now,
                                "event":  "SPAT_STALE",
                                "label":  "LIGHT",
                                "detail": (f"📡 SPaT delay={self.spat_delay:.1f}s "
                                           f"phase={sp.get('phase','?')} "
                                           f"age={age:.2f}s — VM2 using stale phase info"),
                            }
                            self.events.append(stale_evt)

            elapsed = time.time() - t0
            sleep_t = max(0.0, dt - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── Status & events ────────────────────────────────────────────────────────

    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

    def get_throughput_rate(self) -> float:
        """Feature #15: cars/min over the recent window."""
        if len(self.throughput_history) < 2:
            return 0.0
        t0, c0 = self.throughput_history[0]
        t1, c1 = self.throughput_history[-1]
        dur = max(0.1, t1 - t0)
        return (c1 - c0) / dur * 60

    def _check_box_near_misses(self):
        """V6 I4: detect crossing cars from different arms with overlapping paths."""
        crossing = []
        for arm, q in self.queues.items():
            for car in q._road:
                if hasattr(car,'queue_dist_m') and car.queue_dist_m < car._cross_dist_m:
                    progress = 1.0 - car.queue_dist_m / max(car._cross_dist_m, 0.01)
                    crossing.append({'arm':arm,'car':car,'progress':progress})
        # Check each pair from different arms
        for i in range(len(crossing)):
            for j in range(i+1, len(crossing)):
                a, b = crossing[i], crossing[j]
                if a['arm'] == b['arm']: continue
                # Simple proximity check: both ~50% through = potential overlap
                if abs(a['progress'] - b['progress']) < 0.25:
                    self._box_near_miss_count += 1
                    self.events.append({
                        'ts': time.time(), 'event': 'NEAR_MISS_BOX',
                        'arm': f"{a['arm'][0]}-{b['arm'][0]}",
                        'detail': f'!! NEAR-MISS: {a["car"].label} x {b["car"].label} in box'
                    })

    def get_status(self) -> dict:
        total  = self.cars_per_arm * len(self.my_arms)
        done   = sum(q.done_count()   for q in self.queues.values())
        queued = sum(q.queued_count() for q in self.queues.values())
        road   = sum(q.road_count()   for q in self.queues.values())

        spat = self.light.get_spat() if self.light else {
            "phase": "UNKNOWN", "green_arms": [],
            "time_remaining": 0, "preempted": False
        }

        # Feature #15: aggregate throughput across arms
        tput      = self.get_throughput_rate()
        peak_tput = max((q.peak_throughput for q in self.queues.values()), default=0)

        return {
            "elapsed_s":        round(self.elapsed(), 1),
            "tick":             self.tick_n,
            "total_cars":       total,
            "done":             done,
            "queued":           queued,
            "on_road":          road,
            "pct_done":         round(100 * done / total, 1) if total else 0,
            "all_done":         done >= total,
            "light_phase":      spat["phase"],
            "light_green":      spat["green_arms"],
            "light_remaining":  spat.get("time_remaining", 0),
            "preempted":        spat.get("preempted", False),
            "pedestrian":       self._ped_active,
            "corridor_active":  self._corridor_active,
            "corridor_arm":     self._corridor_arm,
            "rush_hour":        self.rush_hour,
            "uncontrolled":     self.uncontrolled,
            "throughput_rate":  round(tput, 1),
            "peak_throughput":  round(peak_tput, 1),
            "arms":             {arm: q.status() for arm, q in self.queues.items()},
            "dead_arms":        [arm for arm, q in self.queues.items()
                                 if getattr(q, "_dead", False)],
            "spat_delay":       self.spat_delay,
            "spat_stale_count": len(self._spat_stale_events),
            "box_near_miss_count": self._box_near_miss_count,
        }

    def get_token_events(self) -> list:
        evts = list(self.events)
        for q in self.queues.values():
            evts.extend(q.get_token_events())
        evts.sort(key=lambda e: e.get("ts", 0))
        return evts

    def is_all_done(self) -> bool:
        """Matches RoadSim API — returns True when all cars have cleared."""
        s = self.get_status()
        return s.get("all_done", False)

    # ── Feature #21: Machine / arm fail ─────────────────────────────────────────

    def trigger_machine_fail(self, arm: str, duration: float = 20.0):
        """
        Simulate one arm's machine going down mid-simulation.
        That arm freezes (no new crossings, no token manager).
        Other arms continue normally — demonstrates distributed resilience.
        """
        if arm not in self.queues:
            return
        q = self.queues[arm]
        q.on_red()   # force red — arm stops processing
        q._dead = True   # mark as failed

        self.events.append({
            "ts":     time.time(),
            "event":  "MACHINE_FAIL",
            "label":  arm,
            "detail": f"⚠ ARM {arm} MACHINE FAIL — simulating VM crash  "
                      f"Other arms continue independently",
        })
        log.info(f"[IntersectionSim] Machine fail triggered for {arm}")

        # Schedule recovery
        def _recover():
            import time as _t
            _t.sleep(duration)
            if self._running:
                q._dead = False
                self.events.append({
                    "ts":     _t.time(),
                    "event":  "MACHINE_RECOVER",
                    "label":  arm,
                    "detail": f"✅ ARM {arm} MACHINE RECOVERED — rejoining simulation",
                })
        threading.Thread(target=_recover, daemon=True).start()

    # ── Feature #25: Right-of-way extended negotiation ────────────────────────

    def _check_row_negotiation(self):
        """
        Feature #25: When uncontrolled mode, log visible CVC 21800 negotiation
        when two arms have cars at the stop line simultaneously.
        """
        if not self.uncontrolled:
            return
        from law import Law
        front_cars = {}
        for arm, q in self.queues.items():
            if q._queue and q._queue[0].queue_dist_m < 8.0:
                front_cars[arm] = q._queue[0]

        if len(front_cars) < 2:
            return

        arms = list(front_cars.keys())
        for i, arm_a in enumerate(arms):
            for arm_b in arms[i+1:]:
                car_a = front_cars[arm_a]
                car_b = front_cars[arm_b]
                # Who yields to whom?
                a_yields = Law.right_of_way_cvc21800(arm_a, arm_b)
                if a_yields:
                    yielder, goer = arm_a, arm_b
                else:
                    yielder, goer = arm_b, arm_a

                evt = {
                    "ts":     time.time(),
                    "event":  "ROW_NEGOTIATION",
                    "arm":    yielder,
                    "detail": (f"⚖ CVC 21800: {front_cars[yielder].label} ({yielder}) YIELDS "
                               f"to {front_cars[goer].label} ({goer}) — vehicle on right has priority"),
                }
                # Only log once per car pair per second
                key = f"ROW_{yielder}_{goer}"
                if not hasattr(self, '_row_logged'):
                    self._row_logged = {}
                if time.time() - self._row_logged.get(key, 0) > 5.0:
                    self._row_logged[key] = time.time()
                    self.events.append(evt)
                    # Force yielder to stop
                    q_yield = self.queues[yielder]
                    if q_yield._queue:
                        q_yield._queue[0].speed_ms = 0.0

    def get_all_events(self) -> list:
        evts = list(self.events)
        for q in self.queues.values():
            for car in q._all:
                evts.extend(car.events[-6:])
        evts.sort(key=lambda e: e.get("ts", 0))
        return evts[-40:]
