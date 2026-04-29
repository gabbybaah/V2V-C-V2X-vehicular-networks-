# =============================================================================
# handshake_v5/car_road.py
#
# Changes from V4:
#  - All NPCs get a behavior profile (smart + legacy)
#  - lateral_offset_m added to every car (smooth lane transitions)
#  - LegacyNPC gets visual_range detection (reacts to what it physically sees)
#  - Emergency corridor: cars move to shoulder (lane 2) to clear lane 1
#  - Mixed spawn — profile colour exported for dashboard
#  - RState.SHOULDER added for shoulder-running cars
# =============================================================================
import time, threading, logging, random, math
from collections import deque
import numpy as np
from config import (CarType, RState, SPEED_LIMIT_MS, SCHOOL_SPEED_MS,
                    EMERGENCY_MS, MIN_GAP_M, FOLLOW_TIME_S, LANE_CLEAR_M,
                    OVERTAKE_BOOST_MS, SIM_TICK_S, SIGNAL_DIST_M,
                    PREEMPT_RANGE_M, PLATOON_CLOSE_M, ONCOMING_CLEAR_M,
                    BREAKDOWN_WARN_RANGE_M, ZONE_WARN_RANGE_M, Msg,
                    CONVOY_SIZE, CONVOY_SPACING_M,
                    REACTION_BUFFER_TICKS, WEATHER_PARAMS, WeatherState,
                    FUEL_RANGE_M, FUEL_LOW_PCT, CAR_LENGTH_M)
from road_geometry import RG, ROAD_LENGTH_M, SCHOOL_ZONE_START_M, SCHOOL_ZONE_END_M
from behavior_profiles import (make_profile, profile_color, profile_label,
                                COMMUTER, random_profile)
from law import Law

log = logging.getLogger("v5.car_road")

_id_counter = 0
def _make_id(label:str)->str:
    global _id_counter; _id_counter += 1
    import hashlib
    return hashlib.sha256(f"{label}{_id_counter}".encode()).hexdigest()[:10]

VISUAL_RANGE_M = 120.0   # legacy cars "see" this far ahead physically


# ── Base road car ─────────────────────────────────────────────────────────────
class BaseRoadCar:
    def __init__(self, label, car_type, lane=0, start_pos=0.0,
                 speed=None, profile_name=None):
        self.label          = label
        self.car_type       = car_type
        self.car_id         = _make_id(label)
        self.lane           = lane
        self.road_pos_m     = start_pos
        self.speed_ms       = speed if speed is not None else SPEED_LIMIT_MS*random.uniform(0.7,1.0)
        self.state          = RState.DRIVING
        self.neighbours     = {}
        self.events         = []
        self.msgs_sent      = 0
        self.msgs_recv      = 0
        self.law_log        = []
        self.trust          = {}
        self._lock          = threading.Lock()
        self.last_intent    = None
        self.token_events   = []
        # V5: lateral position within lane (metres from lane centre, ±LANE_WIDTH/2)
        self.lateral_offset_m = 0.0
        self._lateral_target  = 0.0   # smooth transition target
        # V5: behavior profile
        self.profile_name   = profile_name or COMMUTER
        self.profile        = make_profile(self.profile_name)
        self.profile_color  = profile_color(self.profile_name)
        # V5: lane change smooth tracking
        self._lc_in_progress = False
        self._target_lane    = lane
        self._lc_from_lane   = lane   # lane we came FROM (for strip animation)
        # V5 ML: trajectory buffer (lazy-init to avoid circular import at module load)
        self._traj_buffer    = None
        self._ml_ready       = False   # set True after first buffer init
        self._last_ml_conf   = 0.0    # last confidence value — for dashboard
        self._last_ml_action = -1     # last predicted action index
        self._ml_reload_tick = 0      # counter for periodic model reload check

        # V6 R1: reaction time buffer
        self._ahead_history   = deque(maxlen=REACTION_BUFFER_TICKS)
        self.reaction_delay_s = self.profile.reaction_delay_s
        # V6 D1: indicate-before-lane-change countdown
        self._indicate_ticks  = 0
        # Corridor dedup: remember which ambulance IDs we've already logged
        self._corridor_logged = set()
        # V2X proactive avoidance: stored hazard + zone positions from radio
        self._hazard_pos   = None   # road_pos_m of a known breakdown/hazard ahead
        self._hazard_lane  = None   # its lane
        self._zone_start_m = None   # road_pos_m of next zone start (from ZONE_ALERT)
        # V6 R4: fuel gauge — varied levels so behaviour differs per car
        _fr = random.random()
        if _fr < 0.12:
            self.fuel_pct = random.uniform(0.05, 0.15)   # nearly empty — will slow soon
        elif _fr < 0.30:
            self.fuel_pct = random.uniform(0.15, 0.40)   # low — will run dry mid-sim
        else:
            self.fuel_pct = random.uniform(0.40, 1.0)    # normal range
        self._fuel_out_logged = False
        # V6 R2: weather reference (set by sim each tick)
        self._weather_state   = WeatherState.CLEAR

    def _ensure_buffer(self):
        """Lazy-init trajectory buffer and predictor reference. Hot-reloads model."""
        if self._traj_buffer is None:
            from trajectory_buffer import TrajectoryBuffer
            self._traj_buffer = TrajectoryBuffer()
        if not self._ml_ready:
            try:
                from ml_predictor import get_predictor
                self._predictor  = get_predictor()
                self._ml_ready   = True
            except Exception:
                self._predictor  = None
        else:
            # Every ~100 ticks check if model file was updated, trigger reload
            self._ml_reload_tick += 1
            if self._ml_reload_tick >= 100:
                self._ml_reload_tick = 0
                try:
                    from config import ML_MODEL_PATH
                    import os, time as _t
                    if (os.path.exists(ML_MODEL_PATH) and
                            not self._predictor._loaded):
                        self._predictor._try_load()
                except Exception:
                    pass

    @property
    def speed_kmh(self): return self.speed_ms * 3.6

    def log_event(self, event, detail=""):
        entry = {"ts":time.time(),"event":event,"detail":detail,"label":self.label}
        self.events.append(entry)
        self.token_events.append(entry)
        if len(self.events) > 200: self.events = self.events[-100:]

    def log_law(self, cvc, reason):
        self.law_log.append({"cvc":cvc,"reason":reason,"ts":time.time()})

    def _add_token_event(self, event, detail):
        entry = {"ts":time.time(),"event":event,"detail":detail,"label":self.label}
        self.token_events.append(entry)
        if len(self.token_events) > 100: self.token_events = self.token_events[-60:]

    def status(self):
        return {
            "label":self.label, "car_type":self.car_type,
            "lane":self.lane,   "road_pos_m":round(self.road_pos_m,1),
            "speed_kmh":round(self.speed_kmh,1), "state":self.state,
            "msgs_sent":self.msgs_sent, "msgs_recv":self.msgs_recv,
            "neighbours":len(self.neighbours), "is_player":isinstance(self,PlayerCar),
            "profile":self.profile_name, "profile_color":self.profile_color,
            "lateral_offset_m":round(self.lateral_offset_m,3),
            "ml_conf":round(self._last_ml_conf, 3),
            "ml_action":self._last_ml_action,
            "fuel_pct":round(getattr(self,"fuel_pct",1.0), 3),
            "indicating": getattr(self,"_indicate_ticks",0) > 0,
            "target_lane": getattr(self,"_target_lane", self.lane),
            "weather_state": getattr(self,"_weather_state", 0),
            "machine_id": getattr(self,"machine_id", 0),
            "lc_from_lane": getattr(self,"_lc_from_lane", self.lane),
            "lc_progress": max(0.0, 1.0 - abs(self.lateral_offset_m) / max(0.1, RG.LANE_WIDTH_M)) if self._lc_in_progress else 1.0,
        }

    def _find_ahead(self, cars):
        candidates = [c for c in cars
                      if c.get("lane")==self.lane
                      and c.get("road_pos_m",0) > self.road_pos_m
                      and c.get("car_id") != self.car_id]
        if not candidates: return None
        nearest = min(candidates, key=lambda c: c["road_pos_m"])

        # V6 R1: push to history buffer, return delayed view
        self._ahead_history.append(nearest)
        delay = max(0, min(int(self.reaction_delay_s / SIM_TICK_S), len(self._ahead_history)-1))
        nearest = self._ahead_history[-(delay+1)]

        # ML enhancement: replace position with 2s prediction if confident
        self._ensure_buffer()
        if self._ml_ready and self._traj_buffer is not None:
            cid     = nearest.get("car_id")
            history = self._traj_buffer.get(cid) if cid else None
            if history is not None:
                try:
                    from config import ML_CONFIDENCE_THRESHOLD
                    from ml_predictor import ACTION_HARD_BRAKE, ACTION_SOFT_BRAKE
                    pred_pos, pred_spd, act_probs, conf = self._predictor.predict(history)
                    if conf >= ML_CONFIDENCE_THRESHOLD and pred_pos > self.road_pos_m:
                        pred_action = int(np.argmax(act_probs))
                        nearest = dict(nearest)
                        nearest["road_pos_m"] = pred_pos
                        nearest["speed_kmh"]  = pred_spd * 3.6
                        nearest["ml_conf"]    = conf
                        nearest["ml_action"]  = pred_action
                        # Log predicted brake events proactively
                        if conf >= 0.75 and pred_action in (ACTION_HARD_BRAKE, ACTION_SOFT_BRAKE):
                            evtype = "PREDICTED_HARD_BRAKE" if pred_action == ACTION_HARD_BRAKE \
                                     else "PREDICTED_SOFT_BRAKE"
                            self._add_token_event(evtype,
                                f"🧠 {self.label} predicts {nearest.get('label','?')} "
                                f"will brake — conf={conf:.2f}  CVC 21703")
                        # Expose on self for dashboard
                        self._last_ml_conf   = conf
                        self._last_ml_action = pred_action
                except Exception:
                    pass
        return nearest

    def _find_behind(self, cars):
        candidates = [c for c in cars
                      if c.get("lane")==self.lane
                      and c.get("road_pos_m",0) < self.road_pos_m
                      and c.get("car_id") != self.car_id]
        if not candidates: return None
        return max(candidates, key=lambda c: c["road_pos_m"])

    def _in_school_zone(self):
        return RG.in_zone("SCHOOL_1",self.road_pos_m)

    def _approaching_school_zone(self):
        z = RG.ZONES["SCHOOL_1"]
        return z[0]-ZONE_WARN_RANGE_M <= self.road_pos_m < z[0]

    def _speed_limit_here(self):
        base = RG.speed_limit_at(self.road_pos_m)
        # V6 R2: weather multiplier
        w = WEATHER_PARAMS.get(self._weather_state, WEATHER_PARAMS[WeatherState.CLEAR])
        return base * w["speed"]

    def set_weather(self, state):
        """Called by sim each tick to push weather state."""
        self._weather_state = state

    def _following_dist_mult(self):
        w = WEATHER_PARAMS.get(self._weather_state, WEATHER_PARAMS[WeatherState.CLEAR])
        return w["follow"]

    def _brake_friction(self):
        w = WEATHER_PARAMS.get(self._weather_state, WEATHER_PARAMS[WeatherState.CLEAR])
        return w["brake"]

    def _tick_fuel(self, dt):
        """V6 R4: deplete fuel. Returns True if just ran out."""
        if self.state in (RState.BROKEN_DOWN, RState.DONE): return False
        self.fuel_pct -= max(0.0, self.speed_ms) * dt / FUEL_RANGE_M
        self.fuel_pct  = max(0.0, self.fuel_pct)
        if self.fuel_pct <= 0.0 and not self._fuel_out_logged:
            self._fuel_out_logged = True
            return True
        return False

    def _tick_lateral(self, dt):
        """V6 D1: hold during indicate phase, then smooth slide."""
        if self._indicate_ticks > 0:
            self._indicate_ticks -= 1
            return
        diff = self._lateral_target - self.lateral_offset_m
        if abs(diff) < 0.05:
            self.lateral_offset_m = self._lateral_target
            if self._lc_in_progress:
                self._lc_in_progress = False   # slide complete
        else:
            rate = RG.LANE_WIDTH_M / RG.LATERAL_SNAP_S
            self.lateral_offset_m += math.copysign(min(abs(diff), rate*dt), diff)

    def _start_lane_change(self, new_lane):
        if new_lane == self.lane: return
        self._indicate_ticks  = int(0.8 / SIM_TICK_S)   # 8 ticks blinker
        self._lc_from_lane    = self.lane
        # Place the car's visual at the old lane centre relative to new lane
        # so it slides smoothly across rather than snapping
        self.lateral_offset_m = (self.lane - new_lane) * RG.LANE_WIDTH_M
        self._target_lane     = new_lane
        self.lane             = new_lane
        self._lateral_target  = 0.0
        self._lc_in_progress  = True

    def _calc_oncoming_clear(self, all_cars):
        oncoming_lane = 1 - self.lane
        oncoming = [c for c in all_cars
                    if c.get("lane")==oncoming_lane
                    and c.get("road_pos_m",0) > self.road_pos_m]
        if not oncoming: return 999.0
        return min(oncoming, key=lambda c: c["road_pos_m"])["road_pos_m"] - self.road_pos_m


# ── Emergency corridor helpers ────────────────────────────────────────────────
def _handle_emergency_corridor(car, dt):
    """
    Move car to shoulder (lane 2) to clear lane 1 for emergency vehicle.
    Called when EMERG_PREEMPT received and car is in lane 0 or 1.
    """
    if car.state == RState.DONE: return
    if car.lane == 1:
        # Fast lane → slow lane
        car._start_lane_change(0)
        car.speed_ms = max(2.0, car.speed_ms - 3.0*dt)
        car.state    = RState.YIELDING
    elif car.lane == 0:
        # Slow lane → shoulder
        car._start_lane_change(2)
        car.speed_ms = max(0.5, car.speed_ms - 4.0*dt)
        car.state    = RState.SHOULDER
    # Lane 2 (shoulder) cars stay put


# ── Player Car ────────────────────────────────────────────────────────────────
class PlayerCar(BaseRoadCar):
    def __init__(self, label, lane=0, start_pos=200.0, machine_id=1):
        super().__init__(label, CarType.SMART, lane, start_pos,
                         speed=SPEED_LIMIT_MS*0.85, profile_name=COMMUTER)
        self.machine_id  = machine_id
        self._cmd_queue  = []
        self._cmd_lock   = threading.Lock()
        self.profile_color = "bold bright_white"  # player always white
        self._manual_lane_ts = 0.0  # timestamp of last manual lane command

    def enqueue_command(self, cmd):
        with self._cmd_lock: self._cmd_queue.append(cmd.strip().lower())

    def _pop_command(self):
        with self._cmd_lock:
            if self._cmd_queue: return self._cmd_queue.pop(0)
        return None

    def send_beacon(self, radio):
        radio.send({"type":Msg.BEACON,"from":self.car_id,"label":self.label,
                    "car_type":self.car_type,"lane":self.lane,
                    "road_pos_m":round(self.road_pos_m,1),
                    "speed_kmh":round(self.speed_kmh,1),"state":self.state,
                    "machine_id":self.machine_id,"is_player":True,
                    "profile":self.profile_name,"lateral_offset_m":round(self.lateral_offset_m,3),
                    "lc_from_lane":self._lc_from_lane,
                    "lc_progress":round(max(0.0, 1.0 - abs(self.lateral_offset_m) / max(0.1, RG.LANE_WIDTH_M)) if self._lc_in_progress else 1.0, 3),
                    "fuel_pct":round(getattr(self,"fuel_pct",1.0),3),
                    "ts":time.time()})
        self.msgs_sent += 1

    def receive(self, msg):
        self.msgs_recv += 1
        mtype = msg.get("type"); from_id = msg.get("from")
        if from_id == self.car_id: return
        if from_id and from_id not in self.trust: self.trust[from_id] = 128
        if mtype == Msg.BEACON: self.neighbours[from_id] = msg
        elif mtype == Msg.EMERG_PREEMPT: self._handle_emergency(msg)
        elif mtype == Msg.HARD_BRAKE:    self._handle_chain_brake(msg)
        elif mtype == Msg.HAZARD:        self._handle_hazard(msg)
        elif mtype == Msg.ZONE_ALERT:    self._handle_zone_alert(msg)
        elif mtype == Msg.INTENT_OT:
            self._add_token_event("INTENT_RECV",
                f"◆ {msg.get('label','?')} INTENT_OVERTAKE  CVC 21750")
        elif mtype == Msg.INTENT_LC:
            self._add_token_event("INTENT_RECV",
                f"◆ {msg.get('label','?')} INTENT_LANE_CHG  CVC 22107")

    def _handle_emergency(self, msg):
        if self.state == RState.DONE: return
        emerg_pos  = msg.get("road_pos_m", self.road_pos_m)
        dist_ahead = emerg_pos - self.road_pos_m
        if dist_ahead < 150 and dist_ahead > -50:
            _handle_emergency_corridor(self, 0.1)
        from_id = msg.get("from","")
        if from_id not in self._corridor_logged:
            self._corridor_logged.add(from_id)
            self._add_token_event("CORRIDOR",
                f"🚨 {msg.get('label','?')} → {self.label} moving to shoulder  CVC 21806")

    def _handle_chain_brake(self, msg):
        from_type  = msg.get("car_type", CarType.SMART)
        from_pos   = msg.get("road_pos_m", self.road_pos_m)
        dist_ahead = from_pos - self.road_pos_m
        if 0 < dist_ahead < 600:
            # Scale reaction by distance: 0m = full brake, 600m = very gentle
            weather_brake = WEATHER_PARAMS.get(self._weather_state, WEATHER_PARAMS[0]).get("brake", 1.0)
            urgency   = max(0.0, 1.0 - dist_ahead / 600.0)
            # Braking efficiency reduced in bad weather (less grip = longer stopping)
            v2x_brake = urgency * 4.0 / max(0.3, weather_brake)
            self.speed_ms = max(0.0, self.speed_ms - v2x_brake)
            if from_type == CarType.LEGACY:
                self._add_token_event("LEGACY_NO_WARN",
                    f"⚠ LEGACY {msg.get('label','?')} surprise brake — no V2X!  CVC 21703")
            else:
                self._add_token_event("SMART_CHAIN_WARN",
                    f"📡 V2X chain {msg.get('label','?')}→{self.label}  {dist_ahead:.0f}m  CVC 21703")

    def _handle_hazard(self, msg):
        h_pos  = msg.get("road_pos_m", 0)
        h_lane = msg.get("lane", 0)
        dist   = h_pos - self.road_pos_m
        if 0 < dist < 400:
            self._hazard_pos  = h_pos
            self._hazard_lane = h_lane
            weather_brake = WEATHER_PARAMS.get(self._weather_state, WEATHER_PARAMS[0]).get("brake", 1.0)
            urgency = max(0.0, 1.0 - dist / 400.0)
            self.speed_ms = max(2.0, self.speed_ms - urgency * 3.0 / max(0.3, weather_brake))
            self._add_token_event("HAZARD_WARN",
                f"🔶 V2X HAZARD {msg.get('label','?')} at {h_pos:.0f}m  dist={dist:.0f}m  CVC 22500")

    def _handle_zone_alert(self, msg):
        zone_start = msg.get("zone_start_m", self.road_pos_m + 200)
        if self._zone_start_m is None:
            self._zone_start_m = zone_start
        self._add_token_event("ZONE_ALERT",
            f"🏫 ZONE at {zone_start:.0f}m — {msg.get('speed_limit_kmh',24):.0f} km/h  CVC 22352a")

    def tick(self, dt, radio, all_cars):
        if self.state == RState.DONE: return
        # V5 ML: update trajectory buffer each tick
        self._ensure_buffer()
        if self._traj_buffer is not None:
            self._traj_buffer.update(all_cars)
        self.send_beacon(radio)
        self._tick_lateral(dt)
        # Players never run out of fuel and never break down — fuel display stays at 100%
        self.fuel_pct = 1.0
        if self._approaching_school_zone():
            school_z = RG.ZONES.get("SCHOOL_1", (800, 1100))
            radio.send({"type":Msg.ZONE_ALERT,"from":self.car_id,"label":self.label,
                        "zone_type":"SCHOOL","zone_start_m":school_z[0],
                        "speed_limit_ms":SCHOOL_SPEED_MS,
                        "speed_limit_kmh":round(SCHOOL_SPEED_MS*3.6,1),
                        "ts":time.time()})
        cmd = self._pop_command()
        if cmd: self._execute_command(cmd, radio, all_cars)
        self._auto_drive(dt, radio, all_cars)
        self._enforce_zones(dt, radio)
        # Return from shoulder when emergency has cleared 150m ahead
        if self.state == RState.SHOULDER:
            still_emerg = any(
                v.get("car_type")==CarType.EMERGENCY
                and v.get("road_pos_m",0) > self.road_pos_m  # still ahead
                and v.get("road_pos_m",0) < self.road_pos_m + 150  # within 150m
                for v in self.neighbours.values())
            if not still_emerg:
                self._start_lane_change(0)
                self.state = RState.DRIVING
                self._corridor_logged.clear()
                self._add_token_event("YIELD_DONE", f"✅ {self.label} returning — emergency cleared 150m ahead")
        if self.state == RState.YIELDING:
            still_emerg = any(
                v.get("car_type")==CarType.EMERGENCY
                and v.get("road_pos_m",0) > self.road_pos_m
                and v.get("road_pos_m",0) < self.road_pos_m + 150
                for v in self.neighbours.values())
            if not still_emerg:
                self._start_lane_change(0) if self.lane==2 else None
                self.state = RState.DRIVING
        if self.state not in (RState.SHOULDER,):
            self.road_pos_m += self.speed_ms * dt
        elif self.lane == 2:
            self.road_pos_m += self.speed_ms * 0.3 * dt  # crawl on shoulder
        if self.road_pos_m >= ROAD_LENGTH_M:
            self.state = RState.DONE

    def _enforce_zones(self, dt, radio):
        limit = self._speed_limit_here()
        if self.speed_ms > limit:
            self.speed_ms = max(limit, self.speed_ms - 5.0*dt)
            if self.state != RState.SCHOOL_ZONE and limit <= SCHOOL_SPEED_MS:
                self.state = RState.SCHOOL_ZONE
                self._add_token_event("SCHOOL_ZONE",
                    f"🏫 {self.label} clamped to {limit*3.6:.0f} km/h  CVC 22352a")
        elif self.state == RState.SCHOOL_ZONE:
            self.state = RState.DRIVING

    def _execute_command(self, cmd, radio, all_cars):
        self._add_token_event("CMD", f"▶ YOU: {cmd}")
        if cmd in ("left","l"):
            target = 1
            if self.lane != target:
                ahead  = self._find_ahead(all_cars)
                behind = self._find_behind(all_cars)
                gap_a  = (ahead["road_pos_m"]-self.road_pos_m-4.5) if ahead else 999
                gap_b  = (self.road_pos_m-behind["road_pos_m"]-4.5) if behind else 999
                radio.send({"type":Msg.INTENT_LC,"from":self.car_id,"label":self.label,
                            "new_lane":target,"road_pos_m":self.road_pos_m,"ts":time.time()})
                self.msgs_sent += 1
                ok,cvc,reason = Law.may_change_lane(gap_a,gap_b,signalled=True)
                if ok:
                    self._start_lane_change(target); self.state = RState.LANE_CHANGE
                    self._manual_lane_ts = time.time()
                    self._add_token_event("LANE_CHANGE",
                        f"→ {self.label} → lane {target}  CVC 22107/21658")
                else:
                    self._add_token_event("CMD_BLOCKED",f"✗ Left BLOCKED — {reason}  {cvc}")
        elif cmd in ("right","r"):
            target = 0
            if self.lane != target:
                radio.send({"type":Msg.INTENT_LC,"from":self.car_id,"label":self.label,
                            "new_lane":target,"road_pos_m":self.road_pos_m,"ts":time.time()})
                self.msgs_sent += 1
                self._start_lane_change(target); self.state = RState.LANE_CHANGE
                self._manual_lane_ts = time.time()
                self._add_token_event("LANE_CHANGE",f"→ {self.label} → lane {target}  CVC 22107")
        elif cmd in ("overtake","ot","o"):
            # Find nearest car ahead in any lane, or just accelerate past
            ahead = self._find_ahead(all_cars)
            # Also check fast lane for a gap to move into
            target_lane = 1 if self.lane == 0 else 0
            gap_to_ahead = (ahead["road_pos_m"] - self.road_pos_m) if ahead else 999
            if gap_to_ahead > 200:
                self._add_token_event("CMD_BLOCKED","✗ Overtake — no car close enough to overtake")
            else:
                # Real-world overtake: find space, move, accelerate past
                # CVC is logged for record but does NOT block the manoeuvre
                oncoming_gap = self._calc_oncoming_clear(all_cars)
                _, cvc, _ = Law.may_overtake(self.speed_ms,
                    ahead.get("speed_kmh",0)/3.6 if ahead else 0, oncoming_gap)
                radio.send({"type":Msg.INTENT_OT,"from":self.car_id,"label":self.label,
                            "target":ahead.get("label","?") if ahead else "?",
                            "road_pos_m":self.road_pos_m,"ts":time.time()})
                self.msgs_sent += 1
                self._start_lane_change(target_lane)
                # Boost speed 15 km/h above current (or to speed limit +10%)
                boost = min(SPEED_LIMIT_MS * 1.10, self.speed_ms + OVERTAKE_BOOST_MS * 1.5)
                self.speed_ms = boost
                self.state    = RState.OVERTAKING
                tgt_label     = ahead.get("label","?") if ahead else "open road"
                self._add_token_event("OVERTAKE",
                    f"→ {self.label} OVERTAKING {tgt_label}  [{cvc}]")
        elif cmd in ("brake","b"):
            self.speed_ms = max(0.0, self.speed_ms*0.5); self.state = RState.BRAKING
            radio.send({"type":Msg.HARD_BRAKE,"from":self.car_id,"label":self.label,
                        "car_type":self.car_type,"speed_kmh":round(self.speed_kmh,1),
                        "road_pos_m":self.road_pos_m,"ts":time.time()})
            self.msgs_sent += 1
            self._add_token_event("HARD_BRAKE",f"⚠ {self.label} HARD_BRAKE  CVC 21703")
        elif cmd in ("accelerate","acc","a"):
            self.speed_ms = min(SPEED_LIMIT_MS*1.2, self.speed_ms+3.0); self.state = RState.DRIVING
            self._add_token_event("ACCELERATE",f"↑ {self.label} → {self.speed_kmh:.0f} km/h")
        elif cmd in ("yield","y"):
            self.state = RState.YIELDING; self.speed_ms = max(0.0,self.speed_ms*0.3)
            self._add_token_event("YIELD",f"← {self.label} manual YIELD")
        elif cmd in ("normal","n","resume"):
            self.state = RState.DRIVING
            self._add_token_event("RESUME",f"▶ {self.label} resuming")

    def _auto_drive(self, dt, radio, all_cars):
        if self.state in (RState.YIELDING, RState.SHOULDER): return
        limit = self._speed_limit_here()

        # Zone ripple: pre-decelerate 300m before known zone start
        if (getattr(self,"_zone_start_m",None) is not None
                and self.state not in (RState.BROKEN_DOWN, RState.DONE)):
            dist_to_zone = self._zone_start_m - self.road_pos_m
            if 0 < dist_to_zone < 300:
                target = SCHOOL_SPEED_MS + (dist_to_zone/300.0) * (limit - SCHOOL_SPEED_MS)
                if self.speed_ms > target:
                    self.speed_ms = max(target, self.speed_ms - 1.5*dt)
            elif dist_to_zone <= 0:
                self._zone_start_m = None

        # Hazard avoidance: change lane proactively when breakdown is ahead in same lane
        if (getattr(self,"_hazard_pos",None) is not None
                and getattr(self,"_hazard_lane",None) == self.lane
                and self.state not in (RState.BROKEN_DOWN, RState.DONE, RState.SHOULDER)):
            dist_to_haz = self._hazard_pos - self.road_pos_m
            if 0 < dist_to_haz < 200:
                other_lane = 1 - self.lane
                others = [c for c in all_cars if c.get("lane")==other_lane
                          and abs(c.get("road_pos_m",0)-self.road_pos_m)<LANE_CLEAR_M]
                if not others:
                    self._start_lane_change(other_lane)
                    self._add_token_event("HAZARD_AVOID",
                        f"🔶 {self.label} lane change — hazard at {self._hazard_pos:.0f}m")
            elif dist_to_haz <= 0:
                self._hazard_pos = None

        ahead = self._find_ahead(all_cars)
        if ahead:
            gap = ahead["road_pos_m"] - self.road_pos_m - 4.5
            if ahead.get("car_type") == CarType.BREAKDOWN and gap < 120:
                other = [c for c in all_cars if c.get("lane")==(1-self.lane)
                         and abs(c.get("road_pos_m",0)-self.road_pos_m)<LANE_CLEAR_M]
                if not other: self._start_lane_change(1-self.lane)
                else: self.speed_ms = max(0.0, self.speed_ms-4.0*dt)
                return
            # V2X safe following: target gap = speed × FOLLOW_TIME_S × weather factor
            weather_follow = WEATHER_PARAMS.get(self._weather_state, WEATHER_PARAMS[0]).get("follow", 1.0)
            target_gap = max(MIN_GAP_M, self.speed_ms * FOLLOW_TIME_S * weather_follow)
            if gap < target_gap:
                # How urgently we need to brake — scales 0→1 as gap→0
                urgency = max(0.0, 1.0 - gap / target_gap)
                brake_force = urgency * 6.0 * dt          # up to 6 m/s² decel
                self.speed_ms = max(0.0, self.speed_ms - brake_force)
                if gap < MIN_GAP_M and self.state != RState.BRAKING:
                    self.state = RState.BRAKING
                    radio.send({"type":Msg.HARD_BRAKE,"from":self.car_id,"label":self.label,
                                "car_type":self.car_type,"speed_kmh":round(self.speed_kmh,1),
                                "road_pos_m":self.road_pos_m,"ts":time.time()})
                    self.msgs_sent += 1
                    self._add_token_event("HARD_BRAKE",
                        f"⚠ {self.label} V2X HARD_BRAKE — gap {gap:.1f}m  CVC 21703")
            else:
                self.speed_ms = min(limit, self.speed_ms+1.5*dt)
                if self.state == RState.BRAKING: self.state = RState.DRIVING
        else:
            self.speed_ms = min(limit, self.speed_ms+1.5*dt)
            if self.state == RState.BRAKING:    self.state = RState.DRIVING
            elif self.state == RState.OVERTAKING: self.state = RState.DRIVING
            elif self.state == RState.LANE_CHANGE and not self._lc_in_progress:
                self.state = RState.DRIVING


# ── Smart NPC ─────────────────────────────────────────────────────────────────
class SmartNPC(BaseRoadCar):
    def __init__(self, label, lane=0, start_pos=0.0, profile_name=None):
        pname = profile_name or random_profile()
        super().__init__(label, CarType.SMART, lane, start_pos,
                         speed=SPEED_LIMIT_MS*random.uniform(0.85,1.0),
                         profile_name=pname)
        self._overtake_cooldown = 0.0
        self._brake_sent        = False
        self._zone_announced    = False

    def send_beacon(self, radio):
        radio.send({"type":Msg.BEACON,"from":self.car_id,"label":self.label,
                    "car_type":self.car_type,"lane":self.lane,
                    "road_pos_m":round(self.road_pos_m,1),
                    "speed_kmh":round(self.speed_kmh,1),"state":self.state,
                    "is_player":False,"profile":self.profile_name,
                    "profile_color":self.profile_color,
                    "lateral_offset_m":round(self.lateral_offset_m,3),
                    "ts":time.time()})
        self.msgs_sent += 1

    def receive(self, msg):
        self.msgs_recv += 1
        mtype = msg.get("type"); from_id = msg.get("from")
        if from_id == self.car_id: return
        if mtype == Msg.BEACON: self.neighbours[from_id] = msg
        elif mtype == Msg.EMERG_PREEMPT:
            if self.state != RState.DONE:
                emerg_pos = msg.get("road_pos_m", self.road_pos_m)
                dist_ahead = emerg_pos - self.road_pos_m
                # Only move to shoulder if emergency is CLOSE (within 150m ahead or behind).
                # If it's already >150m ahead we are safe to stay/return — do NOT react.
                if dist_ahead < 150 and dist_ahead > -50:
                    _handle_emergency_corridor(self, 0.1)
                if from_id not in self._corridor_logged:
                    self._corridor_logged.add(from_id)
                    self._add_token_event("CORRIDOR",
                        f"🚨 {msg.get('label','?')} → {self.label} shoulder  CVC 21806")
        elif mtype == Msg.HARD_BRAKE:
            # V2X chain braking — scale reaction by distance to broadcasting car
            from_type  = msg.get("car_type", CarType.SMART)
            from_pos   = msg.get("road_pos_m", self.road_pos_m)
            dist_ahead = from_pos - self.road_pos_m
            if 0 < dist_ahead < 600:
                # Proportional: 0m=full brake, 600m=very gentle
                urgency   = max(0.0, 1.0 - dist_ahead / 600.0)
                v2x_brake = urgency * 4.0           # up to 4 m/s reduction
                self.speed_ms = max(0.0, self.speed_ms - v2x_brake)
                if from_type == CarType.LEGACY:
                    self._add_token_event("LEGACY_NO_WARN",
                        f"⚠ LEGACY {msg.get('label','?')} surprise brake — no V2X!")
                else:
                    self._add_token_event("SMART_CHAIN_WARN",
                        f"📡 V2X chain {msg.get('label','?')}→{self.label}  {dist_ahead:.0f}m  CVC 21703")
        elif mtype == Msg.HAZARD:
            # Store hazard position — proactive avoidance before we see it visually
            h_pos  = msg.get("road_pos_m", 0)
            h_lane = msg.get("lane", 0)
            dist   = h_pos - self.road_pos_m
            if 0 < dist < 400:
                self._hazard_pos  = h_pos
                self._hazard_lane = h_lane
                # Proportional slow-down: closer = harder
                urgency = max(0.0, 1.0 - dist/400.0)
                self.speed_ms = max(2.0, self.speed_ms - urgency*3.0)
                self._add_token_event("HAZARD_WARN",
                    f"🔶 V2X HAZARD {msg.get('label','?')} at {h_pos:.0f}m  dist={dist:.0f}m")
        elif mtype == Msg.ZONE_ALERT:
            # Store zone start position so we can pre-decelerate before entering
            zone_start = msg.get("zone_start_m", self.road_pos_m + 200)
            if self._zone_start_m is None:
                self._zone_start_m = zone_start
            if not self._zone_announced:
                self._zone_announced = True
                self._add_token_event("ZONE_ALERT",
                    f"🏫 ZONE ahead at {zone_start:.0f}m — {msg.get('speed_limit_kmh',24):.0f} km/h  CVC 22352a")

    def tick(self, dt, radio, all_cars):
        if self.state == RState.DONE: return
        # V5 ML: update trajectory buffer before any driving logic
        self._ensure_buffer()
        if self._traj_buffer is not None:
            self._traj_buffer.update(all_cars)
        self.send_beacon(radio)
        self._tick_lateral(dt)
        # V6 R4: fuel
        if self._tick_fuel(dt):
            self.speed_ms = 0.0; self.state = RState.BROKEN_DOWN
            radio.send({'type':Msg.HAZARD,'from':self.car_id,'label':self.label,
                        'road_pos_m':round(self.road_pos_m,1),'lane':self.lane,'ts':time.time()})
            self._add_token_event('FUEL_OUT',f'⛽ {self.label} ran out of fuel — HAZARD active'); return
        if self.fuel_pct <= FUEL_LOW_PCT and self.state not in (RState.BROKEN_DOWN,RState.DONE):
            self.speed_ms = min(self.speed_ms, 11.1)  # cap 40 km/h

        # Zone alert broadcast — include zone_start_m so receivers can pre-decelerate
        if self._approaching_school_zone() and not self._zone_announced:
            self._zone_announced = True
            school_z = RG.ZONES["SCHOOL_1"]
            radio.send({"type":Msg.ZONE_ALERT,"from":self.car_id,"label":self.label,
                        "zone_type":"SCHOOL","zone_start_m":school_z[0],
                        "speed_limit_ms":SCHOOL_SPEED_MS,
                        "speed_limit_kmh":round(SCHOOL_SPEED_MS*3.6,1),
                        "ts":time.time()})
            self.msgs_sent += 1

        # Zone ripple: if we know a zone is coming, start decelerating 300m before
        if (self._zone_start_m is not None
                and self.state not in (RState.BROKEN_DOWN, RState.DONE)):
            dist_to_zone = self._zone_start_m - self.road_pos_m
            if 0 < dist_to_zone < 300:
                # Decelerate proportionally: 300m away=gentle, 0m=full limit enforcement
                target_spd = SCHOOL_SPEED_MS + (dist_to_zone/300.0) * (self._speed_limit_here() - SCHOOL_SPEED_MS)
                if self.speed_ms > target_spd:
                    self.speed_ms = max(target_spd, self.speed_ms - 1.5*dt)
            elif dist_to_zone <= 0:
                self._zone_start_m = None  # we've passed it

        # Hazard avoidance: if we stored a hazard position ahead in our lane, move away early
        if (self._hazard_pos is not None
                and self._hazard_lane == self.lane
                and self.state not in (RState.BROKEN_DOWN, RState.DONE, RState.SHOULDER)):
            dist_to_haz = self._hazard_pos - self.road_pos_m
            if 0 < dist_to_haz < 200:
                other_lane = 1 - self.lane
                other = [c for c in all_cars if c.get("lane")==other_lane
                         and abs(c.get("road_pos_m",0)-self.road_pos_m)<LANE_CLEAR_M]
                if not other:
                    self._start_lane_change(other_lane)
                    self._add_token_event("HAZARD_AVOID",
                        f"🔶 {self.label} lane change — hazard at {self._hazard_pos:.0f}m")
            elif dist_to_haz <= 0:
                self._hazard_pos = None

        # Zone speed enforcement
        limit = self._speed_limit_here()
        if self._in_school_zone():
            if self.speed_ms > limit:
                self.speed_ms = max(limit, self.speed_ms-5.0*dt)
            self.speed_ms = min(self.speed_ms, limit)
            if self.state != RState.SCHOOL_ZONE:
                self.state = RState.SCHOOL_ZONE
                self._add_token_event("SCHOOL_ZONE",
                    f"🏫 {self.label} → {limit*3.6:.0f} km/h  CVC 22352a")
            self.road_pos_m += self.speed_ms*dt
            if self.road_pos_m >= ROAD_LENGTH_M: self.state = RState.DONE
            return
        elif self.state == RState.SCHOOL_ZONE:
            self.state = RState.DRIVING; self._zone_announced = False

        # Return from shoulder
        if self.state == RState.SHOULDER:
            still_emerg = any(
                v.get("car_type")==CarType.EMERGENCY
                and v.get("road_pos_m",0) < self.road_pos_m + 100
                and v.get("road_pos_m",0) >= self.road_pos_m - 30
                for v in self.neighbours.values())
            if not still_emerg:
                self._start_lane_change(0); self.state = RState.DRIVING
                self._corridor_logged.clear()   # allow next ambulance to log again
                self._add_token_event("YIELD_DONE",f"✅ {self.label} back from shoulder")
            else:
                self.road_pos_m += self.speed_ms*0.3*dt
            return

        # Resume from yield
        if self.state == RState.YIELDING:
            still_emerg = any(
                v.get("car_type")==CarType.EMERGENCY
                and v.get("road_pos_m",0) < self.road_pos_m + 100
                and v.get("road_pos_m",0) >= self.road_pos_m - 30
                for v in self.neighbours.values())
            if not still_emerg:
                self.state = RState.DRIVING
                self._add_token_event("YIELD_DONE",f"✅ {self.label} resuming")
            else:
                self.speed_ms = max(0.0,self.speed_ms-2.0*dt)
                self.road_pos_m += self.speed_ms*dt
            return

        # Profile-modulated driving
        prof_result = self.profile.execute(dt, self.speed_ms, 999.0,
                                           self.road_pos_m, limit)
        self.lateral_offset_m = max(-RG.LANE_WIDTH_M/2,
            min(RG.LANE_WIDTH_M/2,
                self.lateral_offset_m + prof_result.get("lateral_delta",0)))

        ahead = self._find_ahead(all_cars)
        if ahead:
            gap = ahead["road_pos_m"] - self.road_pos_m - 4.5

            # Avoid breakdown car
            if ahead.get("car_type")==CarType.BREAKDOWN and gap < 80:
                other = [c for c in all_cars if c.get("lane")==(1-self.lane)
                         and abs(c.get("road_pos_m",0)-self.road_pos_m)<LANE_CLEAR_M]
                if not other: self._start_lane_change(1-self.lane)

            # V2X safe following: proportional brake from target_gap down to 0
            # Weather multiplies required gap — ice doubles it, clear = normal
            weather_follow = WEATHER_PARAMS.get(self._weather_state, WEATHER_PARAMS[0]).get("follow", 1.0)
            target_gap = prof_result.get("target_gap_m", max(MIN_GAP_M, self.speed_ms*FOLLOW_TIME_S))
            required   = max(MIN_GAP_M, target_gap * weather_follow)

            if gap < required:
                # Try lane change first to free up the lane
                other = [c for c in all_cars if c.get("lane")==(1-self.lane)
                         and abs(c.get("road_pos_m",0)-self.road_pos_m)<LANE_CLEAR_M]
                oncoming_gap = self._calc_oncoming_clear(all_cars)
                ok_ot,_,_ = Law.may_overtake(self.speed_ms,
                                              ahead.get("speed_kmh",0)/3.6, oncoming_gap)
                wants_ot = prof_result.get("want_overtake",False)

                if (time.time()>self._overtake_cooldown and not other
                        and ok_ot and (wants_ot or gap < required*0.5)):
                    radio.send({"type":Msg.INTENT_OT,"from":self.car_id,"label":self.label,
                                "target":ahead.get("label","?"),
                                "road_pos_m":round(self.road_pos_m,1),"ts":time.time()})
                    self.msgs_sent += 1
                    self._start_lane_change(1-self.lane)
                    self.speed_ms = min(limit*1.1, self.speed_ms+OVERTAKE_BOOST_MS)
                    self.state = RState.OVERTAKING
                    self._overtake_cooldown = time.time()+8.0
                    self._add_token_event("OVERTAKE",
                        f"◆ {self.label}[{self.profile_name}] INTENT_OT → {ahead.get('label','?')}  CVC 21750")
                else:
                    # Proportional V2X braking — stronger as gap shrinks
                    urgency = max(0.0, 1.0 - gap / required)
                    brake_force = urgency * 5.0 * dt
                    self.speed_ms = max(0.0, self.speed_ms - brake_force)
                    if gap < MIN_GAP_M and not self._brake_sent:
                        radio.send({"type":Msg.HARD_BRAKE,"from":self.car_id,"label":self.label,
                                    "car_type":self.car_type,"speed_kmh":round(self.speed_kmh,1),
                                    "road_pos_m":round(self.road_pos_m,1),"ts":time.time()})
                        self.msgs_sent += 1
                        self._brake_sent = True
                        self._add_token_event("SMART_CHAIN_WARN",
                            f"📡 {self.label} V2X BRAKE — gap {gap:.1f}m  CVC 21703")
            else:
                self._brake_sent = False
                delta = prof_result.get("delta_speed_ms",0)
                self.speed_ms = max(0.0, min(limit, self.speed_ms+delta+1.0*dt))
                if self.state == RState.OVERTAKING:
                    radio.send({"type":Msg.INTENT_LC,"from":self.car_id,"label":self.label,
                                "new_lane":0,"road_pos_m":round(self.road_pos_m,1),"ts":time.time()})
                    self.msgs_sent += 1
                    self._start_lane_change(0); self.state = RState.DRIVING
        else:
            self._brake_sent = False
            delta = prof_result.get("delta_speed_ms",0)
            self.speed_ms = max(0.0, min(limit, self.speed_ms+delta+1.5*dt))
            if self.state in (RState.OVERTAKING,RState.LANE_CHANGE,RState.BRAKING):
                self.state = RState.DRIVING

        self.road_pos_m += self.speed_ms*dt
        if self.road_pos_m >= ROAD_LENGTH_M: self.state = RState.DONE


# ── Legacy NPC ────────────────────────────────────────────────────────────────
class LegacyNPC(BaseRoadCar):
    """
    No V2X radio. Reacts only to visual_range physical proximity.
    Profile modulates driving behaviour.
    Feature #7: no chain-brake warning — surprise to followers.
    """
    def __init__(self, label, lane=0, start_pos=0.0, profile_name=None):
        pname = profile_name or random_profile()
        super().__init__(label, CarType.LEGACY, lane, start_pos,
                         speed=SPEED_LIMIT_MS*random.uniform(0.55,0.75),
                         profile_name=pname)
        self._hard_braked   = False
        self._emergency_seen = False

    def receive(self, msg): pass   # no radio — intentionally empty

    def tick(self, dt, radio, all_cars):
        if self.state == RState.DONE: return
        self._tick_lateral(dt)

        # Visual range emergency detection (sees flashing lights physically)
        emerg_nearby = [c for c in all_cars
                        if c.get("car_type")==CarType.EMERGENCY
                        and abs(c.get("road_pos_m",9999)-self.road_pos_m) < VISUAL_RANGE_M]
        if emerg_nearby and not self._emergency_seen:
            self._emergency_seen = True
            _handle_emergency_corridor(self, dt)
        elif not emerg_nearby and self._emergency_seen:
            self._emergency_seen = False
            self._start_lane_change(min(1, max(0, self.lane)))  # return to driving lanes
            self.state = RState.DRIVING

        if self.state in (RState.SHOULDER,):
            self.road_pos_m += self.speed_ms*0.3*dt
            self.speed_ms = max(0.5, self.speed_ms-1.0*dt)
            if self.road_pos_m >= ROAD_LENGTH_M: self.state = RState.DONE
            return

        # Visual range following — uses profile
        limit = self._speed_limit_here()
        ahead = self._find_ahead(all_cars)
        if ahead:
            gap = ahead["road_pos_m"] - self.road_pos_m - 4.5
            visual_gap = gap if gap <= VISUAL_RANGE_M else 999.0
            prof = self.profile.execute(dt, self.speed_ms, visual_gap,
                                        self.road_pos_m, limit)

            # Apply profile lateral drift
            self.lateral_offset_m = max(-RG.LANE_WIDTH_M/2,
                min(RG.LANE_WIDTH_M/2,
                    self.lateral_offset_m + prof.get("lateral_delta",0)))

            target_gap = prof.get("target_gap_m", MIN_GAP_M*2)
            if gap < MIN_GAP_M*0.4 and not self._hard_braked:
                self._hard_braked = True
                # Legacy: NO V2X broadcast — surprise brake
                self.speed_ms = max(0.0, self.speed_ms-4.5*dt)
            elif gap < target_gap:
                self.speed_ms = max(0.0,
                    self.speed_ms + prof.get("delta_speed_ms",0) - 2.0*dt)
            else:
                self._hard_braked = False
                delta = prof.get("delta_speed_ms",0)
                self.speed_ms = max(0.0, min(limit*0.9, self.speed_ms+delta+0.8*dt))
        else:
            self._hard_braked = False
            prof = self.profile.execute(dt, self.speed_ms, 999.0,
                                        self.road_pos_m, limit)
            self.lateral_offset_m = max(-RG.LANE_WIDTH_M/2,
                min(RG.LANE_WIDTH_M/2,
                    self.lateral_offset_m + prof.get("lateral_delta",0)))
            delta = prof.get("delta_speed_ms",0)
            self.speed_ms = max(0.0, min(limit*0.9, self.speed_ms+delta+0.8*dt))

        self.road_pos_m += self.speed_ms*dt
        if self.road_pos_m >= ROAD_LENGTH_M: self.state = RState.DONE


# ── Emergency NPC ─────────────────────────────────────────────────────────────
class EmergencyNPC(BaseRoadCar):
    def __init__(self, label, lane=1, start_pos=0.0):
        super().__init__(label, CarType.EMERGENCY, lane, start_pos,
                         speed=EMERGENCY_MS*0.7, profile_name=COMMUTER)
        self.profile_color = "bold white on red"
        self._preempt_sent = False; self._last_preempt = 0.0

    def send_beacon(self, radio):
        radio.send({"type":Msg.BEACON,"from":self.car_id,"label":self.label,
                    "car_type":CarType.EMERGENCY,"lane":self.lane,
                    "road_pos_m":round(self.road_pos_m,1),
                    "speed_kmh":round(self.speed_kmh,1),"state":self.state,
                    "is_player":False,"emergency":True,
                    "lateral_offset_m":0.0,"ts":time.time()})
        self.msgs_sent += 1

    def receive(self, msg): pass

    def tick(self, dt, radio, all_cars):
        if self.state == RState.DONE: return
        self.send_beacon(radio)
        if not self._preempt_sent or time.time()-self._last_preempt > 3.0:
            radio.send({"type":Msg.EMERG_PREEMPT,"from":self.car_id,"label":self.label,
                        "road_pos_m":round(self.road_pos_m,1),"lane":self.lane,"ts":time.time()})
            self.msgs_sent += 1
            self._preempt_sent = True; self._last_preempt = time.time()
            if not hasattr(self,"_first_logged"):
                self._first_logged = True
                self.log_event("PREEMPT_SENT","CVC 21806 — corridor broadcast")
                self._add_token_event("PREEMPT",
                    f"🚨 {self.label} EMERG_PREEMPT — corridor forming  CVC 21806")
        # Drive through corridor (lane 1 cleared)
        ahead = self._find_ahead(all_cars)
        if ahead:
            gap = ahead["road_pos_m"]-self.road_pos_m-4.5
            if gap < MIN_GAP_M*0.4: self.speed_ms = max(4.0,self.speed_ms-5.0*dt)
            else: self.speed_ms = min(EMERGENCY_MS,self.speed_ms+6.0*dt)
        else:
            self.speed_ms = min(EMERGENCY_MS,self.speed_ms+6.0*dt)
        self.road_pos_m += self.speed_ms*dt
        if self.road_pos_m >= ROAD_LENGTH_M:
            self.state = RState.DONE
            self.log_event("ROAD_DONE","Emergency cleared road")


# ── Breakdown NPC ─────────────────────────────────────────────────────────────
class BreakdownNPC(BaseRoadCar):
    def __init__(self, label, lane=0, start_pos=0.0, breakdown_pos=None):
        super().__init__(label, CarType.BREAKDOWN, lane, start_pos,
                         speed=SPEED_LIMIT_MS*0.9, profile_name=COMMUTER)
        self.profile_color     = "bright_yellow"
        self._breakdown_pos    = breakdown_pos or (ROAD_LENGTH_M*0.28)
        self._broken           = False
        self._hazard_sent      = False
        self._hazard_repeat    = 0.0

    def send_beacon(self, radio):
        radio.send({"type":Msg.BEACON,"from":self.car_id,"label":self.label,
                    "car_type":CarType.BREAKDOWN,"lane":self.lane,
                    "road_pos_m":round(self.road_pos_m,1),
                    "speed_kmh":round(self.speed_kmh,1),"state":self.state,
                    "broken_down":self._broken,"is_player":False,
                    "lateral_offset_m":round(self.lateral_offset_m,3),
                    "ts":time.time()})
        self.msgs_sent += 1

    def receive(self, msg): pass

    def tick(self, dt, radio, all_cars):
        if self.state == RState.DONE: return
        self.send_beacon(radio)
        if not self._broken and self.road_pos_m >= self._breakdown_pos:
            self._broken = True; self.speed_ms = 0.0; self.state = RState.BROKEN_DOWN
            self.log_event("BREAKDOWN",f"⛔ {self.label} BROKEN DOWN at {self.road_pos_m:.0f}m  CVC 22500")
            self._add_token_event("BREAKDOWN",
                f"⛔ {self.label} broken at {self.road_pos_m:.0f}m — HAZARD active")
        if self._broken:
            if not self._hazard_sent or time.time()-self._hazard_repeat > 2.0:
                radio.send({"type":Msg.HAZARD,"from":self.car_id,"label":self.label,
                            "road_pos_m":round(self.road_pos_m,1),"lane":self.lane,"ts":time.time()})
                self.msgs_sent += 1
                self._hazard_sent = True; self._hazard_repeat = time.time()
        else:
            self.road_pos_m += self.speed_ms*dt
            if self.road_pos_m >= ROAD_LENGTH_M: self.state = RState.DONE


# ── Platoon NPC ───────────────────────────────────────────────────────────────
class PlatoonNPC(BaseRoadCar):
    def __init__(self, label, lane=0, start_pos=0.0, is_lead=True, lead_id=None):
        super().__init__(label, CarType.PLATOON, lane, start_pos,
                         speed=SPEED_LIMIT_MS*0.95, profile_name=COMMUTER)
        self.profile_color   = "green"
        self.is_lead         = is_lead
        self.lead_id         = lead_id
        self._platoon_formed = False
        self._invite_sent    = 0.0
        self._got_invite     = False

    def send_beacon(self, radio):
        radio.send({"type":Msg.BEACON,"from":self.car_id,"label":self.label,
                    "car_type":CarType.PLATOON,"lane":self.lane,
                    "road_pos_m":round(self.road_pos_m,1),
                    "speed_kmh":round(self.speed_kmh,1),"state":self.state,
                    "is_player":False,"is_lead":self.is_lead,
                    "lateral_offset_m":round(self.lateral_offset_m,3),
                    "ts":time.time()})
        self.msgs_sent += 1

    def receive(self, msg):
        self.msgs_recv += 1
        mtype = msg.get("type"); from_id = msg.get("from")
        if from_id == self.car_id: return
        if mtype == Msg.BEACON: self.neighbours[from_id] = msg
        elif mtype == Msg.PLATOON_INVITE and not self.is_lead:
            self._got_invite = True
        elif mtype == Msg.PLATOON_ACK and self.is_lead:
            self._platoon_formed = True
            self._add_token_event("PLATOON_FORMED",
                f"🚗🚗 PLATOON formed: {self.label} ← {msg.get('label','?')}  CVC 21703")
        elif mtype == Msg.EMERG_PREEMPT:
            if self.state != RState.DONE:
                emerg_pos  = msg.get("road_pos_m", self.road_pos_m)
                dist_ahead = emerg_pos - self.road_pos_m
                if dist_ahead < 150 and dist_ahead > -50:
                    _handle_emergency_corridor(self, 0.1)

    def tick(self, dt, radio, all_cars):
        if self.state == RState.DONE: return
        self.send_beacon(radio); self._tick_lateral(dt)
        limit = self._speed_limit_here()
        if self.is_lead:
            if time.time()-self._invite_sent > 5.0:
                radio.send({"type":Msg.PLATOON_INVITE,"from":self.car_id,
                            "label":self.label,"road_pos_m":round(self.road_pos_m,1),
                            "lane":self.lane,"ts":time.time()})
                self.msgs_sent += 1; self._invite_sent = time.time()
            self.speed_ms = min(limit*0.95, self.speed_ms+0.5*dt)
        else:
            if not self._platoon_formed and self._got_invite:
                radio.send({"type":Msg.PLATOON_ACK,"from":self.car_id,
                            "label":self.label,"ts":time.time()})
                self.msgs_sent += 1
                self._platoon_formed = True; self._got_invite = False
                self._add_token_event("PLATOON_ACK",
                    f"🚗🚗 {self.label} joined platoon — {PLATOON_CLOSE_M}m gap  CVC 21703")
            lead = next((c for c in all_cars
                         if c.get("car_type")==CarType.PLATOON and c.get("is_lead")), None)
            if lead:
                if gap < 5.0:
                    self.speed_ms = max(0.0, self.speed_ms - 8.0*dt)
                elif gap < PLATOON_CLOSE_M:
                    self.speed_ms = max(0.0, self.speed_ms-3.0*dt)
                elif gap > PLATOON_CLOSE_M*2: self.speed_ms = min(limit,self.speed_ms+2.0*dt)
                elif gap > PLATOON_CLOSE_M*2: self.speed_ms = min(limit,self.speed_ms+2.0*dt)
                else:
                    target = lead.get("speed_kmh",limit*3.6)/3.6
                    self.speed_ms += (target-self.speed_ms)*0.5*dt
                if self.state not in (RState.SHOULDER, RState.YIELDING):
                    self.state = RState.PLATOONING
            else:
                self.speed_ms = min(limit, self.speed_ms+1.0*dt)
        self.road_pos_m += self.speed_ms*dt
        if self.road_pos_m >= ROAD_LENGTH_M: self.state = RState.DONE


# ── Convoy NPC ────────────────────────────────────────────────────────────────
class ConvoyNPC(BaseRoadCar):
    def __init__(self, label, lane=0, start_pos=0.0,
                 convoy_id=None, convoy_pos=0, convoy_size=3):
        super().__init__(label, CarType.PLATOON, lane, start_pos,
                         speed=SPEED_LIMIT_MS*0.80, profile_name=COMMUTER)
        self.profile_color  = "bold green"
        self.convoy_id      = convoy_id or "CONVOY-1"
        self.convoy_pos     = convoy_pos
        self.convoy_size    = convoy_size
        self._is_lead       = (convoy_pos==0)
        self._ot_cooldown   = 0.0
        self._ot_ready      = False
        self._ot_delay      = convoy_pos*1.5

    @property
    def is_lead(self): return self._is_lead

    def send_beacon(self, radio):
        radio.send({"type":Msg.BEACON,"from":self.car_id,"label":self.label,
                    "car_type":CarType.PLATOON,"lane":self.lane,
                    "road_pos_m":round(self.road_pos_m,1),
                    "speed_kmh":round(self.speed_kmh,1),"state":self.state,
                    "is_player":False,"convoy_id":self.convoy_id,
                    "convoy_pos":self.convoy_pos,"is_lead":self._is_lead,
                    "lateral_offset_m":round(self.lateral_offset_m,3),
                    "ts":time.time()})
        self.msgs_sent += 1

    def receive(self, msg):
        self.msgs_recv += 1; mtype=msg.get("type"); from_id=msg.get("from")
        if from_id==self.car_id: return
        if mtype==Msg.BEACON: self.neighbours[from_id]=msg
        elif mtype==Msg.EMERG_PREEMPT:
            _handle_emergency_corridor(self,0.1)
        elif mtype==Msg.INTENT_OT:
            sc=msg.get("convoy_id"); sp=msg.get("convoy_pos",99)
            if sc==self.convoy_id and sp==self.convoy_pos-1:
                self._ot_ready=True; self._ot_start=time.time()
                self._add_token_event("CONVOY_READY",
                    f"🚛 {self.label} slot {self.convoy_pos} ready")

    def tick(self, dt, radio, all_cars):
        if self.state==RState.DONE: return
        self.send_beacon(radio); self._tick_lateral(dt)
        limit = self._speed_limit_here()
        if self.state==RState.SHOULDER:
            still_emerg=any(v.get("car_type")==CarType.EMERGENCY
                and abs(v.get("road_pos_m",999)-self.road_pos_m)<200
                for v in self.neighbours.values())
            if not still_emerg:
                self._start_lane_change(0); self.state=RState.DRIVING
            self.road_pos_m+=self.speed_ms*0.3*dt; return
        ahead=self._find_ahead(all_cars)
        if ahead:
            gap=ahead["road_pos_m"]-self.road_pos_m-4.5
            slow=ahead.get("speed_kmh",50)/3.6 < self.speed_ms-1.5
            too_close=gap<max(MIN_GAP_M,self.speed_ms*FOLLOW_TIME_S)
            if (too_close or (slow and gap<80)) and self._is_lead and time.time()>self._ot_cooldown:
                oncoming=[c for c in all_cars if c.get("lane")==(1-self.lane)
                          and c.get("road_pos_m",0)>self.road_pos_m]
                og=(min(oncoming,key=lambda c:c["road_pos_m"])["road_pos_m"]-self.road_pos_m) if oncoming else 999.0
                if og>ONCOMING_CLEAR_M*1.5:
                    radio.send({"type":Msg.INTENT_OT,"from":self.car_id,"label":self.label,
                                "target":ahead.get("label","?"),"road_pos_m":round(self.road_pos_m,1),
                                "convoy_id":self.convoy_id,"convoy_pos":self.convoy_pos,"ts":time.time()})
                    self.msgs_sent+=1; self._start_lane_change(1-self.lane)
                    self.speed_ms=min(limit,self.speed_ms+OVERTAKE_BOOST_MS)
                    self.state=RState.OVERTAKING; self._ot_cooldown=time.time()+12.0
                    self._add_token_event("CONVOY_OVERTAKE",
                        f"🚛🚛🚛 {self.convoy_id} CONVOY OVERTAKE by {self.label}  CVC 21750")
                else: self.speed_ms=max(2.0,self.speed_ms-1.5*dt)
            elif too_close and not self._is_lead:
                if self._ot_ready and time.time()-getattr(self,"_ot_start",0)>=self._ot_delay:
                    radio.send({"type":Msg.INTENT_OT,"from":self.car_id,"label":self.label,
                                "target":ahead.get("label","?"),"road_pos_m":round(self.road_pos_m,1),
                                "convoy_id":self.convoy_id,"convoy_pos":self.convoy_pos,"ts":time.time()})
                    self.msgs_sent+=1; self._start_lane_change(1-self.lane)
                    self.speed_ms=min(limit,self.speed_ms+OVERTAKE_BOOST_MS*0.7)
                    self.state=RState.OVERTAKING; self._ot_ready=False
                    self._add_token_event("CONVOY_MEMBER_OT",
                        f"🚛 {self.label} slot {self.convoy_pos} joining")
                else: self.speed_ms=max(2.0,self.speed_ms-1.5*dt)
            elif not too_close:
                self.speed_ms=min(limit*0.82,self.speed_ms+1.0*dt)
                if self.state==RState.OVERTAKING:
                    radio.send({"type":Msg.INTENT_LC,"from":self.car_id,"label":self.label,
                                "new_lane":0,"road_pos_m":round(self.road_pos_m,1),"ts":time.time()})
                    self.msgs_sent+=1; self._start_lane_change(0); self.state=RState.DRIVING
        else:
            self.speed_ms=min(limit*0.82,self.speed_ms+1.0*dt)
            if self.state in (RState.OVERTAKING,RState.BRAKING): self.state=RState.DRIVING
        self.road_pos_m+=self.speed_ms*dt
        if self.road_pos_m>=ROAD_LENGTH_M: self.state=RState.DONE
