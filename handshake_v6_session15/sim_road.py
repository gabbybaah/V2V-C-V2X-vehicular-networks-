# =============================================================================
# handshake_v5/sim_road.py  —  Road Simulation Coordinator
#
# V5 changes:
#   - 5km road, 3 lanes from road_geometry
#   - Mixed NPC spawn (interleaved smart+legacy by position)
#   - Per-machine NPC segment ownership (--vms N flag)
#   - All NPCs broadcast beacons every tick (full road visibility across VMs)
#   - lateral_offset_m in every snapshot entry
#   - Emergency corridor logic lives in car_road.py
#   - Roadworks, split-brain, SPaT delay carried forward
# =============================================================================
import time, random, threading, logging
from road_geometry import RG, ROAD_LENGTH_M
from config import (SIM_TICK_S, CarType, Msg, RState,
                    ROADWORKS_ZONE_START_M, ROADWORKS_ZONE_END_M,
                    ROADWORKS_SPEED_MS, CONVOY_SIZE, CONVOY_SPACING_M,
                    SPLIT_BRAIN_DURATION_S, FLEET,
                    WeatherState, WEATHER_PARAMS, WEATHER_CHANGE_INTERVAL_S,
                    WEATHER_WEIGHTS, NEAR_MISS_M, HEATMAP_BUCKETS)
from radio import Radio
from car_road import (PlayerCar, SmartNPC, LegacyNPC, EmergencyNPC,
                      BreakdownNPC, PlatoonNPC, ConvoyNPC)
from behavior_profiles import random_profile, COMMUTER
from incident_buffer import IncidentBuffer

log = logging.getLogger("v5.sim_road")
REMOTE_TIMEOUT_S = 3.0


# ── Mixed NPC spawn ───────────────────────────────────────────────────────────
def _spawn_npcs(n_smart, n_legacy, n_emerg, seg_start=0.0, seg_end=None,
                n_breakdown=0, n_platoon=0, n_convoy=0, machine_id=1):
    """
    Spawn NPCs near km 0. machine_id differentiates labels so multi-VM cars
    don't share IDs. VM1 labels S01-S09, VM2 labels S11-S19, etc.
    Cars spawn spread out across the first 1500m — plenty of room between them.
    """
    seg_end = seg_end or ROAD_LENGTH_M
    npcs    = []
    rng     = random.Random(42 + machine_id * 17)

    # Label offset so VMs have unique car IDs  (VM1→1, VM2→11, VM3→21…)
    lbl_base = (machine_id - 1) * 10 + 1

    # Spawn zone: spread cars from 120m to 1500m — each VM offset slightly
    spawn_base = 120 + (machine_id - 1) * 250   # VM1@120m, VM2@370m, VM3@620m

    total_road = n_smart + n_legacy
    if total_road > 0:
        # Spread across 1500m with a minimum 100m between cars in same lane
        slot_size = max(120.0, 1500.0 / max(total_road, 1))
        car_types = ([CarType.SMART]*n_smart + [CarType.LEGACY]*n_legacy)
        rng.shuffle(car_types)
        for i, ctype in enumerate(car_types):
            pos  = spawn_base + i * slot_size + rng.uniform(-20.0, 20.0)
            pos  = max(50.0, min(seg_end - 100, pos))
            lane = i % 2
            pname = random_profile()
            li = lbl_base + len(npcs)
            if ctype == CarType.SMART:
                npcs.append(SmartNPC(f"S{li:02d}◆", lane, pos, pname))
            else:
                npcs.append(LegacyNPC(f"L{li:02d}◇", lane, pos, pname))

    # emergency vehicles — spawn behind spawn_base so they come through
    for i in range(n_emerg):
        npcs.append(EmergencyNPC(f"E{lbl_base+i:02d}🚑", lane=1,
                                  start_pos=max(0, spawn_base - 80 - i*120)))

    # breakdown NPCs — scatter on road (no longer tied to removed BREAKDOWN zone)
    bd_positions = [1600.0, 2200.0, 3000.0]
    for i in range(n_breakdown):
        bd_pos = bd_positions[i % len(bd_positions)] + rng.uniform(-100, 100)
        npcs.append(BreakdownNPC(f"B{lbl_base+i:02d}⛔", lane=i%2,
                                  start_pos=bd_pos*0.3,
                                  breakdown_pos=bd_pos))

    # platoon
    if n_platoon >= 2:
        pl_start = 1500.0 + (machine_id-1)*200
        lead = PlatoonNPC(f"PT{lbl_base:02d}★", lane=0, start_pos=pl_start, is_lead=True)
        npcs.append(lead)
        for i in range(1, min(n_platoon, 5)):
            npcs.append(PlatoonNPC(f"PT{lbl_base+i:02d}▶", lane=0,
                                    start_pos=pl_start-i*30.0,
                                    is_lead=False, lead_id=lead.car_id))

    # convoy
    if n_convoy >= CONVOY_SIZE:
        convoy_id  = f"CONVOY-{machine_id}"
        base_start = 2000.0 + (machine_id-1)*200
        for i in range(2):
            sl_pos = base_start + 80 + i*60
            npcs.append(LegacyNPC(f"SL{lbl_base+i:02d}◇", lane=0,
                                   start_pos=sl_pos, profile_name=COMMUTER))
        for i in range(min(n_convoy, 5)):
            npcs.append(ConvoyNPC(
                label      = f"TK{lbl_base+i:02d}🚛",
                lane       = 0,
                start_pos  = base_start - i*CONVOY_SPACING_M,
                convoy_id  = convoy_id,
                convoy_pos = i,
                convoy_size= min(n_convoy, 5),
            ))

    return npcs


# ── RoadSim ───────────────────────────────────────────────────────────────────
class RoadSim:
    def __init__(self, player_configs=None, npc_counts=None, loss=0.0,
                 rush_hour=False, spat_delay=0.0, roadworks=False,
                 total_vms=1, machine_id=1):

        self.loss        = loss
        self.rush_hour   = rush_hour
        self.spat_delay  = spat_delay
        self.roadworks   = roadworks
        self.total_vms   = total_vms
        self.machine_id  = machine_id
        self._running    = False
        self._thread     = None
        self.start_time  = 0.0
        self.tick_n      = 0

        # No segment split — all VMs share the full road
        # Each VM spawns its own labeled cars from km 0
        self.seg_start = 0.0
        self.seg_end   = ROAD_LENGTH_M

        # split-brain state
        self._split_brain     = False
        self._split_brain_end = 0.0
        self._split_brain_events = []

        # SPaT delay queue
        self._spat_delay_queue = []
        self._spat_delay_lock  = threading.Lock()

        # roadworks announced set
        self._roadworks_announced = set()

        # violation tracking
        self._violation_events = []

        # V6 R2: weather
        self._weather_state      = WeatherState.CLEAR
        self._weather_changed_at = 0.0
        self._weather_tick_accum = 0.0

        # V6 R3: near-miss counters
        self._near_miss_count    = 0
        self._collision_count    = 0

        # V6 R5: heatmap buckets (reset each update)
        self._heatmap            = [0.0] * HEATMAP_BUCKETS

        self.radio = Radio("ROAD-SIM", loss=loss)
        if total_vms > 1:
            try:
                from radio_network import Radio as NetworkRadio
                self.radio = NetworkRadio(f"ROAD-VM{machine_id}", loss=loss)
                log.info(f"[RoadSim] VM{machine_id}: using UDP multicast radio")
            except Exception as e:
                log.warning(f"[RoadSim] Network radio failed, using in-process: {e}")

        # Player cars
        self.player_cars = []
        if player_configs:
            for cfg in player_configs:
                pc = PlayerCar(
                    label      = cfg.get("label", f"P{len(self.player_cars)+1}◆"),
                    lane       = cfg.get("lane", 0),
                    start_pos  = cfg.get("start_pos", 150.0),
                    machine_id = cfg.get("machine_id", machine_id),
                )
                self.player_cars.append(pc)
        else:
            self.player_cars.append(PlayerCar(
                f"P{machine_id}◆", lane=0, start_pos=150.0, machine_id=machine_id))

        # NPC counts
        if rush_hour and not npc_counts:
            npc_cfg = {"smart":10,"legacy":8,"emergency":2,
                       "breakdown":1,"platoon":2,"convoy":CONVOY_SIZE}
        else:
            npc_cfg = npc_counts or {"smart":4,"legacy":4,"emergency":1,
                                     "breakdown":0,"platoon":0,"convoy":0}

        self.npcs = _spawn_npcs(
            npc_cfg.get("smart",4), npc_cfg.get("legacy",4),
            npc_cfg.get("emergency",1),
            seg_start   = self.seg_start,
            seg_end     = self.seg_end,
            n_breakdown = npc_cfg.get("breakdown",0),
            n_platoon   = npc_cfg.get("platoon",0),
            n_convoy    = npc_cfg.get("convoy",0),
            machine_id  = machine_id,
        )

        self.all_cars = list(self.player_cars) + self.npcs

        self._remote_cars  = {}
        self._remote_lock  = threading.Lock()
        self._token_events = []
        self._tok_lock     = threading.Lock()
        self._incident_cooldown = {}   # pair_key → last_reported_ts
        # V5 Phase 4: incident replay buffer
        self.incident_buf  = IncidentBuffer()

    # ── Accessors ─────────────────────────────────────────────────────────────
    def get_player(self, machine_id=1):
        for pc in self.player_cars:
            if pc.machine_id == machine_id: return pc
        return self.player_cars[0] if self.player_cars else None

    # ── Split-brain ───────────────────────────────────────────────────────────
    def trigger_split_brain(self, duration=SPLIT_BRAIN_DURATION_S):
        self._split_brain     = True
        self._split_brain_end = time.time() + duration
        evt = {"event":"SPLIT_BRAIN_START",
               "detail":f"🔌 NETWORK PARTITION — {duration:.0f}s blackout","ts":time.time()}
        self._split_brain_events.append(evt)
        with self._tok_lock: self._token_events.append(evt)
        def _reconnect():
            time.sleep(duration+0.1)
            self._split_brain = False
            reconn = {"event":"SPLIT_BRAIN_END",
                      "detail":"🔌 RECONNECTED — V2X resuming","ts":time.time()}
            self._split_brain_events.append(reconn)
            with self._tok_lock: self._token_events.append(reconn)
        threading.Thread(target=_reconnect, daemon=True).start()

    # ── SPaT delay ────────────────────────────────────────────────────────────
    def _inject_spat_delay(self, msg):
        with self._spat_delay_lock:
            self._spat_delay_queue.append((time.time()+self.spat_delay, msg))

    def _flush_delayed_spat(self):
        now   = time.time()
        ready = []
        with self._spat_delay_lock:
            rem = []
            for (rel, msg) in self._spat_delay_queue:
                (ready if now>=rel else rem).append((rel,msg))
            self._spat_delay_queue = rem
        for _, msg in ready:
            for car in self.all_cars:
                if hasattr(car,"receive") and msg.get("from")!=car.car_id:
                    car.receive(msg)
            age = now - msg.get("_ts", now)
            with self._tok_lock:
                self._token_events.append({"event":"SPAT_STALE",
                    "detail":f"📡 SPaT {age:.1f}s late — phase={msg.get('phase','?')}",
                    "ts":now})

    # ── Roadworks ─────────────────────────────────────────────────────────────
    def _handle_roadworks_zone(self, car):
        if not self.roadworks: return
        in_zone = ROADWORKS_ZONE_START_M <= car.road_pos_m <= ROADWORKS_ZONE_END_M
        cid     = car.car_id
        if in_zone:
            if car.speed_ms > ROADWORKS_SPEED_MS and car.car_type != CarType.EMERGENCY:
                car.speed_ms = max(ROADWORKS_SPEED_MS, car.speed_ms-3.0*SIM_TICK_S)
                if car.state != RState.SCHOOL_ZONE:
                    car.state = RState.SCHOOL_ZONE
                    if cid not in self._roadworks_announced:
                        self._roadworks_announced.add(cid)
                        with self._tok_lock:
                            self._token_events.append({"event":"ROADWORKS",
                                "detail":f"🚧 {car.label} slowing to {ROADWORKS_SPEED_MS*3.6:.0f} km/h  CVC 22352",
                                "label":car.label,"ts":time.time()})
            if car.lane not in (0,2):
                car.lane = 0
        elif car.state == RState.SCHOOL_ZONE:
            car.state = RState.DRIVING
            self._roadworks_announced.discard(cid)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        self.radio.start()
        self.start_time = time.time()
        self._running   = True
        self._thread    = threading.Thread(target=self._loop, daemon=True, name="sim-road")
        self._thread.start()

    def stop(self):
        self._running = False
        try: self.radio.stop()
        except: pass

    # ── Main loop ─────────────────────────────────────────────────────────────
    def _loop(self):
        dt = SIM_TICK_S
        while self._running:
            t0 = time.time()
            self.tick_n += 1

            # Drain radio — priority ordered
            msgs = self.radio.drain()
            for msg in msgs:
                if self._split_brain: continue
                if self.spat_delay>0 and msg.get("type")==Msg.SPAT:
                    self._inject_spat_delay(msg); continue
                self._handle_incoming(msg)
                for car in self.all_cars:
                    if hasattr(car,"receive") and msg.get("from")!=car.car_id:
                        car.receive(msg)

            if self.spat_delay>0: self._flush_delayed_spat()

            # Expire stale remote cars
            now = time.time()
            with self._remote_lock:
                stale = [k for k,v in self._remote_cars.items()
                         if now-v["last_seen"]>REMOTE_TIMEOUT_S]
                for k in stale: del self._remote_cars[k]

            # Build full snapshot (local + remote)
            local_snap = [
                {"car_id":c.car_id,"label":c.label,"car_type":c.car_type,
                 "lane":c.lane,"road_pos_m":c.road_pos_m,"speed_kmh":c.speed_kmh,
                 "state":c.state,"is_remote":False,"is_player":isinstance(c,PlayerCar),
                 "profile":c.profile_name,"profile_color":c.profile_color,
                 "lateral_offset_m":c.lateral_offset_m}
                for c in self.all_cars if c.state != RState.DONE
            ]
            with self._remote_lock:
                remote_snap = [
                    {"car_id":cid,"label":v["label"],"car_type":v.get("car_type",CarType.SMART),
                     "lane":v["lane"],"road_pos_m":v["road_pos_m"],"speed_kmh":v["speed_kmh"],
                     "state":v.get("state","DRIVING"),"is_remote":True,
                     "is_player":v.get("is_player",False),
                     "profile":v.get("profile","COMMUTER"),
                     "profile_color":v.get("profile_color","white"),
                     "lateral_offset_m":v.get("lateral_offset_m",0.0)}
                    for cid,v in self._remote_cars.items()
                ]
            snapshot = local_snap + remote_snap

            # V6 R2: tick weather
            self._weather_tick_accum += dt
            if self._weather_tick_accum >= WEATHER_CHANGE_INTERVAL_S:
                self._weather_tick_accum = 0.0
                self._weather_state = random.choices(
                    range(6), weights=WEATHER_WEIGHTS)[0]
                self._weather_changed_at = time.time()
                wname = WEATHER_PARAMS[self._weather_state]['name']
                with self._tok_lock:
                    self._token_events.append({'event':'WEATHER_CHANGE',
                        'detail':f"{WEATHER_PARAMS[self._weather_state]['icon']} Weather changed to {wname} — {WEATHER_PARAMS[self._weather_state]['desc']}",
                        'ts':time.time()})
                # Broadcast weather to other VMs so they stay in sync
                self.radio.send({
                    'type': Msg.WEATHER_SYNC,
                    'weather_state': self._weather_state,
                    'weather_changed_at': self._weather_changed_at,
                    'machine_id': self.machine_id,
                    'ts': time.time(),
                })

            # V6 R3: near-miss detection
            self._check_near_misses(snapshot)

            # V6 R5: update heatmap
            self._update_heatmap(snapshot)

            # Tick all local cars (push weather first)
            for car in self.all_cars:
                if hasattr(car, 'set_weather'):
                    car.set_weather(self._weather_state)
            for car in self.all_cars:
                try:
                    car.tick(dt, self.radio, snapshot)
                    self._handle_roadworks_zone(car)
                except Exception as e:
                    log.warning(f"Car tick error [{car.label}]: {e}")

            # Collect events
            new_evts = []
            for car in self.all_cars:
                if hasattr(car,"token_events") and car.token_events:
                    new_evts.extend(car.token_events[-3:])
                    car.token_events.clear()
            if new_evts:
                with self._tok_lock:
                    self._token_events.extend(new_evts)
                    if len(self._token_events)>300:
                        self._token_events = self._token_events[-200:]

            # Phase 4: push snapshot to incident buffer + check for triggers
            self.incident_buf.push(snapshot, self.tick_n, self.elapsed())
            self.incident_buf.check_events(new_evts)

            # Elapsed-aware sleep (no tick drift)
            elapsed = time.time()-t0
            sleep_t = max(0.0, dt-elapsed)
            if sleep_t>0: time.sleep(sleep_t)

    # ── Incoming message handler ───────────────────────────────────────────────
    def _handle_incoming(self, msg):
        mtype  = msg.get("type") or ""
        sender = msg.get("from","")
        if not sender: return

        if mtype == Msg.BEACON:
            local_ids = {c.car_id for c in self.all_cars}
            if sender in local_ids: return
            with self._remote_lock:
                self._remote_cars[sender] = {
                    "label":     msg.get("label", sender[:8]),
                    "lane":      msg.get("lane",0),
                    "road_pos_m":msg.get("road_pos_m",0.0),
                    "speed_kmh": msg.get("speed_kmh",0.0),
                    "state":     msg.get("state","DRIVING"),
                    "car_type":  msg.get("car_type",CarType.SMART),
                    "is_player": msg.get("is_player",False),
                    "profile":   msg.get("profile","COMMUTER"),
                    "profile_color":msg.get("profile_color","white"),
                    "lateral_offset_m":msg.get("lateral_offset_m",0.0),
                    "machine_id": msg.get("machine_id", 0),
                    "is_remote": True,
                    "last_seen": time.time(),
                }
        elif mtype == Msg.WEATHER_SYNC:
            # Adopt weather from another VM if it's different from ours
            remote_mid = msg.get("machine_id", 0)
            if remote_mid != self.machine_id:
                new_ws = msg.get("weather_state", WeatherState.CLEAR)
                remote_ts = msg.get("weather_changed_at", 0.0)
                if remote_ts > self._weather_changed_at:
                    self._weather_state      = new_ws
                    self._weather_changed_at = remote_ts
                    wname = WEATHER_PARAMS[new_ws]['name']
                    with self._tok_lock:
                        self._token_events.append({'event':'WEATHER_CHANGE',
                            'detail':f"{WEATHER_PARAMS[new_ws]['icon']} Weather (VM{remote_mid}): {wname}",
                            'ts':time.time()})
        elif mtype == Msg.HARD_BRAKE:
            with self._tok_lock:
                self._token_events.append({"event":"BRAKE_CHAIN",
                    "detail":f"⚠ {msg.get('label','?')} HARD_BRAKE  CVC 21703","ts":time.time()})
        elif mtype == Msg.EMERG_PREEMPT:
            with self._tok_lock:
                self._token_events.append({"event":"PREEMPT",
                    "detail":f"🚨 {msg.get('label','?')} EMERG_PREEMPT  CVC 21806","ts":time.time()})
        elif mtype == Msg.HAZARD:
            with self._tok_lock:
                self._token_events.append({"event":"HAZARD_WARN",
                    "detail":f"🔶 {msg.get('label','?')} HAZARD at {msg.get('road_pos_m',0):.0f}m","ts":time.time()})
        elif mtype == Msg.ZONE_ALERT:
            with self._tok_lock:
                self._token_events.append({"event":"ZONE_ALERT",
                    "detail":f"🏫 ZONE via {msg.get('label','?')}  CVC 22352a","ts":time.time()})

    # ── Status ────────────────────────────────────────────────────────────────
    def elapsed(self):
        return time.time()-self.start_time if self.start_time else 0.0

    def get_status(self):
        active = [c for c in self.all_cars if c.state!=RState.DONE]
        done   = [c for c in self.all_cars if c.state==RState.DONE]
        local_list  = [c.status() for c in self.all_cars if c.state!=RState.DONE]
        with self._remote_lock:
            remote_list = [
                {"car_id":cid,"label":v["label"],"car_type":v.get("car_type",CarType.SMART),
                 "lane":v["lane"],"road_pos_m":v["road_pos_m"],"speed_kmh":v["speed_kmh"],
                 "state":v.get("state","DRIVING"),"is_player":v.get("is_player",False),
                 "is_remote":True,"profile":v.get("profile","COMMUTER"),
                 "profile_color":v.get("profile_color","white"),
                 "lateral_offset_m":v.get("lateral_offset_m",0.0),
                 "machine_id":v.get("machine_id",0),
                 "lc_from_lane":v.get("lc_from_lane",v.get("lane",0)),
                 "lc_progress":v.get("lc_progress",1.0),
                 "fuel_pct":v.get("fuel_pct",1.0)}
                for cid,v in self._remote_cars.items()
            ]
        all_sorted = sorted(local_list+remote_list, key=lambda x: -x["road_pos_m"])
        rw_active = (self.roadworks and
            any(ROADWORKS_ZONE_START_M<=c.road_pos_m<=ROADWORKS_ZONE_END_M
                for c in self.all_cars if c.state!=RState.DONE))
        return {
            "elapsed_s":round(self.elapsed(),1), "tick":self.tick_n,
            "total_cars":len(self.all_cars)+len(self._remote_cars),
            "active_cars":len(active)+len(self._remote_cars),
            "done_cars":len(done),
            "done":len(done),
            "road_length_m":ROAD_LENGTH_M, "lane_count":3,
            "seg_start":self.seg_start, "seg_end":self.seg_end,
            "remote_count":len(self._remote_cars),
            "rush_hour":self.rush_hour, "roadworks":self.roadworks,
            "roadworks_active":rw_active,
            "spat_delay":self.spat_delay, "split_brain":self._split_brain,
            "players":[pc.status() for pc in self.player_cars],
            "npcs":[c.status() for c in self.npcs],
            "all_cars_sorted":all_sorted,
            "incident_frames": self.incident_buf.size(),
            "incident_writes": self.incident_buf.write_count(),
            "weather_state":   self._weather_state,
            "weather_changed_at": self._weather_changed_at,
            "near_miss_count": self._near_miss_count,
            "collision_count": self._collision_count,
            "heatmap":         list(self._heatmap),
        }

    def get_token_events(self):
        with self._tok_lock: return list(self._token_events)

    def get_all_events(self):
        evts = list(self._split_brain_events)
        for car in self.all_cars:
            if hasattr(car,"events"): evts.extend(car.events[-8:])
        evts.sort(key=lambda e: e.get("ts",0))
        return evts[-60:]

    def enqueue_command(self, cmd, machine_id=1):
        pc = self.get_player(machine_id)
        if pc: pc.enqueue_command(cmd)

    def _check_near_misses(self, snapshot):
        """V6 R3: scan same-lane pairs for near-miss / collision.
        Each pair is rate-limited to one event per 3 seconds so sustained
        overlap doesn't flood the event log."""
        from config import NEAR_MISS_M, CAR_LENGTH_M
        now = time.time()
        by_lane = {}
        for s in snapshot:
            if s.get('is_remote'): continue
            ln = s.get('lane',0)
            by_lane.setdefault(ln,[]).append(s)
        for ln, cars in by_lane.items():
            cars.sort(key=lambda x: x.get('road_pos_m',0))
            for i in range(len(cars)-1):
                a, b = cars[i], cars[i+1]
                gap = b.get('road_pos_m',0) - a.get('road_pos_m',0) - CAR_LENGTH_M
                if gap <= NEAR_MISS_M:
                    la, lb = a.get('label','?'), b.get('label','?')
                    pair_key = "|".join(sorted([la, lb])) + f"|{ln}"
                    last = self._incident_cooldown.get(pair_key, 0)
                    if now - last < 3.0:
                        continue   # already reported this pair recently
                    self._incident_cooldown[pair_key] = now
                    if gap <= 0.0:
                        self._collision_count += 1
                        with self._tok_lock:
                            self._token_events.append({'event':'COLLISION',
                                'detail':f"💥 COLLISION: {a.get('label','?')} ↔ {b.get('label','?')} lane {ln}",
                                'ts':now})
                    else:
                        self._near_miss_count += 1
                        with self._tok_lock:
                            self._token_events.append({'event':'NEAR_MISS',
                                'detail':f"⚠ NEAR-MISS: {a.get('label','?')} ↔ {b.get('label','?')} gap={gap:.1f}m",
                                'ts':now})

    def _update_heatmap(self, snapshot):
        """V6 R5: compute average speed per bucket."""
        from road_geometry import ROAD_LENGTH_M
        buckets  = [[] for _ in range(HEATMAP_BUCKETS)]
        bkt_size = ROAD_LENGTH_M / HEATMAP_BUCKETS
        for s in snapshot:
            if s.get('state') == RState.DONE: continue
            idx = int(s.get('road_pos_m',0) / bkt_size)
            idx = max(0, min(HEATMAP_BUCKETS-1, idx))
            buckets[idx].append(s.get('speed_kmh',0))
        self._heatmap = [
            round(sum(b)/len(b)) if b else 0
            for b in buckets
        ]

    def is_all_done(self):
        return all(c.state==RState.DONE for c in self.all_cars)
