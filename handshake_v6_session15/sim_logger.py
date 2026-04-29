# =============================================================================
# handshake_v4/sim_logger.py  —  Persistent Simulation Logger
#
# Every simulation run is automatically logged:
#   logs/intersection_YYYYMMDD_HHMMSS.json  — full structured event log
#   logs/intersection_YYYYMMDD_HHMMSS.txt   — human-readable transcript
#   logs/HISTORY.txt                         — cumulative one-line-per-run index
#   logs/VIOLATIONS.txt                      — cumulative CVC violation ledger
#
# SimLogger.start()  — begin collecting events in background thread
# SimLogger.finish() — called on sim stop; writes all files
# SimLogger.attach(sim) — convenience factory
# =============================================================================
import os, json, time, threading, logging, datetime
from collections import defaultdict

log = logging.getLogger("v4.logger")

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _ensure_logs_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── CVC Violation Ledger ──────────────────────────────────────────────────────

class ViolationLedger:
    """
    Tracks all CVC violations per car across a simulation.
    Each violation is a "citation" — logged at end as a court record.
    """
    SEVERITY = {
        "CVC 21453": 2,   # running red
        "CVC 21703": 1,   # following too close
        "CVC 21750": 1,   # unsafe overtake
        "CVC 21658": 1,   # unsafe lane change
        "CVC 22352": 2,   # speed violation
        "CVC 22352a": 3,  # school zone speed
        "CVC 21806": 4,   # failed to yield to emergency
        "CVC 21950": 3,   # failed to yield to pedestrian
        "PROTOCOL":  5,   # rogue / token violation
    }
    SEVERITY_LABEL = {1: "WARNING", 2: "INFRACTION", 3: "VIOLATION",
                      4: "MISDEMEANOR", 5: "PROTOCOL BREACH"}

    def __init__(self):
        self._citations: list = []          # list of citation dicts
        self._by_car: dict    = defaultdict(list)  # car_label → citations
        self._lock            = threading.Lock()

    def cite(self, car_label: str, cvc: str, detail: str, ts: float = None):
        sev   = self.SEVERITY.get(cvc, 1)
        label = self.SEVERITY_LABEL.get(sev, "WARNING")
        entry = {
            "ts":        ts or time.time(),
            "car":       car_label,
            "cvc":       cvc,
            "severity":  sev,
            "category":  label,
            "detail":    detail,
        }
        with self._lock:
            self._citations.append(entry)
            self._by_car[car_label].append(entry)

    def scan_events(self, events: list):
        """
        Post-process all simulation events and auto-generate citations
        for known violation patterns.
        """
        keywords = {
            "VIOLATION":       ("PROTOCOL", "PROTOCOL BREACH"),
            "ROGUE_VIOLATION": ("PROTOCOL", "Rogue car ignored token slot"),
            "CVC_VIOLATION":   ("CVC 21703", "Following distance violation"),
            "SCHOOL_ZONE":     ("CVC 22352a", "Speed in school zone"),
            "LEGACY_NO_WARN":  ("CVC 21703", "Legacy car: no chain-brake warning"),
        }
        seen = set()
        for ev in events:
            etype  = str(ev.get("event", ""))
            detail = str(ev.get("detail", ""))
            car    = str(ev.get("label", "?"))
            key    = (car, etype)
            if key in seen:
                continue
            for kw, (cvc, desc) in keywords.items():
                if kw in etype or kw in detail:
                    self.cite(car, cvc, detail or desc, ts=ev.get("ts"))
                    seen.add(key)
                    break

    def summary(self) -> list:
        """Return sorted list of (label, citation_count, max_severity)."""
        out = []
        for car, cits in self._by_car.items():
            count = len(cits)
            max_s = max(c["severity"] for c in cits)
            out.append({"car": car, "citations": count, "max_severity": max_s,
                        "max_category": self.SEVERITY_LABEL.get(max_s, "?"),
                        "items": cits})
        return sorted(out, key=lambda x: -x["max_severity"])

    def as_text(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║            CVC VIOLATION LEDGER — COURT RECORD           ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        summary = self.summary()
        if not summary:
            lines.append("  No violations recorded.")
            return "\n".join(lines)
        for entry in summary:
            lines.append(f"\n  Car: {entry['car']}  |  Citations: {entry['citations']}  "
                         f"|  Max: {entry['max_category']}")
            for c in entry["items"][:8]:
                t = datetime.datetime.fromtimestamp(c["ts"]).strftime("%H:%M:%S")
                lines.append(f"    [{t}] {c['cvc']:<12} [{c['category']:<16}] {c['detail'][:55]}")
        return "\n".join(lines)


# ── Main Logger ───────────────────────────────────────────────────────────────

class SimLogger:
    """
    Attach to any simulation and call finish() when done.
    Writes timestamped JSON + TXT + appends to HISTORY.txt and VIOLATIONS.txt.
    """

    def __init__(self, sim_type: str, sim, extra_meta: dict = None):
        """
        sim_type : 'intersection' or 'road'
        sim      : IntersectionSim or RoadSim instance
        """
        self.sim_type  = sim_type
        self.sim       = sim
        self.meta      = extra_meta or {}
        self._ts       = _ts()
        self._start_t  = time.time()
        self._snapshots: list = []     # periodic status snapshots
        self._running  = False
        self._thread   = None
        self.ledger    = ViolationLedger()
        # V5: per-car trajectory store — collected every tick for ML training
        # dict: car_label → list of {ts, road_pos_m, speed_kmh, lane, lateral_offset_m, state}
        self._car_trajectories: dict = {}
        self._traj_lock = threading.Lock()
        _ensure_logs_dir()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._snapshot_loop, daemon=True, name="sim-logger")
        self._thread.start()

    def _snapshot_loop(self):
        """Capture a status snapshot every 5 seconds, and per-car trajectories every tick."""
        last_snap = 0.0
        while self._running:
            try:
                now = time.time()
                # Full status snapshot every 5s
                if now - last_snap >= 5.0:
                    snap = self.sim.get_status()
                    snap["_log_ts"] = now
                    self._snapshots.append(snap)
                    if len(self._snapshots) > 200:
                        self._snapshots = self._snapshots[-150:]
                    last_snap = now

                # Per-car trajectory: collect state of every active car each tick
                if self.sim_type == "road" and hasattr(self.sim, "all_cars"):
                    ts = now - self._start_t
                    with self._traj_lock:
                        for car in self.sim.all_cars:
                            if not hasattr(car, "road_pos_m"):
                                continue
                            lbl = getattr(car, "label", "?")
                            if lbl not in self._car_trajectories:
                                self._car_trajectories[lbl] = {
                                    "profile": getattr(car, "profile_name", "COMMUTER"),
                                    "car_type": getattr(car, "car_type", "smart"),
                                    "ticks": []
                                }
                            # Cap at 6000 ticks (10min at 10ticks/s) to bound memory
                            ticks = self._car_trajectories[lbl]["ticks"]
                            if len(ticks) < 6000:
                                ticks.append({
                                    "ts":   round(ts, 2),
                                    "pos":  round(car.road_pos_m, 1),
                                    "spd":  round(car.speed_kmh, 1),
                                    "lane": car.lane,
                                    "lat":  round(car.lateral_offset_m, 3),
                                    "st":   car.state,
                                })
            except Exception:
                pass
            time.sleep(0.1)   # ~10 ticks/sec — matches sim tick rate

    def finish(self):
        """Called when simulation ends. Writes all log files."""
        self._running = False
        elapsed = time.time() - self._start_t

        # Collect all events
        try:
            token_events = self.sim.get_token_events()
        except Exception:
            token_events = []
        try:
            all_events = self.sim.get_all_events() if hasattr(self.sim, "get_all_events") else []
        except Exception:
            all_events = []

        all_combined = sorted(token_events + all_events, key=lambda e: e.get("ts", 0))

        # Final status
        try:
            final_status = self.sim.get_status()
        except Exception:
            final_status = {}

        # Build violation ledger
        self.ledger.scan_events(all_combined)

        # Write files
        json_path = self._write_json(elapsed, final_status, all_combined)
        txt_path  = self._write_txt(elapsed, final_status, all_combined)
        self._append_history(elapsed, final_status, json_path)
        self._append_violations(elapsed, final_status)

        print(f"\n  📋 Simulation log saved:")
        print(f"     {txt_path}")
        print(f"     {json_path}")
        print(f"     {os.path.join(LOGS_DIR, 'HISTORY.txt')} (updated)")
        if self.ledger._citations:
            print(f"     {os.path.join(LOGS_DIR, 'VIOLATIONS.txt')} (updated)")

        # V5: extract training data from this run (background thread)
        if self.sim_type == "road":
            try:
                from training_extractor import process as extract_training
                extract_training(json_path, async_mode=True)
            except Exception as _e:
                log.debug(f"Training extractor skipped: {_e}")

        return txt_path, json_path

    # ── JSON log ──────────────────────────────────────────────────────────────

    def _write_json(self, elapsed: float, final_status: dict,
                    events: list) -> str:
        done    = final_status.get("done", 0)
        total   = final_status.get("total_cars", 0)
        tput    = done / elapsed * 60 if elapsed > 0 and done > 0 else 0

        # Throughput curve (from snapshots)
        tput_curve = []
        for s in self._snapshots:
            t = s.get("_log_ts", 0) - self._start_t
            d = s.get("done", 0)
            tput_curve.append({"t": round(t, 1), "done": d})

        payload = {
            "meta": {
                "sim_type":   self.sim_type,
                "timestamp":  _now_str(),
                "run_id":     self._ts,
                "elapsed_s":  round(elapsed, 2),
                **self.meta,
            },
            "summary": {
                "total_cars":   total,
                "done":         done,
                "pct_done":     round(100 * done / total, 1) if total else 0,
                "throughput_per_min": round(tput, 2),
                "peak_throughput":    final_status.get("peak_throughput", 0),
                "total_events":       len(events),
                "token_issued":       sum(1 for e in events if e.get("event") == "TOKEN_ISSUED"),
                "token_ack":          sum(1 for e in events if e.get("event") == "TOKEN_ACK"),
                "violations":         len(self.ledger._citations),
                "rogue_crossings":    sum(1 for e in events if "ROGUE" in str(e.get("event",""))),
                "emergency_events":   sum(1 for e in events if "PREEMPT" in str(e.get("event","")) or "CORRIDOR" in str(e.get("event",""))),
                "pedestrian_events":  sum(1 for e in events if "PEDESTRIAN" in str(e.get("event",""))),
                "hazard_events":      sum(1 for e in events if "HAZARD" in str(e.get("event",""))),
                "school_zone_events": sum(1 for e in events if "SCHOOL" in str(e.get("event",""))),
                "platoon_events":     sum(1 for e in events if "PLATOON" in str(e.get("event",""))),
                "convoy_events":      sum(1 for e in events if "CONVOY" in str(e.get("event",""))),
                "split_brain_events": sum(1 for e in events if "SPLIT" in str(e.get("event",""))+str(e.get("detail",""))),
                "spat_stale_events":  sum(1 for e in events if "STALE" in str(e.get("event",""))+str(e.get("detail",""))),
                "cvc_violations":     sum(1 for e in events if "VIOLATION" in str(e.get("event",""))),
            },
            "final_status":    final_status,
            "throughput_curve": tput_curve,
            "violation_ledger": self.ledger.summary(),
            "events":          events[-500:],   # last 500 events (full run in JSON)
            "car_trajectories": dict(self._car_trajectories),  # V5 ML training data
        }

        path = os.path.join(LOGS_DIR, f"{self.sim_type}_{self._ts}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    # ── TXT transcript ────────────────────────────────────────────────────────

    def _write_txt(self, elapsed: float, final_status: dict,
                   events: list) -> str:
        done  = final_status.get("done", 0)
        total = final_status.get("total_cars", 0)
        tput  = done / elapsed * 60 if elapsed > 0 and done > 0 else 0
        lines = []

        lines += [
            "=" * 70,
            f"  HANDSHAKE V4 — {self.sim_type.upper()} SIMULATION LOG",
            f"  Run ID  : {self._ts}",
            f"  Date    : {_now_str()}",
            f"  Runtime : {elapsed:.1f}s",
        ]
        for k, v in self.meta.items():
            lines.append(f"  {k:<9}: {v}")
        lines += [
            "=" * 70,
            "",
            "── SUMMARY ──────────────────────────────────────────────────────────",
            f"  Fleet        : {total} cars",
            f"  Cleared      : {done}/{total} ({100*done/total:.0f}%)" if total else "  Cleared: 0",
            f"  Throughput   : {tput:.1f} cars/min",
        ]

        # Token protocol stats
        tok_i = sum(1 for e in events if e.get("event") == "TOKEN_ISSUED")
        tok_a = sum(1 for e in events if e.get("event") == "TOKEN_ACK")
        rogue = sum(1 for e in events if "ROGUE" in str(e.get("event","")))
        conflt= sum(1 for e in events if "CONFLICT" in str(e.get("event","")))
        fallb = sum(1 for e in events if "FALLBACK" in str(e.get("event","")))
        canc  = sum(1 for e in events if e.get("event") == "TOKEN_CANCEL")
        lines += [
            "",
            "── TOKEN EXCHANGE PROTOCOL ──────────────────────────────────────────",
            f"  TOKEN_ISSUED   : {tok_i}",
            f"  TOKEN_ACK      : {tok_a}",
            f"  TOKEN_CANCEL   : {canc}   (emergency mid-negotiation)",
            f"  TOKEN_FALLBACK : {fallb}  (no token within timeout)",
            f"  SLOT_CONFLICT  : {conflt} (duplicate position resolved)",
            f"  ROGUE_CROSS    : {rogue}  (protocol violations)",
        ]

        # CVC events
        emerg = sum(1 for e in events if "PREEMPT" in str(e.get("event","")) or "CORRIDOR" in str(e.get("event","")))
        ped   = sum(1 for e in events if "PEDESTRIAN" in str(e.get("event","")))
        haz   = sum(1 for e in events if "HAZARD" in str(e.get("event","")))
        zone  = sum(1 for e in events if "SCHOOL" in str(e.get("event","")) or "ZONE_ALERT" in str(e.get("event","")))
        plat  = sum(1 for e in events if "PLATOON" in str(e.get("event","")))
        conv  = sum(1 for e in events if "CONVOY" in str(e.get("event","")))
        split = sum(1 for e in events if "SPLIT" in str(e.get("event",""))+str(e.get("detail","")))
        stale = sum(1 for e in events if "STALE" in str(e.get("event",""))+str(e.get("detail","")))
        row   = sum(1 for e in events if "21800" in str(e.get("detail","")) or "ROW" in str(e.get("event","")))
        lines += [
            "",
            "── SCENARIO EVENTS ──────────────────────────────────────────────────",
            f"  Emergency / Corridor : {emerg}  (CVC 21806)",
            f"  Pedestrian crossings : {ped}   (CVC 21950)",
            f"  Hazard warnings      : {haz}  (CVC 22500)",
            f"  Zone alerts          : {zone}  (CVC 22352a)",
            f"  Platoon events       : {plat}",
            f"  Convoy events        : {conv}",
            f"  Right-of-way (21800) : {row}",
            f"  Split-brain events   : {split}",
            f"  SPaT stale events    : {stale}",
        ]

        # Violation ledger
        lines += ["", self.ledger.as_text()]

        # Timeline: key events only
        important = [
            "TOKEN_ISSUED","TOKEN_CANCEL","ROGUE_VIOLATION","AMBULANCE_CORRIDOR",
            "PEDESTRIAN","SPLIT_BRAIN","SPAT_STALE","MACHINE_FAIL",
            "CONVOY_OVERTAKE","ROW_NEGOTIATION","BREAKDOWN","ROADWORKS",
        ]
        key_evts = [e for e in events if e.get("event","") in important
                    or any(k in str(e.get("detail","")) for k in
                           ["🚨","⛔","🚶","⚠","SPLIT","STALE","CONVOY","ROW"])]
        if key_evts:
            lines += ["", "── KEY EVENT TIMELINE ───────────────────────────────────────────────"]
            for ev in key_evts[-60:]:
                t = datetime.datetime.fromtimestamp(ev["ts"]).strftime("%H:%M:%S")
                arm   = ev.get("arm", ev.get("label", ""))
                etype = ev.get("event", "")
                det   = str(ev.get("detail", ""))[:62]
                lines.append(f"  [{t}] {arm:<6} {etype:<22} {det}")

        # Full event log
        lines += ["", "── FULL EVENT LOG ───────────────────────────────────────────────────"]
        for ev in events:
            t = datetime.datetime.fromtimestamp(ev.get("ts", 0)).strftime("%H:%M:%S.%f")[:-3]
            arm   = ev.get("arm", ev.get("label", ""))[:8]
            etype = str(ev.get("event", ""))[:20]
            det   = str(ev.get("detail", ev.get("phase", "")))[:60]
            lines.append(f"  {t}  {arm:<8} {etype:<22} {det}")

        lines += ["", "=" * 70, ""]

        path = os.path.join(LOGS_DIR, f"{self.sim_type}_{self._ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    # ── History index ─────────────────────────────────────────────────────────

    def _append_history(self, elapsed: float, final_status: dict, json_path: str):
        done  = final_status.get("done", 0)
        total = final_status.get("total_cars", 0)
        tput  = done / elapsed * 60 if elapsed > 0 and done > 0 else 0
        viols = len(self.ledger._citations)

        modes = []
        if self.meta.get("rush_hour"):     modes.append("RUSH")
        if self.meta.get("rogue"):         modes.append("ROGUE")
        if self.meta.get("uncontrolled"):  modes.append("CVC21800")
        if self.meta.get("ambulance"):     modes.append("AMBULANCE")
        if self.meta.get("pedestrian"):    modes.append("PED")
        if self.meta.get("split_brain"):   modes.append("SPLIT")
        if self.meta.get("machine_fail"):  modes.append("FAIL")
        if self.meta.get("convoy"):        modes.append("CONVOY")
        if self.meta.get("roadworks"):     modes.append("ROADWORKS")
        if self.meta.get("spat_delay", 0): modes.append(f"DELAY{self.meta['spat_delay']:.1f}s")

        mode_str = "[" + "+".join(modes) + "]" if modes else ""
        line = (f"{_now_str()}  {self.sim_type:<14} "
                f"T={elapsed:>6.0f}s  "
                f"done={done:>4}/{total:<4}  "
                f"{tput:>5.1f}/min  "
                f"viols={viols:<3}  "
                f"{mode_str:<30}  "
                f"{os.path.basename(json_path)}")

        hist_path = os.path.join(LOGS_DIR, "HISTORY.txt")
        with open(hist_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── Violations ledger file ────────────────────────────────────────────────

    def _append_violations(self, elapsed: float, final_status: dict):
        if not self.ledger._citations:
            return
        viol_path = os.path.join(LOGS_DIR, "VIOLATIONS.txt")
        with open(viol_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'─'*70}\n")
            f.write(f"RUN: {self._ts}  {self.sim_type}  T={elapsed:.0f}s\n")
            f.write(self.ledger.as_text() + "\n")


# ── Convenience factory ────────────────────────────────────────────────────────

def attach_logger(sim_type: str, sim, meta: dict = None) -> "SimLogger":
    """Create, attach, and start a SimLogger. Call .finish() when sim ends."""
    sl = SimLogger(sim_type, sim, extra_meta=meta or {})
    sl.start()
    return sl
