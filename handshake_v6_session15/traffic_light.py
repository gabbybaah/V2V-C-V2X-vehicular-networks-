# =============================================================================
# handshake_v4/traffic_light.py  —  Signal Phase and Timing controller
#
# Cycles through NS_GREEN → NS_YELLOW → ALL_RED → EW_GREEN → EW_YELLOW → ALL_RED_2
# Broadcasts SPaT on the shared radio every SPAT_INTERVAL_S seconds.
# Supports emergency preemption (CVC 21806 / MUTCD 4D.27).
#
# DEADLOCK NOTE: All callbacks are invoked OUTSIDE the internal lock.
#   _phase / _green_arms are plain attributes — written under lock, read freely.
# =============================================================================
import time, threading, logging
from config import (Phase, GREEN_FOR, GREEN_S, YELLOW_S, ALL_RED_S,
                    PREEMPT_HOLD_S, Msg)

log = logging.getLogger("v4.light")

PHASE_SEQUENCE = [
    Phase.NS_GREEN, Phase.NS_YELLOW, Phase.ALL_RED,
    Phase.EW_GREEN, Phase.EW_YELLOW, Phase.ALL_RED_2,
]
PHASE_DURATION = {
    Phase.NS_GREEN:  GREEN_S,
    Phase.NS_YELLOW: YELLOW_S,
    Phase.ALL_RED:   ALL_RED_S,
    Phase.EW_GREEN:  GREEN_S,
    Phase.EW_YELLOW: YELLOW_S,
    Phase.ALL_RED_2: ALL_RED_S,
}


class TrafficLight:
    def __init__(self, light_id: str = "INT-01"):
        self.id            = light_id
        self._phase        = Phase.NS_GREEN
        self._green_arms   = list(GREEN_FOR[Phase.NS_GREEN])
        self._phase_start  = time.time()
        self._phase_idx    = 0
        self._preempted    = False
        self._preempt_arm  = None
        self._preempt_end  = 0.0
        self._lock         = threading.Lock()
        self._running      = False
        self._thread       = None
        self._callbacks    = []       # called on phase change
        self.cycles        = 0
        self.preemptions   = 0

    # ── Public read (lock-free for safe polling) ──────────────────────────────

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def green_arms(self) -> list:
        return list(self._green_arms)

    def is_green_for(self, arm: str) -> bool:
        return arm in self._green_arms

    def time_remaining(self) -> float:
        with self._lock:
            if self._preempted:
                return max(0.0, self._preempt_end - time.time())
            dur = PHASE_DURATION.get(self._phase, GREEN_S)
            return max(0.0, dur - (time.time() - self._phase_start))

    def get_spat(self) -> dict:
        return {
            "type":          Msg.SPAT,
            "from":          f"LIGHT-{self.id}",
            "phase":         self._phase,
            "green_arms":    list(self._green_arms),
            "time_remaining": round(self.time_remaining(), 1),
            "preempted":     self._preempted,
            "preempt_arm":   self._preempt_arm,
            "ts":            time.time(),
        }

    # ── Callbacks ────────────────────────────────────────────────────────────

    def on_phase_change(self, cb):
        self._callbacks.append(cb)

    def _fire(self, phase: str, green: list):
        """Invoke callbacks outside lock."""
        for cb in self._callbacks:
            try:
                cb(phase, green)
            except Exception as e:
                log.warning(f"[Light] Callback error: {e}")

    # ── Emergency preemption (CVC 21806) ─────────────────────────────────────

    def preempt(self, arm: str):
        with self._lock:
            if self._preempted:
                return
            self._preempted   = True
            self._preempt_arm = arm
            self._preempt_end = time.time() + PREEMPT_HOLD_S
            self._phase       = Phase.PREEMPTED
            self._green_arms  = [arm]
            self._phase_start = time.time()
            self.preemptions += 1
        log.info(f"[Light {self.id}] PREEMPTED for {arm} (CVC 21806)")
        self._fire(Phase.PREEMPTED, [arm])

    # ── Internal loop ─────────────────────────────────────────────────────────

    def start(self):
        self._running    = True
        self._phase_start = time.time()
        self._thread     = threading.Thread(
            target=self._loop, daemon=True, name=f"light-{self.id}"
        )
        self._thread.start()

    def stop(self):
        self._running = False

    # ── Pedestrian crossing interrupt — Feature #13 ───────────────────────────

    def pedestrian_crossing(self):
        """
        Interrupt current green phase for a pedestrian crossing event.
        All arms go to ALL_RED (Phase.PEDESTRIAN) for PEDESTRIAN_HOLD_S.
        CVC 21950 — all vehicles must yield to pedestrian in crosswalk.
        """
        from config import PEDESTRIAN_HOLD_S, Phase
        with self._lock:
            if self._preempted:
                return  # Emergency has priority over pedestrian
            self._phase       = Phase.PEDESTRIAN
            self._green_arms  = []
            self._phase_start = time.time()
            self._ped_end     = time.time() + PEDESTRIAN_HOLD_S
            self._pedestrian  = True
        log.info(f"[Light {self.id}] PEDESTRIAN CROSSING — all arms hold {PEDESTRIAN_HOLD_S}s")
        self._fire(Phase.PEDESTRIAN, [])

    def _loop(self):
        """Override to handle pedestrian phase."""
        while self._running:
            time.sleep(0.05)
            notify = None

            with self._lock:
                now = time.time()

                if self._preempted:
                    if now >= self._preempt_end:
                        self._preempted   = False
                        self._preempt_arm = None
                        self._phase       = Phase.ALL_RED
                        self._green_arms  = []
                        self._phase_start = now
                        notify = (Phase.ALL_RED, [])

                elif getattr(self, '_pedestrian', False):
                    if now >= getattr(self, '_ped_end', now):
                        self._pedestrian  = False
                        # Resume normal cycle
                        self._phase_idx   = (self._phase_idx + 1) % len(PHASE_SEQUENCE)
                        self._phase       = PHASE_SEQUENCE[self._phase_idx]
                        self._green_arms  = list(GREEN_FOR.get(self._phase, []))
                        self._phase_start = now
                        notify = (self._phase, list(self._green_arms))

                else:
                    dur = PHASE_DURATION.get(self._phase, GREEN_S)
                    if now - self._phase_start >= dur:
                        self._phase_idx  = (self._phase_idx + 1) % len(PHASE_SEQUENCE)
                        self._phase      = PHASE_SEQUENCE[self._phase_idx]
                        self._green_arms = list(GREEN_FOR.get(self._phase, []))
                        self._phase_start = now
                        if self._phase_idx == 0:
                            self.cycles += 1
                        notify = (self._phase, list(self._green_arms))

            if notify:
                phase, green = notify
                log.info(f"[Light {self.id}] → {phase}  green={green}")
                self._fire(phase, green)
