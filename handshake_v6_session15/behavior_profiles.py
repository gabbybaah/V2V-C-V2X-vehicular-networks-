# =============================================================================
# handshake_v5/behavior_profiles.py  —  6 driving personality profiles
#
# Every NPC (smart AND legacy) gets one profile at spawn.
# Profile controls: target following gap, braking style, lane change frequency,
#                   speed consistency, lateral drift.
#
# Smart car with a profile: still obeys CVC law checks — profile modulates
#   THRESHOLDS (how close it tries to get) not the final law gate.
# Legacy car with a profile: acts on the profile directly with no V2X and
#   no law checks — pure physics behaviour.
#
# Profile names / colours (for dashboard):
#   COMMUTER        white
#   TAILGATER       red
#   HESITANT        yellow
#   LATE_BRAKER     magenta
#   AGGR_OT         bold red
#   WANDERER        cyan
# =============================================================================

import random, time, math
from dataclasses import dataclass, field
from typing import Optional

# ── Profile identifiers ───────────────────────────────────────────────────────
COMMUTER   = "COMMUTER"
TAILGATER  = "TAILGATER"
HESITANT   = "HESITANT"
LATE_BRAKER = "LATE_BRAKER"
AGGR_OT    = "AGGR_OT"
WANDERER   = "WANDERER"

ALL_PROFILES = [COMMUTER, TAILGATER, HESITANT, LATE_BRAKER, AGGR_OT, WANDERER]

# Weighted random — commuter most common, aggr_ot least
PROFILE_WEIGHTS = [0.30, 0.15, 0.15, 0.15, 0.10, 0.15]

# Rich terminal colours per profile
PROFILE_COLORS = {
    COMMUTER:    "white",
    TAILGATER:   "red",
    HESITANT:    "yellow",
    LATE_BRAKER: "magenta",
    AGGR_OT:     "bold red",
    WANDERER:    "cyan",
}

# Profile display labels
PROFILE_LABELS = {
    COMMUTER:    "Commuter",
    TAILGATER:   "Tailgater",
    HESITANT:    "Hesitant",
    LATE_BRAKER: "LateBreaker",
    AGGR_OT:     "AggrOT",
    WANDERER:    "Wanderer",
}


def random_profile() -> str:
    return random.choices(ALL_PROFILES, weights=PROFILE_WEIGHTS, k=1)[0]


# ── Base profile ──────────────────────────────────────────────────────────────

class BaseProfile:
    name = COMMUTER

    def __init__(self):
        self.phase           = "CRUISE"
        self.phase_timer     = 0.0
        self._last_action    = ""
        # Distraction model — occasional inattention
        self._distract_timer = random.uniform(30.0, 90.0)
        self._distracted     = False
        self._distract_dur   = 0.0

    # ── Targets returned to car logic ─────────────────────────────────────────
    @property
    def target_headway_s(self) -> float:
        return 3.0  # safe following time in seconds

    @property
    def brake_aggression(self) -> float:
        """Multiplier on braking rate. 1.0 = normal, 2.0 = hard."""
        return 1.0

    @property
    def accel_aggression(self) -> float:
        return 1.0

    @property
    def lane_change_freq(self) -> float:
        """Relative frequency of lane changes. 1.0 = normal."""
        return 1.0

    @property
    def lateral_drift_rate(self) -> float:
        """How much the car wanders within its lane per second."""
        return 0.0

    @property
    def reaction_delay_s(self) -> float:
        """Perception-reaction delay before responding to car-ahead changes."""
        return 0.8

    # ── Execute — called every tick from car.tick() ───────────────────────────
    def execute(self, dt: float, speed_ms: float, gap_m: float,
                road_pos_m: float, speed_limit_ms: float) -> dict:
        """
        Returns dict of driving adjustments:
          delta_speed_ms:  how much to change speed this tick
          target_gap_m:    desired following gap
          want_overtake:   bool — profile wants to overtake
          lateral_delta:   lateral offset change this tick
          event:           optional event string for logging
        """
        self._tick_distraction(dt)

        result = {
            "delta_speed_ms": 0.0,
            "target_gap_m":   max(5.0, speed_ms * self.target_headway_s),
            "want_overtake":  False,
            "lateral_delta":  0.0,
            "event":          None,
        }

        if self._distracted:
            # Slow slightly when distracted
            result["delta_speed_ms"] = -0.5 * dt
            return result

        self._execute_phase(dt, speed_ms, gap_m, road_pos_m,
                            speed_limit_ms, result)
        return result

    def _execute_phase(self, dt, speed_ms, gap_m, road_pos_m,
                       speed_limit_ms, result):
        # Default: aim for speed limit with comfortable gap
        if speed_ms < speed_limit_ms * 0.95:
            result["delta_speed_ms"] = 1.5 * dt * self.accel_aggression
        elif speed_ms > speed_limit_ms:
            result["delta_speed_ms"] = -1.0 * dt

    def _tick_distraction(self, dt: float):
        self._distract_timer -= dt
        if self._distracted:
            self._distract_dur -= dt
            if self._distract_dur <= 0:
                self._distracted = False
                self._distract_timer = random.uniform(30.0, 120.0)
        elif self._distract_timer <= 0:
            # 8% chance of distraction event when timer fires
            if random.random() < 0.08:
                self._distracted   = True
                self._distract_dur = random.uniform(2.0, 5.0)
            else:
                self._distract_timer = random.uniform(30.0, 120.0)


# ── Steady Commuter ───────────────────────────────────────────────────────────

class CommuterProfile(BaseProfile):
    name = COMMUTER
    @property
    def reaction_delay_s(self): return 0.8
    @property
    def target_headway_s(self): return 3.0
    @property
    def brake_aggression(self): return 1.0
    @property
    def accel_aggression(self): return 1.0
    @property
    def lane_change_freq(self): return 0.8

    def _execute_phase(self, dt, speed_ms, gap_m, road_pos_m,
                       speed_limit_ms, result):
        target = speed_limit_ms * 0.92
        diff   = target - speed_ms
        result["delta_speed_ms"] = diff * 0.3 * dt


# ── Tailgater ─────────────────────────────────────────────────────────────────

class TailgaterProfile(BaseProfile):
    name = TAILGATER
    @property
    def reaction_delay_s(self): return 0.5

    def __init__(self):
        super().__init__()
        self._follow_phase   = "CLOSE_FOLLOW"
        self._same_car_timer = 0.0

    @property
    def target_headway_s(self): return 0.8   # very close
    @property
    def brake_aggression(self): return 2.2   # hard and late
    @property
    def accel_aggression(self): return 2.5
    @property
    def lane_change_freq(self): return 0.5   # stays in fast lane

    def _execute_phase(self, dt, speed_ms, gap_m, road_pos_m,
                       speed_limit_ms, result):
        # Always push toward speed limit + 10%
        target = speed_limit_ms * 1.10
        diff   = target - speed_ms
        result["delta_speed_ms"] = diff * 0.5 * dt

        target_gap = max(3.0, speed_ms * 0.8)
        result["target_gap_m"] = target_gap

        if gap_m > target_gap * 2.5:
            # Gap opened — accelerate aggressively to close
            result["delta_speed_ms"] += 2.0 * dt

        # Build overtake if following same car a long time
        self._same_car_timer += dt
        if self._same_car_timer > 12.0:
            result["want_overtake"] = True
            result["event"] = "TAILGATER_OT_BUILD"


# ── Hesitant Merger ───────────────────────────────────────────────────────────

class HesitantProfile(BaseProfile):
    name = HESITANT
    @property
    def reaction_delay_s(self): return 1.3

    def __init__(self):
        super().__init__()
        self._drift_dir      = 1.0
        self._drift_timer    = random.uniform(5.0, 15.0)
        self._abort_count    = 0

    @property
    def target_headway_s(self): return 2.5
    @property
    def brake_aggression(self): return 1.0
    @property
    def accel_aggression(self): return 0.8
    @property
    def lane_change_freq(self): return 0.6
    @property
    def lateral_drift_rate(self): return 0.04   # slow drift toward boundary

    def _execute_phase(self, dt, speed_ms, gap_m, road_pos_m,
                       speed_limit_ms, result):
        target = speed_limit_ms * 0.88
        diff   = target - speed_ms
        result["delta_speed_ms"] = diff * 0.25 * dt

        # Drift toward lane boundary then pull back
        self._drift_timer -= dt
        if self._drift_timer <= 0:
            # Drift one way, then randomly abort or commit
            if random.random() < 0.4:
                # Abort — drift back
                result["lateral_delta"] = -self._drift_dir * self.lateral_drift_rate * dt
                result["event"] = "HESITANT_ABORT"
                self._abort_count += 1
            else:
                result["lateral_delta"] = self._drift_dir * self.lateral_drift_rate * dt
            self._drift_timer = random.uniform(3.0, 8.0)
            self._drift_dir  *= -1
        else:
            result["lateral_delta"] = self._drift_dir * self.lateral_drift_rate * 0.3 * dt


# ── Late Braker ───────────────────────────────────────────────────────────────

class LateBrakerProfile(BaseProfile):
    name = LATE_BRAKER
    @property
    def reaction_delay_s(self): return 1.5

    def __init__(self):
        super().__init__()
        self._brake_phase    = "NORMAL"
        self._brake_gap_trig = 0.0

    @property
    def target_headway_s(self): return 1.5
    @property
    def brake_aggression(self): return 3.0   # very hard when finally brakes
    @property
    def accel_aggression(self): return 1.2
    @property
    def lane_change_freq(self): return 1.0

    def _execute_phase(self, dt, speed_ms, gap_m, road_pos_m,
                       speed_limit_ms, result):
        # Follow fairly close, brake only when very tight
        target_gap = max(4.0, speed_ms * 1.5)
        result["target_gap_m"] = target_gap

        if gap_m < 5.0:
            # NOW brake — very late and hard
            result["delta_speed_ms"] = -4.5 * dt * self.brake_aggression
            result["event"] = "LATE_BRAKE_TRIGGER"
        elif gap_m < target_gap:
            # Mild slow — keep following close
            result["delta_speed_ms"] = -0.8 * dt
        else:
            target = speed_limit_ms * 0.95
            result["delta_speed_ms"] = (target - speed_ms) * 0.3 * dt


# ── Aggressive Overtaker ──────────────────────────────────────────────────────

class AggrOTProfile(BaseProfile):
    name = AGGR_OT
    @property
    def reaction_delay_s(self): return 0.6

    def __init__(self):
        super().__init__()
        self._follow_timer   = 0.0
        self._ot_cooldown    = 0.0

    @property
    def target_headway_s(self): return 1.0
    @property
    def brake_aggression(self): return 2.5
    @property
    def accel_aggression(self): return 3.0
    @property
    def lane_change_freq(self): return 3.0   # very frequent
    @property
    def lateral_drift_rate(self): return 0.01

    def _execute_phase(self, dt, speed_ms, gap_m, road_pos_m,
                       speed_limit_ms, result):
        target = speed_limit_ms * 1.15
        result["delta_speed_ms"] = (target - speed_ms) * 0.6 * dt
        result["target_gap_m"]   = max(3.0, speed_ms * 1.0)

        self._follow_timer += dt
        # After following for 8s, aggressively want overtake
        if self._follow_timer > 8.0 and time.time() > self._ot_cooldown:
            result["want_overtake"] = True
            result["event"]         = "AGGR_OT_INTENT"
            self._follow_timer      = 0.0
            self._ot_cooldown       = time.time() + 6.0


# ── Lane Wanderer ─────────────────────────────────────────────────────────────

class WandererProfile(BaseProfile):
    name = WANDERER
    @property
    def reaction_delay_s(self): return 1.2

    def __init__(self):
        super().__init__()
        self._drift_phase = 0.0   # sinusoidal cycle position
        self._lc_cooldown = 0.0
        self._lc_prob_timer = random.uniform(8.0, 20.0)

    @property
    def target_headway_s(self): return 2.0
    @property
    def brake_aggression(self): return 1.1
    @property
    def accel_aggression(self): return 1.0
    @property
    def lane_change_freq(self): return 2.5
    @property
    def lateral_drift_rate(self): return 0.08

    def _execute_phase(self, dt, speed_ms, gap_m, road_pos_m,
                       speed_limit_ms, result):
        target = speed_limit_ms * 0.90
        result["delta_speed_ms"] = (target - speed_ms) * 0.25 * dt

        # Sinusoidal lateral drift
        self._drift_phase += dt * 0.3
        drift = math.sin(self._drift_phase) * self.lateral_drift_rate * dt
        result["lateral_delta"] = drift

        # Random unannounced lane changes
        self._lc_prob_timer -= dt
        if self._lc_prob_timer <= 0:
            if random.random() < 0.35 and time.time() > self._lc_cooldown:
                result["want_overtake"] = True  # reuse as "want lane change"
                result["event"]         = "WANDERER_LC"
                self._lc_cooldown       = time.time() + 8.0
            self._lc_prob_timer = random.uniform(8.0, 20.0)


# ── Factory ───────────────────────────────────────────────────────────────────

_PROFILE_CLASSES = {
    COMMUTER:    CommuterProfile,
    TAILGATER:   TailgaterProfile,
    HESITANT:    HesitantProfile,
    LATE_BRAKER: LateBrakerProfile,
    AGGR_OT:     AggrOTProfile,
    WANDERER:    WandererProfile,
}


def make_profile(name: str = None) -> BaseProfile:
    """Create a profile instance by name. If name is None, pick randomly."""
    if name is None:
        name = random_profile()
    cls = _PROFILE_CLASSES.get(name, CommuterProfile)
    return cls()


def profile_color(name: str) -> str:
    return PROFILE_COLORS.get(name, "white")


def profile_label(name: str) -> str:
    return PROFILE_LABELS.get(name, name)
