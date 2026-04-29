# =============================================================================
# handshake_v5/road_geometry.py  —  Single source of truth for all road spatial
#                                    constants.  Every file imports from here.
#                                    Nothing hardcodes a zone boundary anywhere
#                                    else.
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class _RoadGeometry:
    # ── Road dimensions ───────────────────────────────────────────────────────
    ROAD_LENGTH_M:       float = 5000.0
    LANE_COUNT:          int   = 3          # 0=slow  1=fast  2=shoulder
    LANE_WIDTH_M:        float = 3.7        # standard lane
    SHOULDER_WIDTH_M:    float = 2.5        # emergency shoulder (lane 2)
    ROAD_TOTAL_WIDTH_M:  float = 9.9        # 2×3.7 + 2.5

    # Lane centre positions (metres from left road edge)
    # Lane 0 (slow/right):  centre at 1.85m
    # Lane 1 (fast/left):   centre at 5.55m
    # Lane 2 (shoulder):    centre at 8.80m
    LANE_CENTERS: Tuple[float, ...] = (1.85, 5.55, 8.80)

    # ── Lateral movement ──────────────────────────────────────────────────────
    LATERAL_SNAP_S:    float = 2.5    # seconds to complete a lane change
    LATERAL_DRIFT_MAX: float = 0.35   # max wander inside lane (metres from centre)

    # ── Zone definitions ──────────────────────────────────────────────────────
    # Format: zone_name → (start_m, end_m, speed_ms, cvc_code, display_icon)
    ZONES: Dict[str, tuple] = field(default_factory=lambda: {
        "ENTRY":       (   0.0,  500.0, 13.4, "CVC 22352",  ""),
        "SCHOOL_1":    ( 800.0, 1100.0,  6.7, "CVC 22352a", "🏫"),
        "OPEN_1":      (1100.0, 2200.0, 13.4, "CVC 22352",  ""),
        "ROADWORKS":   (2400.0, 2900.0,  4.5, "CVC 22352",  "🚧"),
        "HIGHWAY":     (2900.0, 5000.0, 13.4, "CVC 22352",  ""),
    })

    # ── Pre-warn distances ────────────────────────────────────────────────────
    ZONE_WARN_RANGE_M:     float = 200.0
    BREAKDOWN_WARN_RANGE_M: float = 300.0

    # ── Spawn helper positions ────────────────────────────────────────────────
    PLATOON_START_PCT:  float = 0.15   # 15% of segment
    CONVOY_START_PCT:   float = 0.12   # 12% of segment
    EMERG_START_M:      float = 50.0   # near rear of road

    def in_zone(self, zone_name: str, pos_m: float) -> bool:
        z = self.ZONES.get(zone_name)
        if z is None:
            return False
        return z[0] <= pos_m <= z[1]

    def zone_speed(self, zone_name: str) -> float:
        z = self.ZONES.get(zone_name)
        return z[2] if z else 13.4

    def zone_for_pos(self, pos_m: float):
        """Return the active zone name+data for a position, or None."""
        for name, data in self.ZONES.items():
            if data[0] <= pos_m <= data[1] and data[2] < 13.4:
                return name, data
        return None, None

    def speed_limit_at(self, pos_m: float) -> float:
        """Return speed limit in m/s at a given road position."""
        for name, data in self.ZONES.items():
            if data[0] <= pos_m <= data[1]:
                return data[2]
        return 13.4  # default road speed

    def segment_for_machine(self, machine_id: int, total_vms: int) -> Tuple[float, float]:
        """Return (start_m, end_m) that machine_id owns on a total_vms run."""
        seg_len = self.ROAD_LENGTH_M / total_vms
        start   = (machine_id - 1) * seg_len
        end     = machine_id * seg_len
        return start, end

    def marked_zones(self):
        """Return list of (start_m, end_m, icon, label) for zones with icons."""
        out = []
        for name, data in self.ZONES.items():
            if data[4]:  # has icon
                label = name.replace("_", " ")
                out.append((data[0], data[1], data[4], label, data[3]))
        return out


# Singleton instance — import this everywhere
RG = _RoadGeometry()


# ── Quick constants for backward compat imports ───────────────────────────────
ROAD_LENGTH_M       = RG.ROAD_LENGTH_M
LANE_COUNT          = RG.LANE_COUNT
LANE_WIDTH_M        = RG.LANE_WIDTH_M
SHOULDER_WIDTH_M    = RG.SHOULDER_WIDTH_M
LATERAL_SNAP_S      = RG.LATERAL_SNAP_S
LATERAL_DRIFT_MAX   = RG.LATERAL_DRIFT_MAX

SCHOOL_ZONE_START_M = RG.ZONES["SCHOOL_1"][0]
SCHOOL_ZONE_END_M   = RG.ZONES["SCHOOL_1"][1]
SCHOOL_SPEED_MS     = RG.ZONES["SCHOOL_1"][2]

ROADWORKS_ZONE_START_M = RG.ZONES["ROADWORKS"][0]
ROADWORKS_ZONE_END_M   = RG.ZONES["ROADWORKS"][1]
ROADWORKS_SPEED_MS     = RG.ZONES["ROADWORKS"][2]
ROADWORKS_LANES        = 1

ZONE_WARN_RANGE_M      = RG.ZONE_WARN_RANGE_M
BREAKDOWN_WARN_RANGE_M = RG.BREAKDOWN_WARN_RANGE_M
