# =============================================================================
# handshake_v4/law.py  —  California Vehicle Code Rules Engine
# =============================================================================
from config import (Phase, CarType, SPEED_LIMIT_MS, SCHOOL_SPEED_MS,
                    FOLLOW_TIME_S, MIN_GAP_M, LANE_CLEAR_M, ONCOMING_CLEAR_M,
                    PLATOON_CLOSE_M)


class Law:

    @staticmethod
    def may_enter_intersection(phase: str, car_type: str):
        if car_type == CarType.EMERGENCY:
            return True, "CVC 21055", "Emergency vehicle exempt from signal controls"
        if phase in (Phase.NS_GREEN, Phase.EW_GREEN):
            return True, "CVC 21450", "Green signal — may proceed"
        if phase == Phase.PREEMPTED:
            return False, "CVC 21806", "Intersection preempted — yield to emergency"
        if phase == Phase.PEDESTRIAN:
            return False, "CVC 21950", "Pedestrian crossing — must yield"
        return False, "CVC 21453", "Red or yellow signal — must stop"

    @staticmethod
    def must_yield_to_emergency(car_type: str, dist_m: float):
        if car_type == CarType.EMERGENCY:
            return False, "CVC 21806", "This IS the emergency vehicle"
        if dist_m <= 300:
            return True, "CVC 21806", f"Emergency within {dist_m:.0f}m — yield and pull right"
        return False, "CVC 21806", "Emergency not within yield range"

    @staticmethod
    def safe_following(speed_ms: float, gap_m: float, platooning: bool = False):
        if platooning:
            # V2X platooning: tighter gap allowed because of cooperative braking
            required = PLATOON_CLOSE_M
            if gap_m < required:
                return False, "CVC 21703", f"Platoon gap {gap_m:.1f}m < {required:.1f}m"
            return True, "CVC 21703", "Platoon V2X following — gap safe"
        required = max(MIN_GAP_M, speed_ms * FOLLOW_TIME_S)
        if gap_m < required:
            return False, "CVC 21703", f"Gap {gap_m:.1f}m < required {required:.1f}m — VIOLATION"
        return True, "CVC 21703", "Following distance safe"

    @staticmethod
    def may_overtake(my_speed: float, target_speed: float, oncoming_clear_m: float):
        if my_speed <= target_speed:
            return False, "CVC 21750", "Must be faster than vehicle being overtaken"
        if oncoming_clear_m < ONCOMING_CLEAR_M:
            return (False, "CVC 21750",
                    f"Oncoming lane not clear — {oncoming_clear_m:.0f}m < {ONCOMING_CLEAR_M:.0f}m required")
        return True, "CVC 21750", "Overtake permitted — oncoming clear"

    @staticmethod
    def may_change_lane(gap_ahead_m: float, gap_behind_m: float, signalled: bool):
        if not signalled:
            return False, "CVC 22107", "Must signal before lane change"
        if gap_ahead_m < LANE_CLEAR_M:
            return False, "CVC 21658", f"Gap ahead {gap_ahead_m:.1f}m < {LANE_CLEAR_M}m"
        if gap_behind_m < LANE_CLEAR_M:
            return False, "CVC 21658", f"Gap behind {gap_behind_m:.1f}m < {LANE_CLEAR_M}m"
        return True, "CVC 21658", "Lane change safe"

    @staticmethod
    def speed_ok(speed_ms: float, in_school_zone: bool = False,
                 is_emergency: bool = False):
        if is_emergency:
            return True, "CVC 21055", "Emergency vehicle — speed exempt"
        limit = SCHOOL_SPEED_MS if in_school_zone else SPEED_LIMIT_MS
        cvc   = "CVC 22352a" if in_school_zone else "CVC 22352"
        limit_label = "15 mph school zone" if in_school_zone else "30 mph urban"
        if speed_ms <= limit:
            return True, cvc, f"{speed_ms*3.6:.0f} km/h within {limit_label}"
        return False, cvc, f"SPEED VIOLATION {speed_ms*3.6:.0f} km/h exceeds {limit_label}"

    @staticmethod
    def right_of_way_cvc21800(my_arm: str, other_arm: str) -> bool:
        """
        CVC 21800 — at an uncontrolled intersection, vehicle on the RIGHT
        has right of way. Arms in clockwise order: North, East, South, West.
        The arm to the RIGHT of each arm gets priority.
        Right of: North→East, East→South, South→West, West→North
        """
        right_of = {
            "North": "West",   # West is to the right of North-facing driver
            "East":  "North",
            "South": "East",
            "West":  "South",
        }
        return right_of.get(my_arm) == other_arm

    @staticmethod
    def must_yield_pedestrian():
        return True, "CVC 21950", "Pedestrian in crosswalk — all vehicles must yield"
