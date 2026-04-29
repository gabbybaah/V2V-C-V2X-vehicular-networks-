# =============================================================================
# handshake_v6/config.py
# =============================================================================
from road_geometry import (RG, ROAD_LENGTH_M, LANE_COUNT,
                            SCHOOL_ZONE_START_M, SCHOOL_ZONE_END_M,
                            SCHOOL_SPEED_MS, ROADWORKS_ZONE_START_M,
                            ROADWORKS_ZONE_END_M, ROADWORKS_SPEED_MS,
                            ROADWORKS_LANES, ZONE_WARN_RANGE_M,
                            BREAKDOWN_WARN_RANGE_M)

ARMS           = ["North","East","South","West"]
CARS_PER_ARM   = 20
CAR_LENGTH_M   = 4.5
CAR_GAP_STOPPED_M = 2.0
INTER_BOX_M    = 20.0

GREEN_S        = 25.0
YELLOW_S       = 4.0
ALL_RED_S      = 2.0
PLATOON_MAX    = 10
PLATOON_GAP_S  = 1.2

SPEED_LIMIT_MS    = 13.4
QUEUE_CREEP_MS    = 3.0
CROSS_SPEED_MS    = 8.3
DEPART_SPEED_MS   = 13.4
EMERGENCY_MS      = 22.2
FOLLOW_TIME_S     = 4.0     # 4s following time — more space between cars
MIN_GAP_M         = 15.0    # minimum 15m gap — was 5m, prevents most collisions
PLATOON_CLOSE_M   = 3.5
ONCOMING_CLEAR_M  = 150.0

PREEMPT_HOLD_S    = 15.0
PREEMPT_RANGE_M   = 600.0

ROAD_LANES        = LANE_COUNT
BROADCAST_RANGE_M = 1000.0
SIGNAL_DIST_M     = 30.0
LANE_CLEAR_M      = 20.0
HARD_BRAKE_MS2    = 4.5
OVERTAKE_BOOST_MS = 3.0

NEG_WINDOW_S      = 0.8
TOKEN_TIMEOUT_S   = 5.0

PEDESTRIAN_HOLD_S = 6.0
PEDESTRIAN_PROB   = 0.15

TRUST_FULL   = 192
TRUST_HIGH   = 128
TRUST_LOW    = 64
TRUST_BL     = 0

SIM_TICK_S   = 0.1

from behavior_profiles import (ALL_PROFILES, PROFILE_WEIGHTS, PROFILE_COLORS,
                                PROFILE_LABELS, COMMUTER, TAILGATER, HESITANT,
                                LATE_BRAKER, AGGR_OT, WANDERER,
                                random_profile, make_profile,
                                profile_color, profile_label)

DRIVER_FATIGUE_PROB = 0.08

ML_HISTORY_TICKS        = 40
ML_PREDICT_TICKS        = 20
ML_CONFIDENCE_THRESHOLD = 0.5
ML_MIN_HISTORY_TICKS    = 20
ML_GRU_HIDDEN           = 64
ML_GRU_HIDDEN2          = 32
ML_FEATURES             = 5
ML_ACTIONS              = 6
ML_MODEL_PATH           = "model/predictor.pt"
ML_TRAINING_DIR         = "logs/training"

NPC_BROADCAST_INTERVAL  = 1
REMOTE_NPC_TIMEOUT_S    = 3.0

CONVOY_SIZE      = 3
CONVOY_SPACING_M = 12.0

SPLIT_BRAIN_DURATION_S  = 10.0
SPAT_STALE_THRESHOLD_S  = 2.0

class CarType:
    SMART     = "SMART"
    LEGACY    = "LEGACY"
    EMERGENCY = "EMERGENCY"
    BREAKDOWN = "BREAKDOWN"
    ROGUE     = "ROGUE"
    PLATOON   = "PLATOON"

class IState:
    QUEUED   = "QUEUED";  NEG      = "NEG";      TOKEN_OK  = "TOKEN_OK"
    MOVING   = "MOVING";  CROSSING = "CROSSING"; YIELDING  = "YIELDING"
    DONE     = "DONE";    ROGUE_GO = "ROGUE_GO"

class RState:
    DRIVING     = "DRIVING";   OVERTAKING  = "OVERTAKING"
    LANE_CHANGE = "LANE_CHANGE"; BRAKING   = "BRAKING"
    YIELDING    = "YIELDING";  BROKEN_DOWN = "BROKEN_DOWN"
    PLATOONING  = "PLATOONING"; SCHOOL_ZONE = "SCHOOL_ZONE"
    SHOULDER    = "SHOULDER";  DONE        = "DONE"

class Phase:
    NS_GREEN  = "NS_GREEN";  NS_YELLOW = "NS_YELLOW"; ALL_RED   = "ALL_RED"
    EW_GREEN  = "EW_GREEN";  EW_YELLOW = "EW_YELLOW"; ALL_RED_2 = "ALL_RED_2"
    PREEMPTED = "PREEMPTED"; PEDESTRIAN= "PEDESTRIAN"

GREEN_FOR = {
    Phase.NS_GREEN:  ["North","South"], Phase.NS_YELLOW: ["North","South"],
    Phase.EW_GREEN:  ["East","West"],   Phase.EW_YELLOW: ["East","West"],
    Phase.ALL_RED:[], Phase.ALL_RED_2:[], Phase.PREEMPTED:[], Phase.PEDESTRIAN:[],
}

class Msg:
    BEACON        ="BEACON";       NEG_REQUEST   ="NEG_REQUEST"
    PASSAGE_TOKEN ="PASSAGE_TOKEN";TOKEN_ACK     ="TOKEN_ACK"
    TOKEN_CANCEL  ="TOKEN_CANCEL"; EMERG_PREEMPT ="EMERG_PREEMPT"
    SPAT          ="SPAT";         INTENT_OT     ="INTENT_OVERTAKE"
    INTENT_LC     ="INTENT_LANE_CHG"; HARD_BRAKE ="HARD_BRAKE"
    CMD           ="CMD";          YIELD_ACK     ="YIELD_ACK"
    HAZARD        ="HAZARD";       ZONE_ALERT    ="ZONE_ALERT"
    PLATOON_INVITE="PLATOON_INVITE";PLATOON_ACK  ="PLATOON_ACK"
    PEDESTRIAN    ="PEDESTRIAN";   ROGUE_CROSS   ="ROGUE_CROSS"
    WEATHER_SYNC  ="WEATHER_SYNC"

CVC = {
    "21450":"Green signal — may proceed",
    "21453":"Red/yellow signal — shall stop",
    "21806":"Emergency — all yield and pull right",
    "21055":"Emergency vehicles exempt from signals",
    "21703":"Safe following distance",
    "21750":"Overtake — pass left only when safe",
    "21658":"Lane change — only when safe",
    "22107":"Signal intent 100ft before change",
    "22352a":"School zone — 15 mph",
    "21800":"Right of way at uncontrolled intersection",
    "21950":"Yield to pedestrian in crosswalk",
    "22500":"Stopping in hazardous position prohibited",
}

FLEET = {"smart":0.65,"legacy":0.30,"emergency":0.05}
RUSH_HOUR_CARS = 40

MCAST_GRP       = "239.0.0.4"
MCAST_PORT      = 5400
SPAT_INTERVAL_S = 0.4
MACHINE_ARMS = {1:"North",2:"East",3:"South",4:"West"}

# ── V6: Weather ────────────────────────────────────────────────────────────
class WeatherState:
    CLEAR=0; LIGHT_RAIN=1; HEAVY_RAIN=2; FOG=3; LIGHT_SNOW=4; ICE=5

WEATHER_PARAMS = {
    0: {"speed":1.00,"follow":1.00,"brake":1.00,"visual":120,
        "icon":"☀️ ","name":"Clear",      "desc":"Normal conditions",       "bg":"grey11"},
    1: {"speed":0.90,"follow":1.30,"brake":0.85,"visual":100,
        "icon":"🌧️ ","name":"Light Rain", "desc":"Stopping dist +30%",      "bg":"grey15"},
    2: {"speed":0.75,"follow":1.60,"brake":0.70,"visual": 60,
        "icon":"⛈️ ","name":"Heavy Rain", "desc":"Speed -25%  Braking -30%","bg":"grey19"},
    3: {"speed":0.65,"follow":1.50,"brake":0.90,"visual": 30,
        "icon":"🌫️ ","name":"Fog",        "desc":"Visibility 30m  -35% spd","bg":"grey23"},
    4: {"speed":0.70,"follow":1.70,"brake":0.65,"visual": 80,
        "icon":"❄️ ","name":"Light Snow", "desc":"Speed -30%  Braking -35%","bg":"grey15"},
    5: {"speed":0.50,"follow":2.20,"brake":0.40,"visual": 40,
        "icon":"🧊 ","name":"Ice",        "desc":"DANGER: brake power -60%","bg":"grey11"},
}
WEATHER_CHANGE_INTERVAL_S = 60.0
WEATHER_WEIGHTS           = [40,20,10,10,10,10]

# ── V6: Near-miss ──────────────────────────────────────────────────────────
NEAR_MISS_M = 1.5
COLLISION_M = 0.0

# ── V6: Fuel ───────────────────────────────────────────────────────────────
FUEL_RANGE_M = 8000.0
FUEL_LOW_PCT = 0.10

# ── V6: Heatmap ────────────────────────────────────────────────────────────
HEATMAP_BUCKETS = 20

# ── V6: Reaction time ─────────────────────────────────────────────────────
REACTION_BUFFER_TICKS = 15

# ── V6: Intersection maneuvers ────────────────────────────────────────────
MANEUVER_DIST_MULT = {"STRAIGHT":1.0,"TURN_RIGHT":1.3,"TURN_LEFT":1.8}
MANEUVER_ICONS     = {"STRAIGHT":"→","TURN_RIGHT":"↘","TURN_LEFT":"↖"}
MANEUVER_WEIGHTS   = [0.50,0.20,0.30]

# ── V6: Intersection startup delay ────────────────────────────────────────
INT_STARTUP_BASE    = {"SMART":0.8,"LEGACY":1.2,"EMERGENCY":0.0,"ROGUE":0.3}
INT_STARTUP_PER_POS = 0.3

# ── V6: Intersection near-miss ────────────────────────────────────────────
INT_NEAR_MISS_M     = 2.0
OUTBOUND_HOLD_TICKS = 5   # ticks car stays visible in outbound lane
