# =============================================================================
# handshake_v5/radio.py  —  Priority V2X message bus
#
# 4 priority levels:
#   P0 CRITICAL : EMERG_PREEMPT, HARD_BRAKE, TOKEN_CANCEL  — never pruned
#   P1 HIGH     : INTENT_OT, INTENT_LC, HAZARD, ZONE_ALERT, PLATOON
#   P2 NORMAL   : TOKENS, SPAT, CMD
#   P3 LOW      : BEACON (dict keyed by sender — one per car, latest only)
#
# Same API as V4: start() stop() send(msg) drain()→list
# =============================================================================
import threading, time, random, logging
from collections import deque

log = logging.getLogger("v5.radio")

_PRIORITY_MAP = {
    "EMERG_PREEMPT":0,"HARD_BRAKE":0,"TOKEN_CANCEL":0,"SPLIT_BRAIN":0,
    "INTENT_OVERTAKE":1,"INTENT_LANE_CHG":1,"HAZARD":1,"ZONE_ALERT":1,
    "PEDESTRIAN":1,"PLATOON_INVITE":1,"PLATOON_ACK":1,"AMBULANCE":1,
    "PASSAGE_TOKEN":2,"TOKEN_ACK":2,"NEG_REQUEST":2,"SPAT":2,
    "CMD":2,"YIELD_ACK":2,"ROGUE_CROSS":2,
    "BEACON":3,
}
_TTL    = {0:30.0, 1:10.0, 2:5.0, 3:None}
_MAXLEN = {0:200,  1:500,  2:1000}

_LOCK = threading.Lock()
_P0: deque = deque(maxlen=200)
_P1: deque = deque(maxlen=500)
_P2: deque = deque(maxlen=1000)
_P3: dict  = {}   # sender_id → latest beacon (replaced on each send)


class Radio:
    def __init__(self, node_id:str, loss:float=0.0):
        self.node_id = node_id
        self.loss    = loss
        self.stats   = {"sent":0,"recv":0,"dropped":0}

    def start(self): pass
    def stop(self):  pass

    def send(self, msg:dict):
        if self.loss > 0 and random.random() < self.loss:
            self.stats["dropped"] += 1
            return
        sender      = msg.get("from", self.node_id)
        msg["from"] = sender
        if "_ts" not in msg:
            msg["_ts"] = time.time()
        p = _PRIORITY_MAP.get(msg.get("type",""), 2)
        with _LOCK:
            if   p == 3: _P3[sender] = dict(msg)
            elif p == 0: _P0.append(dict(msg))
            elif p == 1: _P1.append(dict(msg))
            else:        _P2.append(dict(msg))
        self.stats["sent"] += 1

    def drain(self) -> list:
        now = time.time()
        out = []
        with _LOCK:
            for m in list(_P0):
                if now - m.get("_ts",now) <= _TTL[0]: out.append(m)
            _P0.clear()
            for m in list(_P1):
                if now - m.get("_ts",now) <= _TTL[1]: out.append(m)
            _P1.clear()
            for m in list(_P2):
                if now - m.get("_ts",now) <= _TTL[2]: out.append(m)
            _P2.clear()
            out.extend(_P3.values())
            _P3.clear()
        if self.loss > 0:
            kept = []
            for m in out:
                if random.random() < self.loss: self.stats["dropped"] += 1
                else: kept.append(m)
            out = kept
        result = [m for m in out if m.get("from") != self.node_id]
        self.stats["recv"] += len(result)
        return result
