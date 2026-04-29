# =============================================================================
# handshake_v4/radio_network.py  —  UDP Multicast Radio (multi-machine mode)
#
# REVERTED to 0.0.0.0 binding which works on VirtualBox Internal Network.
# Interface detection is used for diagnostics only, not for socket binding.
# Multicast group: 239.0.0.4  port: 5400
# =============================================================================
import socket, threading, json, time, random, logging

log = logging.getLogger("v4.radio_net")

MCAST_GRP  = "239.0.0.4"
MCAST_PORT = 5400


def _detect_interface_ip() -> str:
    """Diagnostic only — detect which IP routes toward multicast group."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((MCAST_GRP, MCAST_PORT))
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return "0.0.0.0"


class Radio:
    """
    UDP multicast radio — same API as in-process radio.py.
    send(msg)  — broadcast to multicast group
    drain()    — return all messages from OTHER machines since last call
    start()    — open sockets, join multicast group, begin receiving
    stop()     — close sockets cleanly
    """

    def __init__(self, car_id: str, loss: float = 0.0):
        self.car_id   = car_id
        self.loss     = loss
        self._inbox   = []
        self._lock    = threading.Lock()
        self._running = False
        self.stats    = {"sent": 0, "recv": 0, "dropped": 0}
        self._tx      = None
        self._rx      = None

    def start(self):
        # Detect the actual interface IP — needed for VirtualBox Internal Network
        # which has no default multicast route (0.0.0.0 causes Errno 19)
        local_ip = _detect_interface_ip()

        # ── Send socket ───────────────────────────────────────────────────────
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                  socket.IPPROTO_UDP)
        self._tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
        # Bind outbound to the correct interface
        if local_ip != "0.0.0.0":
            self._tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                                socket.inet_aton(local_ip))

        # ── Receive socket ────────────────────────────────────────────────────
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                  socket.IPPROTO_UDP)
        self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass
        self._rx.bind(("", MCAST_PORT))
        # Join on the specific interface IP — required for Internal Network
        # where 0.0.0.0 gives OSError Errno 19 (no such device / no route)
        mreq = socket.inet_aton(MCAST_GRP) + socket.inet_aton(local_ip)
        self._rx.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self._rx.settimeout(0.3)

        self._running = True
        threading.Thread(target=self._recv_loop, daemon=True,
                         name=f"radio-{self.car_id[:8]}").start()
        log.info(f"[Radio] {self.car_id} joined {MCAST_GRP}:{MCAST_PORT}")

    def stop(self):
        self._running = False
        for s in (self._tx, self._rx):
            try:
                s.close()
            except Exception:
                pass

    def send(self, msg: dict):
        if not self._running:
            return
        if self.loss > 0 and random.random() < self.loss:
            self.stats["dropped"] += 1
            return
        msg["from"] = msg.get("from", self.car_id)
        msg["_ts"]  = time.time()
        try:
            self._tx.sendto(json.dumps(msg).encode(), (MCAST_GRP, MCAST_PORT))
            self.stats["sent"] += 1
        except Exception as e:
            log.debug(f"[Radio] Send error: {e}")

    def drain(self) -> list:
        with self._lock:
            msgs = list(self._inbox)
            self._inbox.clear()
        if self.loss > 0:
            kept = [m for m in msgs if random.random() >= self.loss]
            self.stats["dropped"] += len(msgs) - len(kept)
            msgs = kept
        self.stats["recv"] += len(msgs)
        return msgs

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._rx.recvfrom(65535)
                msg = json.loads(data.decode())
                if msg.get("from") == self.car_id:
                    continue
                with self._lock:
                    self._inbox.append(msg)
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                pass
            except Exception as e:
                if self._running:
                    log.debug(f"[Radio] Recv error: {e}")
