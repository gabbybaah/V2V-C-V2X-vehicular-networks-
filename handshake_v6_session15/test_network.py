#!/usr/bin/env python3
# =============================================================================
# handshake_v4/test_network.py  —  Multicast connectivity test
#
# Run this on BOTH VMs at the same time before starting the simulation.
# It tells you exactly whether the two machines can exchange messages.
#
# Usage:
#   python3 test_network.py
#
# What to look for:
#   "Detected interface IP: 192.168.x.x"  — should NOT be 127.0.0.1
#   "RECEIVED from other machine"          — multicast is working
#   If you only see your own sends and no receives — multicast is blocked
# =============================================================================
import socket, threading, json, time, sys

MCAST_GRP  = "239.0.0.4"
MCAST_PORT = 5400

def detect_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((MCAST_GRP, MCAST_PORT))
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass
    return "0.0.0.0"

local_ip = detect_ip()
print(f"\n{'='*55}")
print(f"  Handshake V4 — Network Test")
print(f"{'='*55}")
print(f"  Multicast group : {MCAST_GRP}:{MCAST_PORT}")
print(f"  Detected IP     : {local_ip}")
if local_ip == "0.0.0.0" or local_ip.startswith("127."):
    print(f"\n  ⚠️  WARNING: IP looks like loopback or undetected.")
    print(f"     Check your network adapter in VirtualBox.")
    print(f"     Should be Bridged or Host-only, not NAT.\n")
else:
    print(f"  ✅  IP looks correct — good interface detected")
print(f"{'='*55}\n")

# ── Receive socket ────────────────────────────────────────────────────────────
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
except AttributeError:
    pass
rx.bind(("", MCAST_PORT))
mreq = socket.inet_aton(MCAST_GRP) + socket.inet_aton(local_ip)
rx.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
rx.settimeout(1.0)

# ── Send socket ───────────────────────────────────────────────────────────────
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
if local_ip != "0.0.0.0":
    tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                  socket.inet_aton(local_ip))

received_from_other = []

def recv_loop():
    while True:
        try:
            data, addr = rx.recvfrom(4096)
            msg = json.loads(data.decode())
            if msg.get("ip") == local_ip:
                continue  # our own message
            received_from_other.append(msg)
            print(f"  ✅  RECEIVED from {msg.get('ip','?')} — "
                  f"\"{msg.get('text','')}\"")
        except socket.timeout:
            continue
        except Exception:
            break

t = threading.Thread(target=recv_loop, daemon=True)
t.start()

print("  Sending test messages every 2 seconds for 20 seconds...")
print("  Start this script on the OTHER VM now if you haven't already.\n")

for i in range(10):
    msg = {"ip": local_ip, "text": f"hello from {local_ip} (msg {i+1}/10)",
           "t": time.time()}
    tx.sendto(json.dumps(msg).encode(), (MCAST_GRP, MCAST_PORT))
    print(f"  → Sent message {i+1}/10 from {local_ip}")
    time.sleep(2.0)

tx.close()
rx.close()

print(f"\n{'='*55}")
if received_from_other:
    print(f"  ✅  MULTICAST WORKING — received {len(received_from_other)} "
          f"message(s) from other VM")
    print(f"  You can now run the simulation.")
else:
    print(f"  ❌  NO MESSAGES RECEIVED FROM OTHER VM")
    print(f"\n  Possible causes:")
    print(f"  1. Other VM is not running this test at the same time")
    print(f"  2. VirtualBox adapter is set to NAT (change to Bridged or Host-only)")
    print(f"  3. A firewall is blocking UDP port 5400")
    print(f"\n  To check/fix firewall on Ubuntu:")
    print(f"     sudo ufw status")
    print(f"     sudo ufw allow 5400/udp")
    print(f"     sudo ufw allow in proto udp to 239.0.0.4")
print(f"{'='*55}\n")
