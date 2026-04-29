#!/usr/bin/env python3
# =============================================================================
# handshake_v4/road.py  —  Road (In-Lane Driving) Simulation Entry Point
#
# SINGLE MACHINE:
#   python road.py                      # 1 player car + NPCs
#   python road.py --players 2          # 2 player cars on 1 machine
#   python road.py --players 4          # 4 player cars on 1 machine
#
# MULTI-MACHINE (1 player car per machine):
#   Machine 1:  python road.py --machine 1
#   Machine 2:  python road.py --machine 2
#   Machine 3:  python road.py --machine 3
#   Machine 4:  python road.py --machine 4
#
# INTERACTIVE COMMANDS (type while simulation runs):
#   left / l          → change to left lane      (CVC 22107 + 21658)
#   right / r         → change to right lane
#   overtake / o      → overtake car ahead        (CVC 21750)
#   brake / b         → hard brake + chain warn   (CVC 21703)
#   accelerate / a    → increase speed
#   yield / y         → manually yield for emergency
#   normal / n        → resume normal driving
#   help              → show commands
#   quit / q          → exit
#
# All commands are broadcast via V2X — other cars see your intent immediately.
# The simulation also runs automatically if you don't type anything.
#
# Network (multi-machine): UDP multicast 239.0.0.4:5400
# VirtualBox: Host-only Adapter (vboxnet0) on all VMs.
# =============================================================================
import sys, os, time, signal, argparse, logging, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("road_v4.log")]
)


def run(machine: int, n_players: int, loss: float,
        use_dashboard: bool, npc_smart: int, npc_legacy: int,
        npc_emerg: int, runtime: float):

    single = (machine == 0)

    _print_header(machine, n_players, loss, single, npc_smart, npc_legacy, npc_emerg)

    # ── Build simulation ───────────────────────────────────────────────────

    if single:
        # All player cars on one machine
        player_configs = [
            {"label": f"P{i+1}◆", "lane": i % 2,
             "start_pos": 5.0 + i * 12.0, "machine_id": i + 1}
            for i in range(n_players)
        ]
        from sim_road import RoadSim
        sim = RoadSim(
            player_configs = player_configs,
            npc_counts     = {"smart": npc_smart, "legacy": npc_legacy,
                              "emergency": npc_emerg},
            loss           = loss,
        )
        machine_id = 1  # local player controls machine 1 car

    else:
        # Multi-machine: this machine has exactly 1 player car
        # NPCs run on Machine 1 alongside its player car
        player_configs = [
            {"label": f"M{machine}◆", "lane": (machine - 1) % 2,
             "start_pos": (machine - 1) * 10.0 + 5.0, "machine_id": machine}
        ]
        npc_cfg = ({"smart": npc_smart, "legacy": npc_legacy, "emergency": npc_emerg}
                   if machine == 1 else {"smart": 0, "legacy": 0, "emergency": 0})

        # Use network radio for multi-machine
        from radio_network import Radio as NetRadio
        from sim_road import RoadSim
        import radio as radio_mod

        # Monkey-patch the radio used by sim_road to the network radio
        # (both have identical API)
        net_radio = NetRadio(f"ROAD-M{machine}", loss=loss)
        net_radio.start()

        sim = RoadSim(
            player_configs = player_configs,
            npc_counts     = npc_cfg,
            loss           = loss,
        )
        # Replace the in-process radio with the network radio
        sim.radio = net_radio
        machine_id = machine

    print(f"  Cars: {[c.label for c in sim.all_cars]}")
    print()

    sim.start()

    def on_stop(sig=None, frame=None):
        sim.stop()
        if not single and machine != 0:
            try:
                net_radio.stop()
            except Exception:
                pass
        _print_summary(sim)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_stop)

    # ── Run display ────────────────────────────────────────────────────────

    if use_dashboard:
        try:
            from dashboard_road import RoadDashboard
            print("  Dashboard starting... (Ctrl+C or type 'quit' to stop)")
            time.sleep(0.3)
            RoadDashboard(sim, machine_id=machine_id).run()
        except ImportError:
            print("  [rich not available — text mode]")
            _text_loop(sim, machine_id, runtime)
        except KeyboardInterrupt:
            pass
    else:
        _text_loop(sim, machine_id, runtime)

    sim.stop()
    if not single and machine != 0:
        try:
            net_radio.stop()
        except Exception:
            pass
    _print_summary(sim)


# ── Text mode ─────────────────────────────────────────────────────────────────

def _text_loop(sim, machine_id: int, runtime: float):
    """Text-only loop: prints status + accepts commands via stdin."""
    print("  Running. Type commands (left/right/overtake/brake/accelerate/yield/normal/quit)")
    print("  Press Enter after each command. Ctrl+C to stop.\n")

    stop_event = threading.Event()

    def _input_loop():
        while not stop_event.is_set():
            try:
                raw = input("  CMD> ")
                cmd = raw.strip().lower()
                if cmd in ("q", "quit", "exit"):
                    stop_event.set()
                    break
                if cmd:
                    sim.enqueue_command(cmd, machine_id=machine_id)
                    print(f"  → Command queued: {cmd}")
            except (EOFError, KeyboardInterrupt):
                stop_event.set()
                break

    input_t = threading.Thread(target=_input_loop, daemon=True)
    input_t.start()

    deadline  = time.time() + runtime
    prev_snap = None
    while time.time() < deadline and not stop_event.is_set():
        time.sleep(1.0)
        s    = sim.get_status()
        cars = s.get("all_cars_sorted", [])
        snap = tuple((c["label"], round(c["road_pos_m"],0)) for c in cars)
        if snap != prev_snap:
            print(f"\n  T+{s['elapsed_s']:4.1f}s  active:{s['active_cars']}  done:{s['done_cars']}")
            for c in cars:
                icon = {"SMART":"◆","LEGACY":"◇","EMERGENCY":"🚨"}.get(c["car_type"],"?")
                pl   = "★ " if c.get("is_player") else "  "
                print(f"  {pl}{icon}{c['label']:<10} L{c['lane']}  "
                      f"{c['road_pos_m']:7.1f}m  {c['speed_kmh']:4.0f}km/h  {c['state']}")
            # Print recent token events
            toks = sim.get_token_events()
            for t in toks[-4:]:
                print(f"    ⚡ [{t['event']:14}] {t['detail'][:64]}")
            prev_snap = snap
        if sim.is_all_done():
            print("\n  All cars done!")
            stop_event.set()
            break

    stop_event.set()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header(machine, n_players, loss, single,
                  npc_smart, npc_legacy, npc_emerg):
    mode = (f"Single Machine — {n_players} player car(s)" if single
            else f"Machine {machine} — 1 player car")
    print(f"\n{'='*60}")
    print(f"  HANDSHAKE V4  —  Road Simulation")
    print(f"  Mode:     {mode}")
    print(f"  NPCs:     {npc_smart} smart  {npc_legacy} legacy  {npc_emerg} emergency")
    print(f"  Loss:     {loss*100:.0f}%")
    if not single:
        print(f"  Network:  UDP multicast 239.0.0.4:5400")
    print(f"{'='*60}\n")


def _print_summary(sim):
    s = sim.get_status()
    e = s["elapsed_s"]
    d = s["done_cars"]
    t = s["total_cars"]
    print(f"\n{'='*60}")
    print(f"  Road Sim done: {d}/{t} cars finished in {e:.0f}s")

    toks = sim.get_token_events()
    event_types = {}
    for ev in toks:
        k = ev.get("event","")
        event_types[k] = event_types.get(k, 0) + 1
    if event_types:
        print(f"  V2X events:")
        for k, v in sorted(event_types.items(), key=lambda x: -x[1]):
            print(f"    {k:<18} {v}")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Handshake V4 — Road Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single machine:
  python road.py                       # 1 player, default NPCs
  python road.py --players 2           # 2 player cars
  python road.py --players 4           # 4 player cars

Multi-machine (1 player each):
  Machine 1:  python road.py --machine 1
  Machine 2:  python road.py --machine 2
  Machine 3:  python road.py --machine 3
  Machine 4:  python road.py --machine 4
        """
    )
    p.add_argument("--machine",      type=int, default=0, choices=[0,1,2,3,4],
                   help="0=single machine, 1-4=this machine number")
    p.add_argument("--players",      type=int, default=1,
                   help="Player cars (single machine only, 1-4)")
    p.add_argument("--loss",         type=float, default=0.0)
    p.add_argument("--npc-smart",    type=int, default=2)
    p.add_argument("--npc-legacy",   type=int, default=3)
    p.add_argument("--npc-emerg",    type=int, default=1)
    p.add_argument("--runtime",      type=float, default=600.0)
    p.add_argument("--no-dashboard", action="store_true")
    args = p.parse_args()

    players = max(1, min(4, args.players)) if args.machine == 0 else 1
    run(args.machine, players, args.loss, not args.no_dashboard,
        args.npc_smart, args.npc_legacy, args.npc_emerg, args.runtime)


if __name__ == "__main__":
    main()
