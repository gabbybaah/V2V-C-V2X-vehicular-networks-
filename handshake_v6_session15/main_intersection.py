#!/usr/bin/env python3
# =============================================================================
# handshake_v4/main_intersection.py  —  Intersection Simulation
#
# SINGLE MACHINE (all 4 arms):
#   python main_intersection.py
#   python main_intersection.py --cars 10 --loss 0.1
#
# TWO MACHINES (2 arms each — most common setup):
#   VM1 (North + South + light host): python main_intersection.py --machine 1 --vms 2
#   VM2 (East + West):                python main_intersection.py --machine 2 --vms 2
#
# FOUR MACHINES (one arm each):
#   Machine 1 (North + light host):  python main_intersection.py --machine 1
#   Machine 2 (South):               python main_intersection.py --machine 2
#   Machine 3 (East):                python main_intersection.py --machine 3
#   Machine 4 (West):                python main_intersection.py --machine 4
#
# Start Machine 1 first in both modes. Others connect via UDP multicast.
#
# Token exchange (NEG_REQUEST → PASSAGE_TOKEN → TOKEN_ACK) is visible
# in the live dashboard and logged to intersection_v4.log.
# =============================================================================
import sys, os, time, signal, argparse, logging, threading
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("intersection_v4.log")]
)

from config import ARMS, MACHINE_ARMS, Phase
from sim_logger import attach_logger


# ── Single-machine run ────────────────────────────────────────────────────────

def run_single(cars_per_arm: int, loss: float, use_dashboard: bool, runtime: float,
               rush_hour: bool = False, include_rogue: bool = False,
               pedestrian: bool = False, ambulance: bool = False,
               uncontrolled: bool = False, machine_fail: int = None,
               spat_delay: float = 0.0, no_log: bool = False, **_):
    from sim_intersection import IntersectionSim
    from dashboard_intersection import IntersectionDashboard
    import threading as _thr

    modes = []
    if rush_hour:     modes.append("RUSH HOUR")
    if include_rogue: modes.append("ROGUE")
    if uncontrolled:  modes.append("CVC 21800")
    mode_str = " [" + "+".join(modes) + "]" if modes else ""
    _print_header(f"ALL ARMS{mode_str}", cars_per_arm, loss)
    sim = IntersectionSim(my_arms=ARMS, cars_per_arm=cars_per_arm,
                          loss=loss, host_light=True,
                          rush_hour=rush_hour, include_rogue=include_rogue,
                          uncontrolled=uncontrolled, spat_delay=spat_delay)
    _print_fleet(sim)

    def on_stop(sig=None, frame=None):
        sim.stop()
        _print_summary(sim)
        sys.exit(0)
    signal.signal(signal.SIGINT, on_stop)

    sim.start()

    # Scenario triggers
    if pedestrian:
        def _ped(): time.sleep(10); sim.trigger_pedestrian()
        _thr.Thread(target=_ped, daemon=True).start()
        print("  🚶 Pedestrian crossing → T+10s  CVC 21950\n")
    if ambulance:
        def _amb(): time.sleep(15); sim.trigger_ambulance_corridor("North")
        _thr.Thread(target=_amb, daemon=True).start()
        print("  🚨 Ambulance corridor → T+15s  CVC 21806\n")
    if machine_fail:
        _arm_map = {1: "North", 2: "East", 3: "South", 4: "West"}
        _fail_arm = _arm_map.get(machine_fail, "North")
        def _mfail(): time.sleep(25); sim.trigger_machine_fail(_fail_arm)
        _thr.Thread(target=_mfail, daemon=True).start()
        print(f"  ⚠  Machine fail → {_fail_arm} arm at T+25s (feature #21)\n")
    if spat_delay > 0:
        print(f"  📡 SPaT delay injected: {spat_delay:.1f}s — stale phase events logged\n")

    # Persistent simulation logger (feature #26)
    _logger = None
    if not no_log:
        _meta = dict(
            sim_type="intersection", cars_per_arm=cars_per_arm, loss=loss,
            rush_hour=rush_hour, rogue=include_rogue, pedestrian=pedestrian,
            ambulance=ambulance, uncontrolled=uncontrolled,
            machine_fail=bool(machine_fail), spat_delay=spat_delay,
        )
        _logger = attach_logger("intersection", sim, _meta)

    if use_dashboard:
        try:
            IntersectionDashboard(sim).run()
        except KeyboardInterrupt:
            pass
    else:
        _text_loop(sim, runtime)

    sim.stop()
    if _logger:
        _logger.finish()
        # M1: extract intersection training data and trigger model training
        try:
            from intersection_extractor import run as extract_run
            extract_run(async_mode=True)
        except Exception as e:
            pass
    _print_summary(sim)


# ── Two-VM run (2 arms per machine) ──────────────────────────────────────────

def run_two_vm(machine: int, cars_per_arm: int, loss: float,
               use_dashboard: bool, runtime: float):
    """
    VM1 = North + South + traffic light host
    VM2 = East  + West
    Start VM1 first. VM2 listens for SPaT and mirrors the phase.
    """
    from sim_intersection import build_fleet
    from dashboard_intersection import IntersectionDashboard
    from traffic_light import TrafficLight
    from arm_queue import ArmQueue
    from config import SPAT_INTERVAL_S

    VM_ARMS   = {1: ["North","South"], 2: ["East","West"]}
    my_arms   = VM_ARMS[machine]
    host_light = (machine == 1)

    _print_header("+".join(my_arms), cars_per_arm, loss, machine=machine)

    try:
        import radio as radio_mod
        from radio_network import Radio, _detect_interface_ip
        radio_mod.Radio = Radio
        detected_ip = _detect_interface_ip()
        print(f"  Network radio : UDP multicast 239.0.0.4:5400")
        print(f"  This machine  : {detected_ip}")
        print()
    except Exception as e:
        from radio import Radio
        print(f"  Network radio failed: {e} — using in-process bus")
        print()

    # One radio + one ArmQueue per local arm
    radios = {}
    queues = {}
    for arm in my_arms:
        r = Radio(f"ARM-{arm}", loss=loss)
        r.start()
        radios[arm] = r
        fleet = build_fleet(arm, cars_per_arm,
                            seed=42 + ARMS.index(arm) * 100)
        queues[arm] = ArmQueue(arm, fleet, r, light=None)  # light attached below

    # Traffic light
    if host_light:
        light = TrafficLight("INT-01")
        print(f"  VM1 hosts traffic light — broadcasting SPaT to VM2\n")
    else:
        light = _RemoteLight()
        print(f"  VM2: waiting for SPaT from VM1...\n")

    # Attach light to all queues
    for q in queues.values():
        q.light = light

    # Phase callback
    def on_phase(phase, green_arms):
        if isinstance(green_arms, str):
            green_arms = [green_arms]
        for arm in my_arms:
            if arm in (green_arms or []):
                queues[arm].on_green()
            else:
                queues[arm].on_red()

    light.on_phase_change(on_phase)

    # SPaT broadcaster (VM1 only)
    if host_light:
        spat_radio = Radio("SPAT-TX", loss=0.0)
        spat_radio.start()
        def _spat_bcast():
            while True:
                s = light.get_spat()
                s["type"] = "SPAT"
                spat_radio.send(s)
                time.sleep(SPAT_INTERVAL_S)
        threading.Thread(target=_spat_bcast, daemon=True).start()
        light.start()
        on_phase(Phase.NS_GREEN, ["North","South"])

    # SPaT listener (VM2 only)
    else:
        spat_rx = Radio("SPAT-RX", loss=0.0)
        spat_rx.start()
        def _spat_listen():
            last = None
            while True:
                time.sleep(0.15)
                for msg in spat_rx.drain():
                    if msg.get("type") == "SPAT":
                        p, g = msg.get("phase"), msg.get("green_arms",[])
                        if p != last:
                            last = p
                            light.update(p, g)
        threading.Thread(target=_spat_listen, daemon=True).start()

    # Sim loop — tick all local arms
    start_time = time.time()
    running    = [True]

    def _loop():
        import config
        dt = config.SIM_TICK_S
        while running[0]:
            t0 = time.time()
            for q in queues.values():
                q.tick(dt)
            elapsed = time.time() - t0
            sleep_t = max(0, dt - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    threading.Thread(target=_loop, daemon=True, name="sim-arms").start()

    # Fake sim so dashboard works with 2 arms
    _cpa = cars_per_arm   # capture before class body (class scope can't see locals)
    class _FakeSim:
        my_arms      = list(queues.keys())
        cars_per_arm = _cpa

        def get_status(self_):
            spat    = light.get_spat()
            total   = cars_per_arm * len(my_arms)
            done    = sum(q.done_count()   for q in queues.values())
            queued  = sum(q.queued_count() for q in queues.values())
            on_road = sum(q.road_count()   for q in queues.values())
            return {
                "elapsed_s":       round(time.time() - start_time, 1),
                "tick":            0,
                "total_cars":      total,
                "done":            done,
                "queued":          queued,
                "on_road":         on_road,
                "pct_done":        round(100*done/total,1) if total else 0,
                "all_done":        done >= total,
                "light_phase":     spat["phase"],
                "light_green":     spat["green_arms"],
                "light_remaining": spat.get("time_remaining", 0),
                "preempted":       spat.get("preempted", False),
                "arms":            {arm: queues[arm].status() for arm in my_arms},
            }

        def get_token_events(self_):
            evts = []
            for q in queues.values():
                evts.extend(q.get_token_events())
            return sorted(evts, key=lambda e: e.get("ts",0))[-60:]

    fake_sim = _FakeSim()

    for arm in my_arms:
        icons = "".join(
            "🚨" if c.car_type=="EMERGENCY"
            else ("◆" if c.car_type=="SMART" else "◇")
            for c in queues[arm]._all
        )
        print(f"  {arm:<6}: {icons}")
    print()

    def on_stop(sig=None, frame=None):
        running[0] = False
        if host_light: light.stop()
        for r in radios.values(): r.stop()
        _print_summary(fake_sim)
        sys.exit(0)
    signal.signal(signal.SIGINT, on_stop)

    if use_dashboard:
        try:
            IntersectionDashboard(fake_sim).run()
        except KeyboardInterrupt:
            pass
    else:
        _text_loop(fake_sim, runtime)

    running[0] = False
    if host_light: light.stop()
    for r in radios.values(): r.stop()
    _print_summary(fake_sim)

# ── Multi-machine run (one arm per machine) ───────────────────────────────────

def run_multi(machine: int, cars_per_arm: int, loss: float,
              use_dashboard: bool, runtime: float):
    from sim_intersection import IntersectionSim, build_fleet
    from dashboard_intersection import IntersectionDashboard
    from traffic_light import TrafficLight
    from arm_queue import ArmQueue
    from config import SPAT_INTERVAL_S

    my_arm     = MACHINE_ARMS[machine]
    host_light = (machine == 1)

    _print_header(my_arm, cars_per_arm, loss, machine=machine)

    # Use network radio for multi-machine
    try:
        from radio_network import Radio, _detect_interface_ip
        detected_ip = _detect_interface_ip()
        print(f"  Network radio : UDP multicast 239.0.0.4:5400")
        print(f"  This machine  : {detected_ip}")
        if detected_ip == "0.0.0.0" or detected_ip.startswith("127."):
            print(f"  ⚠️  WARNING: IP looks like loopback — run test_network.py to diagnose")
        else:
            print(f"  ✅  Interface detected correctly")
        print()
    except Exception as e:
        from radio import Radio
        print(f"  ❌ Network radio failed: {e}")
        print(f"  ❌ Falling back to in-process bus — OTHER MACHINE WILL NOT BE VISIBLE")
        print(f"  Run: python3 test_network.py  to diagnose")
        print()

    # Build arm
    radio = Radio(f"ARM-{my_arm}", loss=loss)
    radio.start()
    fleet = build_fleet(my_arm, cars_per_arm,
                        seed=42 + ARMS.index(my_arm) * 100)

    # Traffic light
    if host_light:
        from traffic_light import TrafficLight
        light = TrafficLight("INT-01")
        print(f"  Machine 1 hosting traffic light — broadcasting SPaT\n")
    else:
        light = _RemoteLight()
        print(f"  Waiting for SPaT from Machine 1...\n")

    queue = ArmQueue(my_arm, fleet, radio, light=light)

    # Phase callback
    def on_phase(phase, green_arms):
        if isinstance(green_arms, str): green_arms = [green_arms]
        if my_arm in (green_arms or []):
            queue.on_green()
        else:
            queue.on_red()

    light.on_phase_change(on_phase)

    # SPaT broadcaster (machine 1 only)
    spat_radio = None
    if host_light:
        spat_radio = Radio("SPAT-TX", loss=0.0)
        spat_radio.start()
        def _spat_bcast():
            while True:
                s = light.get_spat(); s["type"] = "SPAT"
                spat_radio.send(s)
                time.sleep(SPAT_INTERVAL_S)
        threading.Thread(target=_spat_bcast, daemon=True).start()
        light.start()
        on_phase(Phase.NS_GREEN, ["North", "South"])

    # SPaT listener (machines 2/3/4)
    else:
        spat_rx = Radio("SPAT-RX", loss=0.0)
        spat_rx.start()
        def _spat_listen():
            last = None
            while True:
                time.sleep(0.15)
                for msg in spat_rx.drain():
                    if msg.get("type") == "SPAT":
                        p, g = msg.get("phase"), msg.get("green_arms", [])
                        if p != last:
                            last = p
                            light.update(p, g)
        threading.Thread(target=_spat_listen, daemon=True).start()

    # Sim loop
    start_time = time.time()
    running    = [True]

    def _loop():
        import config
        dt = config.SIM_TICK_S
        while running[0]:
            t0 = time.time()
            queue.tick(dt)
            elapsed = time.time() - t0
            sleep_t = max(0, dt - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    threading.Thread(target=_loop, daemon=True, name="sim-arm").start()

    # Fake sim interface for dashboard
    _cpa = cars_per_arm   # capture before class body
    class _FakeSim:
        my_arms = [my_arm]
        cars_per_arm = _cpa

        def get_status(self_):
            spat = light.get_spat()
            total = cars_per_arm
            done  = queue.done_count()
            return {
                "elapsed_s":       round(time.time() - start_time, 1),
                "tick":            0,
                "total_cars":      total,
                "done":            done,
                "queued":          queue.queued_count(),
                "on_road":         queue.road_count(),
                "pct_done":        round(100*done/total, 1) if total else 0,
                "all_done":        done >= total,
                "light_phase":     spat["phase"],
                "light_green":     spat["green_arms"],
                "light_remaining": spat.get("time_remaining", 0),
                "preempted":       spat.get("preempted", False),
                "arms":            {my_arm: queue.status()},
            }

        def get_token_events(self_):
            return queue.get_token_events()

        def get_all_events(self_):
            evts = queue.get_token_events()
            for car in queue._all:
                evts.extend(car.events[-4:])
            return sorted(evts, key=lambda e: e.get("ts", 0))[-30:]

    fake_sim = _FakeSim()

    def on_stop(sig=None, frame=None):
        running[0] = False
        if host_light: light.stop()
        radio.stop()
        _print_summary(fake_sim)
        sys.exit(0)
    signal.signal(signal.SIGINT, on_stop)

    if use_dashboard:
        try:
            IntersectionDashboard(fake_sim).run()
        except KeyboardInterrupt:
            pass
    else:
        _text_loop(fake_sim, runtime)

    running[0] = False
    if host_light: light.stop()
    radio.stop()
    _print_summary(fake_sim)


# ── Remote light proxy ────────────────────────────────────────────────────────

class _RemoteLight:
    def __init__(self):
        self._phase      = Phase.NS_GREEN
        self._green      = ["North", "South"]
        self._preempted  = False
        self._preempt_arm = None
        self._cbs        = []

    def on_phase_change(self, cb): self._cbs.append(cb)

    def update(self, phase, green_arms):
        self._phase     = phase
        self._green     = green_arms or []
        self._preempted = (phase == Phase.PREEMPTED)
        for cb in self._cbs:
            try: cb(phase, green_arms)
            except Exception: pass

    def preempt(self, arm):
        self._preempted  = True
        self._preempt_arm = arm
        self._phase      = Phase.PREEMPTED
        self._green      = [arm]
        for cb in self._cbs:
            try: cb(Phase.PREEMPTED, [arm])
            except Exception: pass

    def is_green_for(self, arm): return arm in self._green
    def is_red_for(self, arm):   return arm not in self._green

    def get_spat(self):
        return {"phase": self._phase, "green_arms": list(self._green),
                "time_remaining": 0.0, "preempted": self._preempted,
                "preempt_arm": self._preempt_arm}


# ── Text loop ─────────────────────────────────────────────────────────────────

def _text_loop(sim, runtime: float):
    print("Running... (Ctrl+C to stop)\n")
    deadline  = time.time() + runtime
    prev_done = -1
    while time.time() < deadline:
        time.sleep(1.0)
        s = sim.get_status()
        done = s["done"]
        if done != prev_done or True:
            tput = done / s["elapsed_s"] * 60 if s["elapsed_s"] > 3 else 0
            print(f"  T+{s['elapsed_s']:5.0f}s │ ✓{done:>3}/{s['total_cars']} "
                  f"({s['pct_done']:3.0f}%) │ Q:{s['queued']:>3} │ "
                  f"{s['light_phase']:<12} │ {tput:.1f}/min")
            toks = sim.get_token_events()
            for ev in toks[-3:]:
                print(f"    [{ev.get('event',''):14}] {ev.get('detail','')[:60]}")
            prev_done = done
        if s.get("all_done"):
            print("  ✓ ALL DONE")
            break


# ── Pretty helpers ────────────────────────────────────────────────────────────

def _print_header(arms_label, cars_per_arm, loss, machine=None):
    print(f"\n{'='*64}")
    print(f"  HANDSHAKE V4  —  Intersection Simulation")
    if machine:
        print(f"  Machine #{machine}: {arms_label}")
    else:
        print(f"  Mode: Single machine, all 4 arms")
    print(f"  Cars / arm:  {cars_per_arm}")
    print(f"  Packet loss: {loss*100:.0f}%")
    print(f"  Light cycle: 25s green / 4s yellow / 2s all-red")
    print(f"  Token:       NEG_REQUEST → PASSAGE_TOKEN → TOKEN_ACK")
    print(f"{'='*64}\n")


def _print_fleet(sim):
    for arm, q in sim.queues.items():
        icons = ''.join(
            '🚨' if c.car_type == 'EMERGENCY'
            else ('◆' if c.car_type == 'SMART' else '◇')
            for c in q._all
        )
        print(f"  {arm:<6}: {icons}")
    print()


def _print_summary(sim):
    s = sim.get_status()
    print(f"\n{'='*64}")
    print(f"  INTERSECTION SIM COMPLETE")
    print(f"  Runtime:    {s['elapsed_s']:.0f}s")
    print(f"  Cleared:    {s['done']}/{s['total_cars']} ({s['pct_done']:.0f}%)")
    if s['elapsed_s'] > 0 and s['done'] > 0:
        print(f"  Throughput: {s['done']/s['elapsed_s']*60:.1f} cars/min")
    toks = sim.get_token_events()
    tok_n = sum(1 for e in toks if e.get("event") == "TOKEN_ISSUED")
    ack_n = sum(1 for e in toks if e.get("event") == "TOKEN_ACK")
    print(f"  Tokens issued:    {tok_n}")
    print(f"  Token acks:       {ack_n}")
    print(f"  Log: intersection_v4.log")
    print(f"{'='*64}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Handshake V4 — Intersection Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single machine (all 4 arms):
  python main_intersection.py
  python main_intersection.py --cars 10 --no-dashboard

Two VMs (2 arms each — start VM1 first):
  python main_intersection.py --machine 1 --vms 2   # VM1: North+South + light
  python main_intersection.py --machine 2 --vms 2   # VM2: East+West

Four machines (1 arm each — start machine 1 first):
  python main_intersection.py --machine 1
  python main_intersection.py --machine 2
  python main_intersection.py --machine 3
  python main_intersection.py --machine 4
        """
    )
    p.add_argument("--machine", type=int, choices=[1,2,3,4], default=None,
                   help="Machine number. Omit for single-machine mode.")
    p.add_argument("--vms",       type=int, choices=[2,4], default=4,
                   help="Number of VMs (2 = 2 arms each, 4 = 1 arm each). Default: 4")
    p.add_argument("--rush-hour",    action="store_true",
                   help="Rush hour: 40 cars/arm, auto-ambulance corridor at T+40s")
    p.add_argument("--rogue",        action="store_true",
                   help="One rogue car per arm ignores token slot [feature #8]")
    p.add_argument("--pedestrian",   action="store_true",
                   help="Trigger pedestrian crossing at T+10s  CVC 21950 [feature #13]")
    p.add_argument("--ambulance",    action="store_true",
                   help="Trigger ambulance corridor at T+15s  CVC 21806 [feature #12]")
    p.add_argument("--uncontrolled", action="store_true",
                   help="CVC 21800 right-of-way mode — no fixed signal [features #14+#25]")
    p.add_argument("--machine-fail", type=int, default=None, choices=[1,2,3,4],
                   help="Arm # to crash at T+25s, recovers at T+45s [feature #21]")
    p.add_argument("--spat-delay",   type=float, default=0.0,
                   help="Inject SPaT delay N seconds, shows stale-phase events [feature #22]")
    p.add_argument("--no-log",       action="store_true",
                   help="Skip writing logs/ files after simulation")
    p.add_argument("--cars",    type=int, default=20,
                   help="Cars per arm (default: 20)")
    p.add_argument("--loss",    type=float, default=0.0,
                   help="Packet loss 0.0–1.0")
    p.add_argument("--runtime", type=float, default=900.0,
                   help="Max runtime seconds (text mode)")
    p.add_argument("--no-dashboard", action="store_true",
                   help="Text-only output")
    args = p.parse_args()

    if args.machine and args.vms == 2:
        run_two_vm(args.machine, args.cars, args.loss,
                   not args.no_dashboard, args.runtime)
    elif args.machine:
        run_multi(args.machine, args.cars, args.loss,
                  not args.no_dashboard, args.runtime)
    else:
        run_single(args.cars, args.loss, not args.no_dashboard, args.runtime,
                   rush_hour    = args.rush_hour,
                   include_rogue= args.rogue,
                   pedestrian   = getattr(args, "pedestrian", False),
                   ambulance    = getattr(args, "ambulance", False),
                   uncontrolled = args.uncontrolled,
                   machine_fail = getattr(args, "machine_fail", None),
                   spat_delay   = getattr(args, "spat_delay", 0.0),
                   no_log       = getattr(args, "no_log", False))


if __name__ == "__main__":
    main()
