#!/usr/bin/env python3
# =============================================================================
# handshake_v4/intersection.py  —  Intersection Simulation Entry Point
#
# SINGLE MACHINE (all 4 arms):
#   python intersection.py
#   python intersection.py --cars 10 --loss 0.1
#   python intersection.py --no-dashboard
#
# FOUR MACHINES (one arm each):
#   Machine 1 (North + traffic light):  python intersection.py --machine 1
#   Machine 2 (East):                   python intersection.py --machine 2
#   Machine 3 (South):                  python intersection.py --machine 3
#   Machine 4 (West):                   python intersection.py --machine 4
#
# Machine 1 must start first — it hosts the traffic light.
# Machines 2-4 receive SPaT broadcasts from Machine 1.
#
# Network: UDP multicast 239.0.0.4:5400
# VirtualBox: set all VMs to Host-only Adapter (vboxnet0)
# =============================================================================
import sys, os, time, signal, argparse, logging, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("intersection_v4.log")]
)

MACHINE_ARM = {1: "North", 2: "East", 3: "South", 4: "West"}
LIGHT_MACHINE = 1


# ── Remote light proxy (Machines 2-4) ────────────────────────────────────────

class RemoteLight:
    """Mirrors the traffic light from Machine 1 via SPaT messages."""
    import threading as _t
    def __init__(self):
        self._phase      = "NS_GREEN"
        self._green_arms = ["North","South"]
        self._preempted  = False
        self._preempt_arm = None
        self._lock       = self._t.Lock()
        self._callbacks  = []

    def on_phase_change(self, cb): self._callbacks.append(cb)

    def update(self, phase: str, green_arms: list, preempted: bool = False):
        with self._lock:
            changed = phase != self._phase
            self._phase       = phase
            self._green_arms  = green_arms or []
            self._preempted   = preempted
        if changed:
            for cb in self._callbacks:
                try: cb(phase, green_arms)
                except Exception: pass

    def preempt(self, arm: str):
        self.update("PREEMPTED", [arm], preempted=True)

    def is_green_for(self, arm: str) -> bool:
        with self._lock: return arm in self._green_arms

    def is_red_for(self, arm: str) -> bool:
        return not self.is_green_for(arm)

    def get_spat(self) -> dict:
        with self._lock:
            return {"phase": self._phase, "green_arms": list(self._green_arms),
                    "time_remaining": 0.0, "preempted": self._preempted,
                    "preempt_arm": self._preempt_arm}


def _spat_broadcaster(light, radio, stop_event, interval=0.4):
    """Machine 1: broadcast SPaT every 400ms."""
    while not stop_event.is_set():
        spat = light.get_spat()
        spat["type"] = "SPAT"
        radio.send(spat)
        stop_event.wait(interval)


def _spat_listener(radio, remote_light, stop_event):
    """Machines 2-4: listen for SPaT and update RemoteLight."""
    last_phase = None
    while not stop_event.is_set():
        for msg in radio.drain():
            if msg.get("type") == "SPAT":
                p = msg.get("phase", "")
                g = msg.get("green_arms", [])
                pre = msg.get("preempted", False)
                if p != last_phase:
                    last_phase = p
                    remote_light.update(p, g, pre)
        stop_event.wait(0.1)


# ── Run ───────────────────────────────────────────────────────────────────────

def run(machine: int, cars_per_arm: int, loss: float,
        use_dashboard: bool, runtime: float):

    single = (machine == 0)
    stop_event = threading.Event()

    # ── Single machine — all 4 arms, no networking ──────────────────────────
    if single:
        from sim_intersection import IntersectionSim
        sim = IntersectionSim(my_arms=["North","East","South","West"],
                               cars_per_arm=cars_per_arm, loss=loss,
                               host_light=True)
        _print_header(machine, cars_per_arm, loss, single=True)
        _print_fleet(sim)
        sim.start()
        _run_display(sim, use_dashboard, runtime, stop_event, machine=0)
        sim.stop()
        _print_summary(sim)
        return

    # ── Multi-machine mode ───────────────────────────────────────────────────
    arm = MACHINE_ARM[machine]
    _print_header(machine, cars_per_arm, loss, single=False, arm=arm)

    from radio_network import Radio as NetRadio
    arm_radio   = NetRadio(f"ARM-{arm}", loss=loss)
    arm_radio.start()

    if machine == LIGHT_MACHINE:
        from traffic_light import TrafficLight
        from sim_intersection import IntersectionSim, build_fleet
        from arm_queue import ArmQueue
        from config import Phase

        light = TrafficLight("INT-01")
        spat_radio = NetRadio("SPAT-TX", loss=0.0)
        spat_radio.start()

        fleet = build_fleet(arm, cars_per_arm, seed=42)
        queue = ArmQueue(arm, fleet, arm_radio, light=light)

        def on_phase(phase, green):
            if isinstance(green, str): green = [green]
            green = green or []
            if arm in green: queue.on_green()
            else:            queue.on_red()

        light.on_phase_change(on_phase)

        t_spat = threading.Thread(
            target=_spat_broadcaster,
            args=(light, spat_radio, stop_event, 0.4),
            daemon=True
        )
        t_spat.start()
        print(f"  [Machine {machine}] Hosting traffic light — broadcasting SPaT")

        # Minimal sim wrapper for the dashboard
        sim = _SingleArmSim(arm, queue, light)

        light.start()
        on_phase(Phase.NS_GREEN, ["North", "South"])

    else:
        from sim_intersection import build_fleet
        from arm_queue import ArmQueue

        remote  = RemoteLight()
        spat_rx = NetRadio("SPAT-RX", loss=0.0)
        spat_rx.start()

        fleet = build_fleet(arm, cars_per_arm, seed=42 + machine * 100)
        queue = ArmQueue(arm, fleet, arm_radio, light=remote)

        def on_phase(phase, green):
            if isinstance(green, str): green = [green]
            green = green or []
            if arm in green: queue.on_green()
            else:            queue.on_red()

        remote.on_phase_change(on_phase)

        t_spat = threading.Thread(
            target=_spat_listener,
            args=(spat_rx, remote, stop_event),
            daemon=True
        )
        t_spat.start()
        print(f"  [Machine {machine}] Arm={arm} — listening for SPaT from Machine 1...")
        light = remote
        sim   = _SingleArmSim(arm, queue, remote)

    # Simulation loop
    sim_stop = threading.Event()
    def _loop():
        import time
        from config import SIM_TICK_S
        dt = SIM_TICK_S
        while not sim_stop.is_set():
            t0 = time.time()
            queue.tick(dt)
            elapsed = time.time() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    sim_thread = threading.Thread(target=_loop, daemon=True, name="sim-arm")
    sim_thread.start()
    print()

    def on_stop(sig=None, frame=None):
        sim_stop.set()
        stop_event.set()
        arm_radio.stop()
        if machine == LIGHT_MACHINE:
            light.stop()
            spat_radio.stop()
        else:
            spat_rx.stop()
        _print_summary(sim)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_stop)

    _run_display(sim, use_dashboard, runtime, stop_event, machine=machine)

    sim_stop.set()
    stop_event.set()
    arm_radio.stop()
    if machine == LIGHT_MACHINE:
        light.stop()
    _print_summary(sim)


# ── Single-arm sim wrapper for multi-machine dashboard ────────────────────────

class _SingleArmSim:
    def __init__(self, arm, queue, light):
        self._arm   = arm
        self._q     = queue
        self._light = light
        self._st    = __import__("time").time()

    def elapsed(self): return __import__("time").time() - self._st

    def get_status(self):
        s    = self._q.status()
        spat = self._light.get_spat()
        done  = s.get("done",0)
        total = s.get("total",0)
        return {
            "elapsed_s":       round(self.elapsed(),1),
            "total_cars":      total,
            "done":            done,
            "queued":          s.get("queued",0),
            "on_road":         s.get("on_road",0),
            "pct_done":        round(100*done/total,1) if total else 0,
            "all_done":        done >= total,
            "light_phase":     spat.get("phase",""),
            "light_green":     spat.get("green_arms",[]),
            "light_remaining": spat.get("time_remaining",0),
            "preempted":       spat.get("preempted",False),
            "arms":            {self._arm: s},
        }

    def get_token_events(self):
        return self._q.get_token_events()

    def get_all_events(self):
        evts = []
        for car in self._q._all:
            evts.extend(car.events[-5:])
        evts.sort(key=lambda e: e.get("ts",0))
        return evts[-30:]


# ── Display ───────────────────────────────────────────────────────────────────

def _run_display(sim, use_dashboard, runtime, stop_event, machine=0):
    if use_dashboard:
        try:
            from dashboard_intersection import IntersectionDashboard
            print("  Dashboard starting... (Ctrl+C to stop)")
            time.sleep(0.4)
            IntersectionDashboard(sim).run()
            return
        except ImportError:
            print("  [rich not available — text mode]")
        except KeyboardInterrupt:
            return

    # Text mode
    print("  Running... (Ctrl+C to stop)\n")
    deadline   = time.time() + runtime
    prev_done  = -1
    prev_phase = ""
    while time.time() < deadline and not stop_event.is_set():
        time.sleep(1.0)
        s     = sim.get_status()
        done  = s["done"]
        phase = s["light_phase"]
        if done != prev_done or phase != prev_phase:
            total = s["total_cars"]
            elapsed = s["elapsed_s"]
            tput = done/elapsed*60 if elapsed>3 else 0
            arm_info = "" if machine == 0 else f"[{MACHINE_ARM.get(machine,'?')}] "
            print(f"  {arm_info}T+{elapsed:4.0f}s │ ✓{done:>3}/{total} "
                  f"({s['pct_done']:3.0f}%) │ Q:{s['queued']:>3} │ "
                  f"{phase:<12} │ {tput:5.1f}/min")
            prev_done  = done
            prev_phase = phase
        if s.get("all_done"):
            print("  ✓ All cars cleared!")
            break


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_header(machine, cars, loss, single, arm=""):
    mode = "Single Machine — All 4 Arms" if single else f"Machine {machine} — Arm: {arm}"
    print(f"\n{'='*60}")
    print(f"  HANDSHAKE V4  —  Intersection Simulation")
    print(f"  Mode:        {mode}")
    print(f"  Cars / arm:  {cars}")
    print(f"  Packet loss: {loss*100:.0f}%")
    if not single:
        print(f"  Network:     UDP multicast 239.0.0.4:5400")
        print(f"  Light host:  Machine {LIGHT_MACHINE}")
    print(f"{'='*60}\n")


def _print_fleet(sim):
    from config import CarType
    for arm, q in sim.queues.items():
        icons = "".join(
            "🚨" if c.car_type==CarType.EMERGENCY
            else ("◆" if c.car_type==CarType.SMART else "◇")
            for c in q._all
        )
        print(f"  {arm:<6}: {icons}")
    print()


def _print_summary(sim):
    s = sim.get_status()
    e = s["elapsed_s"]
    d = s["done"]
    t = s["total_cars"]
    print(f"\n{'='*60}")
    print(f"  DONE: {d}/{t} cleared in {e:.0f}s")
    if e > 0 and d > 0:
        print(f"  Throughput: {d/e*60:.1f} cars/min")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Handshake V4 — Intersection Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single machine:   python intersection.py
Machine 1:        python intersection.py --machine 1
Machine 2:        python intersection.py --machine 2
Machine 3:        python intersection.py --machine 3
Machine 4:        python intersection.py --machine 4
        """
    )
    p.add_argument("--machine",      type=int, default=0, choices=[0,1,2,3,4],
                   help="0=single machine, 1-4=specific arm")
    p.add_argument("--cars",         type=int, default=20)
    p.add_argument("--loss",         type=float, default=0.0)
    p.add_argument("--runtime",      type=float, default=900.0)
    p.add_argument("--no-dashboard", action="store_true")
    args = p.parse_args()
    run(args.machine, args.cars, args.loss, not args.no_dashboard, args.runtime)


if __name__ == "__main__":
    main()
