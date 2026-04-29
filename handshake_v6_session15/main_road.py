#!/usr/bin/env python3
# =============================================================================
# handshake_v5/main_road.py  —  Road simulation entry point
#
# Usage:
#   python3 main_road.py                      # single PC, default NPCs
#   python3 main_road.py --rush               # rush hour (more cars)
#   python3 main_road.py --breakdown          # add breakdown car
#   python3 main_road.py --platoon            # add platoon
#   python3 main_road.py --convoy             # add convoy
#   python3 main_road.py --split-brain        # network partition test
#   python3 main_road.py --spat-delay 2.0     # SPaT delay injection
#   python3 main_road.py --roadworks          # roadworks zone active
#   python3 main_road.py --loss 0.1           # 10% packet loss
#   python3 main_road.py --vms 2 --machine 1  # 2-VM run, this is VM1
#   python3 main_road.py --spectator          # watch without driving
#   python3 main_road.py --no-learn           # skip ML training after run
#   python3 main_road.py --runtime 120        # run for 120s then stop
# =============================================================================
import argparse, sys, time, threading, logging, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

from sim_road       import RoadSim
from dashboard_road import RoadDashboard
from sim_logger     import attach_logger


def _input_thread(sim, machine_id=1):
    """Read player commands from stdin in a background thread."""
    while True:
        try:
            cmd = input()
            if cmd.strip():
                sim.enqueue_command(cmd.strip(), machine_id=machine_id)
        except (EOFError, KeyboardInterrupt):
            break


def main():
    ap = argparse.ArgumentParser(description="Handshake V5 Road Simulation")
    ap.add_argument("--rush",        action="store_true", help="Rush hour mode")
    ap.add_argument("--breakdown",   action="store_true", help="Add breakdown car")
    ap.add_argument("--platoon",     action="store_true", help="Add platoon")
    ap.add_argument("--convoy",      action="store_true", help="Add convoy of trucks")
    ap.add_argument("--split-brain", action="store_true", dest="split_brain",
                    help="Trigger network partition after 10s")
    ap.add_argument("--spat-delay",  type=float, default=0.0, dest="spat_delay",
                    metavar="N",     help="Inject SPaT delay of N seconds")
    ap.add_argument("--roadworks",   action="store_true", help="Enable roadworks zone")
    ap.add_argument("--loss",        type=float, default=0.0, metavar="F",
                    help="Packet loss 0.0–1.0")
    ap.add_argument("--vms",         type=int, default=1, metavar="N",
                    help="Total VMs in this run (1-4)")
    ap.add_argument("--machine",     type=int, default=1, metavar="N",
                    help="This machine's ID (1-based)")
    ap.add_argument("--spectator",   action="store_true",
                    help="Watch without controlling a car")
    ap.add_argument("--no-learn",    action="store_true", dest="no_learn",
                    help="Skip ML training after run")
    ap.add_argument("--runtime",     type=int, default=0, metavar="S",
                    help="Auto-stop after S seconds (0=run until done)")
    ap.add_argument("--smart",       type=int, default=4, metavar="N")
    ap.add_argument("--legacy",      type=int, default=4, metavar="N")
    ap.add_argument("--emerg",       type=int, default=1, metavar="N")
    args = ap.parse_args()

    from config import CONVOY_SIZE
    npc_counts = {
        "smart":     args.smart,
        "legacy":    args.legacy,
        "emergency": args.emerg,
        "breakdown": 1 if args.breakdown else 0,
        "platoon":   2 if args.platoon   else 0,
        "convoy":    CONVOY_SIZE if args.convoy else 0,
    }

    player_configs = None if args.spectator else [
        {"label":f"P{args.machine}◆","lane":0,"start_pos":150.0,"machine_id":args.machine}
    ]

    sim = RoadSim(
        player_configs = player_configs,
        npc_counts     = npc_counts if not args.rush else None,
        loss           = args.loss,
        rush_hour      = args.rush,
        spat_delay     = args.spat_delay,
        roadworks      = args.roadworks,
        total_vms      = args.vms,
        machine_id     = args.machine,
    )

    # Attach logger
    meta = {
        "rush_hour":  args.rush,    "roadworks":  args.roadworks,
        "split_brain":args.split_brain,"convoy":  args.convoy,
        "platoon":    args.platoon,  "spat_delay": args.spat_delay,
        "vms":        args.vms,      "machine_id": args.machine,
        "spectator":  args.spectator,
    }
    logger = attach_logger("road", sim, meta)

    sim.start()

    # Trigger split-brain after 15s if requested
    if args.split_brain:
        def _trigger():
            time.sleep(15.0)
            sim.trigger_split_brain()
            print("\n  🔌 SPLIT-BRAIN triggered — network partitioned for 10s\n")
        threading.Thread(target=_trigger, daemon=True).start()

    # Input thread (skip in spectator mode)
    if not args.spectator:
        inp = threading.Thread(target=_input_thread,
                               args=(sim, args.machine), daemon=True)
        inp.start()

    # Auto-stop thread
    if args.runtime > 0:
        def _autostop():
            time.sleep(args.runtime)
            sim.stop()
        threading.Thread(target=_autostop, daemon=True).start()

    # Dashboard (runs in main thread)
    dash = RoadDashboard(sim, refresh_rate=5)
    try:
        dash.run()
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()
        dash.stop()
        txt, jsn = logger.finish()
        print(f"\n  ✅ Run complete.  Log: {txt}")

        if not args.no_learn:
            # Phase 2: run ML training in background after sim ends
            try:
                from model_trainer import train_from_logs, quick_eval
                def _after_train(msg):
                    print(f"  {msg}")
                    # Quick eval after training completes
                    try: quick_eval()
                    except Exception: pass
                train_from_logs(async_mode=True, callback=_after_train)
            except Exception as e:
                print(f"  ML training skipped: {e}")


if __name__ == "__main__":
    main()
