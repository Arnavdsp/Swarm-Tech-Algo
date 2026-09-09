#!/usr/bin/env python3
"""
On-drone entrypoint. One of these runs on each Raspberry Pi.

    python scripts/run_drone_node.py --id 3 --config config/swarm.yaml
    python scripts/run_drone_node.py --id 0 --sim          # no flight controller
    python scripts/run_drone_node.py --id 1 --no-vision    # lifting only

Two loops share the process:

  * the control loop, at ``formation.control_hz``, which holds this drone's
    hexagon slot and broadcasts telemetry to the mesh;
  * the vision loop, in a background thread, which reads the camera and raises
    wanted-person alerts.

They are deliberately decoupled. Vision on a Pi is slow and jittery; the flight
loop must not wait for it. A vision stall degrades to "this drone stops
reporting sightings", never to "this drone stops holding formation".
"""
import argparse
import json
import os
import signal
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_drone import SwarmConfig, DroneAgent                        # noqa: E402
from swarm_drone.backends import SimBackend                            # noqa: E402
from swarm_drone.comms import make_link                                # noqa: E402
from swarm_drone.formation import HexFormation                        # noqa: E402
from swarm_drone.geometry import hexagon_slots                        # noqa: E402
from swarm_drone.load_lift import LiftPlanner                          # noqa: E402

RUNNING = True


def _stop(signum, frame):
    global RUNNING
    RUNNING = False
    print("\nShutdown requested — landing.")


class VisionThread(threading.Thread):
    """Camera -> pipeline -> mesh, off the flight loop's critical path."""

    def __init__(self, agent, source=0, width=1280, height=720):
        super().__init__(daemon=True)
        self.agent = agent
        self.source = source
        self.width, self.height = width, height
        self.frames = 0
        self.error = None

    def run(self):
        try:
            import cv2
        except ImportError:
            self.error = "opencv not installed; vision disabled"
            return
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            self.error = f"could not open camera {self.source}"
            return
        while RUNNING:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            self.frames += 1
            try:
                self.agent.process_frame(frame)
            except Exception as exc:                      # noqa: BLE001
                # Never let a vision fault reach the flight loop.
                self.error = str(exc)
                time.sleep(0.5)
        cap.release()


def build_pipeline(cfg, drone_id):
    from swarm_drone.vision import (AerialVisionPipeline, RTDETRDetector,
                                    WantedFaceDB, AlertSink, make_embedder)
    embedder = make_embedder(cfg.vision)
    db = WantedFaceDB.load_or_empty(cfg.vision.db_path,
                                    dim=getattr(embedder, "dim", 512))
    sink = AlertSink(cfg.vision.alert_log, cfg.vision.alert_cooldown_s)
    sink.subscribe(lambda a: print(f"  🚨 {a.summary()}"))
    print(f"  vision: watchlist {len(db)} people / {db.n_vectors} embeddings")
    return AerialVisionPipeline(cfg.vision, detector=RTDETRDetector(cfg.vision),
                                embedder=embedder, db=db, sink=sink)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, required=True, help="drone index, 0..n-1")
    ap.add_argument("--config", help="YAML config; defaults are used if omitted")
    ap.add_argument("--mavlink", default="/dev/serial0",
                    help="flight controller connection string")
    ap.add_argument("--sim", action="store_true",
                    help="simulated flight backend, for bench testing")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--camera", default=0)
    ap.add_argument("--link", default="udp", choices=["udp", "loopback"])
    ap.add_argument("--status-hz", type=float, default=1.0)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    cfg = SwarmConfig.from_yaml(args.config) if args.config else SwarmConfig()
    n = cfg.formation.n_drones
    if not 0 <= args.id < n:
        print(f"--id must be 0..{n - 1}")
        return 1

    print(f"── Drone {args.id} of {n} ──────────────────────────────────")

    if args.sim:
        start = hexagon_slots(cfg.formation.radius_m * 1.5, n=n)[args.id]
        backend = SimBackend(cfg.drone, position=start)
        print(f"  backend: simulated, start {np.round(start, 2)}")
    else:
        from swarm_drone.backends import MavlinkBackend
        backend = MavlinkBackend(args.mavlink, spec=cfg.drone,
                                 system_id=args.id + 1)
        print(f"  backend: MAVLink on {args.mavlink}")

    link = make_link(args.id, cfg.comms, kind=args.link)
    print(f"  mesh: {args.link} on port {cfg.comms.port}")

    vision = None
    if not args.no_vision:
        try:
            vision = build_pipeline(cfg, args.id)
        except Exception as exc:                          # noqa: BLE001
            print(f"  vision: unavailable ({exc}) — flying without it")

    agent = DroneAgent(args.id, cfg, backend, link=link, vision=vision)
    agent.slot = args.id

    # Every drone builds the identical virtual structure from the shared config,
    # so slot k means the same point in space on all six Pis without any of them
    # having to agree at runtime.
    formation = HexFormation(cfg.formation, centre=(0.0, 0.0, 0.0))
    planner = LiftPlanner(cfg, formation)
    planner.start(np.stack([agent.position] * n))
    formation.perm = list(range(n))
    agent.arm()

    vision_thread = None
    if vision is not None:
        vision_thread = VisionThread(agent, source=args.camera)
        vision_thread.start()

    dt = 1.0 / cfg.formation.control_hz
    status_every = max(1, int(cfg.formation.control_hz / max(args.status_hz, 0.1)))
    tick = 0
    print("  running — Ctrl-C to land\n")

    while RUNNING:
        loop_start = time.time()
        agent.receive()
        peers = link.peers()

        # Fill the swarm state from the mesh; anything unheard falls back to this
        # drone's own slot target, which keeps the control law well-defined when
        # a neighbour drops out rather than steering toward a stale position.
        positions, velocities = [], []
        for k in range(n):
            if k == args.id:
                positions.append(agent.position)
                velocities.append(agent.velocity)
            elif k in peers:
                positions.append(np.array(peers[k]["pos"], dtype=float))
                velocities.append(np.array(peers[k]["vel"], dtype=float))
            else:
                positions.append(formation.targets()[k])
                velocities.append(np.zeros(3))
        positions = np.stack(positions)
        velocities = np.stack(velocities)

        phase = planner.step(dt, positions, formation.is_formed(positions))
        acc = formation.command(args.id, positions, velocities,
                                alt_setpoint=planner.alt_setpoint, dt=dt)
        share = planner.current_shares(positions)[args.id]
        agent.apply(acc, dt, load_share_n=float(share))
        agent.broadcast({"phase": phase.value, "share_n": round(float(share), 2)})

        tick += 1
        if tick % status_every == 0:
            lost = link.lost_peers(range(n))
            line = {
                "t": round(tick * dt, 1),
                "phase": phase.value,
                "pos": [round(float(v), 2) for v in agent.position],
                "share_n": round(float(share), 1),
                "batt": round(agent.battery_frac, 2),
                "peers": sorted(peers),
            }
            if lost:
                line["lost"] = lost
            if vision_thread is not None:
                line["frames"] = vision_thread.frames
                if vision_thread.error:
                    line["vision_error"] = vision_thread.error
            print(json.dumps(line))

        time.sleep(max(0.0, dt - (time.time() - loop_start)))

    agent.disarm()
    link.close()
    if vision is not None:
        print(f"\nVision: {json.dumps(vision.report(), indent=2)}")
    print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
