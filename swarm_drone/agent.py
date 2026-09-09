"""
A single drone: the code that actually runs on one Raspberry Pi.

The agent owns its flight backend, its mesh link and (optionally) its camera
pipeline.  Its ``tick`` needs only its own state plus whatever it heard from
neighbours, so the same object works standalone on hardware and inside the
simulator's loop.
"""
import time

import numpy as np

from .geometry import clamp_norm

G = 9.80665


class DroneAgent:
    def __init__(self, drone_id, cfg, backend, link=None, vision=None):
        self.id = int(drone_id)
        self.cfg = cfg
        self.backend = backend
        self.link = link
        self.vision = vision              # optional AerialVisionPipeline
        self.slot = int(drone_id)
        self.healthy = True
        self.fault = None
        self.last_cmd = np.zeros(3)
        self.sightings = []               # alerts this drone raised or heard

    # ------------------------------------------------------------------ state
    @property
    def position(self):
        return self.backend.state()[0]

    @property
    def velocity(self):
        return self.backend.state()[1]

    @property
    def battery_frac(self):
        return getattr(self.backend, "battery_frac", 1.0)

    def battery_ok(self):
        return self.battery_frac > self.cfg.drone.reserve_frac

    def telemetry(self):
        pos, vel = self.backend.state()
        return {
            "id": self.id,
            "slot": self.slot,
            "pos": [round(float(v), 3) for v in pos],
            "vel": [round(float(v), 3) for v in vel],
            "batt": round(float(self.battery_frac), 3),
            "ok": bool(self.healthy and self.battery_ok()),
            "fault": self.fault,
        }

    # ---------------------------------------------------------------- control
    def apply(self, acc_cmd, dt, load_share_n=0.0):
        """Clamp a command to the airframe's envelope and send it."""
        acc = np.asarray(acc_cmd, dtype=float).copy()
        horiz_cap = G * np.tan(self.cfg.drone.max_tilt_rad)
        acc[:2] = clamp_norm(acc[:2], horiz_cap)
        acc[2] = np.clip(acc[2], -self.cfg.drone.max_accel_mps2,
                         self.cfg.drone.max_accel_mps2)
        self.last_cmd = acc

        if hasattr(self.backend, "set_load_share"):
            self.backend.set_load_share(load_share_n)
        return self.backend.send_acceleration(acc, dt)

    def arm(self):
        self.backend.arm()

    def disarm(self):
        self.backend.disarm()

    def fail(self, reason):
        """Mark this drone unhealthy — the coordinator will rebalance the lift."""
        self.healthy = False
        self.fault = reason

    # ------------------------------------------------------------------ comms
    def broadcast(self, extra=None):
        if self.link is None:
            return None
        msg = self.telemetry()
        if extra:
            msg.update(extra)
        return self.link.send(msg)

    def receive(self):
        return self.link.poll() if self.link is not None else []

    # ----------------------------------------------------------------- vision
    def process_frame(self, frame, timestamp=None):
        """Run one camera frame through the on-board pipeline.

        Any wanted-person match is broadcast to the mesh so all six drones — and
        the ground station — see the sighting, not just the one that saw it.
        """
        if self.vision is None:
            return []
        timestamp = time.time() if timestamp is None else timestamp
        alerts = self.vision.process(frame, drone_id=self.id, timestamp=timestamp)
        for alert in alerts:
            self.sightings.append(alert)
            self.broadcast({"kind": "sighting", "alert": alert.to_dict()})
        return alerts
