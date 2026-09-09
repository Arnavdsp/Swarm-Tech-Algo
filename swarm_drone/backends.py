"""
Flight-control backends.

``SimBackend`` is a double integrator with tether drag — enough to exercise the
formation controller and the lift state machine on a laptop.  ``MavlinkBackend``
sends the same acceleration commands to a real flight controller over MAVLink
(ArduPilot/PX4) from the Pi's serial or UDP link; pymavlink is imported lazily so
the simulator has no dependency on it.
"""
import numpy as np

from .geometry import clamp_norm

G = 9.80665


class BaseBackend:
    """Interface every backend implements."""

    def arm(self):
        raise NotImplementedError

    def disarm(self):
        raise NotImplementedError

    def state(self):
        """Return (position(3,), velocity(3,))."""
        raise NotImplementedError

    def send_acceleration(self, acc, dt=None):
        raise NotImplementedError


class SimBackend(BaseBackend):
    """Point-mass drone with first-order drag and an optional tether load.

    The tether is modelled as the share of payload weight this drone carries; it
    shows up as a downward acceleration that eats into the climb authority, which
    is exactly the effect the lift state machine has to cope with.
    """

    def __init__(self, spec, position=(0.0, 0.0, 0.0), drag=0.35, seed=None):
        self.spec = spec
        self.pos = np.asarray(position, dtype=float).copy()
        self.vel = np.zeros(3)
        self.drag = drag
        self.armed = False
        self.load_share_n = 0.0
        self.battery_frac = 1.0
        self.rng = np.random.default_rng(seed)
        self.wind = np.zeros(3)

    def arm(self):
        self.armed = True

    def disarm(self):
        self.armed = False
        self.vel[:] = 0.0

    def state(self):
        return self.pos.copy(), self.vel.copy()

    def set_load_share(self, newtons):
        self.load_share_n = float(newtons)

    def set_wind(self, wind):
        self.wind = np.asarray(wind, dtype=float)

    def available_accel(self):
        """Acceleration the motors can still produce given the hanging load."""
        total_mass = self.spec.mass_kg + self.load_share_n / G
        max_a = self.spec.max_thrust_n / total_mass - G
        return max(0.0, max_a)

    def send_acceleration(self, acc, dt=0.05):
        if not self.armed:
            return self.state()
        acc = np.asarray(acc, dtype=float).copy()

        # Horizontal authority is bounded by the tilt limit, vertical by thrust.
        horiz_cap = G * np.tan(self.spec.max_tilt_rad)
        acc[:2] = clamp_norm(acc[:2], horiz_cap)
        acc[2] = np.clip(acc[2], -self.spec.max_accel_mps2,
                         self.available_accel())

        # Tether drag: the load pulls back on whichever drone runs ahead.
        total_mass = self.spec.mass_kg + self.load_share_n / G
        acc *= self.spec.mass_kg / total_mass

        self.vel += (acc - self.drag * self.vel + self.wind) * dt
        speed = np.linalg.norm(self.vel[:2])
        if speed > self.spec.max_speed_mps:
            self.vel[:2] *= self.spec.max_speed_mps / speed
        self.vel[2] = np.clip(self.vel[2], -self.spec.max_climb_mps,
                              self.spec.max_climb_mps)

        self.pos += self.vel * dt
        self.pos[2] = max(0.0, self.pos[2])
        if self.pos[2] == 0.0 and self.vel[2] < 0:
            self.vel[2] = 0.0

        self._drain_battery(acc, dt)
        return self.state()

    def _drain_battery(self, acc, dt):
        """Crude but monotonic: hover draw plus a term for extra thrust."""
        hover_w = self.spec.hover_throttle * self.spec.max_thrust_n * 3.0
        extra = np.linalg.norm(acc) * (self.spec.mass_kg + self.load_share_n / G) * 3.0
        watts = hover_w + extra
        self.battery_frac -= watts * (dt / 3600.0) / max(self.spec.battery_wh, 1e-6)
        self.battery_frac = max(0.0, self.battery_frac)


class MavlinkBackend(BaseBackend):
    """Thin pymavlink adapter for a real Pixhawk-class controller.

    The Pi runs this next to the vision pipeline and streams
    SET_POSITION_TARGET_LOCAL_NED in acceleration mode at the formation control
    rate.  Kept deliberately small — the flight controller owns attitude, this
    only owns where the drone should be.
    """

    def __init__(self, connection_str="udp:127.0.0.1:14550", spec=None,
                 system_id=1, component_id=1):
        try:
            from pymavlink import mavutil
        except ImportError as exc:                     # pragma: no cover
            raise ImportError(
                "MavlinkBackend needs pymavlink: pip install pymavlink") from exc
        self.mavutil = mavutil
        self.spec = spec
        self.master = mavutil.mavlink_connection(connection_str,
                                                 source_system=system_id,
                                                 source_component=component_id)
        self.master.wait_heartbeat()
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.armed = False

    # NED <-> ENU: swap x/y and negate z.
    @staticmethod
    def _enu_to_ned(v):
        return np.array([v[1], v[0], -v[2]], dtype=float)

    @staticmethod
    def _ned_to_enu(v):
        return np.array([v[1], v[0], -v[2]], dtype=float)

    def arm(self):
        self.master.arducopter_arm()
        self.master.motors_armed_wait()
        self.armed = True

    def disarm(self):
        self.master.arducopter_disarm()
        self.armed = False

    def state(self):
        msg = self.master.recv_match(type="LOCAL_POSITION_NED", blocking=False)
        if msg is not None:
            self.pos = self._ned_to_enu([msg.x, msg.y, msg.z])
            self.vel = self._ned_to_enu([msg.vx, msg.vy, msg.vz])
        return self.pos.copy(), self.vel.copy()

    def send_acceleration(self, acc, dt=None):
        ned = self._enu_to_ned(acc)
        # type_mask: ignore position and velocity, use acceleration only.
        type_mask = 0b0000_11_000_111_111
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            self.mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0, 0, 0,
            0, 0, 0,
            float(ned[0]), float(ned[1]), float(ned[2]),
            0, 0)
        return self.state()
