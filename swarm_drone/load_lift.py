"""
Cooperative load lifting for the hexagonal swarm.

Six tethers meet a load hanging under the centre of the hexagon.  Because the
attach points are a regular hexagon too, the horizontal components of the six
tether tensions cancel by symmetry and each drone carries exactly one sixth of
the weight — *provided* the ring stays level and centred.  Everything here is
about checking that assumption and aborting when it stops holding.

    tension_i = (m_load * g / 6) / cos(alpha)

where ``alpha`` is the tether's angle from vertical, set by the hexagon radius,
the attach radius and the tether length.
"""
from enum import Enum

import numpy as np

from .geometry import hexagon_slots, tether_unit_vectors

G = 9.80665


class LiftPhase(Enum):
    """Mission states. The swarm only ever moves one step along this list."""
    IDLE = "idle"
    ARMING = "arming"
    TAKEOFF = "takeoff"
    FORMING = "forming"          # fly into the hexagon at hover altitude
    DESCEND_TO_LOAD = "descend"  # drop to pickup altitude over the load
    TENSIONING = "tensioning"    # ramp thrust to take up tether slack
    LIFTING = "lifting"          # synchronised ascent
    CRUISE = "cruise"            # carry to destination
    LOWERING = "lowering"
    RELEASED = "released"
    LANDING = "landing"
    ABORT = "abort"


def tether_geometry(hex_radius, attach_radius, tether_len):
    """Return (horizontal_offset, vertical_drop, alpha) for one tether.

    ``alpha`` is the angle from vertical. Raises if the tether is too short to
    span the offset at all.
    """
    horiz = float(hex_radius - attach_radius)
    if tether_len <= abs(horiz):
        raise ValueError(
            f"tether {tether_len:.2f} m too short for a {horiz:.2f} m offset; "
            f"shrink the hexagon radius or use longer tethers")
    vert = float(np.sqrt(tether_len ** 2 - horiz ** 2))
    alpha = float(np.arctan2(abs(horiz), vert))
    return horiz, vert, alpha


def tension_per_drone(payload_mass, n, alpha):
    """Tension in one tether when the load hangs evenly. Newtons."""
    return (payload_mass * G / n) / max(np.cos(alpha), 1e-6)


def solve_load_equilibrium(drone_positions, attach_offsets, tether_len,
                           payload_mass, stiffness=None, tol=1e-6, iters=80):
    """Where the load hangs, and how much each tether then carries.

    Each tether is a *unilateral* spring: it pulls once stretched past its
    natural length and does nothing at all when slack. That one-sidedness is the
    whole physics of the problem — a drone that climbs pulls its tether taut and
    takes load off the drones that have gone slack, which is precisely the
    failure the lift monitor has to catch.

    The load is assumed to hang under the swarm's horizontal centroid (true while
    the hexagon is roughly symmetric) so only its height is unknown. Total
    vertical tension rises monotonically as the load hangs lower, so a bisection
    on that height converges without any solver machinery.

    Returns ``(vertical_share_n, load_z, tensions_n)``.
    """
    pos = np.asarray(drone_positions, dtype=float)
    offs = np.asarray(attach_offsets, dtype=float)
    weight = payload_mass * G
    if weight <= 0.0:
        return np.zeros(len(pos)), float(pos[:, 2].min() - tether_len), np.zeros(len(pos))

    if stiffness is None:
        # Near-inextensible fallback (~2 % stretch at full load). Callers that
        # have a LiftConfig pass its elastic element's rate instead.
        stiffness = weight / max(0.02 * tether_len, 1e-3)

    centre_xy = pos[:, :2].mean(axis=0)

    def tension_at(load_z):
        attach = np.column_stack([
            centre_xy[0] + offs[:, 0],
            centre_xy[1] + offs[:, 1],
            np.full(len(offs), load_z)])
        delta = pos - attach
        dist = np.linalg.norm(delta, axis=1)
        stretch = np.maximum(0.0, dist - tether_len)
        tens = stiffness * stretch
        vertical = tens * np.where(dist > 1e-9, delta[:, 2] / np.maximum(dist, 1e-9), 0.0)
        return tens, np.maximum(vertical, 0.0)

    hi = float(pos[:, 2].min())                  # load right up at the drones
    lo = float(pos[:, 2].min() - tether_len - weight / stiffness - 1.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, vertical = tension_at(mid)
        if vertical.sum() < weight:
            hi = mid                             # not enough lift: hang lower
        else:
            lo = mid
        if hi - lo < tol:
            break

    load_z = 0.5 * (lo + hi)
    tens, vertical = tension_at(load_z)
    total = vertical.sum()
    if total > 1e-9:
        # Renormalise away the last bisection residual so the shares sum to the
        # weight exactly; the *ratios* are what the monitor reads.
        vertical = vertical * (weight / total)
    return vertical, load_z, tens


def share_from_geometry(drone_positions, attach_points, payload_mass,
                        tether_len=None):
    """Vertical newtons carried by each drone.

    With ``tether_len`` given this uses the slack-tether equilibrium above.
    Without it, it falls back to the least-norm force balance — every tether
    assumed taut — which is the right model for a rigid frame but understates
    imbalance badly for hanging lines.
    """
    pos = np.asarray(drone_positions, dtype=float)
    attach = np.asarray(attach_points, dtype=float)

    if tether_len is not None:
        offsets = attach - np.array([attach[:, 0].mean(), attach[:, 1].mean(), 0.0])
        return solve_load_equilibrium(pos, offsets, tether_len, payload_mass)[0]

    u = tether_unit_vectors(pos, attach)
    weight = np.array([0.0, 0.0, payload_mass * G])
    tensions, *_ = np.linalg.lstsq(u.T, weight, rcond=None)
    tensions = np.clip(tensions, 0.0, None)
    vertical = tensions * u[:, 2]
    total = vertical.sum()
    if total > 1e-9:
        vertical *= (payload_mass * G) / total
    return vertical


def feasibility(cfg_drone, cfg_lift, cfg_formation):
    """Can these six airframes lift this payload? Returns a report dict."""
    n = cfg_formation.n_drones
    _, _, alpha = tether_geometry(cfg_formation.radius_m,
                                  cfg_lift.attach_radius_m,
                                  cfg_formation.tether_len_m)
    per_drone_n = tension_per_drone(cfg_lift.payload_mass_kg, n, alpha)
    capacity_n = cfg_drone.payload_capacity_kg * G
    margin = capacity_n / per_drone_n if per_drone_n > 0 else float("inf")
    return {
        "n_drones": n,
        "tether_angle_deg": float(np.degrees(alpha)),
        "tension_per_drone_n": float(per_drone_n),
        "share_per_drone_kg": float(per_drone_n / G),
        "capacity_per_drone_kg": float(cfg_drone.payload_capacity_kg),
        "margin": float(margin),
        "required_margin": float(cfg_lift.safety_factor),
        "feasible": bool(margin >= cfg_lift.safety_factor),
        "max_payload_kg": float(n * capacity_n * np.cos(alpha)
                                / (G * cfg_lift.safety_factor)),
    }


class LiftPlanner:
    """Drives the lift state machine and the shared altitude setpoint.

    The ascent rate is set by the *lowest* drone, not by a clock: the setpoint
    only climbs while every drone is within tolerance of the current one.  A
    drone that falls behind therefore stalls the whole climb instead of being
    dragged by its tether.
    """

    def __init__(self, cfg, formation):
        self.cfg = cfg                  # SwarmConfig
        self.formation = formation
        self.phase = LiftPhase.IDLE
        self.alt_setpoint = 0.0
        self.t_in_phase = 0.0
        self.tension_frac = 0.0         # 0..1 ramp during TENSIONING
        self.abort_reason = None
        self.cruise_target = None
        self.imbalance_t = 0.0          # how long the load has been uneven
        self.load_z = 0.0               # solved height of the payload
        self.tensions = np.zeros(cfg.formation.n_drones)

    # ------------------------------------------------------------------ utils
    def _enter(self, phase):
        self.phase = phase
        self.t_in_phase = 0.0

    # Phases that must converge to something. Left out: ARMING and RELEASED
    # (fixed short waits), CRUISE (bounded by its own distance), and the
    # terminal states.
    _TIMED_PHASES = (LiftPhase.TAKEOFF, LiftPhase.FORMING,
                     LiftPhase.DESCEND_TO_LOAD, LiftPhase.TENSIONING,
                     LiftPhase.LIFTING, LiftPhase.LOWERING)

    def _all_at(self, positions, alt, tol):
        z = np.asarray(positions, dtype=float)[:, 2]
        return bool(np.all(np.abs(z - alt) <= tol))

    def load_mass_carried(self):
        """Effective payload mass hanging on the tethers right now."""
        return self.cfg.lift.payload_mass_kg * self.tension_frac

    def abort(self, reason):
        self.abort_reason = reason
        self._enter(LiftPhase.ABORT)

    # -------------------------------------------------------------- lifecycle
    def start(self, positions):
        """Bind slots and begin the mission."""
        report = feasibility(self.cfg.drone, self.cfg.lift, self.cfg.formation)
        if not report["feasible"]:
            self.abort("payload exceeds swarm capacity: margin "
                       f"{report['margin']:.2f} < {report['required_margin']:.2f}")
            return report
        self.formation.bind(positions)
        self._enter(LiftPhase.ARMING)
        return report

    def set_cruise_target(self, xy):
        self.cruise_target = np.asarray(xy, dtype=float)

    def current_shares(self, positions):
        """Vertical newtons carried by each drone for the current pose.

        The load hangs where the tethers actually put it — under the swarm's own
        centroid, one tether length down — not under the commanded centre.  That
        distinction matters during cruise: if all six drones lag the setpoint
        together the load lags with them and the shares stay even, so only a
        drone out of position *relative to the others* shows up as imbalance.
        """
        positions = np.asarray(positions, dtype=float)
        mass = self.load_mass_carried()
        if mass <= 0.0:
            return np.zeros(len(positions))
        offsets = hexagon_slots(self.cfg.lift.attach_radius_m,
                                yaw=self.formation.yaw,
                                n=self.cfg.formation.n_drones)
        shares, self.load_z, self.tensions = solve_load_equilibrium(
            positions, offsets, self.cfg.formation.tether_len_m, mass,
            stiffness=self.cfg.lift.tether_stiffness_n_per_m)
        return shares

    def imbalance_sensitivity(self):
        """Altitude error on one drone that trips the imbalance abort, in metres.

        Reported rather than assumed: it falls straight out of the tether spring
        rate and the payload, and it is the number that decides whether the
        abort threshold is a useful guard or a nuisance. If it comes out smaller
        than ``formation.slot_tolerance_m`` the swarm will abort on ordinary
        station-keeping error — soften the tethers or loosen the threshold.
        """
        n = self.cfg.formation.n_drones
        fair = self.cfg.lift.payload_mass_kg * G / n
        k = self.cfg.lift.tether_stiffness_n_per_m
        if k <= 0:
            return float("inf")
        # One drone rising by dz adds k*dz to its line and sheds it from the
        # others, so its share ratio is about 1 + (k*dz/fair) * (n-1)/n.
        return float(self.cfg.lift.max_share_imbalance * fair
                     / (k * (n - 1) / n))

    def check_safety(self, positions, dt=0.0):
        """Run every tick. Returns None or an abort reason string.

        The imbalance test only counts once the tethers are more than half
        loaded, and it has to stay tripped for ``imbalance_grace_s`` — a single
        gust tilting the ring is not a reason to drop a payload.
        """
        lift = self.cfg.lift
        airborne = np.asarray(positions, dtype=float)[:, 2].min() > 0.5
        if airborne and not self.formation.separation_ok(positions):
            return "min separation violated"

        loaded = self.phase in (LiftPhase.TENSIONING, LiftPhase.LIFTING,
                                LiftPhase.CRUISE, LiftPhase.LOWERING)
        if loaded and self.tension_frac >= 0.5:
            share = self.current_shares(positions)
            fair = share.sum() / len(share)
            worst = share.max() / fair if fair > 0 else 1.0
            if worst > 1.0 + lift.max_share_imbalance:
                self.imbalance_t += dt
                if self.imbalance_t >= lift.imbalance_grace_s:
                    return (f"load imbalance sustained: worst drone at "
                            f"{worst:.2f}x fair share")
            else:
                self.imbalance_t = 0.0
        else:
            self.imbalance_t = 0.0
        return None

    def step(self, dt, positions, formed):
        """Advance the state machine one control tick.

        ``formed`` is the swarm's own report that everyone is inside the slot
        tolerance — passed in rather than recomputed so the caller can use its
        own (possibly filtered) estimate.
        """
        lift = self.cfg.lift
        self.t_in_phase += dt
        z = np.asarray(positions, dtype=float)[:, 2]
        tol = self.cfg.formation.slot_tolerance_m

        if self.phase is LiftPhase.ABORT:
            self.alt_setpoint = max(0.0, self.alt_setpoint - lift.climb_rate_mps * dt)
            self.formation.centre[2] = self.alt_setpoint
            return self.phase

        reason = self.check_safety(positions, dt)
        if reason:
            self.abort(reason)
            self.formation.centre[2] = self.alt_setpoint
            return self.phase

        # A phase that cannot converge — too much wind to hold formation, a
        # tether that never comes taut — must end the mission, not hang in it.
        if (self.phase in self._TIMED_PHASES
                and self.t_in_phase > lift.phase_timeout_s):
            self.abort(f"phase '{self.phase.value}' timed out after "
                       f"{lift.phase_timeout_s:.0f}s without converging")
            self.formation.centre[2] = self.alt_setpoint
            return self.phase

        if self.phase is LiftPhase.ARMING:
            if self.t_in_phase >= 1.0:
                self.alt_setpoint = lift.hover_alt_m
                self._enter(LiftPhase.TAKEOFF)

        elif self.phase is LiftPhase.TAKEOFF:
            if self._all_at(positions, lift.hover_alt_m, tol):
                self._enter(LiftPhase.FORMING)

        elif self.phase is LiftPhase.FORMING:
            if formed and self.t_in_phase >= lift.settle_time_s:
                self.alt_setpoint = lift.pickup_alt_m
                self._enter(LiftPhase.DESCEND_TO_LOAD)

        elif self.phase is LiftPhase.DESCEND_TO_LOAD:
            if formed and self._all_at(positions, lift.pickup_alt_m, tol):
                self._enter(LiftPhase.TENSIONING)

        elif self.phase is LiftPhase.TENSIONING:
            # Ramp the tethers from slack to fully loaded over tension_ramp_s.
            self.tension_frac = min(1.0, self.t_in_phase / lift.tension_ramp_s)
            if self.tension_frac >= 1.0 and formed:
                self._enter(LiftPhase.LIFTING)

        elif self.phase is LiftPhase.LIFTING:
            # The setpoint only rises while the slowest drone keeps up.
            if z.min() >= self.alt_setpoint - tol:
                self.alt_setpoint = min(lift.lift_alt_m,
                                        self.alt_setpoint + lift.climb_rate_mps * dt)
            if self.alt_setpoint >= lift.lift_alt_m - 1e-3 and \
                    self._all_at(positions, lift.lift_alt_m, tol):
                self._enter(LiftPhase.CRUISE)

        elif self.phase is LiftPhase.CRUISE:
            if self.cruise_target is not None:
                here = self.formation.centre[:2]
                delta = self.cruise_target - here
                dist = float(np.linalg.norm(delta))
                step = min(self.cfg.drone.max_speed_mps * 0.5 * dt, dist)
                if dist > 1e-6:
                    self.formation.centre[:2] = here + delta / dist * step
                if dist <= tol:
                    self._enter(LiftPhase.LOWERING)
            elif self.t_in_phase >= lift.settle_time_s:
                self._enter(LiftPhase.LOWERING)

        elif self.phase is LiftPhase.LOWERING:
            # Ramp down at the same rate we came up — a step command here would
            # slacken the tethers and let the load swing.
            if z.max() <= self.alt_setpoint + tol:
                self.alt_setpoint = max(lift.pickup_alt_m,
                                        self.alt_setpoint - lift.climb_rate_mps * dt)
            if self._all_at(positions, lift.pickup_alt_m, tol):
                self.tension_frac = 0.0
                self._enter(LiftPhase.RELEASED)

        elif self.phase is LiftPhase.RELEASED:
            if self.t_in_phase >= 1.0:
                self.alt_setpoint = 0.0
                self._enter(LiftPhase.LANDING)

        # The virtual structure carries the commanded altitude, so every drone's
        # slot target and the shared setpoint always agree.
        self.formation.centre[2] = self.alt_setpoint
        return self.phase

    def status(self, positions):
        return {
            "phase": self.phase.value,
            "alt_setpoint": round(float(self.alt_setpoint), 3),
            "tension_frac": round(float(self.tension_frac), 3),
            "load_carried_kg": round(float(self.load_mass_carried()), 3),
            "shape_quality": round(self.formation.shape_quality(positions), 3),
            "abort_reason": self.abort_reason,
        }
