"""
Hexagonal formation: slot assignment and the distributed position controller.

Six drones hold the vertices of a regular hexagon centred on the payload.  Two
problems have to be solved:

  1. *Which* drone takes *which* vertex.  Solved exactly — with six agents there
     are only 720 permutations, so we minimise total travel by brute force and
     never need scipy.  A greedy fallback covers n > 8.
  2. Holding the shape while it moves.  Each drone runs the same local law using
     only its own state and the two ring neighbours it can hear, so there is no
     leader to lose.
"""
import warnings
from itertools import permutations

import numpy as np

from .geometry import (hexagon_slots, min_pair_distance, clamp_norm,
                       ring_neighbours, rot_z)


def assignment_cost(positions, slots, perm):
    """Total distance if drone i flies to slot perm[i]."""
    p = np.asarray(positions, dtype=float)
    s = np.asarray(slots, dtype=float)
    return float(np.linalg.norm(p - s[list(perm)], axis=1).sum())


def assign_slots(positions, slots):
    """Return ``perm`` where drone i is responsible for ``slots[perm[i]]``.

    Exact (minimum total travel) for n <= 8, greedy nearest-free above that.
    Minimising total travel also minimises the chance of two drones crossing
    paths on the way into formation.
    """
    positions = np.asarray(positions, dtype=float)
    slots = np.asarray(slots, dtype=float)
    n = len(positions)
    if n != len(slots):
        raise ValueError("need one slot per drone")

    if n <= 8:
        best, best_cost = None, float("inf")
        for perm in permutations(range(n)):
            c = assignment_cost(positions, slots, perm)
            if c < best_cost:
                best, best_cost = perm, c
        return list(best)

    cost = np.linalg.norm(positions[:, None, :] - slots[None, :, :], axis=-1)
    perm = [-1] * n
    taken = set()
    for i in np.argsort(cost.min(axis=1)):
        order = np.argsort(cost[i])
        for j in order:
            if j not in taken:
                perm[i] = int(j)
                taken.add(int(j))
                break
    return perm


class HexFormation:
    """Desired hexagon in world coordinates, plus the per-drone control law.

    The formation is a *virtual structure*: a centre, a yaw and a radius.  Every
    drone derives its own target from that shared description, so the shape is
    defined even if a drone temporarily hears nobody.
    """

    def __init__(self, cfg, centre=(0.0, 0.0, 0.0), yaw=None):
        self.cfg = cfg
        self.n = cfg.n_drones
        self.centre = np.asarray(centre, dtype=float)
        self.yaw = cfg.formation_yaw_rad if yaw is None else float(yaw)
        self.radius = cfg.radius_m
        self.perm = list(range(self.n))
        self.neighbours = ring_neighbours(self.n)
        self._integral = np.zeros((self.n, 3))
        if self.stability_margin() <= 0.0:
            warnings.warn(
                f"formation gains are not stable: k_formation="
                f"{cfg.k_formation} vs consensus load "
                f"{cfg.k_consensus * self.laplacian_lambda_max() / 2:.3f}. "
                "The alternating mode will not be corrected — lower "
                "k_consensus or raise k_formation.", RuntimeWarning)

    # ---------------------------------------------------------------- targets
    def slot_offsets(self):
        """Slot positions relative to the formation centre, yaw applied."""
        return hexagon_slots(self.radius, yaw=self.yaw, n=self.n)

    def targets(self):
        """World-frame target for every slot index."""
        return self.centre + self.slot_offsets()

    def target_for(self, drone_index):
        """World-frame target of the drone holding slot ``perm[drone_index]``."""
        return self.targets()[self.perm[drone_index]]

    def bind(self, positions):
        """Assign drones to slots from their current positions. Returns perm."""
        self.perm = assign_slots(positions, self.targets())
        self._integral[:] = 0.0        # slots changed; old error history is void
        return self.perm

    # ------------------------------------------------------------- stability
    def laplacian_lambda_max(self):
        """Largest eigenvalue of the ring graph Laplacian.

        For an even ring this is 4, reached by the *alternating* mode: drone 0
        pushed one way, drone 1 the other, all the way round.
        """
        k = self.n // 2
        return float(2.0 - 2.0 * np.cos(2.0 * np.pi * k / self.n))

    def stability_margin(self):
        """How much the formation gain out-pulls the consensus term. Must be > 0.

        This is the condition that makes the shape hold. The consensus term is
        ``-(k_c/|N|) * L * e``; on the alternating mode ``L`` multiplies the error
        by ``lambda_max = 4``, so with |N| = 2 the consensus term reaches
        ``-2 * k_c * e``.  Set ``k_consensus`` to half ``k_formation`` and the two
        cancel exactly: the swarm settles into a permanently mis-shaped hexagon
        with zero net command and no indication anything is wrong.  Keeping this
        margin positive is what stops that.
        """
        n_nb = max(len(self.neighbours[0]), 1)
        return float(self.cfg.k_formation
                     - self.cfg.k_consensus * self.laplacian_lambda_max() / n_nb)

    # ---------------------------------------------------------------- metrics
    def errors(self, positions):
        """Per-drone distance to its assigned slot."""
        tgt = self.targets()[self.perm]
        return np.linalg.norm(np.asarray(positions, dtype=float) - tgt, axis=1)

    def is_formed(self, positions):
        return bool(np.all(self.errors(positions) <= self.cfg.slot_tolerance_m))

    def shape_quality(self, positions):
        """0..1 score: 1.0 is a perfect hexagon of the commanded radius.

        Reported in telemetry so the ground station sees formation health as one
        number instead of six errors.
        """
        err = self.errors(positions)
        scale = max(self.cfg.slot_tolerance_m, 1e-6)
        return float(np.exp(-np.mean(err) / (4.0 * scale)))

    def separation_ok(self, positions):
        return min_pair_distance(positions) >= self.cfg.min_separation_m

    # ---------------------------------------------------------------- control
    def command(self, i, positions, velocities, alt_setpoint=None, dt=None):
        """Acceleration command for drone ``i``.

        Four terms, all computable from data drone i actually has:
          * formation  — proportional pull toward its own slot;
          * consensus  — matches its formation error to its ring neighbours', so
            a drone lagging behind drags the others back rather than being left;
          * integral   — without it a steady crosswind leaves a permanent offset,
            because a pure proportional law needs a standing error to produce the
            standing force that balances the wind. Clamped against windup;
          * damping    — on its own velocity.
        Altitude gets an extra sync term because during a tethered lift a single
        high drone takes a disproportionate share of the load.

        Pass ``dt`` to advance the integrator; omit it for a pure feedback query.
        """
        cfg = self.cfg
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        tgt = self.targets()[self.perm]

        err_i = tgt[i] - positions[i]
        acc = cfg.k_formation * err_i

        for j in self.neighbours[i]:
            err_j = tgt[j] - positions[j]
            acc += cfg.k_consensus * (err_j - err_i) / len(self.neighbours[i])

        if dt and cfg.k_integral:
            self._integral[i] = clamp_norm(self._integral[i] + err_i * dt,
                                           cfg.integral_limit)
            acc += cfg.k_integral * self._integral[i]

        acc += self.avoidance(i, positions)
        acc -= cfg.k_damping * velocities[i]

        if alt_setpoint is not None:
            acc[2] += cfg.k_altitude_sync * (alt_setpoint - positions[i][2])

        return acc

    def avoidance(self, i, positions):
        """Short-range repulsion from every other drone.

        Slot assignment keeps the steady state collision-free, but the flight
        *into* formation can still bring two drones close; this term pushes them
        apart before the separation monitor has to abort the mission.
        """
        cfg = self.cfg
        positions = np.asarray(positions, dtype=float)
        acc = np.zeros(3)
        for j in range(len(positions)):
            if j == i:
                continue
            d = positions[i] - positions[j]
            dist = float(np.linalg.norm(d))
            if 1e-6 < dist < cfg.avoid_radius_m:
                acc += cfg.k_avoid * (cfg.avoid_radius_m / dist - 1.0) * (d / dist)
        return acc

    def commands(self, positions, velocities, alt_setpoint=None, dt=None):
        """Stacked (n,3) accelerations for the whole swarm."""
        return np.stack([self.command(i, positions, velocities, alt_setpoint, dt)
                         for i in range(self.n)])

    # ------------------------------------------------------------- reshaping
    def scale_to(self, radius):
        """Expand or shrink the hexagon (e.g. to clear an obstacle)."""
        self.radius = float(radius)

    def rotate_to(self, yaw):
        self.yaw = float(yaw)

    def move_to(self, centre):
        self.centre = np.asarray(centre, dtype=float)

    def attach_points(self, attach_radius):
        """Where each tether meets the load, directly under its drone's slot."""
        ring = hexagon_slots(attach_radius, yaw=self.yaw, n=self.n)
        return self.centre + ring
