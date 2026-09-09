"""
Geometry helpers for the 6-drone hexagonal lifting formation.

Everything is in a local ENU frame (x = east, y = north, z = up) with the
payload centroid as the origin.  Angles are radians unless stated otherwise.
"""
import numpy as np

TAU = 2.0 * np.pi


def hexagon_slots(radius, yaw=0.0, n=6, z=0.0):
    """Return the (n,3) vertex positions of a regular n-gon.

    Slot k sits at angle ``yaw + k * 2*pi/n``.  With n=6 the vertices are the
    corners of a regular hexagon of circumradius ``radius`` — the formation the
    swarm holds while carrying a load.
    """
    k = np.arange(n)
    theta = yaw + k * (TAU / n)
    return np.stack([radius * np.cos(theta),
                     radius * np.sin(theta),
                     np.full(n, float(z))], axis=1)


def rot_z(yaw):
    """3x3 rotation about the vertical axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def slot_side_length(radius):
    """Edge length of a regular hexagon = its circumradius."""
    return float(radius)


def min_pair_distance(positions):
    """Smallest distance between any two rows of ``positions``."""
    p = np.asarray(positions, dtype=float)
    if len(p) < 2:
        return float("inf")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def clamp_norm(vec, max_norm):
    """Scale ``vec`` down so its L2 norm never exceeds ``max_norm``."""
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= max_norm or n == 0.0:
        return v
    return v * (max_norm / n)


def wrap_angle(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + np.pi) % TAU - np.pi


def ring_neighbours(n):
    """Neighbour map for a ring communication topology.

    Drone k talks to k-1 and k+1 (mod n).  This is the sparsest topology that
    still keeps the formation graph rigid enough for consensus to converge, and
    it is what the Pi mesh actually uses at range.
    """
    return {k: ((k - 1) % n, (k + 1) % n) for k in range(n)}


def tether_unit_vectors(drone_positions, attach_points):
    """Unit vectors pointing from each attach point up to its drone."""
    d = np.asarray(drone_positions, dtype=float) - np.asarray(attach_points, dtype=float)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return d / norms
