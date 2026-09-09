"""Hexagon geometry, slot assignment and the distributed controller."""
import numpy as np
import pytest

from swarm_drone.config import FormationConfig, SwarmConfig
from swarm_drone.formation import HexFormation, assign_slots, assignment_cost
from swarm_drone.geometry import (hexagon_slots, min_pair_distance,
                                  ring_neighbours, clamp_norm, wrap_angle)


def test_hexagon_is_regular():
    """All six edges equal the circumradius — that is what makes it regular."""
    slots = hexagon_slots(2.0)
    assert slots.shape == (6, 3)
    edges = [np.linalg.norm(slots[i] - slots[(i + 1) % 6]) for i in range(6)]
    assert np.allclose(edges, 2.0)
    assert np.allclose(np.linalg.norm(slots[:, :2], axis=1), 2.0)
    assert np.allclose(slots.mean(axis=0), 0.0, atol=1e-9)


def test_hexagon_yaw_rotates_without_deforming():
    a = hexagon_slots(1.5)
    b = hexagon_slots(1.5, yaw=np.pi / 7)
    assert np.isclose(min_pair_distance(a), min_pair_distance(b))


def test_assignment_is_optimal_and_a_permutation():
    slots = hexagon_slots(2.0)
    positions = slots[[3, 1, 5, 0, 4, 2]] + 0.1
    perm = assign_slots(positions, slots)
    assert sorted(perm) == list(range(6))
    # Nothing beats it: check against every alternative.
    from itertools import permutations
    best = min(assignment_cost(positions, slots, p)
               for p in permutations(range(6)))
    assert np.isclose(assignment_cost(positions, slots, perm), best)


def test_assignment_handles_large_n_greedily():
    rng = np.random.default_rng(0)
    slots = hexagon_slots(5.0, n=12)
    positions = slots + rng.normal(scale=0.2, size=slots.shape)
    perm = assign_slots(positions, slots)
    assert sorted(perm) == list(range(12))


def test_stability_margin_positive_by_default():
    """The shipped gains must satisfy the condition the controller relies on."""
    f = HexFormation(FormationConfig())
    assert f.laplacian_lambda_max() == pytest.approx(4.0)
    assert f.stability_margin() > 0


def test_marginal_gains_are_warned_about():
    """k_consensus == k_formation/2 cancels the alternating mode exactly."""
    cfg = FormationConfig(k_formation=1.4, k_consensus=0.7)
    with pytest.warns(RuntimeWarning, match="not stable"):
        HexFormation(cfg)


def test_controller_converges_to_the_hexagon():
    cfg = FormationConfig()
    f = HexFormation(cfg, centre=(0.0, 0.0, 3.0))
    rng = np.random.default_rng(2)
    pos = f.targets() + rng.normal(scale=0.8, size=(6, 3))
    vel = np.zeros((6, 3))
    f.bind(pos)
    dt = 0.05
    for _ in range(2000):
        acc = f.commands(pos, vel, alt_setpoint=3.0, dt=dt)
        vel += (acc - 0.35 * vel) * dt
        pos = pos + vel * dt
    assert f.is_formed(pos)
    assert f.shape_quality(pos) > 0.95


def test_alternating_error_is_actually_corrected():
    """The exact failure the stability margin exists to prevent."""
    cfg = FormationConfig()
    f = HexFormation(cfg, centre=(0.0, 0.0, 3.0))
    pos = f.targets().copy()
    pos[::2, 0] += 0.4          # every other drone pushed the same way
    pos[1::2, 0] -= 0.4
    vel = np.zeros((6, 3))
    dt = 0.05
    for _ in range(2000):
        acc = f.commands(pos, vel, alt_setpoint=3.0, dt=dt)
        vel += (acc - 0.35 * vel) * dt
        pos = pos + vel * dt
    assert f.errors(pos).max() < 0.05


def test_integral_rejects_a_steady_disturbance():
    """A pure P controller would settle with a permanent offset here."""
    cfg = FormationConfig()
    f = HexFormation(cfg, centre=(0.0, 0.0, 3.0))
    pos = f.targets().copy()
    vel = np.zeros((6, 3))
    wind = np.array([0.8, 0.0, 0.0])
    dt = 0.05
    for _ in range(4000):
        acc = f.commands(pos, vel, alt_setpoint=3.0, dt=dt)
        vel += (acc - 0.35 * vel + wind) * dt
        pos = pos + vel * dt
    assert f.errors(pos).max() < cfg.slot_tolerance_m


def test_avoidance_pushes_close_drones_apart():
    cfg = FormationConfig()
    f = HexFormation(cfg, centre=(0.0, 0.0, 3.0))
    pos = f.targets().copy()
    pos[1] = pos[0] + np.array([0.3, 0.0, 0.0])     # nearly on top of drone 0
    push = f.avoidance(0, pos)
    assert push[0] < 0                               # drone 0 shoved away in -x
    assert np.linalg.norm(push) > 1.0


def test_scaling_and_rotating_keep_the_shape():
    f = HexFormation(FormationConfig())
    f.scale_to(3.0)
    assert np.isclose(min_pair_distance(f.targets()), 3.0)
    f.rotate_to(0.5)
    assert np.isclose(min_pair_distance(f.targets()), 3.0)


def test_ring_neighbours_wrap():
    nb = ring_neighbours(6)
    assert nb[0] == (5, 1)
    assert nb[5] == (4, 0)


def test_clamp_norm_and_wrap_angle():
    assert np.isclose(np.linalg.norm(clamp_norm([3.0, 4.0], 2.5)), 2.5)
    assert np.allclose(clamp_norm([0.3, 0.4], 2.5), [0.3, 0.4])
    assert np.isclose(wrap_angle(3 * np.pi), -np.pi)      # half-open [-pi, pi)
    assert np.isclose(wrap_angle(0.5), 0.5)
    assert np.isclose(wrap_angle(2 * np.pi + 0.3), 0.3)
