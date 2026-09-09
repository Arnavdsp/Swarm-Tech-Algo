"""Force sharing, feasibility gating and the lift state machine."""
import numpy as np
import pytest

from swarm_drone.config import SwarmConfig
from swarm_drone.formation import HexFormation
from swarm_drone.geometry import hexagon_slots
from swarm_drone.load_lift import (G, LiftPhase, LiftPlanner, feasibility,
                                   share_from_geometry, solve_load_equilibrium,
                                   tension_per_drone, tether_geometry)


def _offsets(cfg):
    return hexagon_slots(cfg.lift.attach_radius_m, n=cfg.formation.n_drones)


def _shares(cfg, positions, mass=4.0):
    return solve_load_equilibrium(
        positions, _offsets(cfg), cfg.formation.tether_len_m, mass,
        stiffness=cfg.lift.tether_stiffness_n_per_m)[0]


def test_tether_geometry_rejects_a_too_short_tether():
    with pytest.raises(ValueError, match="too short"):
        tether_geometry(hex_radius=3.0, attach_radius=0.4, tether_len=1.0)


def test_tension_grows_as_the_tether_angle_opens():
    straight = tension_per_drone(6.0, 6, alpha=0.0)
    splayed = tension_per_drone(6.0, 6, alpha=np.radians(40))
    assert straight == pytest.approx(6.0 * G / 6)
    assert splayed > straight


def test_a_level_hexagon_shares_the_load_evenly():
    cfg = SwarmConfig()
    f = HexFormation(cfg.formation, centre=np.array([0.0, 0.0, 6.0]))
    share = _shares(cfg, f.targets())
    assert share.sum() == pytest.approx(4.0 * G, rel=1e-6)
    assert np.allclose(share, share[0], rtol=1e-4)


def test_a_high_drone_takes_more_than_its_share():
    """The physical reason the lift has to keep the ring level."""
    cfg = SwarmConfig()
    f = HexFormation(cfg.formation, centre=np.array([0.0, 0.0, 6.0]))
    pos = f.targets()
    level = _shares(cfg, pos)
    tilted_pos = pos.copy()
    tilted_pos[0, 2] += 0.3
    tilted = _shares(cfg, tilted_pos)
    assert tilted[0] > level[0]
    assert tilted[0] == tilted.max()
    assert np.all(tilted[1:] < level[1:])        # the others are unloaded
    assert tilted.sum() == pytest.approx(4.0 * G, rel=1e-6)


def test_slack_tethers_carry_nothing():
    """A drone below the others has a slack line and takes no load at all."""
    cfg = SwarmConfig()
    f = HexFormation(cfg.formation, centre=np.array([0.0, 0.0, 6.0]))
    pos = f.targets()
    pos[2, 2] -= 1.5
    share = _shares(cfg, pos)
    assert share[2] == pytest.approx(0.0, abs=1e-6)
    assert share.sum() == pytest.approx(4.0 * G, rel=1e-6)


def test_least_norm_fallback_still_balances():
    """Without a tether length the rigid-frame model is used instead."""
    cfg = SwarmConfig()
    f = HexFormation(cfg.formation, centre=np.array([0.0, 0.0, 6.0]))
    pos = f.targets()
    attach = _offsets(cfg) + np.array([0.0, 0.0, 6.0 - cfg.formation.tether_len_m])
    share = share_from_geometry(pos, attach, 4.0)
    assert share.sum() == pytest.approx(4.0 * G, rel=1e-6)
    assert np.allclose(share, share[0], rtol=1e-6)


def test_imbalance_sensitivity_clears_the_slot_tolerance():
    """Otherwise ordinary station-keeping error would abort the lift."""
    cfg = SwarmConfig()
    planner = LiftPlanner(cfg, HexFormation(cfg.formation))
    assert planner.imbalance_sensitivity() > cfg.formation.slot_tolerance_m


def test_feasibility_gates_an_overweight_payload():
    cfg = SwarmConfig()
    assert feasibility(cfg.drone, cfg.lift, cfg.formation)["feasible"]
    cfg.lift.payload_mass_kg = 40.0
    report = feasibility(cfg.drone, cfg.lift, cfg.formation)
    assert not report["feasible"]
    assert report["margin"] < report["required_margin"]


def test_max_payload_is_self_consistent():
    """Loading exactly the reported maximum must still pass the gate."""
    cfg = SwarmConfig()
    cfg.lift.payload_mass_kg = feasibility(
        cfg.drone, cfg.lift, cfg.formation)["max_payload_kg"] * 0.999
    assert feasibility(cfg.drone, cfg.lift, cfg.formation)["feasible"]


def test_planner_refuses_to_start_an_impossible_lift():
    cfg = SwarmConfig()
    cfg.lift.payload_mass_kg = 50.0
    f = HexFormation(cfg.formation)
    planner = LiftPlanner(cfg, f)
    planner.start(f.targets())
    assert planner.phase is LiftPhase.ABORT
    assert "capacity" in planner.abort_reason


def test_climb_setpoint_waits_for_the_slowest_drone():
    cfg = SwarmConfig()
    f = HexFormation(cfg.formation, centre=np.array([0.0, 0.0, 1.2]))
    planner = LiftPlanner(cfg, f)
    planner.phase = LiftPhase.LIFTING
    planner.tension_frac = 1.0
    planner.alt_setpoint = 1.2

    lagging = f.targets()
    lagging[3, 2] -= 2.0                     # one drone well below the rest
    planner.step(0.05, lagging, formed=True)
    assert planner.alt_setpoint == pytest.approx(1.2)

    planner.step(0.05, f.targets(), formed=True)
    assert planner.alt_setpoint > 1.2


def test_sustained_imbalance_aborts_but_a_transient_does_not():
    cfg = SwarmConfig()
    f = HexFormation(cfg.formation, centre=np.array([0.0, 0.0, 4.0]))
    planner = LiftPlanner(cfg, f)
    planner.phase = LiftPhase.LIFTING
    planner.tension_frac = 1.0

    bad = f.targets()
    bad[0, 2] += 3 * planner.imbalance_sensitivity()
    assert planner.check_safety(bad, dt=0.2) is None      # one blip: tolerated
    for _ in range(20):
        reason = planner.check_safety(bad, dt=0.1)
    assert reason is not None and "imbalance" in reason

    # And a ring inside the sensitivity band never trips it, however long.
    planner.imbalance_t = 0.0
    ok = f.targets()
    ok[0, 2] += 0.5 * planner.imbalance_sensitivity()
    for _ in range(50):
        assert planner.check_safety(ok, dt=0.1) is None


def test_a_phase_that_cannot_converge_times_out():
    cfg = SwarmConfig()
    f = HexFormation(cfg.formation, centre=np.array([0.0, 0.0, 3.0]))
    planner = LiftPlanner(cfg, f)
    planner.phase = LiftPhase.FORMING
    stuck = f.targets() + np.array([2.0, 0.0, 0.0])       # never in tolerance
    for _ in range(int(cfg.lift.phase_timeout_s / 0.05) + 40):
        planner.step(0.05, stuck, formed=False)
    assert planner.phase is LiftPhase.ABORT
    assert "timed out" in planner.abort_reason


def test_no_load_means_no_share():
    cfg = SwarmConfig()
    f = HexFormation(cfg.formation)
    planner = LiftPlanner(cfg, f)
    assert np.allclose(planner.current_shares(f.targets()), 0.0)


def test_solved_load_hangs_below_the_swarm():
    cfg = SwarmConfig()
    f = HexFormation(cfg.formation, centre=np.array([0.0, 0.0, 6.0]))
    pos = f.targets()
    _, load_z, tensions = solve_load_equilibrium(
        pos, _offsets(cfg), cfg.formation.tether_len_m, 4.0,
        stiffness=cfg.lift.tether_stiffness_n_per_m)
    assert load_z < pos[:, 2].min() - 1.0
    assert np.all(tensions >= 0.0)
