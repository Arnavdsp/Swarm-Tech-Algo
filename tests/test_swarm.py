"""Agents, the mesh link, and full missions end to end."""
import numpy as np
import pytest

from swarm_drone import SwarmConfig, DroneAgent, SwarmCoordinator
from swarm_drone.backends import SimBackend
from swarm_drone.comms import LoopbackLink
from swarm_drone.config import DroneSpec
from swarm_drone.geometry import hexagon_slots
from swarm_drone.load_lift import LiftPhase
from swarm_drone.vision.alerts import Alert


def build(cfg=None, seed=0, spread=4.0, vision=None):
    cfg = cfg or SwarmConfig()
    rng = np.random.default_rng(seed)
    bus, agents = [], []
    for i in range(cfg.formation.n_drones):
        angle = 2 * np.pi * i / cfg.formation.n_drones + rng.uniform(-0.3, 0.3)
        r = rng.uniform(spread * 0.7, spread)
        start = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
        agents.append(DroneAgent(i, cfg, SimBackend(cfg.drone, position=start,
                                                    seed=seed * 10 + i),
                                 link=LoopbackLink(i, cfg.comms, bus),
                                 vision=vision))
    return cfg, SwarmCoordinator(cfg, agents, centre=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------- capability
def test_payload_capacity_accounts_for_tilt_and_margin():
    spec = DroneSpec()
    assert 0.5 < spec.payload_capacity_kg < spec.max_thrust_n / 9.80665
    # A heavier airframe on the same motors can carry less.
    assert DroneSpec(mass_kg=2.5).payload_capacity_kg < spec.payload_capacity_kg


def test_backend_clamps_to_the_flight_envelope():
    spec = DroneSpec()
    backend = SimBackend(spec, position=(0.0, 0.0, 5.0))
    backend.arm()
    for _ in range(200):
        backend.send_acceleration([100.0, 100.0, 100.0], dt=0.05)
    _, vel = backend.state()
    assert np.linalg.norm(vel[:2]) <= spec.max_speed_mps + 1e-6
    assert vel[2] <= spec.max_climb_mps + 1e-6


def test_a_hanging_load_eats_climb_authority():
    spec = DroneSpec()
    backend = SimBackend(spec)
    free = backend.available_accel()
    backend.set_load_share(20.0)
    assert backend.available_accel() < free


def test_a_disarmed_drone_ignores_commands():
    backend = SimBackend(DroneSpec(), position=(1.0, 2.0, 3.0))
    backend.send_acceleration([5.0, 5.0, 5.0], dt=0.1)
    assert np.allclose(backend.state()[0], [1.0, 2.0, 3.0])


def test_battery_drains_monotonically():
    backend = SimBackend(DroneSpec())
    backend.arm()
    levels = []
    for _ in range(50):
        backend.send_acceleration([0.0, 0.0, 0.5], dt=0.1)
        levels.append(backend.battery_frac)
    assert levels == sorted(levels, reverse=True)
    assert levels[-1] < 1.0


# --------------------------------------------------------------------- comms
def test_loopback_peers_see_each_other():
    cfg = SwarmConfig()
    bus = []
    links = [LoopbackLink(i, cfg.comms, bus) for i in range(3)]
    for i, link in enumerate(links):
        link.send({"pos": [i, i, i]})
    for link in links:
        link.poll()
    assert sorted(links[0].peers()) == [1, 2]
    assert links[0].lost_peers(range(5)) == [3, 4]


def test_a_drone_never_lists_itself_as_a_peer():
    cfg = SwarmConfig()
    bus = []
    a, b = LoopbackLink(0, cfg.comms, bus), LoopbackLink(1, cfg.comms, bus)
    a.send({"x": 1}); b.send({"x": 2})
    a.poll(); b.poll()
    assert 0 not in a.peers()


def test_telemetry_is_json_ready():
    import json
    cfg = SwarmConfig()
    agent = DroneAgent(2, cfg, SimBackend(cfg.drone, position=(1.0, 2.0, 3.0)))
    payload = agent.telemetry()
    assert payload["id"] == 2 and payload["ok"] is True
    json.dumps(payload)


# ------------------------------------------------------------------ missions
def test_a_full_lift_mission_completes():
    cfg, swarm = build()
    report = swarm.start_lift(cruise_target=(8.0, 0.0))
    assert report["feasible"]
    phase = swarm.run(120.0)
    assert phase is LiftPhase.LANDING
    assert swarm.planner.abort_reason is None
    assert swarm.positions()[:, 2].max() < 0.2


def test_the_hexagon_is_actually_held_while_carrying():
    cfg, swarm = build()
    swarm.start_lift(cruise_target=(8.0, 0.0))
    qualities = []
    for _ in range(int(60 / 0.05)):
        phase = swarm.tick(0.05)
        if phase in (LiftPhase.LIFTING, LiftPhase.CRUISE):
            qualities.append(swarm.formation.shape_quality(swarm.positions()))
        if phase in (LiftPhase.LANDING, LiftPhase.ABORT):
            break
    assert qualities, "never reached a carrying phase"
    assert np.mean(qualities) > 0.75


def test_the_load_stays_balanced_once_the_tethers_are_loaded():
    """The balance guarantee applies from half-tension onward, not before.

    Early in the tension ramp the tethers are still taking up slack under a
    fraction of the payload, and the share ratio swings hard. That is harmless
    and is exactly why check_safety waits for tension_frac >= 0.5 — this test
    pins both halves of that behaviour so the guard cannot be silently widened.
    """
    cfg, swarm = build()
    swarm.start_lift(cruise_target=(8.0, 0.0))
    worst_loaded, worst_ramp = 1.0, 1.0
    for _ in range(int(80 / 0.05)):
        phase = swarm.tick(0.05)
        shares = swarm.load_shares()
        if shares.sum() > 1.0:
            ratio = shares.max() / shares.mean()
            if swarm.planner.tension_frac >= 0.5:
                worst_loaded = max(worst_loaded, ratio)
            else:
                worst_ramp = max(worst_ramp, ratio)
        if phase in (LiftPhase.LANDING, LiftPhase.ABORT):
            break
    assert swarm.planner.abort_reason is None
    assert worst_loaded < 1.0 + cfg.lift.max_share_imbalance
    assert worst_ramp > worst_loaded          # the ramp really is the noisy part


def test_separation_is_maintained_throughout():
    cfg, swarm = build(seed=3)
    swarm.start_lift(cruise_target=(6.0, 3.0))
    for _ in range(int(60 / 0.05)):
        phase = swarm.tick(0.05)
        pos = swarm.positions()
        if pos[:, 2].min() > 0.5:
            assert swarm.formation.separation_ok(pos)
        if phase in (LiftPhase.LANDING, LiftPhase.ABORT):
            break


def test_a_drone_failing_mid_lift_aborts_the_mission():
    cfg, swarm = build()
    swarm.start_lift()
    for step in range(int(60 / 0.05)):
        if step == int(15 / 0.05):
            swarm.agents[2].fail("motor failure")
        if swarm.tick(0.05) is LiftPhase.ABORT:
            break
    assert swarm.planner.phase is LiftPhase.ABORT
    assert "2" in swarm.planner.abort_reason


def test_a_flat_battery_counts_as_unavailable():
    cfg, swarm = build()
    swarm.start_lift()
    for _ in range(int(12 / 0.05)):
        swarm.tick(0.05)
    swarm.agents[4].backend.battery_frac = 0.05
    assert swarm.agents[4] not in swarm.healthy_agents()
    swarm.tick(0.05)
    assert swarm.planner.phase is LiftPhase.ABORT


def test_status_is_serialisable_and_complete():
    import json
    cfg, swarm = build()
    swarm.start_lift()
    swarm.run(10.0)
    status = swarm.status()
    for key in ("phase", "shape_quality", "min_separation_m", "batteries"):
        assert key in status
    json.dumps(status)


def test_history_records_every_tick():
    cfg, swarm = build()
    swarm.start_lift()
    for _ in range(20):
        swarm.tick(0.05)
    assert len(swarm.history) == 20
    assert len(swarm.history[0]["positions"]) == cfg.formation.n_drones


# ----------------------------------------------------------- vision + swarm
class _VisionStub:
    """Raises one alert on the third frame, like a real detection would."""

    def __init__(self):
        self.calls = 0

    def process(self, frame, drone_id, timestamp):
        self.calls += 1
        if self.calls == 3:
            return [Alert(person_id="W-7", name="Seven", score=0.91, votes=3,
                          drone_id=drone_id, track_id=1, frame_idx=self.calls,
                          timestamp=timestamp)]
        return []


def test_a_sighting_is_broadcast_to_the_whole_mesh():
    """One drone sees someone; every other drone hears about it."""
    cfg = SwarmConfig()
    bus = []
    agents = [DroneAgent(i, cfg, SimBackend(cfg.drone),
                         link=LoopbackLink(i, cfg.comms, bus),
                         vision=_VisionStub() if i == 3 else None)
              for i in range(6)]
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(3):
        agents[3].process_frame(frame, timestamp=1000.0)

    assert len(agents[3].sightings) == 1
    for other in (a for a in agents if a.id != 3):
        received = [m for m in other.receive() if m.get("kind") == "sighting"]
        assert len(received) == 1
        assert received[0]["alert"]["person_id"] == "W-7"
        assert received[0]["id"] == 3


def test_a_drone_without_a_camera_just_returns_nothing():
    cfg = SwarmConfig()
    agent = DroneAgent(0, cfg, SimBackend(cfg.drone))
    assert agent.process_frame(np.zeros((8, 8, 3), dtype=np.uint8)) == []


# --------------------------------------------------------------------- config
def test_config_round_trips_through_disk(tmp_path):
    cfg = SwarmConfig()
    cfg.lift.payload_mass_kg = 6.25
    path = str(tmp_path / "swarm.json")
    cfg.save(path)
    assert SwarmConfig.from_yaml(path).lift.payload_mass_kg == 6.25


def test_unknown_config_keys_are_ignored():
    cfg = SwarmConfig.from_dict({"lift": {"payload_mass_kg": 3.0, "nonsense": 1},
                                 "formation": {"radius_m": 2.5}})
    assert cfg.lift.payload_mass_kg == 3.0
    assert cfg.formation.radius_m == 2.5


def test_slot_offsets_match_the_attach_ring_orientation():
    cfg = SwarmConfig()
    from swarm_drone.formation import HexFormation
    f = HexFormation(cfg.formation, centre=(1.0, 2.0, 5.0), yaw=0.4)
    ring = f.attach_points(cfg.lift.attach_radius_m)
    expected = f.centre + hexagon_slots(cfg.lift.attach_radius_m, yaw=0.4)
    assert np.allclose(ring, expected)
