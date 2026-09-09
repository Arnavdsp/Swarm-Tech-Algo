"""
Swarm coordinator: runs the hexagon, the lift state machine and the mesh
bookkeeping for all six agents.

On hardware each Pi runs *its own* copy with ``local_id`` set, using neighbour
telemetry from the mesh in place of ground truth.  In the simulator one
coordinator drives all six agents directly.  The control law is identical either
way, which is the whole point of keeping it decentralised.
"""
import numpy as np

from .formation import HexFormation
from .load_lift import LiftPlanner, LiftPhase


class SwarmCoordinator:
    def __init__(self, cfg, agents, centre=(0.0, 0.0, 0.0), local_id=None):
        self.cfg = cfg
        self.agents = list(agents)
        self.local_id = local_id
        self.formation = HexFormation(cfg.formation, centre=centre)
        self.planner = LiftPlanner(cfg, self.formation)
        self.t = 0.0
        self.history = []

    # ------------------------------------------------------------------ state
    def positions(self):
        return np.stack([a.position for a in self.agents])

    def velocities(self):
        return np.stack([a.velocity for a in self.agents])

    def healthy_agents(self):
        return [a for a in self.agents if a.healthy and a.battery_ok()]

    # ---------------------------------------------------------------- mission
    def start_lift(self, cruise_target=None):
        """Bind slots, check feasibility, and arm. Returns the feasibility report."""
        report = self.planner.start(self.positions())
        for i, agent in enumerate(self.agents):
            agent.slot = self.formation.perm[i]
            agent.arm()
        if cruise_target is not None:
            self.planner.set_cruise_target(cruise_target)
        return report

    def load_shares(self):
        """Vertical newtons each drone is currently carrying."""
        return self.planner.current_shares(self.positions())

    def tick(self, dt):
        """One control cycle for the whole swarm."""
        pos, vel = self.positions(), self.velocities()

        # Any drone that dropped off the mesh or below reserve is a lift risk.
        down = [a.id for a in self.agents if not (a.healthy and a.battery_ok())]
        if down and self.planner.phase not in (LiftPhase.IDLE, LiftPhase.ABORT,
                                               LiftPhase.LANDING):
            self.planner.abort(f"drone(s) {down} unavailable mid-lift")

        formed = self.formation.is_formed(pos)
        phase = self.planner.step(dt, pos, formed)

        acc = self.formation.commands(pos, vel,
                                      alt_setpoint=self.planner.alt_setpoint,
                                      dt=dt)
        shares = self.load_shares()

        for i, agent in enumerate(self.agents):
            agent.apply(acc[i], dt, load_share_n=float(shares[i]))
            agent.broadcast()
            agent.receive()

        self.t += dt
        self.history.append({
            "t": round(self.t, 3),
            "phase": phase.value,
            "positions": self.positions().tolist(),
            "alt_setpoint": float(self.planner.alt_setpoint),
            "shape_quality": self.formation.shape_quality(self.positions()),
            "shares_n": shares.tolist(),
        })
        return phase

    def run(self, duration_s, dt=None, on_step=None):
        """Fly the mission until it finishes or ``duration_s`` elapses."""
        dt = dt or 1.0 / self.cfg.formation.control_hz
        steps = int(duration_s / dt)
        for _ in range(steps):
            phase = self.tick(dt)
            if on_step:
                on_step(self)
            if phase in (LiftPhase.LANDING, LiftPhase.ABORT) and \
                    self.positions()[:, 2].max() < 0.05:
                break
        return self.planner.phase

    # -------------------------------------------------------------- reporting
    def status(self):
        pos = self.positions()
        st = self.planner.status(pos)
        st.update({
            "t": round(self.t, 2),
            "min_separation_m": round(float(np.min(
                [np.linalg.norm(pos[i] - pos[j])
                 for i in range(len(pos)) for j in range(i + 1, len(pos))]
            )) if len(pos) > 1 else 0.0, 3),
            "mean_alt_m": round(float(pos[:, 2].mean()), 3),
            "alt_spread_m": round(float(np.ptp(pos[:, 2])), 3),
            "batteries": [round(a.battery_frac, 3) for a in self.agents],
        })
        return st
