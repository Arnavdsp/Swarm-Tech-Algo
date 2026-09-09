#!/usr/bin/env python3
"""
Simulate the six-drone hexagonal load lift end to end.

    python scripts/run_sim.py
    python scripts/run_sim.py --payload 5.5 --radius 2.4 --plot out.png
    python scripts/run_sim.py --fail-drone 3 --fail-at 20   # watch it abort

Prints the phase timeline and, with --plot, writes a four-panel figure:
formation geometry, altitude tracking, per-drone load share and shape quality.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_drone import SwarmConfig, DroneAgent, SwarmCoordinator      # noqa: E402
from swarm_drone.backends import SimBackend                            # noqa: E402
from swarm_drone.comms import LoopbackLink                             # noqa: E402
from swarm_drone.load_lift import feasibility                          # noqa: E402


def build_swarm(cfg, seed=0, spread=4.0):
    """Six drones scattered on the ground around the pickup point."""
    rng = np.random.default_rng(seed)
    bus, agents = [], []
    for i in range(cfg.formation.n_drones):
        angle = 2 * np.pi * i / cfg.formation.n_drones + rng.uniform(-0.35, 0.35)
        radius = rng.uniform(spread * 0.7, spread)
        start = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        backend = SimBackend(cfg.drone, position=start, seed=seed * 10 + i)
        agents.append(DroneAgent(i, cfg, backend,
                                 link=LoopbackLink(i, cfg.comms, bus)))
    return agents


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", type=float, help="payload mass, kg")
    ap.add_argument("--radius", type=float, help="hexagon circumradius, m")
    ap.add_argument("--tether", type=float, help="tether length, m")
    ap.add_argument("--lift-alt", type=float, help="target lift altitude, m")
    ap.add_argument("--cruise", type=float, nargs=2, metavar=("X", "Y"),
                    default=[8.0, 0.0], help="carry the load to this XY")
    ap.add_argument("--wind", type=float, default=0.0,
                    help="steady wind acceleration, m/s^2 along +x")
    ap.add_argument("--fail-drone", type=int, help="fail this drone mid-flight")
    ap.add_argument("--fail-at", type=float, default=20.0, help="failure time, s")
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", help="write a summary figure to this path")
    ap.add_argument("--json", help="dump the flight log to this path")
    args = ap.parse_args()

    cfg = SwarmConfig()
    if args.payload:
        cfg.lift.payload_mass_kg = args.payload
    if args.radius:
        cfg.formation.radius_m = args.radius
    if args.tether:
        cfg.formation.tether_len_m = args.tether
    if args.lift_alt:
        cfg.lift.lift_alt_m = args.lift_alt

    report = feasibility(cfg.drone, cfg.lift, cfg.formation)
    print("── Lift feasibility ──────────────────────────────────────────")
    for k, v in report.items():
        print(f"  {k:24s} {round(v, 3) if isinstance(v, float) else v}")
    print()

    agents = build_swarm(cfg, seed=args.seed)
    if args.wind:
        for a in agents:
            a.backend.set_wind(np.array([args.wind, 0.0, 0.0]))

    swarm = SwarmCoordinator(cfg, agents, centre=(0.0, 0.0, 0.0))
    swarm.start_lift(cruise_target=args.cruise)
    if not report["feasible"]:
        print("Swarm refused the lift — payload beyond capacity.")
        return 1

    dt = 1.0 / cfg.formation.control_hz
    last_phase = None
    print("── Flight ────────────────────────────────────────────────────")
    for step in range(int(args.duration / dt)):
        t = step * dt
        if args.fail_drone is not None and abs(t - args.fail_at) < dt / 2:
            agents[args.fail_drone].fail("simulated motor failure")
            print(f"  t={t:6.2f}  !! drone {args.fail_drone} failed")

        phase = swarm.tick(dt)
        if phase.value != last_phase:
            pos = swarm.positions()
            print(f"  t={swarm.t:6.2f}  {phase.value:<10s} "
                  f"alt_sp={swarm.planner.alt_setpoint:5.2f}m  "
                  f"mean_alt={pos[:, 2].mean():5.2f}m  "
                  f"shape={swarm.formation.shape_quality(pos):.2f}")
            last_phase = phase.value
        if phase.value in ("landing", "abort") and swarm.positions()[:, 2].max() < 0.05:
            break

    print("\n── Final status ──────────────────────────────────────────────")
    print(json.dumps(swarm.status(), indent=2))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"config": cfg.to_dict(), "feasibility": report,
                       "history": swarm.history}, fh, indent=1)
        print(f"\nFlight log → {args.json}")

    if args.plot:
        plot_summary(swarm, cfg, args.plot)
        print(f"Figure     → {args.plot}")
    return 0


def plot_summary(swarm, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = swarm.history
    t = np.array([h["t"] for h in hist])
    pos = np.array([h["positions"] for h in hist])       # (T, n, 3)
    shares = np.array([h["shares_n"] for h in hist])
    quality = np.array([h["shape_quality"] for h in hist])
    alt_sp = np.array([h["alt_setpoint"] for h in hist])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("#0F1117")
    for ax in axes.flat:
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="white")
        ax.spines[["top", "right", "left", "bottom"]].set_color("#333644")
        ax.grid(True, color="#2A2D3A", linewidth=0.7, linestyle="--")

    # Top-left: ground track and the final hexagon.
    ax = axes[0][0]
    for i in range(pos.shape[1]):
        ax.plot(pos[:, i, 0], pos[:, i, 1], linewidth=1.2, alpha=0.85,
                label=f"drone {i}")
    final = pos[-1]
    hexa = np.vstack([final[:, :2], final[:1, :2]])
    ax.plot(hexa[:, 0], hexa[:, 1], "--", color="#4CE87A", linewidth=1.6,
            label="final hexagon")
    ax.scatter(final[:, 0], final[:, 1], color="#4CE87A", zorder=5, s=28)
    ax.set_title("Ground track & final formation", color="white", fontweight="bold")
    ax.set_xlabel("x (m)", color="white"); ax.set_ylabel("y (m)", color="white")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, facecolor="#1A1D27", labelcolor="white",
              edgecolor="#333644", ncol=2)

    # Top-right: altitude vs the shared setpoint.
    ax = axes[0][1]
    for i in range(pos.shape[1]):
        ax.plot(t, pos[:, i, 2], linewidth=1.1, alpha=0.85)
    ax.plot(t, alt_sp, "--", color="#E8734C", linewidth=2, label="setpoint")
    ax.set_title("Altitude tracking", color="white", fontweight="bold")
    ax.set_xlabel("t (s)", color="white"); ax.set_ylabel("z (m)", color="white")
    ax.legend(fontsize=8, facecolor="#1A1D27", labelcolor="white",
              edgecolor="#333644")

    # Bottom-left: load share as a ratio of the fair share. Plotting the ratio
    # rather than newtons is the point — six balanced curves sit on 1.0 and any
    # imbalance is visible against the abort threshold instead of hidden by the
    # weight of the payload.
    ax = axes[1][0]
    fair = shares.sum(axis=1, keepdims=True) / max(shares.shape[1], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(fair > 1e-6, shares / np.maximum(fair, 1e-6), np.nan)
    for i in range(shares.shape[1]):
        ax.plot(t, ratio[:, i], linewidth=1.2, alpha=0.9, label=f"drone {i}")
    ax.axhline(1.0, color="#E8C44C", linestyle="--", linewidth=1.5,
               label="fair share")
    ax.axhline(1.0 + cfg.lift.max_share_imbalance, color="#E8734C",
               linestyle=":", linewidth=1.5, label="abort threshold")
    ax.set_ylim(0.5, 1.6)
    ax.set_title(f"Tether load balance (fair share = "
                 f"{np.nanmax(fair):.1f} N/drone)",
                 color="white", fontweight="bold")
    ax.set_xlabel("t (s)", color="white")
    ax.set_ylabel("share / fair share", color="white")
    ax.legend(fontsize=7, facecolor="#1A1D27", labelcolor="white",
              edgecolor="#333644", ncol=2)

    # Bottom-right: formation quality with phase changes marked.
    ax = axes[1][1]
    ax.plot(t, quality, color="#4C9BE8", linewidth=1.6)
    ax.fill_between(t, quality, alpha=0.15, color="#4C9BE8")
    seen, last = set(), None
    for h in hist:
        if h["phase"] != last:
            ax.axvline(h["t"], color="#666B80", linewidth=0.8, linestyle=":")
            if h["phase"] not in seen:
                ax.text(h["t"], 1.02, h["phase"], rotation=90, fontsize=6,
                        color="#AAB0C0", va="bottom")
                seen.add(h["phase"])
            last = h["phase"]
    ax.set_ylim(0, 1.05)
    ax.set_title("Hexagon shape quality", color="white", fontweight="bold")
    ax.set_xlabel("t (s)", color="white"); ax.set_ylabel("quality (0-1)",
                                                         color="white")

    plt.suptitle(f"6-drone hexagonal lift — {cfg.lift.payload_mass_kg} kg payload, "
                 f"{cfg.formation.radius_m} m hexagon",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=140, facecolor="#0F1117", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
