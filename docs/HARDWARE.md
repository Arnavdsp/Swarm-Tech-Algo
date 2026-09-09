# Hardware and deployment

The six-drone build this stack was written for, and what to change if yours
differs.

---

## Per drone

| Part | What it does | Notes |
|---|---|---|
| Frame | 450–500 mm quad | Sized for the tether attach point below the CG |
| Flight controller | Pixhawk-class, ArduPilot or PX4 | Owns attitude; this stack only commands position/acceleration |
| Companion computer | Raspberry Pi 4 (4 GB) or Pi 5 | Runs `run_drone_node.py` |
| Camera | Pi Camera Module 3, or a USB UVC camera | 1280×720 is plenty; higher resolution costs fps for no detection gain at these object sizes |
| Radio | Wi-Fi (Pi's own, or an external adapter in ad-hoc/mesh mode) | Carries the 10 Hz JSON heartbeat |
| Tether | Dyneema line **plus a compliant element** | See below — this is not optional |
| Battery | 4S 5000 mAh (~77 Wh) | `reserve_frac` keeps 25 % unusable |

Pi ↔ flight controller is serial (`/dev/serial0`, 921600) or USB. Set
`--mavlink` to match.

### Numbers the defaults assume

```
mass_kg          1.45     airframe + Pi + camera, no payload share
max_thrust_n     32.0     all four motors, static, 100 %
max_tilt_rad     0.35     ~20°, deliberately tight while tethered
```

which gives ≈1.16 kg of spare lift per drone, ≈4.6 kg of payload for the swarm
at a 1.35 safety factor. **Measure your own static thrust** — a thrust-stand
number, not the motor manufacturer's — and put it in `config/swarm.yaml`. The
feasibility gate is only as honest as `max_thrust_n`.

---

## The tether — read this before flying

Six drones on near-inextensible lines is a trap. If one drone sits two
centimetres higher than the others, its line goes taut while theirs go slack,
and it takes essentially the entire payload. It will then hit its thrust limit,
sag, and hand the problem to the next drone. That oscillation is how these rigs
break.

Put a **compliant element in every leg** — a bungee section, a spring, or a
sprung winch. `tether_stiffness_n_per_m` in the config is that element's rate.
The default 10 N/m stretches ~0.65 m under a fair share of a 4 kg load and
spreads a 0.1 m altitude error over about 11 % of imbalance, so the abort fires
on real faults and not on ordinary station-keeping.

Check your own numbers before you fly:

```python
from swarm_drone import SwarmConfig
from swarm_drone.formation import HexFormation
from swarm_drone.load_lift import LiftPlanner, feasibility

cfg = SwarmConfig.from_yaml("config/swarm.yaml")
planner = LiftPlanner(cfg, HexFormation(cfg.formation))

print(feasibility(cfg.drone, cfg.lift, cfg.formation))
print("abort trips at", planner.imbalance_sensitivity(), "m of altitude error")
print("slot tolerance ", cfg.formation.slot_tolerance_m, "m")
```

If the abort sensitivity is **below** the slot tolerance, the swarm will abort
on normal flying. Soften the tethers, or accept a larger
`max_share_imbalance`.

### Geometry

`radius_m` is both the circumradius and the hexagon's edge length — the
separation between adjacent drones. Keep it above `min_separation_m` plus a
generous downwash allowance; 2 m for 450 mm frames is comfortable.

`tether_len_m` must exceed `radius_m − attach_radius_m` or the tether cannot
span the offset at all; `tether_geometry()` raises if you get this wrong.
Longer tethers give a smaller splay angle and therefore lower tension:

| tether | splay | tension per drone (4 kg) |
|---|---|---|
| 2.5 m | 38.3° | 8.33 N |
| 3.5 m | 26.3° | 7.29 N |
| 5.0 m | 18.1° | 6.88 N |

Longer also means more pendulum swing. 3.5 m is the default compromise.

---

## Network

The mesh is UDP broadcast on port 47600. Every drone needs an address on the
same subnet with broadcast reachable.

Static IPs by drone index keep debugging sane:

```
drone 0 → 10.0.0.10   drone 3 → 10.0.0.13
drone 1 → 10.0.0.11   drone 4 → 10.0.0.14
drone 2 → 10.0.0.12   drone 5 → 10.0.0.15
```

Many consumer APs rate-limit or drop broadcast traffic. If peers appear and
disappear, that is usually why — use an ad-hoc/IBSS network or a dedicated
mesh radio rather than an AP.

To bench-test several nodes on one machine you must use a real broadcast
address; unicast to `127.0.0.1` reaches only one socket even with
`SO_REUSEPORT`:

```bash
python scripts/run_drone_node.py --id 0 --sim --no-vision &
python scripts/run_drone_node.py --id 1 --sim --no-vision &
```

---

## Vision throughput

RT-DETR-L at 640 px on a Pi 4 CPU is roughly **2–3 fps**. That is the real
constraint on this system, and the pipeline is built around it rather than
pretending otherwise:

- `frame_stride: 3` runs the face stage on every third frame. The tracker
  accumulates votes across frames, so sampling costs latency, not accuracy.
- `min_person_px: 28` skips crops too small to embed usefully.
- `votes_to_alert: 3` at ~1 face-pass per second means about a second of
  consistent observation before anything fires.

To go faster, in rough order of payoff:

1. Export RT-DETR to **ONNX** or **NCNN** and run it through `onnxruntime`
   (2–3× on ARM).
2. Add a **Coral TPU** or **Hailo-8L** accelerator.
3. Drop to a smaller input size — but note that 416 px makes small objects
   *smaller*, which is the whole difficulty here. Measure before committing.
4. Run detection on the drone and face matching on a ground station, sending up
   only crops. Costs bandwidth, and bandwidth is what the mesh is short of.

### Camera

Fix the camera looking down and slightly forward. Faces are only usable from
the air at low altitude and shallow angles — expect useful face matching below
roughly 15–20 m with a standard lens, and detection-only above that. The blur
score recorded on every alert (`meta.blur`) tells you afterwards whether a match
was made on a sharp crop or on mush.

---

## Bring-up order

Do not skip steps. Each one catches faults the next one would hide.

1. **Simulate.** `python scripts/run_sim.py --plot out/sim.png`. Then with your
   own config, and with `--wind` and `--fail-drone` to see the aborts fire.
2. **One drone, tethered to a fixed weight**, hovering. Confirms thrust,
   attitude limits and the MAVLink link.
3. **Two drones, no load.** Confirms the mesh, slot assignment and separation.
4. **Six drones, no load.** Confirms formation quality and the consensus gains
   under real wind. Watch `shape_quality` in the telemetry.
5. **Six drones, light load** (~25 % of the rated payload). Watch the share
   ratios; this is where an under-spec'd compliant element shows up.
6. **Full payload.**

Fly over open ground with the payload clear of people. A six-drone tethered lift
has six single points of failure, and the abort path lowers the load — it does
not catch it.

---

## Responsible use of the face matching

The detection half is unremarkable. The face-matching half identifies people
from the air without their knowledge, and that is a different kind of system.

- Who may be enrolled, and on whose authority, is a **legal** question, not a
  technical one. Settle it before you build a watchlist.
- Every alert is written to `data/alerts.jsonl` with its score, vote count,
  crop sharpness and the drone's pose. Keep that record — an identification
  nobody can review afterwards is not one anyone should act on.
- The thresholds are deliberately conservative and the margin gate suppresses
  near-ties outright. Loosening them to "catch more" mostly catches the wrong
  people.
- A match is a lead for a human to check, not a conclusion.
