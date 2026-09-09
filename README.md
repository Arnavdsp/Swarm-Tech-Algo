# Swarm Technology Algorithms

**Swarm Technology** is a field in robotics and artificial intelligence that
models the **collective behavior of decentralized, self-organized systems**. This
repository holds two things: the classical swarm-intelligence algorithms the
project started from, and the drone stack they grew into — six Raspberry
Pi–based quadcopters that lift a load together in a hexagonal formation and
search from the air with an RT-DETR vision pipeline.

Robots interact locally with one another and the environment, exhibiting
**emergent intelligence** that mirrors natural swarms. The project merges
**distributed algorithms**, **sensor-based navigation**, and **real-time
control** to demonstrate how small simple agents achieve complex group
behaviors.

---

## What's here

**1. Classical swarm-intelligence algorithms** — three foundational
optimizers, each self-contained and runnable, implemented for clarity and
educational value.

**2. A six-drone cooperative lifting swarm** — the drones fly into a regular
hexagon around a payload and lift it together on tethers, with distributed
formation control, live load-share monitoring, and a mission state machine that
aborts on real faults.

**3. Aerial detection with wanted-person matching** — RT-DETR with custom
Normalized Wasserstein Distance post-processing for tiny objects seen from
altitude, feeding a face-matching stage that checks people against a watchlist
and raises alerts across the whole swarm.

---

## Repository structure

```
Swarm-Tech-Algo/
├── ant_colony_optimization.py       classical algorithms
├── particle_swarm_optimization.py
├── artificial_bee_colony.py
│
├── swarm_drone/                     the drone stack
│   ├── geometry.py                  hexagon vertices, ring topology
│   ├── config.py                    every tunable, with defaults
│   ├── formation.py                 slot assignment + distributed controller
│   ├── load_lift.py                 tether physics + mission state machine
│   ├── agent.py                     one drone
│   ├── swarm.py                     six drones
│   ├── comms.py                     UDP broadcast mesh
│   ├── backends.py                  simulation + MAVLink
│   └── vision/
│       ├── nwd.py                   NWD similarity, NMS, reranking
│       ├── detector.py              RT-DETR + NWD post-processing
│       ├── tracker.py               NWD tracking, per-track identity votes
│       ├── embedders.py             InsightFace / ONNX / test stub
│       ├── face_db.py               watchlist + matching gates
│       ├── alerts.py                dedupe, JSONL record, fan-out
│       └── pipeline.py              frame in, alerts out
│
├── scripts/
│   ├── run_sim.py                   simulate the hexagonal lift
│   ├── run_drone_node.py            on-drone entrypoint (one per Pi)
│   ├── run_video_pipeline.py        video → detections + matches
│   ├── build_face_db.py             build a watchlist from photos
│   └── benchmark_nwd_vs_iou.py      NWD vs IoU suppression, measured
│
├── config/          swarm.yaml, vision.yaml
├── docs/            ARCHITECTURE.md, HARDWARE.md
├── tests/           96 tests, no GPU or model weights needed
└── requirements.txt / requirements-pi.txt
```

---

## Quick start

```bash
git clone https://github.com/Arnavdsp/Swarm-Tech-Algo.git
cd Swarm-Tech-Algo
pip install numpy                       # enough for the algorithms and the sim
```

**Classical algorithms** — each script is self-contained:

```bash
python ant_colony_optimization.py
python particle_swarm_optimization.py
python artificial_bee_colony.py
```

**Simulate the six-drone lift** — no hardware, no GPU:

```bash
pip install matplotlib
python scripts/run_sim.py --plot out/sim.png
```

```
── Lift feasibility ──────────────────────────────────────────
  tether_angle_deg         26.29
  share_per_drone_kg       0.744
  capacity_per_drone_kg    1.155
  margin                   1.554   (required 1.35)
  feasible                 True
  max_payload_kg           4.60

── Flight ────────────────────────────────────────────────────
  t=  1.00  takeoff    alt_sp= 3.00m  shape=0.04
  t=  3.55  forming    alt_sp= 3.00m  shape=0.60
  t=  8.20  tensioning alt_sp= 1.20m  shape=0.84
  t= 11.25  lifting    alt_sp= 1.20m  shape=0.91
  t= 23.45  cruise     alt_sp= 6.00m  shape=0.98
  t= 41.50  landing    alt_sp= 0.00m  shape=0.32
```

Try the failure paths — they are the interesting part:

```bash
python scripts/run_sim.py --payload 9.0            # refused: beyond capacity
python scripts/run_sim.py --fail-drone 3 --fail-at 12   # aborts mid-lift
python scripts/run_sim.py --wind 2.0               # aborts: can't hold formation
```

**Run the tests:**

```bash
pip install pytest && python -m pytest tests/ -q     # 96 passed
```

---

## 1. Classical swarm intelligence

- **Ant Colony Optimization (ACO)** — simulates ant foraging with artificial
  pheromone trails to solve the Travelling Salesman Problem.
- **Particle Swarm Optimization (PSO)** — models bird flocks and fish schools to
  find optima in continuous search spaces.
- **Artificial Bee Colony (ABC)** — mimics employed, onlooker and scout bees to
  balance exploration against exploitation.

---

## 2. Hexagonal cooperative lifting

Six drones hold the vertices of a regular hexagon around the payload. Because
the six tethers meet the load at six evenly spaced bearings, their horizontal
components cancel by symmetry and each drone carries exactly one sixth of the
weight — as long as the ring stays level and centred.

**Slot assignment** is exact: with six drones there are only 720 possible
assignments, so the one with minimum total travel is found by brute force. No
scipy, no heuristic.

**Control is decentralised.** Each drone computes its own command from its own
state plus the two ring neighbours it can hear:

```
acc = k_f·(slot − pos)  +  k_c·Σ(neighbour error − own error)/|N|
    + k_i·∫error dt     +  avoidance  −  k_d·velocity
```

There is no leader. The hexagon is a *virtual structure* — a centre, a yaw and a
radius in shared config — so a drone that hears nobody still flies its slot
correctly.

Two things in that law are load-bearing and easy to get wrong:

- **The consensus gain has a stability condition.** Set `k_consensus` to half
  `k_formation` and the consensus term cancels the formation term *exactly* for
  the alternating mode (every other drone displaced the opposite way). The swarm
  then settles into a permanently mis-shaped hexagon with zero net command and
  no error indication. `HexFormation.stability_margin()` reports the margin and
  the constructor warns if it goes non-positive.
- **The integral term is not optional.** A steady crosswind needs steady force
  to oppose it, and a proportional law can only make force from error — so
  without an integrator the swarm settles outside tolerance and stalls in
  `FORMING`.

**Tether physics is modelled properly.** Each tether is a *unilateral* spring:
it pulls when stretched and does nothing when slack.
`solve_load_equilibrium` bisects on the load's height until the tensions balance
the weight. The consequence is real and worth internalising: with a
near-inextensible line, **a couple of centimetres of altitude error puts nearly
the whole payload on the highest drone**. That is why every practical rig puts a
compliant element (bungee, spring, sprung winch) in each leg —
`tether_stiffness_n_per_m` is that element, and
`LiftPlanner.imbalance_sensitivity()` tells you what altitude error your settings
make the abort fire at.

**The mission** runs `ARMING → TAKEOFF → FORMING → DESCEND → TENSIONING →
LIFTING → CRUISE → LOWERING → RELEASED → LANDING`, with `ABORT` reachable from
anywhere. The climb is paced by the *lowest* drone rather than a clock; the
imbalance abort waits for `imbalance_grace_s` so a gust does not drop a payload;
and any phase that cannot converge times out with a stated reason instead of
hanging.

Details and gain derivations: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Airframe numbers, tether sizing and bring-up order:
[`docs/HARDWARE.md`](docs/HARDWARE.md).

---

## 3. Aerial detection and wanted-person matching

```
frame → RT-DETR → NWD rerank → NWD-NMS → person boxes
                                             │
                                    NWD tracker (stable ids)
                                             │
                       every Nth frame: crop → face embed → watchlist
                                             │
                          votes accumulate on the track; K agreeing
                          frames raise one alert, then cooldown
```

### Why NWD instead of IoU

Aerial footage is full of objects a handful of pixels across. Shift a 6×6 box by
three pixels and IoU collapses from 1.0 to about **0.14** — NMS thresholds
become knife-edge and tracking breaks the moment boxes stop overlapping. NWD
models each box as a 2-D Gaussian and compares distributions instead:

```
NWD(a,b) = exp( −W2(a,b) / C )
W2(a,b)  = ‖centre_a − centre_b‖² + ‖half-extent_a − half-extent_b‖²
```

Same three-pixel shift: **NWD ≈ 0.72**. It degrades smoothly with pixel error
instead of falling off a cliff. `C` sets the scale in pixels — 12.8 matches the
VisDrone training runs.

It is used in three places: suppression, density-aware confidence re-ranking (a
low-confidence tiny object inside a confident cluster of the same class is
usually real; the same box alone in an empty field usually is not), and
frame-to-frame association in the tracker.

Measure it on your own footage rather than taking the claim on faith:

```bash
python scripts/benchmark_nwd_vs_iou.py video.mp4 --max-frames 300 \
    --plot out/nwd_vs_iou.png
```

### Wanted-person matching

Build a watchlist from a folder of photos — one directory per person, several
photos each:

```
faces/
  W-0042_Jane_Doe/   img1.jpg img2.jpg img3.jpg  meta.json
  W-0117_John_Roe/   a.png b.png
```

```bash
python scripts/build_face_db.py faces/ -o data/wanted_db.npz
python scripts/build_face_db.py --inspect data/wanted_db.npz
```

Then run footage through the pipeline:

```bash
python scripts/run_video_pipeline.py video.mp4 \
    --db data/wanted_db.npz -o out/annotated.mp4 --drone-id 3
```

Three design choices keep this from producing confident nonsense:

- **Voting, not per-frame matching.** One frame of a face at 40 px from 30 m up
  is not evidence. The tracker holds identity across frames, so opinions
  accumulate and only `votes_to_alert` agreeing frames raise anything.
- **A margin gate, not just a threshold.** The best-scoring person must beat the
  *runner-up* by `match_margin`. Two enrolled people who look alike produce two
  near-equal scores, and a near-tie is exactly where a system like this
  misidentifies someone — so it reports nothing instead. `build_face_db.py`
  flags confusable enrolments up front.
- **An auditable record.** Every alert goes to `data/alerts.jsonl` with its
  score, vote count, crop sharpness and the drone's pose.

A sighting is broadcast on the mesh, so all six drones and the ground station
learn about it — not just the one that happened to be looking the right way.

### On the drone

```bash
python scripts/run_drone_node.py --id 3 --config config/swarm.yaml
python scripts/run_drone_node.py --id 0 --sim --no-vision   # bench test
```

The camera loop runs in a background thread. Detection on a Pi is slow (~2–3 fps
at 640 px) and jittery, and the 20 Hz control loop must never wait on it: a
vision stall degrades to *this drone stops reporting sightings*, never to *this
drone stops holding formation*.

### Responsible use

The detection half is unremarkable. The face-matching half identifies people
from the air without their knowledge, and that is a different kind of system.
Who may be enrolled, and on whose authority, is a legal question rather than a
technical one — settle it before building a watchlist. The thresholds here are
deliberately conservative; loosening them to "catch more" mostly catches the
wrong people. A match is a lead for a human to check, not a conclusion.

---

## Configuration

Everything is a dataclass with a working default, overridable from YAML:

```python
from swarm_drone import SwarmConfig
cfg = SwarmConfig.from_yaml("config/swarm.yaml")
cfg.lift.payload_mass_kg = 5.5
```

See [`config/swarm.yaml`](config/swarm.yaml) and
[`config/vision.yaml`](config/vision.yaml) for every knob with commentary.

---

## Testing

```bash
python -m pytest tests/ -q        # 96 passed in ~10s
```

No GPU, no model weights, no network. Every heavy import (torch, ultralytics,
insightface, pymavlink, cv2) is lazy; the vision tests use a detector stub and a
dependency-free embedder; the flight tests run the real controller against the
simulated backend.

---

## License

MIT. Free to use, modify, and distribute with attribution.

---

## Contact

Developed by Arnav Deshpande
Email: arnavhpd@gmail.com
LinkedIn: [arnav-deshpande-26a792290](https://www.linkedin.com/in/arnav-deshpande-26a792290)
GitHub: [Arnavdsp](https://github.com/Arnavdsp)
