# Architecture

Two capabilities share one codebase and one mesh: **cooperative lifting** in a
hexagonal formation, and **aerial detection** with wanted-person matching. They
are independent — a drone can do either, both, or neither — but they run in the
same process on the same Pi, so this document covers how they stay out of each
other's way.

---

## 1. Layout

```
swarm_drone/
├── geometry.py      hexagon vertices, ring topology, vector helpers
├── config.py        every tunable, as dataclasses with defaults
├── formation.py     slot assignment + the distributed position controller
├── load_lift.py     tether physics, feasibility, the mission state machine
├── agent.py         one drone: backend + mesh + camera
├── swarm.py         coordinator that ties six agents together
├── comms.py         UDP broadcast mesh (and a loopback link for tests)
├── backends.py      SimBackend (physics) and MavlinkBackend (real hardware)
└── vision/
    ├── nwd.py         Normalized Wasserstein Distance: matrix, NMS, reranking
    ├── detector.py    RT-DETR wrapper + NWD post-processing
    ├── tracker.py     NWD association, track identity, per-track votes
    ├── embedders.py   InsightFace / ONNX / hash-stub face embedding
    ├── face_db.py     the watchlist and its two matching gates
    ├── alerts.py      dedupe, JSONL record, fan-out
    └── pipeline.py    frame in, alerts out
```

Nothing in the lifting half imports the vision half. `agent.py` is the only
place they meet, and there the vision pipeline is optional.

---

## 2. Hexagonal cooperative lifting

### Why a hexagon

Six drones on a regular hexagon put the tethers at six evenly spaced bearings
around the load. The horizontal components of the six tensions cancel by
symmetry, so each drone's motors fight gravity rather than each other, and each
carries exactly one sixth of the weight — as long as the ring stays level and
centred. Everything else in this section is about verifying that condition and
reacting when it stops holding.

### Slot assignment

Six drones, six vertices, 720 possible assignments. `assign_slots` evaluates all
of them and picks minimum total travel. That is trivially cheap at n = 6, is
exactly optimal, and needs no scipy; above n = 8 it falls back to greedy
nearest-free.

Minimising total travel also minimises path crossings on the way in, which is
what the avoidance term would otherwise have to sort out.

### The controller

Each drone computes its own acceleration from data it actually has — its own
state, plus whatever its two ring neighbours last broadcast:

```
acc_i =  k_f · (target_i − pos_i)                    formation
       + k_c · Σ_j∈N(i) (err_j − err_i) / |N(i)|     consensus
       + k_i · ∫ err_i dt                            integral (clamped)
       + repulsion from anything closer than 1.5 m   avoidance
       − k_d · vel_i                                 damping
```

There is no leader. The formation is a *virtual structure* — a centre, a yaw and
a radius, all in shared config — so every drone derives the same six targets
independently. A drone that hears nobody still flies its slot correctly; it just
stops contributing to consensus.

**The gain condition that matters.** The consensus term is `−(k_c/|N|)·L·e`,
where `L` is the ring graph Laplacian. On a 6-ring `L`'s largest eigenvalue is
4, belonging to the *alternating* mode: drone 0 displaced one way, drone 1 the
other, all the way round. With `|N| = 2`, that mode sees `−2·k_c·e`. Set
`k_c = k_f/2` and it cancels the formation term **exactly**: the swarm settles
into a permanently mis-shaped hexagon with zero net command on every drone and
no indication anything is wrong.

`HexFormation.stability_margin()` returns `k_f − k_c·λ_max/|N|` and the
constructor warns if it is not positive. Shipped defaults give 0.7.

**Why there is an integral term.** A steady crosswind needs a steady force to
oppose it. A proportional law can only produce force from error, so it settles
with a permanent offset — enough, at moderate wind, to sit outside the slot
tolerance forever and stall the mission in `FORMING`. The integrator supplies
that standing force at zero error. It is clamped (`integral_limit`) so it can
command at most ~1.35 m/s², about 8° of tilt, and it is reset on `bind()` since
a re-assignment makes the old error history meaningless.

### Tether physics

Each tether is a **unilateral spring**: it pulls once stretched beyond its
natural length and does nothing at all when slack. That one-sidedness is the
entire problem. `solve_load_equilibrium` bisects on the load's height until the
vertical tensions balance the weight, then reports each drone's share.

The consequence is stark, and it is real: with a near-inextensible line, **two
centimetres of altitude error puts essentially the whole payload on the highest
drone**. Which is why every practical multi-drone lift rig puts a compliant
element — bungee, spring, sprung winch — in each leg. `tether_stiffness_n_per_m`
is that element. At the default 10 N/m the imbalance abort trips at about 0.28 m
of altitude error, comfortably outside the 0.25 m slot tolerance, so ordinary
station-keeping does not trigger it. `LiftPlanner.imbalance_sensitivity()`
computes that number for whatever settings you choose; if it comes out below
`slot_tolerance_m`, the swarm will abort on normal flying.

### The mission

```
IDLE → ARMING → TAKEOFF → FORMING → DESCEND_TO_LOAD → TENSIONING
     → LIFTING → CRUISE → LOWERING → RELEASED → LANDING
                                                    ABORT ⟵ from anywhere
```

Three properties are worth calling out:

- **The climb is paced by the slowest drone.** In `LIFTING` the shared altitude
  setpoint only rises while the *lowest* drone is within tolerance of it. A
  drone that falls behind stalls the climb rather than being dragged up by its
  tether.
- **The imbalance abort waits.** It only counts once the tethers are past half
  tension, and it must stay tripped for `imbalance_grace_s`. A gust tilting the
  ring for a moment is not a reason to drop a payload. During the first half of
  the tension ramp the share ratio genuinely does swing to ~1.9 — under a
  fraction of the load, while slack is still coming out — and that is fine.
- **Phases time out.** A phase that cannot converge (too much wind to hold
  formation, a tether that never comes taut) aborts after
  `phase_timeout_s` with a reason, rather than hanging in it indefinitely.

Feasibility is checked *before* arming: `feasibility()` compares each drone's
spare lift against the tether tension the geometry demands, and `start()`
refuses outright if the margin is below `safety_factor`.

### The mesh

UDP broadcast, JSON, 10 Hz, no master. Each drone keeps a peer table with a
1.5 s timeout. A malformed or dropped packet is ignored, never fatal. Peers that
go quiet are reported by `lost_peers()`, and the coordinator treats a lost or
flat-battery drone mid-lift as an abort condition.

For bench-testing several nodes on one host, `SO_REUSEPORT` is set and you must
use a real broadcast address (`255.255.255.255`, or `127.255.255.255` on
loopback) — unicast to `127.0.0.1` is delivered to only one of the sockets.

---

## 3. Aerial vision

### Pipeline

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
three pixels and IoU falls from 1.0 to about 0.14 — so NMS thresholds become
knife-edge and frame-to-frame association breaks the moment boxes stop
overlapping. NWD models each box as a 2-D Gaussian and compares distributions:

```
NWD(a,b) = exp( −W2(a,b) / C )
W2(a,b)  = ‖centre_a − centre_b‖² + ‖half-extent_a − half-extent_b‖²
```

Same 3-pixel shift, NWD ≈ 0.72. It degrades smoothly with pixel error instead of
falling off a cliff. `C` sets the scale in pixels — roughly the mean object size;
12.8 matches the VisDrone training runs.

NWD is used in three places: suppression (`nwd_nms`), density-aware confidence
re-ranking (`nwd_confidence_rerank`), and frame-to-frame association in the
tracker (`nwd_match`).

When NWD suppression is on, the detector is asked for effectively raw output
(`iou=0.99`) so its own NMS does not pre-empt ours.

### Why voting, not per-frame matching

A single frame of a face at 40 px from 30 m up is not evidence. The tracker
gives a stable identity across frames, so identity opinions **accumulate on the
track** and only `votes_to_alert` agreeing frames raise anything. This is also
why the face stage can run on a stride: at 3 fps of face passes, three agreeing
frames is about a second of consistent observation.

### The two matching gates

`WantedFaceDB.search` requires both:

1. **Threshold** — cosine similarity above `match_threshold`.
2. **Margin** — the best-scoring person must beat the *runner-up* by
   `match_margin`.

The margin gate is the one that matters. Two enrolled people who look alike
produce two near-equal scores, and a near-tie is exactly the situation where a
system like this misidentifies someone. It reports nothing instead.
`scripts/build_face_db.py` flags confusable pairs at enrolment time so you find
out then rather than in flight.

Scoring is best-per-person, so someone enrolled from fifty photos cannot
out-vote a better match from someone enrolled from one.

### Vision never blocks flight

On the Pi the camera loop runs in a background thread (`VisionThread` in
`scripts/run_drone_node.py`). Detection is slow and jittery; the 20 Hz control
loop must not wait on it. Any exception in the vision thread is caught and
recorded. A vision stall degrades to *this drone stops reporting sightings* —
never to *this drone stops holding formation*.

### Sightings are a swarm event

When a drone raises an alert it broadcasts it on the mesh, so all six drones and
the ground station see it — not just the one that happened to be looking the
right way.

---

## 4. Testing

96 tests, no GPU, no model weights, a few seconds. Every heavy import
(torch, ultralytics, insightface, pymavlink, cv2) is lazy, the vision tests use a
detector stub and a dependency-free `HashEmbedder`, and the flight tests run the
real controller against `SimBackend`.

The tests that earn their place are the ones pinning behaviour that is easy to
break silently:

- `test_alternating_error_is_actually_corrected` — the consensus null-space bug.
- `test_integral_rejects_a_steady_disturbance` — the wind droop.
- `test_slack_tethers_carry_nothing` — the unilateral constraint.
- `test_imbalance_sensitivity_clears_the_slot_tolerance` — the abort is a guard,
  not a nuisance.
- `test_the_margin_gate_suppresses_a_near_tie` — the misidentification gate.
- `test_one_frame_is_never_enough` — voting is not optional.
