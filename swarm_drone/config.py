"""
Configuration objects for the swarm.

Plain dataclasses with sane defaults so every script runs with zero setup;
``from_yaml`` lets you override anything from ``config/*.yaml`` on the Pi.
"""
from dataclasses import dataclass, field, asdict, fields
import json
import os


@dataclass
class DroneSpec:
    """Physical envelope of one airframe. Defaults describe the Pi-class quad."""
    mass_kg: float = 1.45            # airframe + Pi + camera, no payload share
    max_thrust_n: float = 32.0       # all-motor static thrust at 100 %
    hover_throttle: float = 0.42     # fraction of max thrust needed to hover
    max_tilt_rad: float = 0.35       # ~20 deg, hard limit while tethered
    max_speed_mps: float = 3.0
    max_climb_mps: float = 1.2
    max_accel_mps2: float = 2.5
    battery_wh: float = 77.0
    reserve_frac: float = 0.25       # never dip below this state of charge

    @property
    def payload_capacity_kg(self):
        """Spare lift after supporting its own weight, at the tilt limit."""
        import math
        g = 9.80665
        usable = self.max_thrust_n * math.cos(self.max_tilt_rad) * 0.85
        return max(0.0, usable / g - self.mass_kg)


@dataclass
class FormationConfig:
    """Hexagon geometry and the gains of the distributed position controller."""
    n_drones: int = 6
    radius_m: float = 2.0            # hexagon circumradius == edge length
    tether_len_m: float = 3.5
    formation_yaw_rad: float = 0.0
    k_formation: float = 1.4         # pull toward own slot
    k_consensus: float = 0.35        # match ring neighbours' formation error
                                     # (must satisfy k_consensus * 2 < k_formation;
                                     #  see HexFormation.stability_margin)
    k_damping: float = 1.1           # velocity damping
    k_integral: float = 0.45         # rejects steady wind; 0 disables
    integral_limit: float = 3.0      # anti-windup clamp, m*s. k_integral times
                                     # this is the standing acceleration the
                                     # integrator may command: 1.35 m/s^2, or
                                     # about 8 deg of tilt held against wind.
    k_altitude_sync: float = 1.6     # keep the ring level during the lift
    k_avoid: float = 2.4             # inter-drone repulsion
    avoid_radius_m: float = 1.5      # repulsion starts inside this range
    slot_tolerance_m: float = 0.25   # "in formation" radius
    min_separation_m: float = 1.0    # abort threshold
    control_hz: float = 20.0


@dataclass
class LiftConfig:
    """Cooperative-lift envelope and the mission state machine's thresholds."""
    payload_mass_kg: float = 4.0
    attach_radius_m: float = 0.45    # radius of the load's attachment ring
    hover_alt_m: float = 3.0
    pickup_alt_m: float = 1.2
    lift_alt_m: float = 6.0
    climb_rate_mps: float = 0.4      # synchronised ascent rate
    tension_ramp_s: float = 3.0      # seconds to take up tether slack
    max_share_imbalance: float = 0.35  # abort if any drone carries 35 % over fair share
    imbalance_grace_s: float = 1.0   # ...and has done so for this long
    settle_time_s: float = 1.5
    safety_factor: float = 1.35      # required capacity / required lift
    phase_timeout_s: float = 45.0    # a phase that cannot converge aborts
    tether_stiffness_n_per_m: float = 10.0
    """Spring rate of the elastic element in each tether, N/m.

    A bare Dyneema line is effectively inextensible, and with six of them a
    couple of centimetres of altitude error dumps the entire payload onto the
    highest drone. Real cooperative-lift rigs put a compliant element (bungee,
    spring, or a sprung winch) in each leg for exactly that reason. 10 N/m
    stretches about 0.65 m under a fair share of a 4 kg load, and puts the
    imbalance abort at roughly 0.28 m of altitude error -- comfortably outside
    the 0.25 m slot tolerance, so ordinary station-keeping does not trip it.
    Check your own numbers with LiftPlanner.imbalance_sensitivity().
    """


@dataclass
class CommsConfig:
    """UDP mesh between the Pis."""
    bind_host: str = "0.0.0.0"
    port: int = 47600
    broadcast_addr: str = "255.255.255.255"
    heartbeat_hz: float = 10.0
    peer_timeout_s: float = 1.5


@dataclass
class VisionConfig:
    """RT-DETR + NWD detection and the wanted-person face matcher."""
    weights: str = "rtdetr-l.pt"
    imgsz: int = 640
    device: str = "auto"             # "auto" | "cpu" | "0"
    conf_thresh: float = 0.20
    nwd_thresh: float = 0.65
    nwd_c: float = 12.8
    use_nwd_nms: bool = True
    use_reranking: bool = True
    iou_thresh: float = 0.45         # only used when use_nwd_nms is False
    max_det: int = 500
    class_names: tuple = ("pedestrian", "people", "bicycle",
                          "car", "tricycle", "motor")
    person_classes: tuple = (0, 1)   # which class ids get a face pass
    frame_stride: int = 3            # run the face stage every Nth frame
    min_person_px: int = 28          # skip person crops smaller than this
    crop_pad: float = 0.06           # context added around a person box
    face_backend: str = "insightface"  # "insightface" | "onnx" | "null"
    face_model: str = "buffalo_l"
    face_det_size: int = 320
    match_threshold: float = 0.38    # cosine similarity for a candidate match
    match_margin: float = 0.05       # best must beat runner-up by this much
    votes_to_alert: int = 3          # consistent hits on one track before alerting
    alert_cooldown_s: float = 30.0
    db_path: str = "data/wanted_db.npz"
    alert_log: str = "data/alerts.jsonl"


@dataclass
class SwarmConfig:
    drone: DroneSpec = field(default_factory=DroneSpec)
    formation: FormationConfig = field(default_factory=FormationConfig)
    lift: LiftConfig = field(default_factory=LiftConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        sub = {"drone": DroneSpec, "formation": FormationConfig,
               "lift": LiftConfig, "comms": CommsConfig, "vision": VisionConfig}
        kwargs = {}
        for name, klass in sub.items():
            valid = {f.name for f in fields(klass)}
            given = {k: v for k, v in (data.get(name) or {}).items() if k in valid}
            kwargs[name] = klass(**given)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path):
        """Load a YAML (or JSON) override file. PyYAML is optional."""
        with open(path) as fh:
            text = fh.read()
        try:
            import yaml
            data = yaml.safe_load(text)
        except ImportError:
            data = json.loads(text)
        return cls.from_dict(data)
