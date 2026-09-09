"""
Alerts: what leaves the drone when the pipeline believes it has seen someone.

Two jobs. Dedupe — a person walking through frame for twenty seconds is one
event, not six hundred, and the mesh has no bandwidth for the latter. And an
append-only record: every alert is written to JSONL with the frame, the score,
the number of votes behind it and which drone raised it, so a match can be
reviewed afterwards rather than taken on trust.
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict


@dataclass
class Alert:
    person_id: str
    name: str
    score: float
    votes: int
    drone_id: int
    track_id: int
    frame_idx: int
    timestamp: float
    box: list = field(default_factory=list)
    geo: dict = field(default_factory=dict)     # drone pose when it fired
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["score"] = round(float(self.score), 4)
        d["timestamp"] = round(float(self.timestamp), 3)
        d["iso_time"] = time.strftime("%Y-%m-%dT%H:%M:%S",
                                      time.localtime(self.timestamp))
        return d

    def summary(self):
        return (f"[drone {self.drone_id}] {self.name} ({self.person_id}) "
                f"score={self.score:.3f} votes={self.votes} frame={self.frame_idx}")


class AlertSink:
    """Deduplicates, logs, and fans out alerts to callbacks."""

    def __init__(self, log_path=None, cooldown_s=30.0):
        self.log_path = log_path
        self.cooldown_s = float(cooldown_s)
        self._last = {}          # (drone_id, person_id) -> timestamp
        self.callbacks = []
        self.alerts = []
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def subscribe(self, fn):
        """Register a callback: mesh broadcast, buzzer, ground-station uplink."""
        self.callbacks.append(fn)
        return fn

    def should_emit(self, drone_id, person_id, now=None):
        now = time.time() if now is None else now
        last = self._last.get((drone_id, person_id))
        return last is None or (now - last) >= self.cooldown_s

    def emit(self, alert):
        """Publish an alert unless the same person is still in cooldown."""
        key = (alert.drone_id, alert.person_id)
        if not self.should_emit(alert.drone_id, alert.person_id, alert.timestamp):
            return None
        self._last[key] = alert.timestamp
        self.alerts.append(alert)

        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(alert.to_dict()) + "\n")

        for fn in self.callbacks:
            try:
                fn(alert)
            except Exception as exc:                    # noqa: BLE001
                # A broken uplink must never take down the detection loop.
                print(f"[alerts] callback {fn!r} failed: {exc}")
        return alert

    def recent(self, n=10):
        return self.alerts[-n:]

    def by_person(self):
        out = {}
        for a in self.alerts:
            out.setdefault(a.person_id, []).append(a)
        return out

    @staticmethod
    def read_log(path):
        if not os.path.exists(path):
            return []
        with open(path) as fh:
            return [json.loads(line) for line in fh if line.strip()]
