"""Shared fixtures: synthetic frames and a detector stub.

The vision tests must run without torch, ultralytics or a GPU, so the detector
is replaced by a stub that returns known boxes. Everything downstream of the
detector — NWD post-processing, tracking, matching, voting, alerting — is the
real code.
"""
import numpy as np
import pytest

from swarm_drone.config import VisionConfig
from swarm_drone.vision.detector import Detections


@pytest.fixture
def rng():
    return np.random.default_rng(20240607)


@pytest.fixture
def person_patch(rng):
    """A distinctive texture standing in for one person's appearance."""
    return rng.integers(0, 255, (60, 30, 3)).astype(np.uint8)


@pytest.fixture
def other_patch(rng):
    return rng.integers(0, 255, (60, 30, 3)).astype(np.uint8)


@pytest.fixture
def decoy_patch(rng):
    """Someone on the watchlist who never appears in frame."""
    return rng.integers(0, 255, (60, 30, 3)).astype(np.uint8)


@pytest.fixture
def frame_factory(person_patch, other_patch):
    """Build a 640x360 frame with two 'people' shifted by ``dx`` pixels."""
    def build(dx=0):
        frame = np.full((360, 640, 3), 40, dtype=np.uint8)
        frame[100:160, 50 + dx:80 + dx] = person_patch
        frame[200:260, 400 + dx:430 + dx] = other_patch
        return frame
    return build


class StubDetector:
    """Returns the two known boxes, shifted to follow ``frame_factory``."""

    def __init__(self, names):
        self.names = names
        self.dx = 0

    def detect(self, frame):
        d = self.dx
        return Detections(
            np.array([[50 + d, 100, 80 + d, 160],
                      [400 + d, 200, 430 + d, 260]], dtype=np.float32),
            np.array([0.80, 0.75], dtype=np.float32),
            np.array([0, 0]),
            self.names)


@pytest.fixture
def stub_detector():
    return StubDetector(tuple(VisionConfig().class_names))


@pytest.fixture
def vision_cfg(tmp_path):
    """Offline-friendly config: hash embeddings, no crop padding, temp paths."""
    return VisionConfig(
        face_backend="hash", crop_pad=0.0, frame_stride=1, votes_to_alert=3,
        match_threshold=0.5, match_margin=0.05, min_person_px=10,
        db_path=str(tmp_path / "wanted.npz"),
        alert_log=str(tmp_path / "alerts.jsonl"),
    )
