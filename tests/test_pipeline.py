"""Tracking, voting and alerting — the whole vision path minus the detector."""
import json

import numpy as np
import pytest

from swarm_drone.vision.alerts import Alert, AlertSink
from swarm_drone.vision.detector import Detections, RTDETRDetector
from swarm_drone.vision.embedders import HashEmbedder
from swarm_drone.vision.face_db import WantedFaceDB
from swarm_drone.vision.pipeline import AerialVisionPipeline
from swarm_drone.vision.tracker import NWDTracker


# ---------------------------------------------------------------- detections
def test_postprocess_suppresses_duplicates(vision_cfg):
    det = RTDETRDetector(vision_cfg)
    boxes = np.array([[10, 10, 20, 20], [11, 11, 21, 21],
                      [100, 100, 112, 114]], dtype=np.float32)
    out = det.postprocess(boxes, np.array([0.9, 0.7, 0.5], dtype=np.float32),
                          np.array([0, 0, 3]))
    assert len(out) == 2
    assert out.scores[0] >= out.scores[1]          # sorted by confidence
    assert out.label(1).startswith("car")


def test_filter_classes_keeps_only_people(stub_detector, frame_factory):
    dets = stub_detector.detect(frame_factory())
    dets.classes = np.array([0, 3])
    assert len(dets.filter_classes([0, 1])) == 1


def test_crops_are_clipped_to_the_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = Detections(np.array([[-20, -20, 30, 30], [80, 80, 200, 200]],
                               dtype=np.float32),
                      np.array([0.9, 0.8], dtype=np.float32), np.array([0, 0]))
    for _, crop in dets.crops(frame, pad=0.5):
        assert crop.size > 0


def test_crops_skip_tiny_boxes():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = Detections(np.array([[10, 10, 14, 14]], dtype=np.float32),
                      np.array([0.9], dtype=np.float32), np.array([0]))
    assert dets.crops(frame, min_px=28) == []


# -------------------------------------------------------------------- tracks
def test_tracker_keeps_identity_across_a_moving_object():
    tracker = NWDTracker()
    boxes = np.array([[10, 10, 22, 26], [200, 120, 214, 140]], dtype=np.float32)
    for k in range(6):
        shift = np.array([k * 2.0, k * 1.0, k * 2.0, k * 1.0])
        tracker.update(Detections(boxes + shift,
                                  np.array([0.8, 0.6], dtype=np.float32),
                                  np.array([0, 0])))
    ids = [t.id for t in tracker.all_tracks()]
    assert len(ids) == 2 and len(set(ids)) == 2
    assert all(t.hits == 6 for t in tracker.all_tracks())


def test_tracker_drops_a_track_after_enough_misses():
    tracker = NWDTracker(max_misses=2)
    box = np.array([[10, 10, 22, 26]], dtype=np.float32)
    tracker.update(Detections(box, np.array([0.9], dtype=np.float32),
                              np.array([0])))
    for _ in range(4):
        tracker.update(Detections())
    assert tracker.all_tracks() == []


def test_track_leader_reflects_the_votes():
    tracker = NWDTracker()
    tracker.update(Detections(np.array([[10, 10, 22, 26]], dtype=np.float32),
                              np.array([0.9], dtype=np.float32),
                              np.array([0])))
    track = tracker.all_tracks()[0]
    for _ in range(3):
        track.vote("W-A", 0.8)
    track.vote("W-B", 0.95)
    pid, votes, mean = track.leader()
    assert pid == "W-A" and votes == 3
    assert mean == pytest.approx(0.8)


# -------------------------------------------------------------------- alerts
def _alert(**kw):
    base = dict(person_id="W-1", name="One", score=0.8, votes=3, drone_id=0,
                track_id=1, frame_idx=10, timestamp=1000.0)
    base.update(kw)
    return Alert(**base)


def test_cooldown_suppresses_a_repeat_sighting(tmp_path):
    sink = AlertSink(str(tmp_path / "a.jsonl"), cooldown_s=30.0)
    assert sink.emit(_alert(timestamp=1000.0)) is not None
    assert sink.emit(_alert(timestamp=1010.0)) is None
    assert sink.emit(_alert(timestamp=1040.0)) is not None
    assert len(sink.alerts) == 2


def test_cooldown_is_per_drone_and_per_person(tmp_path):
    sink = AlertSink(str(tmp_path / "a.jsonl"), cooldown_s=30.0)
    sink.emit(_alert(timestamp=1000.0))
    assert sink.emit(_alert(timestamp=1001.0, drone_id=4)) is not None
    assert sink.emit(_alert(timestamp=1001.0, person_id="W-2")) is not None


def test_alerts_are_logged_as_readable_jsonl(tmp_path):
    path = str(tmp_path / "a.jsonl")
    sink = AlertSink(path, cooldown_s=0.0)
    sink.emit(_alert(geo={"lat": 1.5, "lon": 2.5}))
    rows = AlertSink.read_log(path)
    assert len(rows) == 1
    assert rows[0]["person_id"] == "W-1"
    assert rows[0]["geo"]["lat"] == 1.5
    assert "iso_time" in rows[0]
    json.dumps(rows)                       # must stay serialisable


def test_a_failing_callback_does_not_stop_the_others(tmp_path, capsys):
    sink = AlertSink(None, cooldown_s=0.0)
    got = []
    sink.subscribe(lambda a: (_ for _ in ()).throw(RuntimeError("uplink down")))
    sink.subscribe(got.append)
    sink.emit(_alert())
    assert len(got) == 1
    assert "uplink down" in capsys.readouterr().out


# ------------------------------------------------------------------ pipeline
def _pipeline(vision_cfg, stub_detector, person_patch, decoy=None):
    embedder = HashEmbedder()
    db = WantedFaceDB(dim=embedder.dim)
    db.enroll("W-042", embedder.embed(person_patch)[0].vector, name="Subject A",
              meta={"case": "TEST-1"})
    if decoy is not None:
        db.enroll("W-099", embedder.embed(decoy)[0].vector, name="Decoy B")
    sink = AlertSink(vision_cfg.alert_log, vision_cfg.alert_cooldown_s)
    return AerialVisionPipeline(vision_cfg, detector=stub_detector,
                                embedder=embedder, db=db, sink=sink), sink


def test_a_watchlisted_person_raises_exactly_one_alert(
        vision_cfg, stub_detector, frame_factory, person_patch, decoy_patch):
    """Two people in frame, one of them enrolled — exactly one alert, for them.

    The decoy is enrolled but never appears, so this also checks that simply
    being on the watchlist does not produce a sighting.
    """
    pipe, sink = _pipeline(vision_cfg, stub_detector, person_patch, decoy_patch)
    for k in range(10):
        stub_detector.dx = k * 3
        pipe.process(frame_factory(k * 3), drone_id=2, timestamp=1000.0 + k,
                     geo={"lat": 12.97, "lon": 77.59, "alt_m": 30.0})
    assert len(sink.alerts) == 1
    alert = sink.alerts[0]
    assert alert.person_id == "W-042"
    assert alert.drone_id == 2
    assert alert.votes >= vision_cfg.votes_to_alert
    assert alert.geo["alt_m"] == 30.0


def test_one_frame_is_never_enough(vision_cfg, stub_detector, frame_factory,
                                   person_patch):
    """Voting is the point: a single frame's opinion must not raise an alert."""
    pipe, sink = _pipeline(vision_cfg, stub_detector, person_patch)
    pipe.process(frame_factory(), drone_id=0, timestamp=1000.0)
    assert sink.alerts == []


def test_nobody_on_the_watchlist_means_no_alerts(
        vision_cfg, stub_detector, frame_factory, other_patch):
    pipe, sink = _pipeline(vision_cfg, stub_detector, other_patch)
    other = np.random.default_rng(1).integers(0, 255, (60, 30, 3)).astype(np.uint8)
    pipe.db = WantedFaceDB(dim=pipe.embedder.dim)
    pipe.db.enroll("W-999", pipe.embedder.embed(other)[0].vector, name="Nobody")
    for k in range(10):
        stub_detector.dx = k * 3
        pipe.process(frame_factory(k * 3), timestamp=1000.0 + k)
    assert sink.alerts == []


def test_an_empty_watchlist_skips_the_face_stage(
        vision_cfg, stub_detector, frame_factory):
    embedder = HashEmbedder()
    pipe = AerialVisionPipeline(vision_cfg, detector=stub_detector,
                                embedder=embedder,
                                db=WantedFaceDB(dim=embedder.dim),
                                sink=AlertSink(None))
    pipe.process(frame_factory(), timestamp=1.0)
    assert pipe.stats["detections"] == 2
    assert pipe.stats["faces"] == 0


def test_frame_stride_skips_the_face_stage(vision_cfg, stub_detector,
                                           frame_factory, person_patch):
    vision_cfg.frame_stride = 5
    pipe, _ = _pipeline(vision_cfg, stub_detector, person_patch)
    for k in range(4):
        pipe.process(frame_factory(), timestamp=float(k))
    assert pipe.stats["faces"] == 0        # nothing embedded before frame 5
    pipe.process(frame_factory(), timestamp=5.0)
    assert pipe.stats["faces"] > 0


def test_report_summarises_the_run(vision_cfg, stub_detector, frame_factory,
                                   person_patch):
    pipe, _ = _pipeline(vision_cfg, stub_detector, person_patch)
    for k in range(6):
        stub_detector.dx = k * 3
        pipe.process(frame_factory(k * 3), timestamp=float(k))
    report = pipe.report()
    assert report["frames"] == 6
    assert report["dets_per_frame"] == pytest.approx(2.0)
    assert report["watchlist"]["people"] == 1
    json.dumps(report)
