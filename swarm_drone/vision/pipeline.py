"""
The on-board vision pipeline: video frame in, wanted-person alerts out.

Per frame:

    RT-DETR  ->  NWD rerank + NWD-NMS  ->  person boxes
                                              |
                              NWD tracker (stable track ids)
                                              |
                      every Nth frame: crop -> face embed -> watchlist search
                                              |
                             votes accumulate on the track; K agreeing
                             frames raise one alert, then cooldown

The face stage runs on a stride because it is the expensive half — on a Pi 4 the
detector alone is roughly 2-3 fps at 640 px, and embedding every person in every
frame would collapse that. Sampling is fine precisely because the tracker gives
us a stable identity to accumulate votes against.
"""
import time

import numpy as np

from .alerts import Alert, AlertSink
from .detector import RTDETRDetector, Detections
from .embedders import make_embedder
from .face_db import WantedFaceDB
from .tracker import NWDTracker


class AerialVisionPipeline:
    def __init__(self, cfg, detector=None, embedder=None, db=None, sink=None,
                 tracker=None):
        self.cfg = cfg
        self.detector = detector or RTDETRDetector(cfg)
        self.embedder = embedder if embedder is not None else make_embedder(cfg)
        self.db = db if db is not None else WantedFaceDB.load_or_empty(
            cfg.db_path, dim=getattr(self.embedder, "dim", 512))
        self.sink = sink or AlertSink(cfg.alert_log, cfg.alert_cooldown_s)
        self.tracker = tracker or NWDTracker(C=cfg.nwd_c)
        self.frame_idx = 0
        self.stats = {"frames": 0, "detections": 0, "faces": 0,
                      "candidates": 0, "alerts": 0, "detect_s": 0.0,
                      "face_s": 0.0}

    # ------------------------------------------------------------------ frame
    def process(self, frame, drone_id=0, timestamp=None, geo=None):
        """Run one frame. Returns the alerts raised by *this* frame."""
        self.frame_idx += 1
        self.stats["frames"] += 1
        timestamp = time.time() if timestamp is None else timestamp

        t0 = time.time()
        detections = self.detector.detect(frame)
        self.stats["detect_s"] += time.time() - t0
        self.stats["detections"] += len(detections)
        self.last_detections = detections

        people = detections.filter_classes(self.cfg.person_classes)
        self.tracker.update(people)
        self.last_people = people

        if self.frame_idx % max(1, self.cfg.frame_stride) != 0:
            return []
        if len(self.db) == 0 or len(people) == 0:
            return []

        return self._face_pass(frame, people, drone_id, timestamp, geo)

    def _face_pass(self, frame, people, drone_id, timestamp, geo):
        cfg = self.cfg
        alerts = []
        t0 = time.time()

        for det_index, crop in people.crops(frame, pad=cfg.crop_pad,
                                            min_px=cfg.min_person_px):
            track = self.tracker.track_for_detection(det_index)
            if track is None:
                continue

            for face in self.embedder.embed(crop):
                self.stats["faces"] += 1
                match = self.db.search(face.vector,
                                       threshold=cfg.match_threshold,
                                       margin=cfg.match_margin)
                if match is None:
                    continue
                self.stats["candidates"] += 1
                track.vote(match.person_id, match.score)

                leader = track.leader()
                if leader is None:
                    continue
                pid, votes, mean_score = leader
                if votes < cfg.votes_to_alert or track.alerted:
                    continue

                alert = Alert(
                    person_id=pid,
                    name=self.db.name_of(pid),
                    score=mean_score,
                    votes=votes,
                    drone_id=drone_id,
                    track_id=track.id,
                    frame_idx=self.frame_idx,
                    timestamp=timestamp,
                    box=[round(float(v), 1) for v in track.box],
                    geo=geo or {},
                    meta={"det_score": round(float(face.det_score), 3),
                          "blur": round(float(face.blur), 1),
                          "match_margin": round(float(match.margin), 4)},
                )
                emitted = self.sink.emit(alert)
                if emitted is not None:
                    track.alerted = True
                    self.stats["alerts"] += 1
                    alerts.append(emitted)

        self.stats["face_s"] += time.time() - t0
        return alerts

    # -------------------------------------------------------------- rendering
    def annotate(self, frame, detections=None, show_tracks=True):
        """Draw detections, track ids and any confirmed identity onto a frame."""
        import cv2
        detections = detections if detections is not None else \
            getattr(self, "last_detections", Detections())
        img = frame.copy()
        palette = [(255, 100, 100), (255, 180, 60), (100, 255, 100),
                   (100, 100, 255), (255, 255, 100), (180, 100, 255)]

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.boxes_xyxy[i])
            colour = palette[int(detections.classes[i]) % len(palette)]
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            self._label(cv2, img, detections.label(i), x1, y1, colour)

        if show_tracks:
            for t in self.tracker.live_tracks():
                x1, y1, x2, y2 = map(int, t.box)
                leader = t.leader()
                if leader and leader[1] >= self.cfg.votes_to_alert:
                    pid, votes, score = leader
                    # A confirmed identity gets a thick red box and a name.
                    cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                                  (0, 0, 255), 3)
                    self._label(cv2, img,
                                f"WANTED: {self.db.name_of(pid)} {score:.2f}",
                                x1, y2 + 18, (0, 0, 255), scale=0.55)
                else:
                    cv2.putText(img, f"#{t.id}", (x1, y2 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        return img

    @staticmethod
    def _label(cv2, img, text, x, y, colour, scale=0.45):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), colour, -1)
        cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1)

    # ------------------------------------------------------------------ stats
    def report(self):
        s = dict(self.stats)
        f = max(s["frames"], 1)
        s["detect_fps"] = round(f / s["detect_s"], 2) if s["detect_s"] else None
        s["dets_per_frame"] = round(s["detections"] / f, 2)
        s["watchlist"] = self.db.summary()
        return s
