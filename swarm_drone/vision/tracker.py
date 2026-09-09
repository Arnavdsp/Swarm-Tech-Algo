"""
NWD-based multi-object tracker.

Purpose here is identity persistence, not trajectory accuracy: we need to know
that the person in this frame is the same person as two frames ago, so that face
matches can be *voted on* over time instead of trusted one frame at a time. One
frame of a face at 40 px from 30 m up is not evidence; the same identity winning
five frames of one track is.

Association uses NWD rather than IoU for the reason in ``nwd.py`` — from
altitude the boxes are small enough that a real target's boxes often do not
overlap between frames at all.
"""
import itertools

import numpy as np

from .nwd import nwd_match, DEFAULT_C


class Track:
    """One tracked object and the identity votes accumulated on it."""

    _ids = itertools.count(1)

    def __init__(self, box, score, cls, frame_idx):
        self.id = next(Track._ids)
        self.box = np.asarray(box, dtype=np.float32)
        self.score = float(score)
        self.cls = int(cls)
        self.hits = 1
        self.age = 0
        self.misses = 0
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.votes = {}                # person_id -> accumulated score
        self.vote_counts = {}          # person_id -> number of frames
        self.alerted = False
        self.history = [self.box.copy()]

    def update(self, box, score, cls, frame_idx):
        self.box = np.asarray(box, dtype=np.float32)
        self.score = float(score)
        self.cls = int(cls)
        self.hits += 1
        self.misses = 0
        self.last_frame = frame_idx
        self.history.append(self.box.copy())
        if len(self.history) > 60:
            self.history.pop(0)

    def mark_missed(self):
        self.misses += 1

    def vote(self, person_id, score):
        """Record one frame's identity opinion for this track."""
        self.votes[person_id] = self.votes.get(person_id, 0.0) + float(score)
        self.vote_counts[person_id] = self.vote_counts.get(person_id, 0) + 1

    def leader(self):
        """(person_id, n_votes, mean_score) for the front-runner, or None."""
        if not self.vote_counts:
            return None
        pid = max(self.vote_counts, key=lambda k: (self.vote_counts[k], self.votes[k]))
        n = self.vote_counts[pid]
        return pid, n, self.votes[pid] / n

    @property
    def centre(self):
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    def to_dict(self):
        return {"track_id": self.id, "box": [round(float(v), 1) for v in self.box],
                "cls": self.cls, "score": round(self.score, 3),
                "hits": self.hits, "votes": dict(self.vote_counts)}


class NWDTracker:
    """Greedy NWD association with a small miss tolerance."""

    def __init__(self, nwd_threshold=0.35, C=DEFAULT_C, max_misses=8,
                 min_hits=2):
        self.nwd_threshold = nwd_threshold
        self.C = C
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.tracks = []
        self.frame_idx = 0

    def update(self, detections):
        """Advance one frame. Returns the list of live tracks."""
        self.frame_idx += 1
        boxes = np.asarray(detections.boxes_xyxy, dtype=np.float32).reshape(-1, 4)

        if not self.tracks:
            for i in range(len(boxes)):
                self.tracks.append(Track(boxes[i], detections.scores[i],
                                         detections.classes[i], self.frame_idx))
            return self.live_tracks()

        prev = np.stack([t.box for t in self.tracks]) if self.tracks \
            else np.zeros((0, 4), np.float32)
        pairs, unmatched_prev, unmatched_curr = nwd_match(
            prev, boxes, C=self.C, threshold=self.nwd_threshold)

        for ti, di, _sim in pairs:
            self.tracks[ti].update(boxes[di], detections.scores[di],
                                   detections.classes[di], self.frame_idx)
        for ti in unmatched_prev:
            self.tracks[ti].mark_missed()
        for di in unmatched_curr:
            self.tracks.append(Track(boxes[di], detections.scores[di],
                                     detections.classes[di], self.frame_idx))

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        self._match_index = {di: ti for ti, di, _ in pairs}
        return self.live_tracks()

    def track_for_detection(self, det_index):
        """Which track a detection index was assigned to on the last update."""
        idx = getattr(self, "_match_index", {})
        if det_index in idx:
            return self.tracks[idx[det_index]]
        # A detection that started a new track is the closest new one by centre.
        recent = [t for t in self.tracks if t.last_frame == self.frame_idx]
        return recent[-1] if recent else None

    def live_tracks(self):
        return [t for t in self.tracks
                if t.misses == 0 and t.hits >= self.min_hits]

    def all_tracks(self):
        return list(self.tracks)

    def reset(self):
        self.tracks = []
        self.frame_idx = 0
