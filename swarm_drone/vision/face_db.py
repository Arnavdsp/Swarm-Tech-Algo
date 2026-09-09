"""
The watchlist: enrolled identities and the nearest-neighbour search against them.

Storage is a single ``.npz`` — an (N, D) matrix of unit-norm embeddings plus a
parallel list of person ids — and matching is one matmul.  At watchlist sizes
that fit on a drone (hundreds to a few thousand faces) that is faster than any
index, needs no extra dependency, and is trivially auditable.

Several embeddings per person is the point, not an accident: a person enrolled
from five photos at different angles matches far more reliably than one from a
single mugshot, and ``search`` scores against every one of them.

RESPONSIBLE USE: this identifies people from the air without their knowledge.
Who may be enrolled, and on whose authority, is a legal question — not a
technical one — before you point it at anybody.
"""
import json
import os
import time

import numpy as np

from .embedders import l2_normalize


class Match:
    """One search result."""

    __slots__ = ("person_id", "name", "score", "margin", "meta")

    def __init__(self, person_id, name, score, margin=0.0, meta=None):
        self.person_id = person_id
        self.name = name
        self.score = float(score)
        self.margin = float(margin)
        self.meta = meta or {}

    def __repr__(self):
        return (f"Match({self.person_id}:{self.name} "
                f"score={self.score:.3f} margin={self.margin:.3f})")

    def to_dict(self):
        return {"person_id": self.person_id, "name": self.name,
                "score": round(self.score, 4), "margin": round(self.margin, 4),
                "meta": self.meta}


class WantedFaceDB:
    """Watchlist of enrolled identities."""

    def __init__(self, dim=512):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.owner = []                  # row -> person_id
        self.people = {}                 # person_id -> {name, meta, n, added}

    # ------------------------------------------------------------------- size
    def __len__(self):
        return len(self.people)

    @property
    def n_vectors(self):
        return len(self.vectors)

    def ids(self):
        return list(self.people)

    def name_of(self, person_id):
        return self.people.get(person_id, {}).get("name", str(person_id))

    # ---------------------------------------------------------------- enrol
    def enroll(self, person_id, vectors, name=None, meta=None):
        """Add one or more embeddings for a person. Repeat calls accumulate."""
        v = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if v.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim}-d embeddings, got {v.shape[1]}")
        v = l2_normalize(v)

        self.vectors = np.vstack([self.vectors, v]) if len(self.vectors) else v
        self.owner.extend([person_id] * len(v))

        rec = self.people.setdefault(person_id, {
            "name": name or str(person_id), "meta": {}, "n": 0,
            "added": time.time()})
        if name:
            rec["name"] = name
        if meta:
            rec["meta"].update(meta)
        rec["n"] += len(v)
        return rec

    def remove(self, person_id):
        """Delete a person and every embedding of them. Returns rows removed."""
        if person_id not in self.people:
            return 0
        keep = [i for i, o in enumerate(self.owner) if o != person_id]
        removed = len(self.owner) - len(keep)
        self.vectors = self.vectors[keep] if keep else np.zeros((0, self.dim), np.float32)
        self.owner = [self.owner[i] for i in keep]
        del self.people[person_id]
        return removed

    # ---------------------------------------------------------------- search
    def search(self, vector, threshold=0.38, margin=0.05, top_k=3):
        """Best identity for one embedding, or None.

        Two gates have to pass. ``threshold`` is the usual cosine cut-off. The
        ``margin`` gate is the one that matters in practice: the best person must
        beat the best *different* person by that much. Two enrolled people who
        look alike produce two near-equal scores, and a near-tie is exactly when
        a system like this gets someone wrong — so it reports nothing instead.
        """
        if self.n_vectors == 0:
            return None
        q = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        sims = self.vectors @ q

        # Best score per person, so a person enrolled from 10 photos does not
        # simply out-vote one enrolled from 2.
        per_person = {}
        for score, pid in zip(sims, self.owner):
            if score > per_person.get(pid, -1.0):
                per_person[pid] = float(score)

        ranked = sorted(per_person.items(), key=lambda kv: -kv[1])[:max(top_k, 2)]
        best_id, best = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
        gap = best - runner_up

        if best < threshold or (len(ranked) > 1 and gap < margin):
            return None
        rec = self.people[best_id]
        return Match(best_id, rec["name"], best, gap, dict(rec["meta"]))

    def search_many(self, vectors, **kw):
        return [self.search(v, **kw) for v in np.atleast_2d(vectors)]

    def rank(self, vector, top_k=5):
        """Full ranked list — for tuning the threshold, not for the live path."""
        if self.n_vectors == 0:
            return []
        q = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        sims = self.vectors @ q
        per_person = {}
        for score, pid in zip(sims, self.owner):
            per_person[pid] = max(per_person.get(pid, -1.0), float(score))
        ranked = sorted(per_person.items(), key=lambda kv: -kv[1])[:top_k]
        return [(pid, self.name_of(pid), s) for pid, s in ranked]

    # ------------------------------------------------------------------- i/o
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            owner=np.array(self.owner, dtype=object),
            people=json.dumps(self.people),
            dim=self.dim,
        )
        return path

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        db = cls(dim=int(data["dim"]))
        db.vectors = data["vectors"].astype(np.float32)
        db.owner = list(data["owner"])
        db.people = json.loads(str(data["people"]))
        return db

    @classmethod
    def load_or_empty(cls, path, dim=512):
        if path and os.path.exists(path):
            return cls.load(path)
        return cls(dim=dim)

    def summary(self):
        return {
            "people": len(self.people),
            "embeddings": self.n_vectors,
            "dim": self.dim,
            "per_person": {pid: rec["n"] for pid, rec in self.people.items()},
        }
