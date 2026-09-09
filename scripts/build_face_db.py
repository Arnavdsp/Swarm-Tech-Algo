#!/usr/bin/env python3
"""
Build the wanted-person watchlist from a folder of reference photos.

Layout — one directory per person, several photos each:

    faces/
      W-0042_Jane_Doe/     img1.jpg img2.jpg img3.jpg
      W-0117_John_Roe/     a.png b.png

The directory name becomes the person id; anything after the first underscore
becomes the display name (underscores -> spaces).  Optional ``meta.json`` inside
a person's folder is stored alongside them (case number, issuing authority,
expiry date — whatever your process needs to justify the entry later).

    python scripts/build_face_db.py faces/ -o data/wanted_db.npz
    python scripts/build_face_db.py faces/ -o data/wanted_db.npz --backend hash
    python scripts/build_face_db.py --inspect data/wanted_db.npz

Enrol several photos per person from different angles. A watchlist built from
one frontal mugshot each will miss people and, worse, will confuse similar-
looking people at exactly the threshold where you are least able to tell.

LEGAL: enrolling someone means the swarm will identify them from the air without
their knowledge. Whether you may do that, and for whom, is not a question this
script can answer for you.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_drone.config import VisionConfig                           # noqa: E402
from swarm_drone.vision.embedders import make_embedder                # noqa: E402
from swarm_drone.vision.face_db import WantedFaceDB                   # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_person_dir(name):
    """'W-0042_Jane_Doe' -> ('W-0042', 'Jane Doe')."""
    if "_" in name:
        pid, rest = name.split("_", 1)
        return pid, rest.replace("_", " ")
    return name, name


def load_image(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read {path}")
    return img


def inspect(path):
    db = WantedFaceDB.load(path)
    print(json.dumps(db.summary(), indent=2))
    print("\nEnrolled:")
    for pid in db.ids():
        rec = db.people[pid]
        print(f"  {pid:<14s} {rec['name']:<28s} {rec['n']} embedding(s)  "
              f"{rec['meta'] or ''}")
    return 0


def cross_check(db, top_k=3):
    """Warn about pairs of enrolled people who look alike to the model.

    Two watchlist entries with a high mutual similarity are the ones the matcher
    will confuse, and the margin gate will suppress *both*. Better to know at
    enrolment than to wonder later why a person never triggers.
    """
    if db.n_vectors < 2:
        return []
    sims = db.vectors @ db.vectors.T
    owner = np.array(db.owner, dtype=object)
    flagged = []
    for i in range(len(sims)):
        for j in range(i + 1, len(sims)):
            if owner[i] != owner[j] and sims[i, j] > 0.5:
                flagged.append((owner[i], owner[j], float(sims[i, j])))
    flagged.sort(key=lambda x: -x[2])
    return flagged[:top_k]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="folder of per-person directories")
    ap.add_argument("-o", "--out", default="data/wanted_db.npz")
    ap.add_argument("--backend", default="insightface",
                    choices=["insightface", "onnx", "hash"])
    ap.add_argument("--model", default="buffalo_l",
                    help="insightface model name, or path to an ONNX file")
    ap.add_argument("--det-size", type=int, default=640,
                    help="face detector input size for enrolment (bigger than "
                         "the in-flight value: reference photos are worth the time)")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing database instead of rebuilding")
    ap.add_argument("--inspect", metavar="DB", help="print a database and exit")
    args = ap.parse_args()

    if args.inspect:
        return inspect(args.inspect)
    if not args.root:
        ap.error("give a folder of person directories, or --inspect a database")

    cfg = VisionConfig(face_backend=args.backend, face_model=args.model,
                       face_det_size=args.det_size)
    embedder = make_embedder(cfg)

    db = (WantedFaceDB.load(args.out)
          if args.append and os.path.exists(args.out)
          else WantedFaceDB(dim=getattr(embedder, "dim", 512)))

    people = sorted(d for d in os.listdir(args.root)
                    if os.path.isdir(os.path.join(args.root, d)))
    if not people:
        print(f"No person directories under {args.root}")
        return 1

    total, skipped = 0, []
    for person_dir in people:
        pid, name = parse_person_dir(person_dir)
        folder = os.path.join(args.root, person_dir)

        meta = {}
        meta_path = os.path.join(folder, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)

        vectors = []
        for fname in sorted(os.listdir(folder)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXT:
                continue
            path = os.path.join(folder, fname)
            try:
                faces = embedder.embed(load_image(path))
            except Exception as exc:                       # noqa: BLE001
                skipped.append((path, str(exc)))
                continue
            if not faces:
                skipped.append((path, "no face detected"))
                continue
            # Several faces in a reference photo is ambiguous — take the
            # largest, which is almost always the subject rather than a
            # bystander, and say so.
            if len(faces) > 1:
                faces = [max(faces, key=lambda f: (
                    (f.box[2] - f.box[0]) * (f.box[3] - f.box[1])
                    if f.box is not None else 0.0))]
                print(f"  ! {fname}: multiple faces, kept the largest")
            vectors.append(faces[0].vector)

        if not vectors:
            print(f"  ✗ {pid:<14s} {name:<28s} no usable photos — not enrolled")
            continue
        db.enroll(pid, np.stack(vectors), name=name, meta=meta)
        total += len(vectors)
        print(f"  ✓ {pid:<14s} {name:<28s} {len(vectors)} embedding(s)")

    db.save(args.out)
    print(f"\nWatchlist → {args.out}")
    print(json.dumps(db.summary(), indent=2))

    if skipped:
        print(f"\nSkipped {len(skipped)} photo(s):")
        for path, why in skipped[:15]:
            print(f"  - {os.path.basename(path)}: {why}")

    confusable = cross_check(db)
    if confusable:
        print("\n⚠ These enrolled people look alike to the model. The margin gate "
              "will suppress matches on both — review before flying:")
        for a, b, s in confusable:
            print(f"  {a} vs {b}: similarity {s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
