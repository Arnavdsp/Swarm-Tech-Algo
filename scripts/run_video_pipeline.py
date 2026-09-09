#!/usr/bin/env python3
"""
Run the detection + wanted-person pipeline over a video file.

This is the offline form of what each drone runs in flight — same detector, same
NWD post-processing, same tracker and watchlist — so a mission can be replayed
and reviewed from the recorded footage.

    python scripts/run_video_pipeline.py video.mp4 -o out/annotated.mp4
    python scripts/run_video_pipeline.py video.mp4 --db data/wanted_db.npz \
        --drone-id 3 --max-frames 600
    python scripts/run_video_pipeline.py video.mp4 --weights runs/VisDrone/best.pt

Every alert is written to JSONL and the run ends with a summary of who was seen,
where and on how many frames' agreement.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_drone.config import VisionConfig                            # noqa: E402
from swarm_drone.vision import (AerialVisionPipeline, RTDETRDetector,   # noqa: E402
                                WantedFaceDB, AlertSink, make_embedder)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="out/annotated.mp4")
    ap.add_argument("--db", default="data/wanted_db.npz")
    ap.add_argument("--alerts", default="data/alerts.jsonl")
    ap.add_argument("--weights", default="rtdetr-l.pt")
    ap.add_argument("--finetuned", help="VisDrone checkpoint with an 'rtdetr' key")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--nwd-thresh", type=float, default=0.65)
    ap.add_argument("--nwd-c", type=float, default=12.8)
    ap.add_argument("--iou-nms", action="store_true",
                    help="use plain IoU-NMS instead of NWD suppression")
    ap.add_argument("--stride", type=int, default=3,
                    help="run the face stage every Nth frame")
    ap.add_argument("--votes", type=int, default=3,
                    help="agreeing frames required before an alert")
    ap.add_argument("--match-threshold", type=float, default=0.38)
    ap.add_argument("--face-backend", default="insightface",
                    choices=["insightface", "onnx", "hash", "null"])
    ap.add_argument("--drone-id", type=int, default=0)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--skip-frames", type=int, default=1,
                    help="process every Nth frame of the video")
    ap.add_argument("--no-video", action="store_true",
                    help="detect only, don't write an annotated video")
    args = ap.parse_args()

    import cv2

    cfg = VisionConfig(
        weights=args.finetuned or args.weights, imgsz=args.imgsz,
        device=args.device, conf_thresh=args.conf,
        nwd_thresh=args.nwd_thresh, nwd_c=args.nwd_c,
        use_nwd_nms=not args.iou_nms, frame_stride=args.stride,
        votes_to_alert=args.votes, match_threshold=args.match_threshold,
        face_backend=args.face_backend, db_path=args.db, alert_log=args.alerts,
    )

    detector = RTDETRDetector(cfg)
    if args.finetuned:
        detector.cfg.weights = args.weights
        detector.load_finetuned(args.finetuned)
        print(f"Loaded fine-tuned weights from {args.finetuned}")

    embedder = make_embedder(cfg)
    db = WantedFaceDB.load_or_empty(args.db, dim=getattr(embedder, "dim", 512))
    if len(db) == 0:
        print(f"⚠ No watchlist at {args.db} — running detection only. "
              f"Build one with scripts/build_face_db.py")
    else:
        print(f"Watchlist: {len(db)} people, {db.n_vectors} embeddings")

    sink = AlertSink(args.alerts, cfg.alert_cooldown_s)
    sink.subscribe(lambda a: print(f"  🚨 {a.summary()}"))
    pipeline = AerialVisionPipeline(cfg, detector=detector, embedder=embedder,
                                    db=db, sink=sink)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input: {os.path.basename(args.video)}  {width}x{height} "
          f"@{fps:.0f}fps  {total} frames")

    writer = None
    if not args.no_video:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / max(args.skip_frames, 1), (width, height))

    frame_idx, processed, t0 = 0, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and processed >= args.max_frames:
            break
        frame_idx += 1
        if frame_idx % max(args.skip_frames, 1) != 0:
            continue

        pipeline.process(frame, drone_id=args.drone_id,
                         timestamp=frame_idx / fps)
        processed += 1

        if writer is not None:
            annotated = pipeline.annotate(frame)
            cv2.putText(annotated,
                        f"{'NWD' if cfg.use_nwd_nms else 'IoU'}-NMS | "
                        f"frame {frame_idx} | {len(pipeline.last_detections)} dets "
                        f"| {len(sink.alerts)} alerts",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2)
            writer.write(annotated)

        if processed % 50 == 0:
            rate = processed / (time.time() - t0)
            print(f"  {processed} frames  {rate:.1f} fps  "
                  f"{len(sink.alerts)} alerts", end="\r")

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t0
    print(f"\n\nProcessed {processed} frames in {elapsed:.1f}s "
          f"({processed / max(elapsed, 1e-6):.1f} fps)")
    if writer is not None:
        print(f"Annotated video → {args.out}")
    print(json.dumps(pipeline.report(), indent=2))

    if sink.alerts:
        print(f"\n── Wanted-person alerts ({len(sink.alerts)}) ──")
        for pid, alerts in sink.by_person().items():
            first = alerts[0]
            print(f"  {pid} ({first.name}): {len(alerts)} sighting(s), "
                  f"first at frame {first.frame_idx} "
                  f"(t={first.timestamp:.1f}s) score {first.score:.3f}")
        print(f"\nAlert log → {args.alerts}")
    elif len(db):
        print("\nNo watchlist matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
