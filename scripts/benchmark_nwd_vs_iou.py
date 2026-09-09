#!/usr/bin/env python3
"""
Compare NWD suppression against IoU-NMS on the same footage.

Runs both post-processing paths over one shared detector pass — same raw
predictions, two suppression rules — so the difference in the numbers is the
suppression and nothing else. This is the notebook's comparison, made
repeatable.

    python scripts/benchmark_nwd_vs_iou.py video.mp4 --max-frames 300
    python scripts/benchmark_nwd_vs_iou.py video.mp4 --plot out/nwd_vs_iou.png

What to look for: on aerial footage NWD should keep more small detections at the
same confidence, because IoU-NMS suppresses neighbouring tiny objects whose boxes
happen to touch. More detections is not automatically better — check the video
before believing the plot.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_drone.config import VisionConfig                            # noqa: E402
from swarm_drone.vision.detector import RTDETRDetector                 # noqa: E402
from swarm_drone.vision.nwd import nwd_nms, nwd_confidence_rerank      # noqa: E402


def iou_matrix(boxes):
    """Pairwise IoU for xyxy boxes."""
    b = np.asarray(boxes, dtype=np.float32)
    area = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = area[:, None] + area[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def iou_nms(boxes, scores, classes, threshold=0.45):
    """Plain IoU-NMS, same loop shape as nwd_nms so the comparison is fair."""
    n = len(boxes)
    if n == 0:
        return []
    sim = iou_matrix(boxes)
    same = np.asarray(classes)[:, None] == np.asarray(classes)[None, :]
    suppressed = np.zeros(n, dtype=bool)
    kept = np.zeros(n, dtype=bool)
    keep = []
    for i in np.argsort(-np.asarray(scores)):
        if suppressed[i]:
            continue
        keep.append(int(i))
        kept[i] = True
        suppressed |= same[i] & (sim[i] > threshold) & ~kept
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--weights", default="rtdetr-l.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--nwd-thresh", type=float, default=0.65)
    ap.add_argument("--nwd-c", type=float, default=12.8)
    ap.add_argument("--iou-thresh", type=float, default=0.45)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--small-px", type=float, default=32.0,
                    help="boxes with a side under this count as 'small'")
    ap.add_argument("--plot")
    ap.add_argument("--json", default="out/nwd_vs_iou.json")
    args = ap.parse_args()

    import cv2

    cfg = VisionConfig(weights=args.weights, imgsz=args.imgsz, device=args.device,
                       conf_thresh=args.conf, nwd_c=args.nwd_c, max_det=500)
    detector = RTDETRDetector(cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open {args.video}")
        return 1

    nwd_counts, iou_counts = [], []
    nwd_small, iou_small = [], []
    t0 = time.time()
    frames = 0
    while frames < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        # One detector pass, effectively unsuppressed; both rules see it.
        results = detector.model.predict(
            source=frame, imgsz=cfg.imgsz, conf=cfg.conf_thresh, iou=0.99,
            max_det=cfg.max_det, device=detector.device, verbose=False)
        r = results[0]
        if len(r.boxes) == 0:
            nwd_counts.append(0); iou_counts.append(0)
            nwd_small.append(0); iou_small.append(0)
            continue

        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        classes = r.boxes.cls.cpu().numpy().astype(int)

        reranked = nwd_confidence_rerank(boxes, scores, classes, C=cfg.nwd_c)
        keep_nwd = nwd_nms(boxes, reranked, classes,
                           nwd_threshold=args.nwd_thresh, C=cfg.nwd_c)
        keep_iou = iou_nms(boxes, scores, classes, threshold=args.iou_thresh)

        def n_small(keep):
            if not keep:
                return 0
            b = boxes[keep]
            return int(np.sum(np.minimum(b[:, 2] - b[:, 0],
                                         b[:, 3] - b[:, 1]) < args.small_px))

        nwd_counts.append(len(keep_nwd)); iou_counts.append(len(keep_iou))
        nwd_small.append(n_small(keep_nwd)); iou_small.append(n_small(keep_iou))

        if frames % 25 == 0:
            print(f"  {frames} frames  NWD {np.mean(nwd_counts):.1f}  "
                  f"IoU {np.mean(iou_counts):.1f}", end="\r")
    cap.release()

    summary = {
        "frames": frames,
        "seconds": round(time.time() - t0, 1),
        "nwd": {"mean": float(np.mean(nwd_counts)), "max": int(np.max(nwd_counts)),
                "total": int(np.sum(nwd_counts)),
                "mean_small": float(np.mean(nwd_small))},
        "iou": {"mean": float(np.mean(iou_counts)), "max": int(np.max(iou_counts)),
                "total": int(np.sum(iou_counts)),
                "mean_small": float(np.mean(iou_small))},
        "settings": {"conf": args.conf, "nwd_thresh": args.nwd_thresh,
                     "nwd_c": args.nwd_c, "iou_thresh": args.iou_thresh,
                     "small_px": args.small_px},
    }
    delta = summary["nwd"]["total"] - summary["iou"]["total"]
    summary["delta_total"] = int(delta)
    summary["delta_pct"] = round(
        100.0 * delta / max(summary["iou"]["total"], 1), 1)

    print("\n" + json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump({**summary, "nwd_counts": nwd_counts,
                   "iou_counts": iou_counts}, fh, indent=1)
    print(f"\nResults → {args.json}")

    if args.plot:
        plot(nwd_counts, iou_counts, nwd_small, iou_small, summary, args.plot)
        print(f"Figure  → {args.plot}")
    return 0


def plot(nwd, iou, nwd_small, iou_small, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("#0F1117")
    for ax in axes.flat:
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="white")
        ax.spines[["top", "right", "left", "bottom"]].set_color("#333644")
        ax.yaxis.grid(True, color="#2A2D3A", linewidth=0.8, linestyle="--")

    ax = axes[0][0]
    ax.plot(nwd, color="#4CE87A", linewidth=1.4, label="NWD-NMS")
    ax.plot(iou, color="#4C9BE8", linewidth=1.4, label="IoU-NMS")
    ax.set_title("Detections per frame", color="white", fontweight="bold")
    ax.set_xlabel("frame", color="white"); ax.set_ylabel("count", color="white")
    ax.legend(fontsize=9, facecolor="#1A1D27", labelcolor="white",
              edgecolor="#333644")

    ax = axes[0][1]
    diff = np.array(nwd) - np.array(iou)
    ax.fill_between(range(len(diff)), 0, diff, where=diff > 0, alpha=0.5,
                    color="#4CE87A", label="NWD keeps more")
    ax.fill_between(range(len(diff)), 0, diff, where=diff < 0, alpha=0.5,
                    color="#E8734C", label="IoU keeps more")
    ax.axhline(0, color="white", linewidth=0.6, alpha=0.4)
    ax.set_title("Difference (NWD - IoU)", color="white", fontweight="bold")
    ax.set_xlabel("frame", color="white"); ax.set_ylabel("count", color="white")
    ax.legend(fontsize=9, facecolor="#1A1D27", labelcolor="white",
              edgecolor="#333644")

    ax = axes[1][0]
    ax.plot(nwd_small, color="#4CE87A", linewidth=1.4, label="NWD-NMS")
    ax.plot(iou_small, color="#4C9BE8", linewidth=1.4, label="IoU-NMS")
    ax.set_title(f"Small objects kept (< {summary['settings']['small_px']:.0f} px)",
                 color="white", fontweight="bold")
    ax.set_xlabel("frame", color="white"); ax.set_ylabel("count", color="white")
    ax.legend(fontsize=9, facecolor="#1A1D27", labelcolor="white",
              edgecolor="#333644")

    ax = axes[1][1]
    ax.axis("off")
    rows = [
        ("metric", "NWD", "IoU"),
        ("mean dets/frame", f"{summary['nwd']['mean']:.1f}",
         f"{summary['iou']['mean']:.1f}"),
        ("max dets/frame", f"{summary['nwd']['max']}", f"{summary['iou']['max']}"),
        ("total dets", f"{summary['nwd']['total']}", f"{summary['iou']['total']}"),
        ("mean small/frame", f"{summary['nwd']['mean_small']:.1f}",
         f"{summary['iou']['mean_small']:.1f}"),
        ("difference", f"{summary['delta_pct']:+.1f}%", ""),
    ]
    for i, row in enumerate(rows):
        weight = "bold" if i == 0 else "normal"
        for j, cell in enumerate(row):
            ax.text(0.04 + j * 0.33, 0.86 - i * 0.12, cell, color="white",
                    fontsize=11, fontweight=weight, family="monospace",
                    transform=ax.transAxes)

    plt.suptitle(f"NWD vs IoU suppression — {summary['frames']} frames "
                 f"@ conf {summary['settings']['conf']}",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=140, facecolor="#0F1117", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
