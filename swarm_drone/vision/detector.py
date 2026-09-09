"""
RT-DETR detector with NWD post-processing.

Wraps ultralytics' RT-DETR so the rest of the stack sees one small ``Detections``
object.  When ``use_nwd_nms`` is on we ask the model for effectively raw output
(iou=0.99, so its own NMS barely fires) and do the suppression ourselves with
NWD — the pipeline from the Kaggle notebook, made reusable.

ultralytics and torch are imported lazily: everything else in this package,
tests included, runs on a machine that has neither.
"""
from dataclasses import dataclass, field

import numpy as np

from .nwd import nwd_nms, nwd_confidence_rerank


@dataclass
class Detections:
    """One frame's detections."""
    boxes_xyxy: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    scores: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32))
    classes: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=int))
    names: tuple = ()

    def __len__(self):
        return len(self.boxes_xyxy)

    def label(self, i):
        c = int(self.classes[i])
        name = self.names[c] if c < len(self.names) else str(c)
        return f"{name} {self.scores[i]:.2f}"

    def labels(self):
        return [self.label(i) for i in range(len(self))]

    def filter_classes(self, keep):
        keep = set(int(k) for k in keep)
        m = np.array([int(c) in keep for c in self.classes], dtype=bool)
        if len(m) == 0:
            return Detections(names=self.names)
        return Detections(self.boxes_xyxy[m], self.scores[m], self.classes[m],
                          self.names)

    def crops(self, frame, pad=0.06, min_px=0):
        """Cut each box out of ``frame``, padded. Returns (index, crop) pairs."""
        h, w = frame.shape[:2]
        out = []
        for i, (x1, y1, x2, y2) in enumerate(self.boxes_xyxy):
            bw, bh = x2 - x1, y2 - y1
            if max(bw, bh) < min_px:
                continue
            px, py = bw * pad, bh * pad
            xa, ya = int(max(0, x1 - px)), int(max(0, y1 - py))
            xb, yb = int(min(w, x2 + px)), int(min(h, y2 + py))
            if xb > xa and yb > ya:
                out.append((i, frame[ya:yb, xa:xb]))
        return out


def _patch_torch_load():
    """PyTorch 2.6 flipped ``weights_only`` to True and broke .pt checkpoints."""
    import torch
    if not getattr(torch, "_swarm_load_patched", False):
        original = torch.load

        def patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original(*args, **kwargs)

        torch.load = patched
        torch._swarm_load_patched = True


def resolve_device(spec="auto"):
    if spec != "auto":
        return spec
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class RTDETRDetector:
    """RT-DETR + optional NWD-NMS. Call ``detect(frame)``."""

    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.names = tuple(cfg.class_names)
        self.device = resolve_device(cfg.device)
        self._model = model

    @property
    def model(self):
        if self._model is None:
            _patch_torch_load()
            from ultralytics import RTDETR
            self._model = RTDETR(self.cfg.weights)
        return self._model

    def load_finetuned(self, ckpt_path):
        """Load VisDrone-finetuned weights saved as a dict with an 'rtdetr' key."""
        import torch
        _patch_torch_load()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt.get("rtdetr", ckpt.get("model", ckpt))
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        missing = self.model.model.load_state_dict(state, strict=False)
        return missing

    def detect(self, frame):
        """Run one frame and return ``Detections``."""
        cfg = self.cfg
        results = self.model.predict(
            source=frame,
            imgsz=cfg.imgsz,
            conf=cfg.conf_thresh,
            # With NWD suppression on, keep the model's own NMS out of the way.
            iou=0.99 if cfg.use_nwd_nms else cfg.iou_thresh,
            max_det=cfg.max_det,
            device=self.device,
            verbose=False,
        )
        r = results[0]
        if len(r.boxes) == 0:
            return Detections(names=self.names)

        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        classes = r.boxes.cls.cpu().numpy().astype(int)
        return self.postprocess(boxes, scores, classes)

    def postprocess(self, boxes, scores, classes):
        """NWD rerank + NWD-NMS. Split out so it can be unit-tested without a GPU."""
        cfg = self.cfg
        if cfg.use_reranking:
            scores = nwd_confidence_rerank(boxes, scores, classes, C=cfg.nwd_c)
        if cfg.use_nwd_nms:
            keep = nwd_nms(boxes, scores, classes,
                           nwd_threshold=cfg.nwd_thresh, C=cfg.nwd_c)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        order = np.argsort(-scores)
        return Detections(boxes[order], scores[order], classes[order], self.names)
