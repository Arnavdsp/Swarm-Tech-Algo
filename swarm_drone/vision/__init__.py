"""Aerial vision: RT-DETR + NWD detection and wanted-person face matching."""

from .nwd import (compute_nwd_matrix, nwd_matrix_xyxy, nwd_nms,
                  nwd_confidence_rerank, nwd_match,
                  xyxy_to_cxcywh, cxcywh_to_xyxy)
from .detector import RTDETRDetector, Detections
from .tracker import NWDTracker, Track
from .face_db import WantedFaceDB, Match
from .embedders import (make_embedder, InsightFaceEmbedder, OnnxEmbedder,
                        HashEmbedder, l2_normalize)
from .alerts import Alert, AlertSink
from .pipeline import AerialVisionPipeline

__all__ = [
    "compute_nwd_matrix", "nwd_matrix_xyxy", "nwd_nms",
    "nwd_confidence_rerank", "nwd_match", "xyxy_to_cxcywh", "cxcywh_to_xyxy",
    "RTDETRDetector", "Detections",
    "NWDTracker", "Track",
    "WantedFaceDB", "Match",
    "make_embedder", "InsightFaceEmbedder", "OnnxEmbedder", "HashEmbedder",
    "l2_normalize",
    "Alert", "AlertSink", "AerialVisionPipeline",
]
