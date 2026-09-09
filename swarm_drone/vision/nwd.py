"""
Normalized Wasserstein Distance (NWD) for tiny-object detection.

Aerial footage is full of objects a handful of pixels across.  IoU is a bad
similarity measure there: shift a 6x6 box by three pixels and IoU collapses from
1.0 to ~0.14, so NMS thresholds become knife-edge and matching is unstable.
NWD models each box as a 2-D Gaussian and compares distributions instead, which
degrades smoothly with the pixel shift:

    NWD(a, b) = exp( -W2(a, b) / C )
    W2(a, b)  = || (cx,cy)_a - (cx,cy)_b ||^2 + || (w/2,h/2)_a - (w/2,h/2)_b ||^2

``C`` sets the scale in pixels — roughly the average object size in the dataset.
12.8 matches the VisDrone training runs.

Pure numpy, no torch: this module is importable (and testable) on the Pi without
a deep-learning stack installed.
"""
import numpy as np

DEFAULT_C = 12.8


def xyxy_to_cxcywh(boxes):
    """[x1,y1,x2,y2] -> [cx,cy,w,h]."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2.0,
                     (b[:, 1] + b[:, 3]) / 2.0,
                     b[:, 2] - b[:, 0],
                     b[:, 3] - b[:, 1]], axis=1)


def cxcywh_to_xyxy(boxes):
    """[cx,cy,w,h] -> [x1,y1,x2,y2]."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.stack([b[:, 0] - b[:, 2] / 2.0,
                     b[:, 1] - b[:, 3] / 2.0,
                     b[:, 0] + b[:, 2] / 2.0,
                     b[:, 1] + b[:, 3] / 2.0], axis=1)


def compute_nwd_matrix(boxes1, boxes2, C=DEFAULT_C):
    """(N,M) NWD similarity in [0,1]. Inputs are cxcywh in pixels."""
    b1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    b2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)), dtype=np.float32)

    mu1, sig1 = b1[:, :2], b1[:, 2:] / 2.0
    mu2, sig2 = b2[:, :2], b2[:, 2:] / 2.0

    centre = ((mu1[:, None, :] - mu2[None, :, :]) ** 2).sum(-1)
    shape = ((sig1[:, None, :] - sig2[None, :, :]) ** 2).sum(-1)
    # Clamp rather than add an epsilon inside the root: an epsilon would make
    # NWD(a, a) come out at 0.99998 instead of exactly 1, which quietly shifts
    # every threshold calibrated against it.
    w2 = np.sqrt(np.maximum(centre + shape, 0.0))
    return np.exp(-w2 / C).astype(np.float32)


def nwd_matrix_xyxy(boxes1_xyxy, boxes2_xyxy, C=DEFAULT_C):
    """Convenience wrapper taking xyxy boxes."""
    return compute_nwd_matrix(xyxy_to_cxcywh(boxes1_xyxy),
                              xyxy_to_cxcywh(boxes2_xyxy), C=C)


def nwd_nms(boxes_xyxy, scores, classes, nwd_threshold=0.65, C=DEFAULT_C,
            class_agnostic=False):
    """NWD-based non-maximum suppression. Returns kept indices, best score first.

    Same rule as IoU-NMS — walk detections in descending confidence and drop
    anything too similar to a survivor — but with NWD as the similarity.  The
    full (N,N) similarity matrix is built once instead of per-iteration, which
    is what makes this usable at 500 detections per frame.
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    n = len(boxes_xyxy)
    if n == 0:
        return []
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes).astype(int)

    sim = compute_nwd_matrix(xyxy_to_cxcywh(boxes_xyxy),
                             xyxy_to_cxcywh(boxes_xyxy), C=C)
    if class_agnostic:
        same = np.ones((n, n), dtype=bool)
    else:
        same = classes[:, None] == classes[None, :]

    suppressed = np.zeros(n, dtype=bool)
    kept = np.zeros(n, dtype=bool)
    keep = []
    for i in np.argsort(-scores):
        if suppressed[i]:
            continue
        keep.append(int(i))
        kept[i] = True
        drop = same[i] & (sim[i] > nwd_threshold) & ~kept
        suppressed |= drop
    return keep


def nwd_confidence_rerank(boxes_xyxy, scores, classes, C=DEFAULT_C, top_k=5,
                          blend=0.3, min_sim=0.05):
    """Density-aware re-scoring of detections.

    A tiny object detected at 0.25 confidence in the middle of a cluster of
    confident same-class detections is usually real; the same box alone in an
    empty field usually isn't.  Each score is blended with a NWD-weighted average
    of its ``top_k`` nearest same-class neighbours' scores.

        new = (1 - blend) * old + blend * sum(nwd_k * conf_k) / sum(nwd_k)

    Only neighbours above ``min_sim`` count. Without that floor the exponential
    never quite reaches zero, and a lone detection on the far side of the frame
    would get boosted by a confident cluster it has nothing to do with.
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).copy()
    n = len(boxes_xyxy)
    if n < 2:
        return scores
    classes = np.asarray(classes).astype(int)

    cxcywh = xyxy_to_cxcywh(boxes_xyxy)
    sim = compute_nwd_matrix(cxcywh, cxcywh, C=C)
    np.fill_diagonal(sim, 0.0)
    same = classes[:, None] == classes[None, :]
    sim = np.where(same & (sim >= min_sim), sim, 0.0)

    out = scores.copy()
    for i in range(n):
        row = sim[i]
        n_neighbours = int((row > 0).sum())
        if n_neighbours == 0:
            continue
        k = min(top_k, n_neighbours)
        idx = np.argpartition(-row, k - 1)[:k]
        w, c = row[idx], scores[idx]
        if w.sum() > 0:
            out[i] = (1.0 - blend) * scores[i] + blend * float((w * c).sum() / w.sum())
    return out


def nwd_match(prev_boxes_xyxy, curr_boxes_xyxy, C=DEFAULT_C, threshold=0.4):
    """Greedy one-to-one matching between two frames' boxes.

    Used by the tracker: NWD keeps small fast-moving targets associated across
    frames where an IoU gate would break the track the moment the boxes stop
    overlapping.  Returns (pairs, unmatched_prev, unmatched_curr).
    """
    sim = nwd_matrix_xyxy(prev_boxes_xyxy, curr_boxes_xyxy, C=C)
    pairs = []
    used_p, used_c = set(), set()
    if sim.size:
        order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
        for p, c in order:
            p, c = int(p), int(c)
            if sim[p, c] < threshold:
                break
            if p in used_p or c in used_c:
                continue
            pairs.append((p, c, float(sim[p, c])))
            used_p.add(p)
            used_c.add(c)
    unmatched_p = [i for i in range(len(np.atleast_2d(prev_boxes_xyxy))) if i not in used_p]
    unmatched_c = [j for j in range(len(np.atleast_2d(curr_boxes_xyxy))) if j not in used_c]
    return pairs, unmatched_p, unmatched_c
