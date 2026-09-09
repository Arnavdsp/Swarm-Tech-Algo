"""NWD similarity, suppression, re-ranking and frame-to-frame matching."""
import numpy as np
import pytest

from swarm_drone.vision.nwd import (compute_nwd_matrix, cxcywh_to_xyxy,
                                    nwd_confidence_rerank, nwd_match, nwd_nms,
                                    nwd_matrix_xyxy, xyxy_to_cxcywh)


def test_box_conversions_round_trip():
    boxes = np.array([[10, 20, 30, 50], [0, 0, 4, 4]], dtype=np.float32)
    assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(boxes)), boxes)


def test_identical_boxes_score_one():
    b = np.array([[10, 10, 20, 20]], dtype=np.float32)
    assert nwd_matrix_xyxy(b, b)[0, 0] == pytest.approx(1.0)


def test_nwd_degrades_gracefully_where_iou_collapses():
    """The whole reason NWD is used on aerial footage."""
    a = np.array([[0, 0, 6, 6]], dtype=np.float32)      # a 6 px object
    b = np.array([[3, 3, 9, 9]], dtype=np.float32)      # shifted 3 px
    iou = 9.0 / (36 + 36 - 9)                            # ~0.14
    nwd = float(nwd_matrix_xyxy(a, b)[0, 0])
    assert iou < 0.15
    assert nwd > 0.6


def test_nwd_falls_off_with_distance():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    near = np.array([[2, 0, 12, 10]], dtype=np.float32)
    far = np.array([[200, 0, 210, 10]], dtype=np.float32)
    assert nwd_matrix_xyxy(a, near)[0, 0] > nwd_matrix_xyxy(a, far)[0, 0]


def test_empty_input_is_shaped_not_crashed():
    empty = np.zeros((0, 4), dtype=np.float32)
    assert compute_nwd_matrix(empty, empty).shape == (0, 0)
    assert nwd_nms(empty, np.zeros(0), np.zeros(0)) == []
    assert len(nwd_confidence_rerank(empty, np.zeros(0), np.zeros(0))) == 0


def test_nms_drops_duplicates_and_keeps_the_best():
    boxes = np.array([[10, 10, 20, 20],
                      [11, 11, 21, 21],      # near-duplicate of the first
                      [100, 100, 112, 114]], dtype=np.float32)
    keep = nwd_nms(boxes, np.array([0.9, 0.7, 0.5]), np.array([0, 0, 3]))
    assert keep == [0, 2]                     # highest score survives


def test_nms_is_class_aware_unless_told_otherwise():
    boxes = np.array([[10, 10, 20, 20], [11, 11, 21, 21]], dtype=np.float32)
    scores = np.array([0.9, 0.7])
    assert len(nwd_nms(boxes, scores, np.array([0, 3]))) == 2         # different
    assert len(nwd_nms(boxes, scores, np.array([0, 3]),
                       class_agnostic=True)) == 1


def test_nms_never_suppresses_something_it_already_kept():
    rng = np.random.default_rng(4)
    boxes = rng.uniform(0, 60, size=(40, 4))
    boxes[:, 2:] = boxes[:, :2] + rng.uniform(4, 12, size=(40, 2))
    keep = nwd_nms(boxes.astype(np.float32), rng.random(40),
                   np.zeros(40, dtype=int))
    assert len(keep) == len(set(keep))


def test_rerank_boosts_a_detection_inside_a_confident_cluster():
    boxes = np.array([[10, 10, 18, 18], [14, 12, 22, 20],
                      [12, 16, 20, 24]], dtype=np.float32)
    scores = np.array([0.9, 0.85, 0.30])
    out = nwd_confidence_rerank(boxes, scores, np.zeros(3, dtype=int))
    assert out[2] > scores[2]


def test_rerank_leaves_an_isolated_detection_alone():
    """Without the min_sim floor a far-away cluster would still lift this."""
    boxes = np.array([[10, 10, 18, 18], [14, 12, 22, 20],
                      [900, 700, 908, 708]], dtype=np.float32)
    scores = np.array([0.9, 0.9, 0.30])
    out = nwd_confidence_rerank(boxes, scores, np.zeros(3, dtype=int))
    assert out[2] == pytest.approx(0.30)


def test_rerank_preserves_length_and_stays_bounded():
    rng = np.random.default_rng(5)
    boxes = rng.uniform(0, 200, size=(25, 4)).astype(np.float32)
    boxes[:, 2:] = boxes[:, :2] + 10
    scores = rng.random(25).astype(np.float32)
    out = nwd_confidence_rerank(boxes, scores, np.zeros(25, dtype=int))
    assert out.shape == scores.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_match_associates_a_small_shift():
    prev = np.array([[10, 10, 20, 20], [200, 200, 214, 216]], dtype=np.float32)
    curr = prev + 2.0
    pairs, un_p, un_c = nwd_match(prev, curr, threshold=0.4)
    assert sorted((p, c) for p, c, _ in pairs) == [(0, 0), (1, 1)]
    assert un_p == [] and un_c == []


def test_match_reports_a_new_object_as_unmatched():
    prev = np.array([[10, 10, 20, 20]], dtype=np.float32)
    curr = np.array([[10, 10, 20, 20], [500, 400, 512, 414]], dtype=np.float32)
    pairs, un_p, un_c = nwd_match(prev, curr, threshold=0.4)
    assert [(p, c) for p, c, _ in pairs] == [(0, 0)]
    assert un_c == [1]


def test_match_is_one_to_one():
    prev = np.array([[10, 10, 20, 20], [12, 12, 22, 22]], dtype=np.float32)
    curr = np.array([[11, 11, 21, 21]], dtype=np.float32)
    pairs, _, _ = nwd_match(prev, curr, threshold=0.3)
    assert len(pairs) == 1


def test_c_controls_the_similarity_scale():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[8, 0, 18, 10]], dtype=np.float32)
    assert nwd_matrix_xyxy(a, b, C=40.0)[0, 0] > nwd_matrix_xyxy(a, b, C=5.0)[0, 0]
