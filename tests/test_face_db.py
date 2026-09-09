"""Watchlist enrolment, matching gates and persistence."""
import numpy as np
import pytest

from swarm_drone.vision.embedders import HashEmbedder, l2_normalize, blur_score
from swarm_drone.vision.face_db import WantedFaceDB


@pytest.fixture
def db(rng):
    d = WantedFaceDB(dim=64)
    for pid in ("W-001", "W-002", "W-003"):
        d.enroll(pid, rng.normal(size=(4, 64)), name=f"Person {pid}")
    return d


def test_enrolment_accumulates(rng):
    d = WantedFaceDB(dim=32)
    d.enroll("W-1", rng.normal(size=32), name="One")
    d.enroll("W-1", rng.normal(size=(3, 32)))
    assert len(d) == 1
    assert d.n_vectors == 4
    assert d.people["W-1"]["n"] == 4
    assert d.name_of("W-1") == "One"      # a later enrol keeps the first name


def test_stored_vectors_are_unit_norm(rng):
    d = WantedFaceDB(dim=16)
    d.enroll("W-1", rng.normal(size=(5, 16)) * 37.0)
    assert np.allclose(np.linalg.norm(d.vectors, axis=1), 1.0, atol=1e-5)


def test_wrong_dimension_is_rejected():
    d = WantedFaceDB(dim=16)
    with pytest.raises(ValueError, match="16-d"):
        d.enroll("W-1", np.zeros(32))


def test_a_near_copy_of_an_enrolled_face_matches(db, rng):
    query = db.vectors[4] + rng.normal(scale=0.03, size=64)
    match = db.search(query, threshold=0.3, margin=0.05)
    assert match is not None
    assert match.person_id == db.owner[4]
    assert match.score > 0.9


def test_a_stranger_does_not_match(db, rng):
    assert db.search(rng.normal(size=64), threshold=0.5) is None


def test_the_margin_gate_suppresses_a_near_tie(rng):
    """Two enrolled people who look alike must produce no match, not a coin flip."""
    d = WantedFaceDB(dim=64)
    base = rng.normal(size=64)
    d.enroll("W-A", base, name="A")
    d.enroll("W-B", base + rng.normal(scale=0.01, size=64), name="B")
    assert d.search(base, threshold=0.3, margin=0.05) is None
    # With the gate off, it happily returns one of them.
    assert d.search(base, threshold=0.3, margin=0.0) is not None


def test_many_photos_do_not_out_vote_a_better_match(rng):
    """Scoring is best-per-person, so enrolment count cannot swamp similarity."""
    d = WantedFaceDB(dim=64)
    target = l2_normalize(rng.normal(size=64))
    d.enroll("W-CORRECT", target, name="Correct")
    d.enroll("W-CROWD", rng.normal(size=(50, 64)), name="Crowd")
    match = d.search(target, threshold=0.3, margin=0.02)
    assert match is not None and match.person_id == "W-CORRECT"


def test_empty_database_matches_nothing(rng):
    assert WantedFaceDB(dim=8).search(rng.normal(size=8)) is None


def test_rank_orders_candidates(db):
    ranked = db.rank(db.vectors[0], top_k=3)
    assert ranked[0][0] == db.owner[0]
    assert [s for _, _, s in ranked] == sorted(
        [s for _, _, s in ranked], reverse=True)


def test_remove_deletes_the_person_and_their_vectors(db):
    removed = db.remove("W-002")
    assert removed == 4
    assert "W-002" not in db.ids()
    assert db.n_vectors == 8
    assert "W-002" not in db.owner
    assert db.remove("nobody") == 0


def test_save_and_load_round_trip(db, tmp_path, rng):
    path = str(tmp_path / "wanted.npz")
    db.save(path)
    loaded = WantedFaceDB.load(path)
    assert loaded.summary() == db.summary()
    assert np.allclose(loaded.vectors, db.vectors)
    query = db.vectors[0] + rng.normal(scale=0.02, size=64)
    assert (loaded.search(query, threshold=0.3).person_id
            == db.search(query, threshold=0.3).person_id)


def test_load_or_empty_tolerates_a_missing_file(tmp_path):
    d = WantedFaceDB.load_or_empty(str(tmp_path / "nope.npz"), dim=128)
    assert len(d) == 0 and d.dim == 128


def test_hash_embedder_is_deterministic_and_discriminative(rng):
    e = HashEmbedder()
    a = rng.integers(0, 255, (40, 30, 3)).astype(np.uint8)
    b = rng.integers(0, 255, (40, 30, 3)).astype(np.uint8)
    noisy = np.clip(a.astype(int) + rng.integers(-10, 10, a.shape),
                    0, 255).astype(np.uint8)
    va, vb, vn = (e.embed(x)[0].vector for x in (a, b, noisy))
    assert float(va @ e.embed(a)[0].vector) == pytest.approx(1.0, abs=1e-5)
    assert float(va @ vn) > 0.9
    assert abs(float(va @ vb)) < 0.5


def test_embedder_returns_nothing_for_an_empty_crop():
    assert HashEmbedder().embed(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    assert HashEmbedder().embed(None) == []


def test_blur_score_ranks_a_smeared_crop_below_a_sharp_one(rng):
    sharp = rng.integers(0, 255, (60, 60, 3)).astype(np.uint8)
    smeared = np.repeat(np.repeat(sharp[::10, ::10], 10, axis=0), 10, axis=1)
    assert blur_score(sharp) > blur_score(smeared)
