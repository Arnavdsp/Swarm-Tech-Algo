"""
Face embedding backends.

All of them turn a person/face crop into a unit-norm vector; matching is then a
dot product against the watchlist.  Three implementations:

  * ``InsightFaceEmbedder`` — the real one. SCRFD detects a face inside the
    person crop, ArcFace (buffalo_l) embeds it to 512-d. Runs on the Pi's CPU at
    roughly 8-12 crops/s, which is why the pipeline only samples every Nth frame.
  * ``OnnxEmbedder``        — any ONNX face model, for a custom/quantised net.
  * ``HashEmbedder``        — deterministic pixel-hash, no ML. Not for real use;
    it makes the pipeline and the database testable offline.

Every backend returns ``[]`` when it finds no face, so a person crop that is all
back-of-head simply produces no candidate rather than a bad match.
"""
import numpy as np

EMBED_DIM = 512


def l2_normalize(v, axis=-1, eps=1e-10):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)


class FaceEmbedding:
    """One detected face: its embedding, box (in crop coords) and quality."""

    __slots__ = ("vector", "box", "det_score", "blur")

    def __init__(self, vector, box=None, det_score=1.0, blur=1.0):
        self.vector = l2_normalize(vector)
        self.box = box
        self.det_score = float(det_score)
        self.blur = float(blur)


def blur_score(crop):
    """Variance of the Laplacian. Low = motion-blurred, don't trust a match on it.

    Aerial video at altitude is often too smeared to identify anyone; scoring it
    up front is cheaper than trusting a match made on mush.

    The input is cast to float32 before the Laplacian: OpenCV 5 rejects the
    uint8-in/CV_64F-out combination that the usual recipe uses, and a scoring
    metric is never worth crashing an enrolment run over — hence the fallback.
    """
    c = np.asarray(crop)
    try:
        import cv2
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
        return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())
    except Exception:                                   # noqa: BLE001
        g = c.mean(axis=2) if c.ndim == 3 else c
        return float(np.var(np.diff(g.astype(np.float32), axis=0)))


class BaseEmbedder:
    dim = EMBED_DIM

    def embed(self, crop):
        """Return a list of FaceEmbedding found in ``crop`` (often 0 or 1)."""
        raise NotImplementedError


class InsightFaceEmbedder(BaseEmbedder):
    """SCRFD detection + ArcFace recognition via insightface's FaceAnalysis."""

    def __init__(self, model_name="buffalo_l", det_size=320, ctx_id=0,
                 min_det_score=0.5):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:                       # pragma: no cover
            raise ImportError(
                "InsightFaceEmbedder needs: pip install insightface onnxruntime"
            ) from exc
        self.app = FaceAnalysis(name=model_name,
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self.min_det_score = min_det_score

    def embed(self, crop):
        if crop is None or crop.size == 0:
            return []
        faces = self.app.get(crop)
        out = []
        for f in faces:
            if f.det_score < self.min_det_score:
                continue
            out.append(FaceEmbedding(f.normed_embedding,
                                     box=np.asarray(f.bbox, dtype=np.float32),
                                     det_score=float(f.det_score),
                                     blur=blur_score(crop)))
        return out


class OnnxEmbedder(BaseEmbedder):
    """Generic ONNX recogniser: whole crop in, one embedding out.

    Assumes a face-cropped, aligned input — pair it with your own detector, or
    feed it person crops if the model tolerates them.
    """

    def __init__(self, model_path, input_size=112, providers=None):
        try:
            import onnxruntime as ort
        except ImportError as exc:                       # pragma: no cover
            raise ImportError("OnnxEmbedder needs: pip install onnxruntime") from exc
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.dim = int(self.session.get_outputs()[0].shape[-1])

    def _preprocess(self, crop):
        import cv2
        img = cv2.resize(crop, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        return img.transpose(2, 0, 1)[None]

    def embed(self, crop):
        if crop is None or crop.size == 0:
            return []
        vec = self.session.run(None, {self.input_name: self._preprocess(crop)})[0][0]
        return [FaceEmbedding(vec, det_score=1.0, blur=blur_score(crop))]


class HashEmbedder(BaseEmbedder):
    """Deterministic, dependency-free stand-in used by the tests.

    Downsamples the crop to an 8x8x3 signature and projects it through a fixed
    random matrix.  Same crop -> same vector, similar crops -> similar vectors,
    which is all the pipeline logic needs to be exercised. It identifies nobody.
    """

    def __init__(self, dim=EMBED_DIM, seed=1234):
        self.dim = dim
        self.proj = np.random.default_rng(seed).normal(
            size=(192, dim)).astype(np.float32)

    def embed(self, crop):
        if crop is None or crop.size == 0:
            return []
        c = np.asarray(crop, dtype=np.float32)
        if c.ndim == 2:
            c = np.stack([c] * 3, axis=-1)
        h, w = c.shape[:2]
        if h < 1 or w < 1:
            return []
        ys = np.linspace(0, h - 1, 8).astype(int)
        xs = np.linspace(0, w - 1, 8).astype(int)
        sig = c[np.ix_(ys, xs)].reshape(-1) / 255.0
        sig -= sig.mean()      # centre, or every crop looks like every other one
        return [FaceEmbedding(sig @ self.proj, det_score=1.0, blur=blur_score(c))]


def make_embedder(cfg):
    """Build the embedder named in ``VisionConfig.face_backend``."""
    backend = (cfg.face_backend or "null").lower()
    if backend == "insightface":
        return InsightFaceEmbedder(model_name=cfg.face_model,
                                   det_size=cfg.face_det_size)
    if backend == "onnx":
        return OnnxEmbedder(cfg.face_model)
    if backend in ("hash", "null", "none"):
        return HashEmbedder()
    raise ValueError(f"unknown face backend: {cfg.face_backend}")
