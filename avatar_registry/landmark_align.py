"""Photo–mesh landmark alignment for 2D avatar packs."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python-headless is required for avatar_registry.landmark_align") from exc

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# MediaPipe Face Landmarker indices (person-centric naming).
_PHOTO_LANDMARKS: dict[str, int] = {
    "forehead": 10,
    "nose": 1,
    "left_eye": 33,
    "right_eye": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "chin": 152,
}

_LANDMARK_ORDER = list(_PHOTO_LANDMARKS.keys())

# MediaPipe face oval indices for soft runtime compositing mask.
_FACE_OVAL_INDICES: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
)


def ensure_face_landmarker_model(model_path: Path | None = None) -> Path:
    path = Path(model_path) if model_path else _MODEL_PATH
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, path)
    return path


def detect_photo_landmarks(
    image_rgb: np.ndarray,
    *,
    model_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Return named 2D landmarks in pixel coordinates (x, y)."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    height, width = image_rgb.shape[:2]
    model_file = ensure_face_landmarker_model(model_path)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_file)),
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError("No face detected in photo for landmark alignment")

    landmarks = result.face_landmarks[0]
    points: dict[str, np.ndarray] = {}
    for name, index in _PHOTO_LANDMARKS.items():
        lm = landmarks[index]
        points[name] = np.array([lm.x * width, lm.y * height], dtype=np.float64)
    return points


def detect_face_landmark_points(
    image_rgb: np.ndarray,
    *,
    model_path: Path | None = None,
) -> np.ndarray:
    """Return all face landmark pixel coordinates as (N, 2) float64."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    height, width = image_rgb.shape[:2]
    model_file = ensure_face_landmarker_model(model_path)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_file)),
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError("No face detected in photo for face mask")

    landmarks = result.face_landmarks[0]
    return np.array(
        [[lm.x * width, lm.y * height] for lm in landmarks],
        dtype=np.float64,
    )


def build_face_mask_from_points(
    landmarks: np.ndarray,
    image_height: int,
    image_width: int,
    *,
    feather_sigma: float = 4.0,
) -> np.ndarray:
    """Soft uint8 mask (0-255) from a precomputed MediaPipe face oval polygon."""
    polygon = landmarks[list(_FACE_OVAL_INDICES)].astype(np.float32)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    if feather_sigma > 0.0:
        ksize = int(max(3, round(feather_sigma * 4))) | 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_sigma)
    return mask


def build_face_mask(
    image_rgb: np.ndarray,
    *,
    feather_sigma: float = 4.0,
    model_path: Path | None = None,
) -> np.ndarray:
    """Soft uint8 mask (0-255) from MediaPipe face oval polygon."""
    height, width = image_rgb.shape[:2]
    landmarks = detect_face_landmark_points(image_rgb, model_path=model_path)
    return build_face_mask_from_points(landmarks, height, width, feather_sigma=feather_sigma)


def face_vertex_mask(vertices_3d: np.ndarray, *, z_percentile: float = 55.0) -> np.ndarray:
    """Vertices on the forward-facing skin surface."""
    z_cut = np.percentile(vertices_3d[:, 2], z_percentile)
    return vertices_3d[:, 2] >= z_cut


def assign_mesh_vertices_to_photo_landmarks(
    coarse_2d: np.ndarray,
    photo_landmarks: dict[str, np.ndarray],
    vertices_3d: np.ndarray,
    *,
    current_2d: np.ndarray | None = None,
) -> dict[str, int]:
    """Map each photo landmark to the nearest coarse mesh vertex (unique assignment)."""
    current = current_2d if current_2d is not None else coarse_2d
    mask = face_vertex_mask(vertices_3d)
    used: set[int] = set()
    assignments: dict[str, int] = {}

    for name in _LANDMARK_ORDER:
        target = photo_landmarks[name]
        candidates = np.where(mask)[0]
        candidates = np.asarray([idx for idx in candidates if idx not in used], dtype=np.int64)
        if candidates.size == 0:
            candidates = np.setdiff1d(np.arange(coarse_2d.shape[0]), list(used))

        dists = np.sum((current[candidates] - target) ** 2, axis=1)
        pick = int(candidates[int(np.argmin(dists))])
        assignments[name] = pick
        used.add(pick)

    return assignments


def estimate_alignment_matrix(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    *,
    method: str = "similarity",
) -> tuple[np.ndarray, float]:
    """Estimate 2x3 matrix mapping coarse mesh 2D coords to photo pixel coords."""
    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 landmark pairs for alignment")

    if method == "similarity":
        matrix, _ = cv2.estimateAffinePartial2D(src, dst)
    elif method == "affine":
        matrix, _ = cv2.estimateAffine2D(src, dst)
    else:
        raise ValueError(f"Unknown alignment method: {method}")

    if matrix is None:
        raise RuntimeError("Failed to estimate photo–mesh alignment transform")

    aligned = apply_affine_points(src, matrix)
    rms = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return matrix.astype(np.float64), rms


def apply_affine_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (matrix @ np.concatenate([pts, ones], axis=1).T).T


def apply_affine_to_vertices(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return apply_affine_points(points, matrix).astype(np.float32)


def apply_affine_to_deltas(deltas: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    linear = np.asarray(matrix[:2, :2], dtype=np.float32)
    return np.einsum("ij,...j->...i", linear, deltas.astype(np.float32))


def align_mesh_to_photo(
    coarse_2d: np.ndarray,
    deltas_2d: np.ndarray,
    vertices_3d: np.ndarray,
    photo_rgb: np.ndarray,
    *,
    method: str = "similarity",
    iterations: int = 3,
    model_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Align coarse mesh 2D coords and morph deltas to detected photo landmarks."""
    photo_landmarks = detect_photo_landmarks(photo_rgb, model_path=model_path)
    current = coarse_2d.astype(np.float64)
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    assignments: dict[str, int] = {}

    for _ in range(max(1, iterations)):
        assignments = assign_mesh_vertices_to_photo_landmarks(
            coarse_2d,
            photo_landmarks,
            vertices_3d,
            current_2d=current,
        )
        src = np.stack([coarse_2d[assignments[name]] for name in _LANDMARK_ORDER], axis=0)
        dst = np.stack([photo_landmarks[name] for name in _LANDMARK_ORDER], axis=0)
        matrix, _ = estimate_alignment_matrix(src, dst, method=method)
        current = apply_affine_points(coarse_2d, matrix)

    src = np.stack([coarse_2d[assignments[name]] for name in _LANDMARK_ORDER], axis=0)
    dst = np.stack([photo_landmarks[name] for name in _LANDMARK_ORDER], axis=0)
    _, rms = estimate_alignment_matrix(src, dst, method=method)

    aligned_vertices = apply_affine_to_vertices(coarse_2d, matrix)
    aligned_deltas = apply_affine_to_deltas(deltas_2d, matrix)
    meta = {
        "alignment_method": method,
        "alignment_iterations": max(1, iterations),
        "alignment_rms_px": round(rms, 3),
        "mesh_landmark_indices": assignments,
        "photo_landmark_names": list(_LANDMARK_ORDER),
    }
    return aligned_vertices, aligned_deltas, meta
