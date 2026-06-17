"""Retarget LAM skin.glb mouth/jaw morph deltas onto the photo mouth region.

LAM skin.glb's jawOpen morph concentrates its motion on lower-neck/throat
vertices (which project far below the lips after 2D alignment) and even points
the wrong way (upward) in image space. Baking those raw deltas into the avatar
pack makes the warp shatter the chin/collar instead of opening the mouth.

This module replaces mouth/jaw morph deltas with photo-landmark-driven motion:
- the animated vertices are restricted to a lip+chin region of the mesh, and
- the open/close direction is derived from the photo's mouth->chin axis so the
  mouth opens downward regardless of the noisy GLB source direction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python-headless is required for avatar_registry.mouth_retarget") from exc

# Full lip contour (MediaPipe Face Landmarker indices).
LIP_LANDMARK_INDICES: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    269, 270, 267, 271, 272,
)
CHIN_LANDMARK_INDEX = 152


def is_mouth_jaw_morph(morph_name: str) -> bool:
    lower = morph_name.lower()
    return lower.startswith("mouth") or lower.startswith("jaw")


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    contour = polygon.reshape(-1, 1, 2).astype(np.float32)
    x_lo, y_lo = polygon.min(axis=0)
    x_hi, y_hi = polygon.max(axis=0)
    in_bbox = (
        (points[:, 0] >= x_lo)
        & (points[:, 0] <= x_hi)
        & (points[:, 1] >= y_lo)
        & (points[:, 1] <= y_hi)
    )
    inside = np.zeros(points.shape[0], dtype=bool)
    for idx in np.flatnonzero(in_bbox):
        pt = points[idx]
        inside[idx] = cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0
    return inside


def build_mouth_animation_region(
    photo_landmarks: np.ndarray,
    mouth_landmark_indices: Sequence[int] = LIP_LANDMARK_INDICES,
    chin_landmark_index: int = CHIN_LANDMARK_INDEX,
    chin_extend_ratio: float = 0.2,
) -> np.ndarray:
    """Convex hull of lip landmarks + chin for mouth/jaw morph retargeting."""
    lip = photo_landmarks[list(mouth_landmark_indices)].astype(np.float32)
    chin = photo_landmarks[chin_landmark_index].astype(np.float32)
    mouth_h = max(float(lip[:, 1].max() - lip[:, 1].min()), 1.0)
    chin_ext = chin + np.array([0.0, mouth_h * chin_extend_ratio], dtype=np.float32)
    hull = cv2.convexHull(np.vstack([lip, chin, chin_ext]).astype(np.float32))
    return hull.reshape(-1, 2)


def build_mouth_animation_mask(
    neutral_2d: np.ndarray,
    photo_landmarks: np.ndarray,
) -> np.ndarray:
    region = build_mouth_animation_region(photo_landmarks)
    return _points_in_polygon(neutral_2d, region)


def _mouth_morph_vertex_weights(
    morph_name: str,
    neutral_2d: np.ndarray,
    mouth_mask: np.ndarray,
    photo_landmarks: np.ndarray,
) -> np.ndarray:
    """Spatial weights inside the mouth animation region per morph semantics."""
    lip = photo_landmarks[list(LIP_LANDMARK_INDICES)]
    y_min = float(lip[:, 1].min())
    y_max = float(lip[:, 1].max())
    x_center = float(lip[:, 0].mean())
    mouth_h = max(y_max - y_min, 1.0)
    mouth_w = max(float(lip[:, 0].max() - lip[:, 0].min()), 1.0)

    weights = np.zeros(neutral_2d.shape[0], dtype=np.float32)
    for v in np.flatnonzero(mouth_mask):
        y_norm = float(np.clip((neutral_2d[v, 1] - y_min) / mouth_h, 0.0, 1.0))
        x_norm = float((neutral_2d[v, 0] - x_center) / max(mouth_w * 0.5, 1.0))

        if morph_name in ("jawOpen", "mouthClose", "jawForward"):
            weights[v] = 0.2 + 0.8 * y_norm
        elif morph_name.endswith("Left") and morph_name.startswith("mouth"):
            weights[v] = 0.15 + 0.85 * max(0.0, -x_norm)
        elif morph_name.endswith("Right") and morph_name.startswith("mouth"):
            weights[v] = 0.15 + 0.85 * max(0.0, x_norm)
        elif morph_name == "jawLeft":
            weights[v] = 0.2 + 0.8 * max(0.0, -x_norm)
        elif morph_name == "jawRight":
            weights[v] = 0.2 + 0.8 * max(0.0, x_norm)
        elif morph_name in ("mouthPucker", "mouthFunnel"):
            weights[v] = float(np.clip(1.0 - 0.65 * np.hypot(x_norm, y_norm - 0.5), 0.15, 1.0))
        else:
            weights[v] = 1.0
    return weights


def _photo_mouth_axes(photo_landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (down_dir, right_dir, mouth_height, mouth_width) in photo pixels."""
    lip = photo_landmarks[list(LIP_LANDMARK_INDICES)].astype(np.float64)
    chin = photo_landmarks[CHIN_LANDMARK_INDEX].astype(np.float64)
    center = lip.mean(axis=0)
    down = chin - center
    down_norm = np.linalg.norm(down)
    down_dir = down / down_norm if down_norm > 1e-6 else np.array([0.0, 1.0])
    right_dir = np.array([down_dir[1], -down_dir[0]])  # rotate -90deg
    mouth_h = max(float(lip[:, 1].max() - lip[:, 1].min()), 1.0)
    mouth_w = max(float(lip[:, 0].max() - lip[:, 0].min()), 1.0)
    return down_dir.astype(np.float32), right_dir.astype(np.float32), mouth_h, mouth_w


def retarget_mouth_jaw_morph_deltas_2d(
    neutral_2d: np.ndarray,
    morph_deltas_2d: np.ndarray,
    morph_names: Sequence[str],
    photo_landmarks: np.ndarray,
    *,
    mouth_open_scale: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Move mouth/jaw morph motion from neck mesh vertices onto the photo mouth region.

    For jawOpen / mouthClose the direction is taken from the photo mouth->chin axis
    (so the mouth opens downward), since the GLB source direction is unreliable.
    Other mouth/jaw morphs keep the GLB source direction but are localized + rescaled.
    """
    mouth_mask = build_mouth_animation_mask(neutral_2d, photo_landmarks)
    if not mouth_mask.any():
        raise RuntimeError("Mouth animation mask is empty; check landmark alignment.")

    down_dir, right_dir, mouth_h, _mouth_w = _photo_mouth_axes(photo_landmarks)
    mouth_vertices = np.flatnonzero(mouth_mask)
    retargeted = morph_deltas_2d.copy()
    stats: list[dict] = []

    for morph_idx, morph_name in enumerate(morph_names):
        if not is_mouth_jaw_morph(morph_name):
            continue

        delta = morph_deltas_2d[morph_idx]
        mag = np.linalg.norm(delta, axis=1)
        if float(mag.max()) < 1e-5:
            continue

        positive = mag > 0.0
        if int(positive.sum()) < 5:
            continue

        # Weighted average of the largest GLB displacements as the source motion.
        source_threshold = float(np.percentile(mag[positive], 90))
        source_mask = mag >= max(source_threshold, float(mag.max()) * 0.08)
        source_weights = mag[source_mask]
        source_delta = np.average(delta[source_mask], axis=0, weights=source_weights)
        source_mag = float(np.linalg.norm(source_delta))
        if source_mag < 1e-5:
            continue

        # Direction + magnitude: override vertical mouth motion with the photo axis.
        if morph_name == "jawOpen":
            target_delta = down_dir * (mouth_h * 1.1 * mouth_open_scale)
        elif morph_name == "mouthClose":
            target_delta = -down_dir * (mouth_h * 0.85 * mouth_open_scale)
        else:
            target_delta = source_delta.astype(np.float32)

        vertex_weights = _mouth_morph_vertex_weights(
            morph_name, neutral_2d, mouth_mask, photo_landmarks
        )
        new_delta = np.zeros_like(delta)
        new_delta[mouth_vertices] = (
            target_delta[None, :] * vertex_weights[mouth_vertices, None]
        )
        retargeted[morph_idx] = new_delta
        stats.append(
            {
                "name": morph_name,
                "source_mag_px": round(source_mag, 2),
                "target_mag_px": round(float(np.linalg.norm(target_delta)), 2),
                "mouth_vertices": int(mouth_mask.sum()),
            }
        )

    info = {
        "method": "mouth_jaw_retarget_photoaxis",
        "mouth_vertex_count": int(mouth_mask.sum()),
        "mouth_height_px": round(mouth_h, 2),
        "mouth_open_scale": mouth_open_scale,
        "morph_count": len(stats),
        "morphs": stats,
    }
    return retargeted, info
