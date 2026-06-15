"""Piecewise-affine mesh warp renderer."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python-headless is required for avatar_registry.warp") from exc


def render_mesh_frame(
    texture: np.ndarray,
    neutral_2d: np.ndarray,
    deformed_2d: np.ndarray,
    triangles: np.ndarray,
    *,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Warp texture through deformed triangles (painter's algorithm, ROI-based)."""
    height, width = texture.shape[:2]
    output = np.full((height, width, 3), background, dtype=np.uint8)
    covered = np.zeros((height, width), dtype=bool)

    neutral = neutral_2d.astype(np.float32)
    deformed = deformed_2d.astype(np.float32)
    tris = triangles.astype(np.int32)

    depth_order = np.argsort(deformed[tris].mean(axis=1)[:, 1])
    for tri_idx in depth_order:
        idx = tris[tri_idx]
        src = neutral[idx]
        dst = deformed[idx]
        if _is_degenerate(src) or _is_degenerate(dst):
            continue

        x_min = max(int(np.floor(dst[:, 0].min())) - 1, 0)
        y_min = max(int(np.floor(dst[:, 1].min())) - 1, 0)
        x_max = min(int(np.ceil(dst[:, 0].max())) + 2, width)
        y_max = min(int(np.ceil(dst[:, 1].max())) + 2, height)
        roi_w = x_max - x_min
        roi_h = y_max - y_min
        if roi_w <= 1 or roi_h <= 1:
            continue

        offset = np.array([x_min, y_min], dtype=np.float32)
        matrix = cv2.getAffineTransform(src - offset, dst - offset)
        warped = cv2.warpAffine(
            texture,
            matrix,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background,
        )

        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, (dst - offset).astype(np.int32), 255)
        triangle_mask = mask.astype(bool)
        if not triangle_mask.any():
            continue

        out_roi = output[y_min:y_max, x_min:x_max]
        covered_roi = covered[y_min:y_max, x_min:x_max]
        newly = triangle_mask & ~covered_roi
        out_roi[newly] = warped[newly]
        covered_roi[triangle_mask] = True

    return output


def _is_degenerate(triangle: np.ndarray) -> bool:
    area = cv2.contourArea(triangle.astype(np.float32))
    return area < 1.0
