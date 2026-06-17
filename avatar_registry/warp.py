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
    composite: bool = False,
    face_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Warp texture through deformed triangles (painter's algorithm, ROI-based)."""
    height, width = texture.shape[:2]
    output = np.full((height, width, 3), background, dtype=np.uint8)
    covered = np.zeros((height, width), dtype=bool)

    neutral = _clip_triangle_points(neutral_2d.astype(np.float32), width, height)
    if deformed_2d.ndim == 2:
        deformed = _clip_triangle_points(deformed_2d.astype(np.float32), width, height)
    else:
        raise ValueError(f"Expected deformed_2d shape (N, 2), got {deformed_2d.shape}")

    tris = triangles.astype(np.int32)
    depth_order = np.argsort(deformed[tris].mean(axis=1)[:, 1])
    for tri_idx in depth_order:
        idx = tris[tri_idx]
        _warp_triangle_into(
            texture,
            output,
            covered,
            neutral[idx],
            deformed[idx],
            background=background,
        )

    if composite:
        output = _composite_with_mask(output, texture, face_mask)

    return output


def _clip_triangle_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = points.astype(np.float32, copy=True)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped


def _warp_triangle_into(
    texture: np.ndarray,
    output: np.ndarray,
    covered: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
    *,
    background: tuple[int, int, int],
) -> None:
    """Warp one triangle from texture (src) onto output (dst) using covered-mask compositing."""
    if _is_degenerate(src_tri) or _is_degenerate(dst_tri):
        return

    height, width = texture.shape[:2]
    x_min = int(max(0, np.floor(dst_tri[:, 0].min())))
    y_min = int(max(0, np.floor(dst_tri[:, 1].min())))
    x_max = int(min(width - 1, np.ceil(dst_tri[:, 0].max())))
    y_max = int(min(height - 1, np.ceil(dst_tri[:, 1].max())))
    if x_min >= x_max or y_min >= y_max:
        return

    roi_w = x_max - x_min + 1
    roi_h = y_max - y_min + 1

    dst_local = dst_tri.copy()
    dst_local[:, 0] -= x_min
    dst_local[:, 1] -= y_min

    matrix = cv2.getAffineTransform(dst_local.astype(np.float32), src_tri.astype(np.float32))
    warped = _remap_affine(texture, matrix, roi_w, roi_h, border_value=background)

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst_local).astype(np.int32), 255)
    triangle_mask = mask.astype(bool)
    if not triangle_mask.any():
        return

    out_roi = output[y_min : y_max + 1, x_min : x_max + 1]
    covered_roi = covered[y_min : y_max + 1, x_min : x_max + 1]
    newly = triangle_mask & ~covered_roi
    out_roi[newly] = warped[newly]
    covered_roi[triangle_mask] = True


def _remap_affine(
    image: np.ndarray,
    matrix: np.ndarray,
    width: int,
    height: int,
    *,
    border_value: tuple[int, int, int],
) -> np.ndarray:
    """Sample image using a 2x3 affine that maps ROI pixels to full-image coordinates."""
    map_x, map_y = _affine_coordinate_maps(matrix, width, height)
    border = border_value
    if len(border) == 3:
        border = int(border[0])
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def _affine_coordinate_maps(matrix: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    ones = np.ones_like(grid_x)
    stacked = np.stack([grid_x, grid_y, ones], axis=0).reshape(3, -1)
    mapped = matrix.astype(np.float32) @ stacked
    map_x = mapped[0].reshape(height, width)
    map_y = mapped[1].reshape(height, width)
    return map_x, map_y


def _composite_with_mask(
    warped: np.ndarray,
    texture: np.ndarray,
    face_mask: np.ndarray | None,
) -> np.ndarray:
    if face_mask is None:
        return texture.copy()
    alpha = face_mask.astype(np.float32)
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    blended = warped.astype(np.float32) * alpha + texture.astype(np.float32) * (1.0 - alpha)
    return blended.astype(np.uint8)


def _is_degenerate(triangle: np.ndarray) -> bool:
    area = cv2.contourArea(triangle.astype(np.float32))
    return area < 1.0
