"""Minimal ASCII FBX parser for LAM/OAC face template (mesh + ARKit shapes)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FbxFaceMesh:
    neutral_vertices: np.ndarray  # (N, 3)
    triangles: np.ndarray  # (T, 3) int32
    shapes: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (M, 3) target positions
    shape_indices: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (M,) int32


def _parse_float_list(block: str) -> list[float]:
    values: list[float] = []
    for token in re.split(r"[\s,]+", block.strip()):
        if token:
            values.append(float(token))
    return values


def _extract_array_section(text: str, key: str) -> tuple[int, list[float]] | None:
    pattern = rf"{key}: \*(\d+) \{{\s*\n\s*a: ([\s\S]*?)\n\s*\}}"
    match = re.search(pattern, text)
    if not match:
        return None
    count = int(match.group(1))
    values = _parse_float_list(match.group(2))
    if len(values) < count:
        raise ValueError(f"{key}: expected {count} values, got {len(values)}")
    return count, values[:count]


def _extract_index_section(text: str, key: str = "Indexes") -> tuple[int, list[int]] | None:
    pattern = rf"{key}: \*(\d+) \{{\s*\n\s*a: ([\s\S]*?)\n\s*\}}"
    match = re.search(pattern, text)
    if not match:
        return None
    count = int(match.group(1))
    raw = re.split(r"[\s,]+", match.group(2).strip())
    values = [int(x) for x in raw if x]
    if len(values) < count:
        raise ValueError(f"{key}: expected {count} values, got {len(values)}")
    return count, values[:count]


def _polygon_indices_to_triangles(polygon_indices: list[int]) -> np.ndarray:
    triangles: list[list[int]] = []
    polygon: list[int] = []
    for idx in polygon_indices:
        if idx < 0:
            polygon.append(-(idx + 1))
            if len(polygon) == 3:
                triangles.append(polygon)
            elif len(polygon) > 3:
                for i in range(1, len(polygon) - 1):
                    triangles.append([polygon[0], polygon[i], polygon[i + 1]])
            polygon = []
        else:
            polygon.append(idx)
    if not triangles:
        raise ValueError("No triangles parsed from PolygonVertexIndex")
    return np.asarray(triangles, dtype=np.int32)


def _split_geometry_blocks(text: str) -> list[tuple[str, str, str]]:
    blocks: list[tuple[str, str, str]] = []
    pattern = re.compile(
        r'Geometry: \d+, "Geometry::([^"]+)", "(Mesh|Shape)" \{(.*?)\n\t\}',
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        blocks.append((match.group(1), match.group(2), match.group(3)))
    return blocks


def load_fbx_face_mesh(
    fbx_path: str | Path,
    *,
    mesh_name: str = "网格",
) -> FbxFaceMesh:
    """Load neutral mesh and ARKit blend-shape targets from ASCII FBX."""
    text = Path(fbx_path).read_text(encoding="utf-8", errors="replace")
    blocks = _split_geometry_blocks(text)

    neutral_vertices: np.ndarray | None = None
    triangles: np.ndarray | None = None
    shapes: dict[str, np.ndarray] = {}
    shape_indices: dict[str, np.ndarray] = {}

    for name, kind, body in blocks:
        if kind == "Mesh" and name == mesh_name:
            vert_section = _extract_array_section(body, "Vertices")
            poly_section = _extract_array_section(body, "PolygonVertexIndex")
            if vert_section is None or poly_section is None:
                raise ValueError(f"Mesh '{mesh_name}' missing Vertices or PolygonVertexIndex")
            _, vert_flat = vert_section
            if len(vert_flat) % 3 != 0:
                raise ValueError("Vertices length is not divisible by 3")
            neutral_vertices = np.asarray(vert_flat, dtype=np.float64).reshape(-1, 3)
            _, poly_flat = poly_section
            poly_indices = [int(v) for v in poly_flat]
            triangles = _polygon_indices_to_triangles(poly_indices)
        elif kind == "Shape":
            idx_section = _extract_index_section(body, "Indexes")
            vert_section = _extract_array_section(body, "Vertices")
            if idx_section is None or vert_section is None:
                continue
            idx_count, indices = idx_section
            _, vert_flat = vert_section
            if len(vert_flat) % 3 != 0:
                raise ValueError(f"Shape '{name}' vertices not divisible by 3")
            positions = np.asarray(vert_flat, dtype=np.float64).reshape(-1, 3)
            if positions.shape[0] != idx_count:
                raise ValueError(
                    f"Shape '{name}': index count {idx_count} != vertex count {positions.shape[0]}"
                )
            shapes[name] = positions
            shape_indices[name] = np.asarray(indices, dtype=np.int32)

    if neutral_vertices is None or triangles is None:
        raise ValueError(f"Mesh '{mesh_name}' not found in FBX")

    return FbxFaceMesh(
        neutral_vertices=neutral_vertices,
        triangles=triangles,
        shapes=shapes,
        shape_indices=shape_indices,
    )


def shape_deltas_3d(mesh: FbxFaceMesh, shape_name: str) -> np.ndarray:
    """Dense per-vertex 3D delta for one blend shape."""
    if shape_name not in mesh.shapes:
        raise KeyError(f"Shape not found: {shape_name}")
    delta = np.zeros_like(mesh.neutral_vertices)
    indices = mesh.shape_indices[shape_name]
    targets = mesh.shapes[shape_name]
    delta[indices] = targets - mesh.neutral_vertices[indices]
    return delta
