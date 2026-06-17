"""Load LAM per-person face mesh from skin.glb (neutral + ARKit morph deltas)."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_COMPONENT_TYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


@dataclass
class LamFaceMesh:
    neutral_vertices: np.ndarray  # (N, 3)
    triangles: np.ndarray  # (T, 3) int32
    morph_deltas_3d: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N, 3)


@dataclass
class _GltfBundle:
    json: dict
    blob: bytes


def _load_glb(path: Path) -> _GltfBundle:
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"Not a GLB file: {path}")
    offset = 12
    json_chunk = None
    bin_chunk = b""
    while offset < len(data):
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == b"JSON":
            json_chunk = json.loads(chunk.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk
    if json_chunk is None:
        raise ValueError(f"GLB missing JSON chunk: {path}")
    return _GltfBundle(json=json_chunk, blob=bin_chunk)


def _quat_to_matrix(quat: list[float]) -> np.ndarray:
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _node_local_matrix(node: dict) -> np.ndarray:
    if node.get("matrix"):
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    matrix = np.eye(4, dtype=np.float64)
    if node.get("rotation"):
        matrix[:3, :3] = _quat_to_matrix(node["rotation"])
    if node.get("scale"):
        scale = np.diag([*node["scale"], 1.0])
        matrix = matrix @ scale
    if node.get("translation"):
        matrix[:3, 3] = np.asarray(node["translation"], dtype=np.float64)
    return matrix


def _node_world_matrix(gltf: dict, node_index: int) -> np.ndarray:
    nodes = gltf["nodes"]
    parent_map: dict[int, int] = {}
    for parent_idx, node in enumerate(nodes):
        for child in node.get("children", []) or []:
            parent_map[child] = parent_idx

    chain = [node_index]
    while node_index in parent_map:
        node_index = parent_map[node_index]
        chain.append(node_index)

    world = np.eye(4, dtype=np.float64)
    for idx in reversed(chain):
        world = world @ _node_local_matrix(nodes[idx])
    return world


def _read_accessor(
    gltf: _GltfBundle,
    accessor_index: int,
    *,
    fill_sparse: bool = True,
) -> np.ndarray:
    accessor = gltf.json["accessors"][accessor_index]
    dtype = _COMPONENT_TYPES[accessor["componentType"]]
    components = _TYPE_COMPONENTS[accessor["type"]]
    count = accessor["count"]

    if accessor.get("bufferView") is None:
        dense = np.zeros((count, components), dtype=np.float64)
    else:
        view = gltf.json["bufferViews"][accessor["bufferView"]]
        start = (view.get("byteOffset") or 0) + (accessor.get("byteOffset") or 0)
        stride = view.get("byteStride") or (dtype().itemsize * components)
        if stride == dtype().itemsize * components:
            dense = np.frombuffer(
                gltf.blob,
                dtype=dtype,
                count=count * components,
                offset=start,
            ).reshape(count, components)
        else:
            rows = []
            for i in range(count):
                row_start = start + i * stride
                rows.append(
                    np.frombuffer(gltf.blob, dtype=dtype, count=components, offset=row_start).copy()
                )
            dense = np.stack(rows, axis=0)
        dense = dense.astype(np.float64)

    if fill_sparse and accessor.get("sparse"):
        sparse = accessor["sparse"]
        sparse_count = sparse["count"]
        idx_info = sparse["indices"]
        val_info = sparse["values"]
        idx_dtype = _COMPONENT_TYPES[idx_info["componentType"]]
        idx_view = gltf.json["bufferViews"][idx_info["bufferView"]]
        idx_start = (idx_view.get("byteOffset") or 0) + (idx_info.get("byteOffset") or 0)
        indices = np.frombuffer(
            gltf.blob,
            dtype=idx_dtype,
            count=sparse_count,
            offset=idx_start,
        ).astype(np.int64)

        val_view = gltf.json["bufferViews"][val_info["bufferView"]]
        val_start = (val_view.get("byteOffset") or 0) + (val_info.get("byteOffset") or 0)
        values = np.frombuffer(
            gltf.blob,
            dtype=dtype,
            count=sparse_count * components,
            offset=val_start,
        ).reshape(sparse_count, components).astype(np.float64)

        if dense.shape[0] != count:
            dense = np.zeros((count, components), dtype=np.float64)
        dense[indices] = values

    if components == 1:
        return dense.reshape(-1)
    return dense


def _indices_to_triangles(indices: np.ndarray) -> np.ndarray:
    if len(indices) % 3 != 0:
        raise ValueError(f"Index count {len(indices)} is not divisible by 3")
    return np.asarray(indices, dtype=np.int32).reshape(-1, 3)


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([points.astype(np.float64), ones], axis=1)
    return (matrix @ homo.T).T[:, :3]


def _transform_vectors(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return (matrix[:3, :3] @ vectors.astype(np.float64).T).T


def load_lam_skin_glb(glb_path: str | Path) -> LamFaceMesh:
    """Load neutral mesh, triangles, and dense 3D morph deltas from LAM skin.glb."""
    glb_path = Path(glb_path)
    gltf = _load_glb(glb_path)

    mesh_index = 0
    mesh = gltf.json["meshes"][mesh_index]
    primitive = mesh["primitives"][0]
    target_names = (mesh.get("extras") or {}).get("targetNames")
    if not target_names:
        raise ValueError(f"No morph targetNames in {glb_path}")

    mesh_node_index = next(
        i for i, node in enumerate(gltf.json["nodes"]) if node.get("mesh") == mesh_index
    )
    world_matrix = _node_world_matrix(gltf.json, mesh_node_index)

    positions = _read_accessor(gltf, primitive["attributes"]["POSITION"])
    neutral = _transform_points(world_matrix, positions)

    indices = _read_accessor(gltf, primitive["indices"], fill_sparse=False)
    triangles = _indices_to_triangles(indices)

    morph_deltas_3d: dict[str, np.ndarray] = {}
    num_vertices = neutral.shape[0]
    for target, name in zip(primitive.get("targets", []), target_names):
        if "POSITION" not in target:
            continue
        delta = _read_accessor(gltf, target["POSITION"])
        if delta.ndim == 1:
            delta = delta.reshape(-1, 3)
        if delta.shape[0] != num_vertices:
            raise ValueError(f"Unexpected morph delta shape for {name}: {delta.shape}")
        morph_deltas_3d[name] = _transform_vectors(world_matrix, delta).astype(np.float32)

    return LamFaceMesh(
        neutral_vertices=neutral.astype(np.float32),
        triangles=triangles,
        morph_deltas_3d=morph_deltas_3d,
    )


def resolve_skin_glb(lam_dir: str | Path) -> Path:
    lam_dir = Path(lam_dir)
    direct = lam_dir / "skin.glb"
    if direct.is_file():
        return direct
    nested = lam_dir / "arkitWithBSData" / "skin.glb"
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"skin.glb not found under {lam_dir}")
