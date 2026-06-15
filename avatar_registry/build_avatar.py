#!/usr/bin/env python3
"""Build a 2D warp avatar pack from LAM/OAC FBX template + neutral photo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatar_registry.avatar_pack import AvatarPack, save_avatar_pack
from avatar_registry.fbx_parser import load_fbx_face_mesh, shape_deltas_3d
from export_utils import ARKIT_BLENDSHAPE_NAMES


def project_xy(vertices_3d: np.ndarray) -> np.ndarray:
    """Orthographic front view: image x = mesh x, image y = -mesh y."""
    return np.column_stack([vertices_3d[:, 0], -vertices_3d[:, 1]])


def fit_projected_to_image(
    projected: np.ndarray,
    width: int,
    height: int,
    *,
    margin: float = 0.08,
) -> np.ndarray:
    """Scale and center projected mesh into the photo canvas."""
    min_xy = projected.min(axis=0)
    max_xy = projected.max(axis=0)
    span = float(max(max_xy - min_xy))
    if span <= 0:
        raise ValueError("Projected mesh span is zero")

    scale = (1.0 - 2.0 * margin) * min(width, height) / span
    mesh_center = (min_xy + max_xy) * 0.5
    image_center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
    return (projected - mesh_center) * scale + image_center


def build_morph_2d_deltas(mesh, blendshape_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (52, N, 2) deltas aligned with ARKIT_BLENDSHAPE_NAMES order."""
    num_vertices = mesh.neutral_vertices.shape[0]
    deltas = np.zeros((len(blendshape_names), num_vertices, 2), dtype=np.float32)
    missing: list[str] = []

    for i, name in enumerate(blendshape_names):
        if name not in mesh.shapes:
            missing.append(name)
            continue
        delta_3d = shape_deltas_3d(mesh, name)
        deltas[i, :, 0] = delta_3d[:, 0]
        deltas[i, :, 1] = -delta_3d[:, 1]

    return deltas, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2D avatar pack from LAM FBX + photo")
    parser.add_argument(
        "--fbx",
        default=str(ROOT.parent / "sample_oac" / "template_file.fbx"),
        help="LAM/OAC face template FBX",
    )
    parser.add_argument(
        "--photo",
        default=str(ROOT.parent / "LAM_Audio2Expression" / "assets" / "sample_input" / "barbara.jpg"),
        help="Neutral face photo for texture",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "avatar_registry" / "packs" / "barbara"),
        help="Output avatar pack directory",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.08,
        help="Image margin ratio when fitting mesh (default: 0.08)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fbx_path = Path(args.fbx)
    photo_path = Path(args.photo)
    if not fbx_path.is_file():
        print(f"FBX not found: {fbx_path}", file=sys.stderr)
        return 1
    if not photo_path.is_file():
        print(f"Photo not found: {photo_path}", file=sys.stderr)
        return 1

    mesh = load_fbx_face_mesh(fbx_path)
    texture = np.asarray(Image.open(photo_path).convert("RGB"))
    height, width = texture.shape[:2]

    neutral_proj = project_xy(mesh.neutral_vertices)
    neutral_2d = fit_projected_to_image(neutral_proj, width, height, margin=args.margin)

    morph_2d_deltas, missing = build_morph_2d_deltas(mesh, ARKIT_BLENDSHAPE_NAMES)

    pack = AvatarPack(
        neutral_2d=neutral_2d.astype(np.float32),
        triangles=mesh.triangles,
        morph_2d_deltas=morph_2d_deltas,
        texture=texture,
        blendshape_names=ARKIT_BLENDSHAPE_NAMES,
        meta={
            "source_fbx": str(fbx_path.resolve()),
            "source_photo": str(photo_path.resolve()),
            "image_width": width,
            "image_height": height,
            "fit_margin": args.margin,
            "missing_shapes": missing,
            "num_shapes_in_fbx": len(mesh.shapes),
        },
    )

    out_dir = save_avatar_pack(pack, args.output)
    print(
        f"saved avatar pack: {out_dir}\n"
        f"  vertices={pack.num_vertices}  triangles={pack.triangles.shape[0]}\n"
        f"  missing_shapes={missing or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
