#!/usr/bin/env python3
"""Build a 2D warp avatar pack from LAM skin.glb + neutral photo."""

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
from avatar_registry.glb_loader import LamFaceMesh, load_lam_skin_glb, resolve_skin_glb
from avatar_registry.landmark_align import (
    align_mesh_to_photo,
    build_face_mask_from_points,
    detect_face_landmark_points,
)
from avatar_registry.mouth_retarget import retarget_mouth_jaw_morph_deltas_2d
from export_utils import ARKIT_BLENDSHAPE_NAMES


def project_xy(vertices_3d: np.ndarray) -> np.ndarray:
    """Orthographic front view: image x = mesh x, image y = -mesh y."""
    return np.column_stack([vertices_3d[:, 0], -vertices_3d[:, 1]])


def compute_fit_transform(
    projected: np.ndarray,
    width: int,
    height: int,
    *,
    margin: float = 0.08,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (scale, mesh_center, image_center) for 3D-projected -> pixel mapping."""
    min_xy = projected.min(axis=0)
    max_xy = projected.max(axis=0)
    span = float(max(max_xy - min_xy))
    if span <= 0:
        raise ValueError("Projected mesh span is zero")

    scale = (1.0 - 2.0 * margin) * min(width, height) / span
    mesh_center = (min_xy + max_xy) * 0.5
    image_center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
    return scale, mesh_center, image_center


def fit_projected_to_image(
    projected: np.ndarray,
    width: int,
    height: int,
    *,
    margin: float = 0.08,
) -> np.ndarray:
    """Scale and center projected mesh into the photo canvas."""
    scale, mesh_center, image_center = compute_fit_transform(
        projected, width, height, margin=margin
    )
    return (projected - mesh_center) * scale + image_center


def build_morph_2d_deltas(
    mesh: LamFaceMesh,
    blendshape_names: list[str],
    *,
    fit_scale: float,
) -> tuple[np.ndarray, list[str]]:
    """Return (52, N, 2) pixel-space deltas aligned with ARKIT_BLENDSHAPE_NAMES order."""
    num_vertices = mesh.neutral_vertices.shape[0]
    deltas = np.zeros((len(blendshape_names), num_vertices, 2), dtype=np.float32)
    missing: list[str] = []

    for i, name in enumerate(blendshape_names):
        delta_3d = mesh.morph_deltas_3d.get(name)
        if delta_3d is None:
            missing.append(name)
            continue
        deltas[i, :, 0] = delta_3d[:, 0] * fit_scale
        deltas[i, :, 1] = -delta_3d[:, 1] * fit_scale

    return deltas, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2D avatar pack from LAM skin.glb + photo")
    parser.add_argument(
        "--lam-dir",
        default=str(ROOT.parent / "LAM_Audio2Expression" / "assets" / "sample_lam" / "barbara" / "arkitWithBSData"),
        help="Directory containing skin.glb (or parent with arkitWithBSData/)",
    )
    parser.add_argument(
        "--skin-glb",
        default="",
        help="Optional direct path to skin.glb (overrides --lam-dir)",
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
        help="Image margin ratio when coarse bbox fitting (default: 0.08)",
    )
    parser.add_argument(
        "--align",
        choices=("landmark", "bbox"),
        default="landmark",
        help="Photo alignment: landmark (default) or bbox-only fit",
    )
    parser.add_argument(
        "--align-method",
        choices=("affine", "similarity"),
        default="similarity",
        help="Landmark transform type (default: similarity)",
    )
    parser.add_argument(
        "--mouth-retarget",
        dest="mouth_retarget",
        action="store_true",
        default=True,
        help="Retarget mouth/jaw morph deltas onto the photo mouth region (default: on)",
    )
    parser.add_argument(
        "--no-mouth-retarget",
        dest="mouth_retarget",
        action="store_false",
        help="Disable mouth/jaw morph retargeting (use raw GLB deltas)",
    )
    parser.add_argument(
        "--mouth-open-scale",
        type=float,
        default=1.0,
        help="Scale factor for retargeted jawOpen/mouthClose magnitude (default: 1.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    skin_glb = Path(args.skin_glb) if args.skin_glb else resolve_skin_glb(args.lam_dir)
    photo_path = Path(args.photo)
    if not skin_glb.is_file():
        print(f"skin.glb not found: {skin_glb}", file=sys.stderr)
        return 1
    if not photo_path.is_file():
        print(f"Photo not found: {photo_path}", file=sys.stderr)
        return 1

    mesh = load_lam_skin_glb(skin_glb)
    texture = np.asarray(Image.open(photo_path).convert("RGB"))
    height, width = texture.shape[:2]

    neutral_proj = project_xy(mesh.neutral_vertices)
    fit_scale, _, _ = compute_fit_transform(neutral_proj, width, height, margin=args.margin)
    coarse_2d = fit_projected_to_image(neutral_proj, width, height, margin=args.margin)

    morph_2d_deltas, missing = build_morph_2d_deltas(
        mesh, ARKIT_BLENDSHAPE_NAMES, fit_scale=fit_scale
    )

    alignment_meta: dict = {}
    neutral_2d = coarse_2d.astype(np.float32)
    if args.align == "landmark":
        try:
            neutral_2d, morph_2d_deltas, alignment_meta = align_mesh_to_photo(
                coarse_2d,
                morph_2d_deltas,
                mesh.neutral_vertices,
                texture,
                method=args.align_method,
            )
        except Exception as exc:
            print(f"landmark alignment failed, using bbox fit: {exc}", file=sys.stderr)
            alignment_meta = {
                "alignment_method": "bbox_fallback",
                "alignment_error": str(exc),
            }

    face_mask: np.ndarray | None = None
    retarget_meta: dict = {}
    if args.align == "landmark":
        try:
            full_landmarks = detect_face_landmark_points(texture)
            face_mask = build_face_mask_from_points(full_landmarks, height, width)
            if args.mouth_retarget:
                morph_2d_deltas, retarget_meta = retarget_mouth_jaw_morph_deltas_2d(
                    neutral_2d.astype(np.float32),
                    morph_2d_deltas,
                    ARKIT_BLENDSHAPE_NAMES,
                    full_landmarks,
                    mouth_open_scale=args.mouth_open_scale,
                )
        except Exception as exc:
            print(f"face mask / mouth retarget failed: {exc}", file=sys.stderr)
            retarget_meta = {"mouth_retarget_error": str(exc)}

    pack = AvatarPack(
        neutral_2d=neutral_2d.astype(np.float32),
        triangles=mesh.triangles,
        morph_2d_deltas=morph_2d_deltas,
        texture=texture,
        blendshape_names=ARKIT_BLENDSHAPE_NAMES,
        face_mask=face_mask,
        meta={
            "source_type": "lam_glb",
            "source_skin_glb": str(skin_glb.resolve()),
            "source_photo": str(photo_path.resolve()),
            "image_width": width,
            "image_height": height,
            "fit_margin": args.margin,
            "fit_scale": float(fit_scale),
            "align_mode": args.align,
            "mouth_retarget": bool(args.mouth_retarget),
            "mouth_retarget_info": retarget_meta,
            **alignment_meta,
            "missing_shapes": missing,
            "morph_names": sorted(mesh.morph_deltas_3d.keys()),
            "num_morphs": len(mesh.morph_deltas_3d),
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
