#!/usr/bin/env python3
"""Sanity-check an avatar pack and optionally render a single morph debug frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatar_registry.avatar_pack import load_avatar_pack
from avatar_registry.warp import render_mesh_frame
from export_utils import ARKIT_BLENDSHAPE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify avatar pack integrity")
    parser.add_argument(
        "--pack",
        default=str(ROOT / "avatar_registry" / "packs" / "barbara"),
        help="Avatar pack directory",
    )
    parser.add_argument(
        "--morph",
        default="jawOpen",
        help="Blendshape name for single-morph debug render",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional PNG output path for debug render",
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Composite warped face over original photo using pack face_mask",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pack_dir = Path(args.pack)
    pack = load_avatar_pack(pack_dir)

    meta_path = pack_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = meta.get("missing_shapes", [])
    expected_missing = ["tongueOut"]

    print(f"pack: {pack_dir}")
    print(f"  vertices={pack.num_vertices}")
    print(f"  triangles={pack.triangles.shape[0]}")
    print(f"  morph_deltas={pack.morph_2d_deltas.shape}")
    print(f"  source_type={meta.get('source_type')}")
    print(f"  missing_shapes={missing}")

    assert pack.num_vertices == 20018, f"unexpected vertex count {pack.num_vertices}"
    assert pack.triangles.shape[0] == 39904, f"unexpected triangle count {pack.triangles.shape[0]}"
    assert pack.morph_2d_deltas.shape == (52, pack.num_vertices, 2)
    assert missing == expected_missing, f"expected missing {expected_missing}, got {missing}"

    if args.morph not in ARKIT_BLENDSHAPE_NAMES:
        raise ValueError(f"Unknown morph: {args.morph}")

    weights = np.zeros(52, dtype=np.float32)
    morph_idx = ARKIT_BLENDSHAPE_NAMES.index(args.morph)
    if morph_idx >= 0 and args.morph not in missing:
        weights[morph_idx] = 1.0

    deformed = pack.deform(weights)
    if args.morph not in missing:
        disp = np.linalg.norm(deformed - pack.neutral_2d, axis=1)
        print(f"  morph={args.morph} max_disp_px={float(disp.max()):.2f} mean_disp_px={float(disp.mean()):.4f}")

    if args.output:
        image = render_mesh_frame(
            pack.texture,
            pack.neutral_2d,
            deformed,
            pack.triangles,
            composite=args.composite,
            face_mask=pack.face_mask,
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"  debug_render={out_path}")

    print("verify: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
