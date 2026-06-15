#!/usr/bin/env python3
"""Render animation frames from avatar pack + bsData.json."""

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


def load_bsdata_weights(json_path: Path) -> tuple[np.ndarray, float]:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    fps = float(data.get("metadata", {}).get("fps", 30.0))
    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"No frames in {json_path}")
    weights = np.asarray([frame["weights"] for frame in frames], dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] != 52:
        raise ValueError(f"Expected weights (N, 52), got {weights.shape}")
    return weights, fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render avatar pack with bsData.json")
    parser.add_argument(
        "--pack",
        default=str(ROOT / "avatar_registry" / "packs" / "barbara"),
        help="Avatar pack directory",
    )
    parser.add_argument(
        "--bsdata",
        default=str(ROOT / "bsData_onnx.json"),
        help="ARKit blendshape animation JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "avatar_registry" / "output" / "barbara_frames"),
        help="Directory for rendered PNG frames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit frames (0 = all)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Render every Nth frame (default: 1)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pack_dir = Path(args.pack)
    bsdata_path = Path(args.bsdata)
    output_dir = Path(args.output_dir)

    if not pack_dir.is_dir():
        print(f"Avatar pack not found: {pack_dir}", file=sys.stderr)
        return 1
    if not bsdata_path.is_file():
        print(f"bsData not found: {bsdata_path}", file=sys.stderr)
        return 1

    pack = load_avatar_pack(pack_dir)
    weights, fps = load_bsdata_weights(bsdata_path)

    if args.max_frames > 0:
        weights = weights[: args.max_frames]
    weights = weights[:: max(1, args.frame_step)]

    output_dir.mkdir(parents=True, exist_ok=True)
    deformed = pack.deform(weights)

    try:
        import cv2
    except ImportError:
        print("opencv-python-headless is required", file=sys.stderr)
        return 1

    for frame_idx, verts in enumerate(deformed):
        image = render_mesh_frame(
            pack.texture,
            pack.neutral_2d,
            verts,
            pack.triangles,
        )
        out_path = output_dir / f"frame_{frame_idx:05d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print(
        f"rendered {len(deformed)} frames @ {fps:.1f} fps\n"
        f"  pack={pack_dir}\n"
        f"  output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
