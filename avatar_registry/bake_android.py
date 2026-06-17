"""Bake a LAM 3D Gaussian avatar into a flat ``.splat`` binary for the Android app.

Re-parsing PLY/GLB on-device is fragile, so we pre-bake everything the GLES
renderer needs into one little-endian file that maps straight into a
``ByteBuffer``. The on-device pipeline (deform -> project -> 2D covariance ->
sort -> instanced draw) then matches ``gaussian_splat.py`` exactly.

Layout (all little-endian; floats are f32):
    magic        : 4 bytes  b"SPL1"
    N            : u32       number of Gaussians
    M            : u32       number of baked morphs (only the ones used by bsData)
    F            : u32       number of animation frames
    fps          : f32
    camera       : 10 f32    eye[3], center[3], up[3], fovy_rad
    morph_names  : M * (u8 len + len ascii bytes)
    base         : N*3 f32   neutral + offset (canonical positions)
    cov6         : N*6 f32   symmetric 3x3 covariance (xx,xy,xz,yy,yz,zz)
    color        : N*3 f32   degree-0 SH used directly as RGB
    opacity      : N   f32   sigmoid-activated
    dynamic      : N   u8    1 = moves during the clip (mouth/jaw set)
    deltas       : M*N*3 f32 per-morph position deltas
    weights      : F*M   f32 per-frame ARKit weights (aligned to baked morphs)

Usage:
    python -m avatar_registry.bake_android \
        --avatar avatar_registry/lam3d_avatar/vfhq_case1 \
        --output android/SplatBench/app/src/main/assets/vfhq_case1.splat
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from avatar_registry.gaussian_splat import (
    ROOT,
    GaussianAvatar,
    _load_bsdata,
    auto_camera,
)

MAGIC = b"SPL1"


def bake(avatar_dir: str | Path, out_path: str | Path, *, dyn_threshold: float = 0.02) -> None:
    avatar = GaussianAvatar(avatar_dir)
    names, fps, weights = _load_bsdata(avatar.dir / "bsData.json")
    weights = np.asarray(weights, np.float32)
    F = weights.shape[0]

    # Keep only morphs that are actually exercised by this clip (shrinks the file
    # and the per-frame device work). Order is preserved for weights+deltas.
    used = [k for k in range(min(len(names), weights.shape[1]))
            if avatar.morph_deltas.get(names[k]) is not None and float(np.ptp(weights[:, k])) > 1e-6]
    morph_names = [names[k] for k in used]
    M = len(morph_names)

    base = (avatar.neutral + avatar.offset).astype(np.float32)        # (N,3)
    N = base.shape[0]
    cov = avatar.cov3d.astype(np.float32)                            # (N,3,3)
    cov6 = np.stack(
        [cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]],
        axis=1,
    ).astype(np.float32)                                             # (N,6)
    color = avatar.color.astype(np.float32)                          # (N,3)
    opacity = avatar.opacity.astype(np.float32)                      # (N,)

    dyn = avatar.dynamic_mask(weights, names, rel_threshold=dyn_threshold).astype(np.uint8)

    deltas = np.stack([avatar.morph_deltas[n] for n in morph_names], axis=0).astype(np.float32)  # (M,N,3)
    w_used = weights[:, used].astype(np.float32)                     # (F,M)

    eye, center, up, fovy = auto_camera(base)
    cam = np.array([*eye, *center, *up, fovy], np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", N, M, F))
        f.write(struct.pack("<f", float(fps)))
        f.write(cam.tobytes())
        for nm in morph_names:
            b = nm.encode("ascii")
            f.write(struct.pack("<B", len(b)))
            f.write(b)
        f.write(np.ascontiguousarray(base).tobytes())
        f.write(np.ascontiguousarray(cov6).tobytes())
        f.write(np.ascontiguousarray(color).tobytes())
        f.write(np.ascontiguousarray(opacity).tobytes())
        f.write(np.ascontiguousarray(dyn).tobytes())
        f.write(np.ascontiguousarray(deltas).tobytes())
        f.write(np.ascontiguousarray(w_used).tobytes())

    size_mb = out_path.stat().st_size / 1e6
    print(f"baked {out_path}")
    print(f"  gaussians={N}  morphs={M}  frames={F}  fps={fps:g}  dynamic={int(dyn.sum())}")
    print(f"  morphs: {', '.join(morph_names)}")
    print(f"  size={size_mb:.1f} MB")


def main() -> int:
    p = argparse.ArgumentParser(description="Bake a LAM avatar to .splat for Android")
    p.add_argument("--avatar", default=str(ROOT / "avatar_registry" / "lam3d_avatar" / "vfhq_case1"))
    p.add_argument("--output", default=str(ROOT / "android" / "SplatBench" / "app" / "src" / "main"
                                          / "assets" / "vfhq_case1.splat"))
    p.add_argument("--dyn-threshold", type=float, default=0.02)
    args = p.parse_args()
    bake(args.avatar, args.output, dyn_threshold=args.dyn_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
