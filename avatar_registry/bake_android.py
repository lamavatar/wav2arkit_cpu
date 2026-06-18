"""Bake a LAM 3D Gaussian avatar into a flat ``.splat`` binary for the Android app.

Two formats are supported:

SPL1 (legacy): geometry + per-frame ARKit weights baked from ``bsData.json``.
SPL2:          geometry only. Per-frame weights come from on-device ONNX
               inference at runtime (wav2arkit), so no ``bsData.json`` is needed.

Layout SPL1 (all little-endian; floats are f32):
    magic        : 4 bytes  b"SPL1"
    N            : u32       number of Gaussians
    M            : u32       number of baked morphs
    F            : u32       number of animation frames
    fps          : f32
    camera       : 10 f32    eye[3], center[3], up[3], fovy_rad
    morph_names  : M * (u8 len + len ascii bytes)
    base         : N*3 f32
    cov6         : N*6 f32
    color        : N*3 f32
    opacity      : N   f32
    dynamic      : N   u8
    deltas       : M*N*3 f32
    weights      : F*M   f32

Layout SPL2 (geometry only — no F, no weights):
    magic        : 4 bytes  b"SPL2"
    N            : u32
    M            : u32
    fps          : f32       default expression fps (30)
    camera       : 10 f32
    morph_names  : M * (u8 len + len ascii bytes)
    base         : N*3 f32
    cov6         : N*6 f32
    color        : N*3 f32
    opacity      : N   f32
    dynamic      : N   u8
    deltas       : M*N*3 f32

Usage:
    # SPL2 (ONNX lip-sync, mouth morph subset) — no bsData.json required
    python -m avatar_registry.bake_android --format spl2 --morph-set mouth

    # SPL1 (legacy baked animation)
    python -m avatar_registry.bake_android --format spl1
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
from export_utils import ARKIT_BLENDSHAPE_NAMES
from postprocess import MOUTH_BLENDSHAPES

MAGIC_SPL1 = b"SPL1"
MAGIC_SPL2 = b"SPL2"
DEFAULT_FPS = 30.0


def _common_geometry(avatar: GaussianAvatar):
    base = (avatar.neutral + avatar.offset).astype(np.float32)  # (N,3)
    cov = avatar.cov3d.astype(np.float32)                       # (N,3,3)
    cov6 = np.stack(
        [cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]],
        axis=1,
    ).astype(np.float32)
    color = avatar.color.astype(np.float32)
    opacity = avatar.opacity.astype(np.float32)
    return base, cov6, color, opacity


def bake_spl1(avatar_dir: str | Path, out_path: str | Path, *, dyn_threshold: float = 0.02) -> None:
    avatar = GaussianAvatar(avatar_dir)
    names, fps, weights = _load_bsdata(avatar.dir / "bsData.json")
    weights = np.asarray(weights, np.float32)
    F = weights.shape[0]

    used = [k for k in range(min(len(names), weights.shape[1]))
            if avatar.morph_deltas.get(names[k]) is not None and float(np.ptp(weights[:, k])) > 1e-6]
    morph_names = [names[k] for k in used]
    M = len(morph_names)

    base, cov6, color, opacity = _common_geometry(avatar)
    N = base.shape[0]
    dyn = avatar.dynamic_mask(weights, names, rel_threshold=dyn_threshold).astype(np.uint8)
    deltas = np.stack([avatar.morph_deltas[n] for n in morph_names], axis=0).astype(np.float32)
    w_used = weights[:, used].astype(np.float32)

    eye, center, up, fovy = auto_camera(base)
    cam = np.array([*eye, *center, *up, fovy], np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MAGIC_SPL1)
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
    print(f"baked {out_path} (SPL1)")
    print(f"  gaussians={N}  morphs={M}  frames={F}  fps={fps:g}  dynamic={int(dyn.sum())}")
    print(f"  morphs: {', '.join(morph_names)}")
    print(f"  size={size_mb:.1f} MB")


def _morph_candidates(avatar: GaussianAvatar, morph_set: str) -> list[str]:
    if morph_set == "mouth":
        pool = list(MOUTH_BLENDSHAPES)
    else:
        pool = list(ARKIT_BLENDSHAPE_NAMES)
    # Keep only morphs present in the GLB skin, preserve pool order.
    return [n for n in pool if avatar.morph_deltas.get(n) is not None]


def _synthetic_weight_sequence(morph_names: list[str]) -> np.ndarray:
    """One-hot ramp so dynamic_mask captures each morph's peak displacement.

    Frame 0 = neutral; frame k+1 fully activates morph k. This makes every
    baked morph contribute its full delta to the dynamic (mouth) mask without
    needing a real bsData.json clip.
    """
    m = len(morph_names)
    seq = np.zeros((m + 1, m), np.float32)
    for k in range(m):
        seq[k + 1, k] = 1.0
    return seq


def bake_spl2(
    avatar_dir: str | Path,
    out_path: str | Path,
    *,
    morph_set: str = "mouth",
    dyn_threshold: float = 0.02,
    fps: float = DEFAULT_FPS,
) -> None:
    avatar = GaussianAvatar(avatar_dir)
    morph_names = _morph_candidates(avatar, morph_set)
    if not morph_names:
        raise SystemExit(f"no morphs from set '{morph_set}' present in skin.glb")
    M = len(morph_names)

    base, cov6, color, opacity = _common_geometry(avatar)
    N = base.shape[0]

    # Dynamic mask from a synthetic one-hot ramp (no bsData.json needed).
    synth = _synthetic_weight_sequence(morph_names)
    dyn = avatar.dynamic_mask(synth, morph_names, rel_threshold=dyn_threshold).astype(np.uint8)

    deltas = np.stack([avatar.morph_deltas[n] for n in morph_names], axis=0).astype(np.float32)

    eye, center, up, fovy = auto_camera(base)
    cam = np.array([*eye, *center, *up, fovy], np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MAGIC_SPL2)
        f.write(struct.pack("<II", N, M))
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

    size_mb = out_path.stat().st_size / 1e6
    print(f"baked {out_path} (SPL2)")
    print(f"  gaussians={N}  morphs={M}  frames=0  fps={fps:g}  dynamic={int(dyn.sum())}")
    print(f"  morph_set={morph_set}  morphs: {', '.join(morph_names)}")
    print(f"  size={size_mb:.1f} MB")


def main() -> int:
    p = argparse.ArgumentParser(description="Bake a LAM avatar to .splat for Android")
    p.add_argument("--avatar", default=str(ROOT / "avatar_registry" / "lam3d_avatar" / "third"))
    p.add_argument("--output", default=str(ROOT / "android" / "SplatBench" / "app" / "src" / "main"
                                          / "assets" / "third.splat"))
    p.add_argument("--format", choices=["spl1", "spl2"], default="spl2")
    p.add_argument("--morph-set", choices=["mouth", "full"], default="mouth",
                   help="SPL2 only: which ARKit morph subset to bake")
    p.add_argument("--dyn-threshold", type=float, default=0.02)
    args = p.parse_args()
    if args.format == "spl1":
        bake_spl1(args.avatar, args.output, dyn_threshold=args.dyn_threshold)
    else:
        bake_spl2(args.avatar, args.output, morph_set=args.morph_set, dyn_threshold=args.dyn_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
