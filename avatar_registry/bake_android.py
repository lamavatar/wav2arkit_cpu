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
    [optional head-bone clip — omitted when ``animation.glb`` is absent]
    magic        : 4 bytes  b"HEAD"
    H            : u32       number of baked head-matrix keyframes (one loop)
    clip_dur     : f32       source clip duration in seconds
    anim_name    : u8 len + len ascii bytes (clip name in animation.glb)
    pivot        : 3 f32     rotation pivot (mesh-space centroid); omitted in legacy HEAD
    head_mats    : H*16 f32  row-major 4x4 rotation-only delta from frame 0 per keyframe

Usage:
    # SPL2 (ONNX lip-sync, mouth morph subset) — no bsData.json required
    python -m avatar_registry.bake_android --format spl2 --morph-set mouth

    # SPL2 with head-bone clip from animation.glb (first clip by default)
    python -m avatar_registry.bake_android --format spl2 --morph-set full --animation yumi_h5_a3_speak

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
    Skeleton,
    _load_bsdata,
    auto_camera,
)
from export_utils import ARKIT_BLENDSHAPE_NAMES
from postprocess import MOUTH_BLENDSHAPES

MAGIC_SPL1 = b"SPL1"
MAGIC_SPL2 = b"SPL2"
MAGIC_HEAD = b"HEAD"
DEFAULT_FPS = 30.0
ARKIT_SUBDIR = "arkitWithBSData"


def resolve_avatar_dir(avatar_dir: str | Path) -> Path:
    """Return the directory that actually contains ``offset.ply`` / ``skin.glb``.

    LAM packs in this repo store assets under ``<person>/arkitWithBSData/`` while
    ``--avatar`` often points at ``<person>/`` (e.g. ``lam3d_avatar/third``).
    """
    root = Path(avatar_dir)
    if (root / "offset.ply").is_file() and (root / "skin.glb").is_file():
        return root
    sub = root / ARKIT_SUBDIR
    if (sub / "offset.ply").is_file() and (sub / "skin.glb").is_file():
        return sub
    return root


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
    avatar_dir = resolve_avatar_dir(avatar_dir)
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


def _orthonormalize_rotation(r3: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(r3.astype(np.float64))
    return (u @ vt).astype(np.float32)


def _rotation_delta_mats(mats_abs: np.ndarray) -> np.ndarray:
    """Frame-0-relative rotation-only deltas (no translation / depth shift)."""
    m0_inv = np.linalg.inv(mats_abs[0].astype(np.float64)).astype(np.float32)
    out = np.zeros_like(mats_abs)
    for i, m_abs in enumerate(mats_abs):
        delta = (m_abs @ m0_inv).astype(np.float32)
        r = _orthonormalize_rotation(delta[:3, :3])
        out[i] = np.eye(4, dtype=np.float32)
        out[i, :3, :3] = r
    return out


def _bake_head_sequence(
    avatar_dir: Path,
    pivot: np.ndarray,
    *,
    animation: str = "",
    fps: float = DEFAULT_FPS,
) -> tuple[str, float, np.ndarray, np.ndarray] | None:
    """Sample one loop of head-bone rotations from ``animation.glb``.

    Returns (clip_name, clip_duration_sec, pivot[3], rotation_mats[H,4,4]) or
    None when the file/clip is missing. Baked matrices are rotation-only
    deltas from keyframe 0 (pivot rotation; no whole-face pan/zoom).
    """
    glb = avatar_dir / "animation.glb"
    if not glb.is_file():
        return None
    skel = Skeleton(glb)
    if not skel.anim_names:
        return None
    clip = animation or skel.anim_names[0]
    anim_idx = skel.anim_index(clip)
    clip_dur = float(skel.animations[anim_idx]["duration"])
    if clip_dur <= 0.0:
        clip_dur = 1.0
    h = max(1, int(np.ceil(clip_dur * fps)))
    mats_abs = np.stack(
        [skel.head_skin_matrix(anim_idx, min(i / fps, clip_dur)) for i in range(h)],
        axis=0,
    ).astype(np.float32)
    mats = _rotation_delta_mats(mats_abs)
    return clip, clip_dur, pivot.astype(np.float32), mats


def _write_head_trailer(f, clip: str, clip_dur: float, pivot: np.ndarray, mats: np.ndarray) -> None:
    h = mats.shape[0]
    f.write(MAGIC_HEAD)
    f.write(struct.pack("<If", h, float(clip_dur)))
    name_b = clip.encode("ascii")
    f.write(struct.pack("<B", len(name_b)))
    f.write(name_b)
    f.write(np.ascontiguousarray(pivot.astype(np.float32)).tobytes())
    f.write(np.ascontiguousarray(mats.reshape(h, 16)).tobytes())


def bake_spl2(
    avatar_dir: str | Path,
    out_path: str | Path,
    *,
    morph_set: str = "mouth",
    dyn_threshold: float = 0.02,
    fps: float = DEFAULT_FPS,
    animation: str = "",
) -> None:
    avatar_dir = resolve_avatar_dir(avatar_dir)
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
    head = _bake_head_sequence(avatar_dir, base.mean(axis=0), animation=animation, fps=fps)

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
        if head is not None:
            clip, clip_dur, pivot, mats = head
            _write_head_trailer(f, clip, clip_dur, pivot, mats)

    size_mb = out_path.stat().st_size / 1e6
    print(f"baked {out_path} (SPL2)")
    print(f"  gaussians={N}  morphs={M}  frames=0  fps={fps:g}  dynamic={int(dyn.sum())}")
    print(f"  morph_set={morph_set}  morphs: {', '.join(morph_names)}")
    if head is not None:
        clip, clip_dur, pivot, mats = head
        print(f"  head_bone: {clip}  keyframes={mats.shape[0]}  clip_dur={clip_dur:g}s  "
              f"pivot=({pivot[0]:.3f},{pivot[1]:.3f},{pivot[2]:.3f})  rot-only")
    else:
        print("  head_bone: (none — no animation.glb or empty clip list)")
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
    p.add_argument(
        "--animation", default="",
        help="SPL2: head-bone clip in animation.glb (default: first clip when present)",
    )
    p.add_argument(
        "--list-animations", action="store_true",
        help="print animation.glb clip names for --avatar and exit",
    )
    args = p.parse_args()
    if args.list_animations:
        glb = resolve_avatar_dir(args.avatar) / "animation.glb"
        if not glb.is_file():
            raise SystemExit(f"animation.glb not found: {glb}")
        print("animations:", Skeleton(glb).anim_names)
        return 0
    if args.format == "spl1":
        bake_spl1(args.avatar, args.output, dyn_threshold=args.dyn_threshold)
    else:
        bake_spl2(
            args.avatar, args.output,
            morph_set=args.morph_set,
            dyn_threshold=args.dyn_threshold,
            animation=args.animation,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
