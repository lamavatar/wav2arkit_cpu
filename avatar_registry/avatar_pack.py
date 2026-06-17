"""2D avatar pack I/O (neutral mesh + 52 morph deltas + texture)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from export_utils import ARKIT_BLENDSHAPE_NAMES


@dataclass
class AvatarPack:
    neutral_2d: np.ndarray  # (N, 2) float32, image pixel coords
    triangles: np.ndarray  # (T, 3) int32
    morph_2d_deltas: np.ndarray  # (52, N, 2) float32
    texture: np.ndarray  # (H, W, 3) uint8 RGB
    blendshape_names: list[str]
    meta: dict
    face_mask: np.ndarray | None = None  # (H, W) uint8 0-255, optional

    @property
    def num_vertices(self) -> int:
        return int(self.neutral_2d.shape[0])

    def deform(self, weights: np.ndarray) -> np.ndarray:
        """Apply blendshape weights. weights: (52,) or (N_frames, 52)."""
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim == 1:
            if w.shape[0] != 52:
                raise ValueError(f"Expected 52 weights, got {w.shape[0]}")
            return self.neutral_2d + np.tensordot(w, self.morph_2d_deltas, axes=(0, 0))
        if w.ndim == 2 and w.shape[1] == 52:
            return self.neutral_2d[None, :, :] + np.einsum("fk,kni->fni", w, self.morph_2d_deltas)
        raise ValueError(f"Invalid weights shape: {w.shape}")


def save_avatar_pack(pack: AvatarPack, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "neutral_2d.npy", pack.neutral_2d.astype(np.float32))
    np.save(output_dir / "triangles.npy", pack.triangles.astype(np.int32))
    np.save(output_dir / "morph_2d_deltas.npy", pack.morph_2d_deltas.astype(np.float32))

    Image.fromarray(pack.texture).save(output_dir / "texture.png")

    if pack.face_mask is not None:
        np.save(output_dir / "face_mask.npy", pack.face_mask.astype(np.uint8))

    meta = dict(pack.meta)
    meta["blendshape_names"] = pack.blendshape_names
    meta["num_vertices"] = pack.num_vertices
    meta["num_triangles"] = int(pack.triangles.shape[0])
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return output_dir


def load_avatar_pack(pack_dir: str | Path) -> AvatarPack:
    pack_dir = Path(pack_dir)
    meta_path = pack_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Avatar pack not found: {pack_dir}")

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)

    texture = np.asarray(Image.open(pack_dir / "texture.png").convert("RGB"))
    face_mask_path = pack_dir / "face_mask.npy"
    face_mask = np.load(face_mask_path) if face_mask_path.is_file() else None
    return AvatarPack(
        neutral_2d=np.load(pack_dir / "neutral_2d.npy"),
        triangles=np.load(pack_dir / "triangles.npy"),
        morph_2d_deltas=np.load(pack_dir / "morph_2d_deltas.npy"),
        texture=texture,
        blendshape_names=meta.get("blendshape_names", ARKIT_BLENDSHAPE_NAMES),
        meta=meta,
        face_mask=face_mask,
    )
