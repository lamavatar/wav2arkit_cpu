"""ARKit blendshape JSON export for wav2arkit_cpu inference output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

ARKIT_BLENDSHAPE_NAMES: list[str] = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]


def export_blendshape_animation(
    blendshape_weights: np.ndarray,
    output_path: str | Path,
    blendshape_names: Optional[list[str]] = None,
    fps: float = 30.0,
    rotation_data: Optional[np.ndarray] = None,
) -> None:
    """Write ARKit-compatible blendshape animation JSON."""
    names = blendshape_names or ARKIT_BLENDSHAPE_NAMES
    output_path = Path(output_path)

    if blendshape_weights.ndim != 2 or blendshape_weights.shape[1] != 52:
        raise ValueError(
            f"Expected blendshape_weights shape (N, 52), got {blendshape_weights.shape}"
        )
    if len(names) != 52:
        raise ValueError(f"Requires 52 blendshape names, got {len(names)}")
    if rotation_data is not None and len(rotation_data) != len(blendshape_weights):
        raise ValueError("Rotation data length must match animation frames")

    animation_data = {
        "names": names,
        "metadata": {
            "fps": fps,
            "frame_count": len(blendshape_weights),
            "blendshape_names": names,
        },
        "frames": [
            {
                "weights": blendshape_weights[frame_idx].tolist(),
                "time": frame_idx / fps,
                "rotation": rotation_data[frame_idx].tolist() if rotation_data is not None else [],
            }
            for frame_idx in range(blendshape_weights.shape[0])
        ],
    }

    if output_path.suffix != ".json":
        output_path = output_path.with_suffix(".json")

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(animation_data, json_file, indent=2, ensure_ascii=False)
