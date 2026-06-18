"""LAM Audio2Expression streaming post-processing (ported for wav2arkit_cpu)."""

from __future__ import annotations

import math
import warnings
from typing import Optional

import librosa
import numpy as np
from scipy.signal import savgol_filter

from export_utils import ARKIT_BLENDSHAPE_NAMES

# LAM infer_streaming_audio: previous_expression kept at max_frame_length = 64
MAX_CONTEXT_FRAMES = 64

# Realtime streaming: 200 ms audio overlap + 1 s chunk => 1.2 s ONNX input window
DEFAULT_OVERLAP_MS = 200

ARKIT_LEFT_RIGHT_PAIR = [
    ("jawLeft", "jawRight"),
    ("mouthLeft", "mouthRight"),
    ("mouthSmileLeft", "mouthSmileRight"),
    ("mouthFrownLeft", "mouthFrownRight"),
    ("mouthDimpleLeft", "mouthDimpleRight"),
    ("mouthStretchLeft", "mouthStretchRight"),
    ("mouthPressLeft", "mouthPressRight"),
    ("mouthLowerDownLeft", "mouthLowerDownRight"),
    ("mouthUpperUpLeft", "mouthUpperUpRight"),
    ("cheekSquintLeft", "cheekSquintRight"),
    ("noseSneerLeft", "noseSneerRight"),
    ("browDownLeft", "browDownRight"),
    ("browOuterUpLeft", "browOuterUpRight"),
    ("eyeBlinkLeft", "eyeBlinkRight"),
    ("eyeLookDownLeft", "eyeLookDownRight"),
    ("eyeLookInLeft", "eyeLookInRight"),
    ("eyeLookOutLeft", "eyeLookOutRight"),
    ("eyeLookUpLeft", "eyeLookUpRight"),
    ("eyeSquintLeft", "eyeSquintRight"),
    ("eyeWideLeft", "eyeWideRight"),
]

MOUTH_BLENDSHAPES = [
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
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "noseSneerLeft",
    "noseSneerRight",
    "cheekPuff",
]

BLINK_PATTERNS = [
    np.array([0.365, 0.950, 0.956, 0.917, 0.367, 0.119, 0.025]),
    np.array([0.235, 0.910, 0.945, 0.778, 0.191, 0.235, 0.089]),
    np.array([0.870, 0.950, 0.949, 0.696, 0.191, 0.073, 0.007]),
    np.array([0.000, 0.557, 0.953, 0.942, 0.426, 0.148, 0.018]),
]


def overlap_samples_from_ms(overlap_ms: float, sample_rate: int = 16000) -> int:
    return max(0, int(overlap_ms * sample_rate / 1000))


def compute_chunk_volume(
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    fps: float = 30.0,
) -> np.ndarray:
    """Per-frame RMS volume (matches LAM infer_streaming_audio librosa RMS)."""
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    hop_length = int(sample_rate / fps)
    frame_length = min(hop_length, len(audio))
    volume = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    frame_count = math.ceil(len(audio) / sample_rate * fps)
    if volume.shape[0] > frame_count:
        volume = volume[:frame_count]
    return volume.astype(np.float32, copy=False)


def symmetrize_blendshapes(
    bs_params: np.ndarray,
    mode: str = "average",
) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}
    if bs_params.ndim != 2 or bs_params.shape[1] != 52:
        raise ValueError("Input must be of shape (N, 52)")

    symmetric_bs = bs_params.copy()
    for left, right in ARKIT_LEFT_RIGHT_PAIR:
        left_idx = name_to_idx.get(left)
        right_idx = name_to_idx.get(right)
        if left_idx is None or right_idx is None:
            continue
        left_col = symmetric_bs[:, left_idx]
        right_col = symmetric_bs[:, right_idx]
        if mode == "average":
            new_vals = (left_col + right_col) / 2
        elif mode == "max":
            new_vals = np.maximum(left_col, right_col)
        elif mode == "min":
            new_vals = np.minimum(left_col, right_col)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        symmetric_bs[:, left_idx] = new_vals
        symmetric_bs[:, right_idx] = new_vals
    return symmetric_bs


def apply_random_eye_blinks_context(
    animation_params: np.ndarray,
    processed_frames: int = 0,
    intensity_range: tuple[float, float] = (0.8, 1.0),
) -> np.ndarray:
    remaining_frames = animation_params.shape[0] - processed_frames
    if remaining_frames <= 7:
        return animation_params

    min_blink_interval = 40
    max_blink_interval = 100

    previous_blink_indices = np.where(animation_params[:processed_frames, 8] > 0.5)[0]
    last_processed_blink = (
        previous_blink_indices[-1] - 7 if previous_blink_indices.size > 0 else processed_frames
    )

    blink_interval = np.random.randint(min_blink_interval, max_blink_interval)
    first_blink_start = max(0, blink_interval - last_processed_blink)

    if first_blink_start <= (remaining_frames - 7):
        blink_pattern = BLINK_PATTERNS[np.random.randint(0, 4)]
        intensity = np.random.uniform(*intensity_range)
        blink_start = processed_frames + first_blink_start
        blink_end = blink_start + 7
        animation_params[blink_start:blink_end, 8] = blink_pattern * intensity
        animation_params[blink_start:blink_end, 9] = blink_pattern * intensity

        remaining_after_blink = animation_params.shape[0] - blink_end
        if remaining_after_blink > min_blink_interval:
            second_intensity = np.random.uniform(*intensity_range)
            second_interval = np.random.randint(min_blink_interval, max_blink_interval)
            if (remaining_after_blink - 7) > second_interval:
                second_pattern = BLINK_PATTERNS[np.random.randint(0, 4)]
                second_blink_start = blink_end + second_interval
                second_blink_end = second_blink_start + 7
                animation_params[second_blink_start:second_blink_end, 8] = (
                    second_pattern * second_intensity
                )
                animation_params[second_blink_start:second_blink_end, 9] = (
                    second_pattern * second_intensity
                )

    return animation_params


def apply_savitzky_golay_smoothing(
    input_data: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
    axis: int = 0,
) -> np.ndarray:
    if input_data.ndim != 2:
        raise ValueError(f"Expected 2D input, got {input_data.ndim}D array")
    if input_data.shape[0] < window_length:
        return input_data
    original_dtype = input_data.dtype
    working_data = input_data.astype(np.float64)
    smoothed_data = savgol_filter(
        working_data,
        window_length=window_length,
        polyorder=polyorder,
        axis=axis,
        mode="mirror",
    )
    return np.clip(smoothed_data, 0.0, 1.0).astype(original_dtype)


def _blend_region_start(
    array: np.ndarray,
    region: np.ndarray,
    processed_boundary: int,
    blend_frames: int,
) -> None:
    blend_length = min(blend_frames, region[0] - processed_boundary)
    if blend_length <= 0:
        return
    pre_frame = array[region[0] - 1]
    for i in range(blend_length):
        weight = (i + 1) / (blend_length + 1)
        array[region[0] + i] = pre_frame * (1 - weight) + array[region[0] + i] * weight


def _blend_region_end(
    array: np.ndarray,
    region: np.ndarray,
    blend_frames: int,
) -> None:
    blend_length = min(blend_frames, array.shape[0] - region[-1] - 1)
    if blend_length <= 0:
        return
    post_frame = array[region[-1] + 1]
    for i in range(blend_length):
        weight = (i + 1) / (blend_length + 1)
        array[region[-1] - i] = post_frame * (1 - weight) + array[region[-1] - i] * weight


def find_low_value_regions(
    signal: np.ndarray,
    threshold: float,
    min_region_length: int = 5,
) -> list[np.ndarray]:
    low_value_indices = np.where(signal < threshold)[0]
    if len(low_value_indices) == 0:
        return []

    contiguous_regions: list[np.ndarray] = []
    region_start_idx = 0
    for i in range(1, len(low_value_indices)):
        if low_value_indices[i] != low_value_indices[i - 1] + 1:
            region = low_value_indices[region_start_idx:i]
            if len(region) >= min_region_length:
                contiguous_regions.append(region)
            region_start_idx = i
    region = low_value_indices[region_start_idx:]
    if len(region) >= min_region_length:
        contiguous_regions.append(region)
    return contiguous_regions


def smooth_mouth_movements(
    blend_shapes: np.ndarray,
    processed_frames: int,
    volume: Optional[np.ndarray] = None,
    silence_threshold: float = 0.001,
    min_silence_duration: int = 7,
    blend_window: int = 3,
) -> np.ndarray:
    if volume is None:
        return blend_shapes

    silent_regions = find_low_value_regions(
        volume,
        threshold=silence_threshold,
        min_region_length=min_silence_duration,
    )
    mouth_blend_indices = [ARKIT_BLENDSHAPE_NAMES.index(name) for name in MOUTH_BLENDSHAPES]

    for region_indices in silent_regions:
        for region_idx in region_indices.tolist():
            blend_shapes[region_idx, mouth_blend_indices] *= 0.1
        try:
            _blend_region_start(blend_shapes, region_indices, processed_frames, blend_window)
            _blend_region_end(blend_shapes, region_indices, blend_window)
        except IndexError as exc:
            warnings.warn(f"Edge blending skipped at region {region_indices}: {exc}")

    return blend_shapes


def _blend_animation_segment(
    array: np.ndarray,
    transition_start: int,
    blend_window: int,
    reference_frame: np.ndarray,
) -> None:
    actual_blend_length = min(blend_window, array.shape[0] - transition_start)
    for frame_offset in range(actual_blend_length):
        current_idx = transition_start + frame_offset
        blend_weight = (frame_offset + 1) / (actual_blend_length + 1)
        array[current_idx] = reference_frame * (1 - blend_weight) + array[current_idx] * blend_weight


def apply_frame_blending(
    blend_shapes: np.ndarray,
    processed_frames: int,
    initial_blend_window: int = 3,
    subsequent_blend_window: int = 5,
) -> np.ndarray:
    if processed_frames > 0:
        _blend_animation_segment(
            blend_shapes,
            transition_start=processed_frames,
            blend_window=subsequent_blend_window,
            reference_frame=blend_shapes[processed_frames - 1],
        )
    else:
        _blend_animation_segment(
            blend_shapes,
            transition_start=0,
            blend_window=initial_blend_window,
            reference_frame=np.zeros_like(blend_shapes[0]),
        )
    return blend_shapes


def apply_expression_postprocessing(
    expression_params: np.ndarray,
    processed_frames: int = 0,
    audio_volume: Optional[np.ndarray] = None,
) -> np.ndarray:
    """LAM Audio2ExpressionInfer.apply_expression_postprocessing pipeline."""
    expression_params = smooth_mouth_movements(
        expression_params, processed_frames, audio_volume
    )
    expression_params = apply_frame_blending(expression_params, processed_frames)
    expression_params = apply_savitzky_golay_smoothing(expression_params, window_length=5)
    expression_params = symmetrize_blendshapes(expression_params)
    expression_params = apply_random_eye_blinks_context(
        expression_params, processed_frames=processed_frames
    )
    return expression_params
