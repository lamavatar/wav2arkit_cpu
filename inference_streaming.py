#!/usr/bin/env python3
"""
Streaming audio → ARKit blendshape inference with wav2arkit_cpu (ONNX).

Model: https://huggingface.co/myned-ai/wav2arkit_cpu
  - Input:  audio_waveform float32 [batch, samples] @ 16kHz
  - Output: blendshapes float32 [batch, frames, 52] @ 30fps
  - Identity 11 baked into the exported graph (no id_idx input)

Examples:
    cd ~/work/wav2arkit_cpu
    python inference_streaming.py \\
        --audio /path/to/speech.wav \\
        --output bsData_onnx.json

    python inference_streaming.py \\
        --audio speech.wav --chunk-seconds 0.5 --output out.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import librosa
import numpy as np

from export_utils import export_blendshape_animation
from postprocess import DEFAULT_OVERLAP_MS
from session import AUDIO_SR, DEFAULT_ONNX_PATH, OUTPUT_FPS, Wav2ArkitOnnxSession


def export_json(bs_array: np.ndarray, json_path: str | Path, fps: float = OUTPUT_FPS) -> None:
    export_blendshape_animation(bs_array, json_path, fps=fps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="wav2arkit_cpu ONNX streaming inference")
    p.add_argument(
        "--audio",
        default="BarackObama_english.wav",
        help="Input audio file path (any format librosa supports)",
    )
    p.add_argument(
        "--output",
        default="bsData_onnx.json",
        help="Output ARKit blendshape JSON path (default: bsData_onnx.json)",
    )
    p.add_argument(
        "--model",
        default=str(DEFAULT_ONNX_PATH),
        help=f"Path to wav2arkit_cpu.onnx (default: {DEFAULT_ONNX_PATH})",
    )
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=1.0,
        help="Streaming chunk size in seconds (default: 1.0)",
    )
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="Run single ONNX pass on full audio (no chunking)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-chunk timing logs",
    )
    p.add_argument(
        "--intra-op-threads",
        type=int,
        default=2,
        help="ORT intra_op_num_threads (default: 4)",
    )
    p.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Dummy inference runs after session load (default: 1, 0=off)",
    )
    p.add_argument(
        "--overlap-ms",
        type=float,
        default=DEFAULT_OVERLAP_MS,
        help=(
            "Audio overlap for LAM-style streaming in milliseconds "
            f"(default: {DEFAULT_OVERLAP_MS}, 0=disable overlap/post-process)"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"Audio not found: {audio_path}", file=sys.stderr)
        return 1

    audio, _sr = librosa.load(str(audio_path), sr=AUDIO_SR, mono=True)
    duration = len(audio) / AUDIO_SR
    expected_frames = math.ceil(OUTPUT_FPS * duration)

    session = Wav2ArkitOnnxSession(
        model_path=args.model,
        intra_op_threads=args.intra_op_threads,
        warmup_runs=args.warmup_runs,
    )
    if not args.quiet and args.warmup_runs > 0:
        overlap_note = (
            f"overlap={args.overlap_ms:.0f}ms"
            if not args.no_streaming and args.overlap_ms > 0
            else "overlap=off"
        )
        print(
            f"session ready  threads={args.intra_op_threads}  "
            f"warmup={args.warmup_runs}x1s  {overlap_note}"
        )

    if args.no_streaming:
        import time

        t0 = time.perf_counter()
        blendshapes = session.infer_full(audio)
        elapsed = time.perf_counter() - t0
        if not args.quiet:
            print(
                f"full pass  samples={len(audio)}  frames={blendshapes.shape[0]}  "
                f"time={elapsed * 1000:.1f}ms"
            )
    else:
        chunk_samples = max(1, int(args.chunk_seconds * AUDIO_SR))
        blendshapes, timings = session.infer_streaming(
            audio,
            chunk_samples=chunk_samples,
            overlap_ms=args.overlap_ms,
            verbose=not args.quiet,
        )
        if not args.quiet and timings:
            total = sum(timings)
            rt_note = f"({duration / total:.1f}x realtime)" if total > 0 else ""
            print(f"total inference {total * 1000:.1f}ms  {rt_note}")

    if blendshapes.shape[0] > expected_frames:
        blendshapes = blendshapes[:expected_frames]
    elif blendshapes.shape[0] < expected_frames:
        pad = np.zeros((expected_frames - blendshapes.shape[0], 52), dtype=np.float32)
        blendshapes = np.concatenate([blendshapes, pad], axis=0)

    export_json(blendshapes, args.output)
    print(
        f"audio={audio_path.name}  duration={duration:.2f}s  "
        f"frames={blendshapes.shape[0]}  saved={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
