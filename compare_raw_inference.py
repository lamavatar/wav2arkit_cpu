#!/usr/bin/env python3
"""Compare LAM PyTorch vs wav2arkit ONNX inference outputs.

Modes:
  default           full-pass raw (no postprocess)
  --streaming       1 s chunks, 200 ms overlap (1.2 s ONNX window)
  --postprocess     LAM apply_expression_postprocessing / ONNX infer_streaming postprocess
  --save-baseline   write compare_outputs/baseline.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent
LAM_ROOT = ROOT.parent / "LAM_Audio2Expression"
DEFAULT_AUDIO = LAM_ROOT / "assets" / "sample_audio" / "BarackObama_english.wav"
DEFAULT_ONNX_FP32 = ROOT / "models" / "wav2arkit_cpu.onnx"
DEFAULT_ONNX_INT8 = ROOT / "models" / "wav2arkit_cpu_int8.onnx"

ARKIT_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]

AUDIO_SR = 16000
OUTPUT_FPS = 30.0
CHUNK_SAMPLES = AUDIO_SR
LAM_MAX_FRAME_LENGTH = 64
ONNX_IDENTITY_IDX = 11


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=AUDIO_SR, mono=True)
    return audio.astype(np.float32)


def expected_frames(num_samples: int) -> int:
    return math.ceil(OUTPUT_FPS * num_samples / AUDIO_SR)


def trim_frames(arr: np.ndarray, num_samples: int) -> np.ndarray:
    return arr[: expected_frames(num_samples)]


def _load_lam_infer(id_idx: int):
    import torch

    if str(LAM_ROOT) not in sys.path:
        sys.path.insert(0, str(LAM_ROOT))

    from engines.defaults import default_config_parser, default_setup
    from engines.infer import INFER

    prev_cwd = os.getcwd()
    os.chdir(LAM_ROOT)
    try:
        cfg = default_config_parser("configs/lam_audio2exp_config_streaming.py", None)
        cfg = default_setup(cfg)
        cfg.id_idx = id_idx
        infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))
        infer.model.eval()
        return infer, cfg
    finally:
        os.chdir(prev_cwd)


def run_lam_full_raw(audio: np.ndarray, id_idx: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    infer, cfg = _load_lam_infer(id_idx)
    num_identity = cfg.model.backbone.num_identity_classes
    with torch.no_grad():
        input_dict = {
            "id_idx": F.one_hot(torch.tensor(id_idx), num_identity).cpu()[None, ...],
            "input_audio_array": torch.from_numpy(audio).float().cpu()[None, ...],
        }
        return infer.model(input_dict)["pred_exp"].squeeze(0).cpu().numpy()


def run_lam_streaming(
    audio: np.ndarray,
    id_idx: int,
    *,
    postprocess: bool,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    infer, cfg = _load_lam_infer(id_idx)
    num_identity = cfg.model.backbone.num_identity_classes
    window_samples = cfg.audio_sr * LAM_MAX_FRAME_LENGTH // 30

    prev_cwd = os.getcwd()
    os.chdir(LAM_ROOT)
    try:
        if str(LAM_ROOT) not in sys.path:
            sys.path.insert(0, str(LAM_ROOT))
        from models.utils import DEFAULT_CONTEXT

        parts: list[np.ndarray] = []
        context = DEFAULT_CONTEXT.copy()

        for i in range(math.ceil(len(audio) / CHUNK_SAMPLES)):
            chunk = audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]

            if postprocess:
                out, context = infer.infer_streaming_audio(chunk, AUDIO_SR, context)
                parts.append(out["expression"])
                continue

            in_audio = chunk.copy()
            start_frame = int(LAM_MAX_FRAME_LENGTH - in_audio.shape[0] / cfg.audio_sr * 30)
            if context["is_initial_input"] or context["previous_audio"] is None:
                blank_len = window_samples - in_audio.shape[0]
                input_audio = np.concatenate([np.zeros(blank_len, np.float32), in_audio])
            else:
                clip_len = window_samples - in_audio.shape[0]
                input_audio = np.concatenate([context["previous_audio"][-clip_len:], in_audio])
            context["previous_audio"] = input_audio

            with torch.no_grad():
                input_dict = {
                    "id_idx": F.one_hot(torch.tensor(id_idx), num_identity).cpu()[None, ...],
                    "input_audio_array": torch.from_numpy(input_audio).float().cpu()[None, ...],
                }
                out_exp = infer.model(input_dict)["pred_exp"].squeeze(0).cpu().numpy()[start_frame:]
            parts.append(out_exp)
            context["is_initial_input"] = False

        return np.concatenate(parts, axis=0)
    finally:
        os.chdir(prev_cwd)


def run_onnx_full_raw(audio: np.ndarray, model_path: Path) -> np.ndarray:
    from session import Wav2ArkitOnnxSession

    session = Wav2ArkitOnnxSession(model_path=model_path, warmup_runs=0)
    return session.infer_full(audio)


def run_onnx_streaming(
    audio: np.ndarray,
    model_path: Path,
    *,
    postprocess: bool,
    overlap_ms: float,
) -> np.ndarray:
    from postprocess import DEFAULT_OVERLAP_MS, overlap_samples_from_ms
    from session import StreamingContext, Wav2ArkitOnnxSession

    session = Wav2ArkitOnnxSession(model_path=model_path, warmup_runs=0)

    if postprocess:
        out, _ = session.infer_streaming(
            audio,
            chunk_samples=CHUNK_SAMPLES,
            overlap_ms=overlap_ms,
            verbose=False,
            time_inference=False,
        )
        return out

    overlap_samples = overlap_samples_from_ms(overlap_ms, AUDIO_SR)
    parts: list[np.ndarray] = []
    ctx = StreamingContext()

    for i in range(math.ceil(len(audio) / CHUNK_SAMPLES)):
        chunk = audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES].astype(np.float32)
        if ctx.is_initial_input or ctx.previous_audio is None:
            input_audio = np.concatenate([np.zeros(overlap_samples, np.float32), chunk])
        else:
            input_audio = np.concatenate([ctx.previous_audio[-overlap_samples:], chunk])
        ctx.previous_audio = input_audio.copy()
        out = session.infer_chunk(input_audio)
        chunk_frames = session.frames_for_samples(len(chunk))
        start = out.shape[0] - chunk_frames
        if start < 0:
            start = 0
        parts.append(out[start:])
        ctx.is_initial_input = False

    return np.concatenate(parts, axis=0)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 5) -> tuple[int, float]:
    best_lag, best_corr = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = a[-lag:], b[: len(b) + lag]
        elif lag > 0:
            x, y = a[: len(a) - lag], b[lag:]
        else:
            x, y = a, b
        if len(x) < 3:
            continue
        c = pearson_corr(x, y)
        if not np.isnan(c) and c > best_corr:
            best_corr, best_lag = c, lag
    return best_lag, best_corr


def compare_arrays(name_a: str, arr_a: np.ndarray, name_b: str, arr_b: np.ndarray) -> dict[str, Any]:
    n = min(arr_a.shape[0], arr_b.shape[0])
    a = arr_a[:n]
    b = arr_b[:n]
    diff = a - b
    abs_diff = np.abs(diff)

    per_ch_mae = abs_diff.mean(axis=0)
    top_idx = np.argsort(per_ch_mae)[::-1][:8]
    jaw_idx = ARKIT_NAMES.index("jawOpen")
    lag, jaw_corr = best_lag_corr(a[:, jaw_idx], b[:, jaw_idx])

    mouth_names = [
        "jawOpen", "mouthClose", "mouthPucker", "mouthShrugUpper",
        "mouthLowerDownLeft", "mouthLowerDownRight",
    ]
    mouth_mae = float(np.mean([per_ch_mae[ARKIT_NAMES.index(n)] for n in mouth_names]))

    return {
        "frames_compared": n,
        "shape_a": list(arr_a.shape),
        "shape_b": list(arr_b.shape),
        "overall_mae": float(abs_diff.mean()),
        "overall_rmse": float(np.sqrt((diff ** 2).mean())),
        "mouth_mae": mouth_mae,
        "max_abs": float(abs_diff.max()),
        "jawOpen_corr": pearson_corr(a[:, jaw_idx], b[:, jaw_idx]),
        "jawOpen_best_lag": lag,
        "jawOpen_best_lag_corr": jaw_corr,
        "top_channels": [
            {
                "name": ARKIT_NAMES[i],
                "mae": float(per_ch_mae[i]),
                "a_mean": float(a[:, i].mean()),
                "b_mean": float(b[:, i].mean()),
            }
            for i in top_idx
        ],
        "label_a": name_a,
        "label_b": name_b,
    }


def print_report(title: str, stats: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    print(f"  {stats['label_a']} shape: {stats['shape_a']}")
    print(f"  {stats['label_b']} shape: {stats['shape_b']}")
    print(f"  frames compared: {stats['frames_compared']}")
    print(f"  overall MAE:  {stats['overall_mae']:.6f}")
    print(f"  mouth MAE:    {stats['mouth_mae']:.6f}")
    print(f"  overall RMSE: {stats['overall_rmse']:.6f}")
    print(f"  max |diff|:   {stats['max_abs']:.6f}")
    print(
        f"  jawOpen corr: {stats['jawOpen_corr']:.4f}  "
        f"(best lag={stats['jawOpen_best_lag']}, corr={stats['jawOpen_best_lag_corr']:.4f})"
    )
    print("  top channel MAE:")
    for ch in stats["top_channels"]:
        print(
            f"    {ch['name']:22s}  MAE={ch['mae']:.5f}  "
            f"{stats['label_a']}_mean={ch['a_mean']:.4f}  "
            f"{stats['label_b']}_mean={ch['b_mean']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare LAM vs wav2arkit ONNX inference")
    p.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    p.add_argument("--onnx-fp32", type=Path, default=DEFAULT_ONNX_FP32)
    p.add_argument("--onnx-int8", type=Path, default=DEFAULT_ONNX_INT8)
    p.add_argument("--lam-id", type=int, default=ONNX_IDENTITY_IDX, help="LAM identity index")
    p.add_argument("--streaming", action="store_true", help="1 s chunk streaming (200 ms overlap)")
    p.add_argument("--postprocess", action="store_true", help="Apply LAM/ONNX streaming postprocess")
    p.add_argument("--overlap-ms", type=float, default=None, help="ONNX overlap ms (default: 200)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for eye-blink postprocess")
    p.add_argument("--save-dir", type=Path, default=ROOT / "compare_outputs")
    p.add_argument("--save-baseline", action="store_true", help="Write baseline.json under save-dir")
    p.add_argument("--skip-int8", action="store_true")
    p.add_argument(
        "--also-lam-id0",
        action="store_true",
        help="Also compare LAM id=0 (legacy default)",
    )
    return p.parse_args()


def mode_label(streaming: bool, postprocess: bool) -> str:
    if streaming and postprocess:
        return "streaming_postprocess"
    if streaming:
        return "streaming_raw"
    if postprocess:
        return "full_postprocess"
    return "full_raw"


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        print(f"Audio not found: {args.audio}", file=sys.stderr)
        return 1

    if args.postprocess and not args.streaming:
        print("Note: --postprocess without --streaming uses streaming path internally for LAM/ONNX parity.")

    from postprocess import DEFAULT_OVERLAP_MS

    overlap_ms = DEFAULT_OVERLAP_MS if args.overlap_ms is None else args.overlap_ms
    np.random.seed(args.seed)

    audio = load_audio(args.audio)
    exp_frames = expected_frames(len(audio))
    mode = mode_label(args.streaming, args.postprocess)
    use_streaming = args.streaming or args.postprocess

    print(f"audio: {args.audio.name}")
    print(f"samples: {len(audio)}  duration: {len(audio)/AUDIO_SR:.3f}s  expected_frames: {exp_frames}")
    print(f"mode: {mode}  lam_id: {args.lam_id}  onnx_identity: {ONNX_IDENTITY_IDX}")
    print(f"overlap_ms: {overlap_ms}  seed: {args.seed}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.save_dir / "input_audio.npy", audio)

    baseline: dict[str, Any] = {
        "audio": str(args.audio),
        "mode": mode,
        "lam_id": args.lam_id,
        "onnx_identity_index": ONNX_IDENTITY_IDX,
        "overlap_ms": overlap_ms,
        "seed": args.seed,
        "chunk_seconds": 1.0,
        "max_context_frames": 64,
        "comparisons": [],
    }

    def run_pair(lam_id: int) -> tuple[np.ndarray, np.ndarray]:
        np.random.seed(args.seed)
        if use_streaming:
            lam = trim_frames(
                run_lam_streaming(audio, lam_id, postprocess=args.postprocess),
                len(audio),
            )
        else:
            lam = trim_frames(run_lam_full_raw(audio, lam_id), len(audio))

        np.random.seed(args.seed)
        if use_streaming:
            onnx = trim_frames(
                run_onnx_streaming(
                    audio,
                    args.onnx_fp32,
                    postprocess=args.postprocess,
                    overlap_ms=overlap_ms,
                ),
                len(audio),
            )
        else:
            onnx = trim_frames(run_onnx_full_raw(audio, args.onnx_fp32), len(audio))
        return lam, onnx

    tag = "postprocess" if args.postprocess else "raw"
    path_tag = "stream" if use_streaming else "full"

    print(f"\n[1] LAM id={args.lam_id} ({path_tag} {tag})...")
    lam_main, onnx_fp32 = run_pair(args.lam_id)
    np.save(args.save_dir / f"lam_id{args.lam_id}_{path_tag}_{tag}.npy", lam_main)
    np.save(args.save_dir / f"onnx_fp32_{path_tag}_{tag}.npy", onnx_fp32)
    print(f"  LAM: {lam_main.shape}  ONNX FP32: {onnx_fp32.shape}")

    stats_main = compare_arrays(f"LAM id={args.lam_id}", lam_main, "ONNX FP32", onnx_fp32)
    print_report(f"LAM id={args.lam_id}  vs  ONNX FP32 ({mode})", stats_main)
    baseline["comparisons"].append(stats_main)

    if args.also_lam_id0 and args.lam_id != 0:
        np.random.seed(args.seed)
        print(f"\n[1b] LAM id=0 ({path_tag} {tag})...")
        lam0, _ = run_pair(0)
        np.save(args.save_dir / f"lam_id0_{path_tag}_{tag}.npy", lam0)
        stats0 = compare_arrays("LAM id=0", lam0, "ONNX FP32", onnx_fp32)
        print_report(f"LAM id=0  vs  ONNX FP32 ({mode})", stats0)
        baseline["comparisons"].append(stats0)

    if not args.skip_int8 and args.onnx_int8.is_file():
        np.random.seed(args.seed)
        print(f"\n[2] ONNX INT8 ({path_tag} {tag})...")
        if use_streaming:
            onnx_int8 = trim_frames(
                run_onnx_streaming(
                    audio,
                    args.onnx_int8,
                    postprocess=args.postprocess,
                    overlap_ms=overlap_ms,
                ),
                len(audio),
            )
        else:
            onnx_int8 = trim_frames(run_onnx_full_raw(audio, args.onnx_int8), len(audio))
        np.save(args.save_dir / f"onnx_int8_{path_tag}_{tag}.npy", onnx_int8)
        stats_int8 = compare_arrays("ONNX FP32", onnx_fp32, "ONNX INT8", onnx_int8)
        print_report(f"ONNX FP32  vs  ONNX INT8 ({mode})", stats_int8)
        baseline["comparisons"].append(stats_int8)

        np.random.seed(args.seed)
        stats_lam_int8 = compare_arrays(f"LAM id={args.lam_id}", lam_main, "ONNX INT8", onnx_int8)
        print_report(f"LAM id={args.lam_id}  vs  ONNX INT8 ({mode})", stats_lam_int8)
        baseline["comparisons"].append(stats_lam_int8)

    if args.save_baseline:
        baseline_path = args.save_dir / "baseline.json"
        with baseline_path.open("w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        print(f"\nBaseline saved: {baseline_path}")

    print(f"\nSaved arrays under: {args.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
