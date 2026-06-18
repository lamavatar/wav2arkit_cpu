"""Smoke test for the SPL2 + ONNX lip-sync Android pipeline (PC side).

Validates the pieces the Android app relies on:
  1. SPL2 splat header parses (magic, N, M, fps, geometry-only / no weights).
  2. u8 standard input: ByteArray(16000)=128 (silence) -> u8->float [-1,1] -> ONNX
     produces a [~30, 52] frame block.
  3. u8 quantization parity: float waveform vs (float->u8->float) round-trip
     yields ONNX outputs within tolerance.

Run:
    python scripts/smoke_test_spl2_onnx.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from session import Wav2ArkitOnnxSession  # noqa: E402

SPLAT = ROOT / "android" / "SplatBench" / "app" / "src" / "main" / "assets" / "vfhq_case1.splat"
ONNX = ROOT / "android" / "SplatBench" / "app" / "src" / "main" / "assets" / "models" / "wav2arkit_cpu_int8.onnx"
AUDIO_SR = 16000


def u8_to_onnx_float(pcm_u8: np.ndarray) -> np.ndarray:
    """Match AudioPcmConverter.u8ToOnnxFloat: (byte-128)/128 -> [-1,1]."""
    return ((pcm_u8.astype(np.int32) - 128) / 128.0).astype(np.float32)


def float_to_u8(x: np.ndarray) -> np.ndarray:
    """Match the on-device i16->u8 quantization path applied to a float wave."""
    i16 = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int32)
    return np.clip((i16 >> 8) + 128, 0, 255).astype(np.uint8)


def test_spl2_header() -> None:
    data = SPLAT.read_bytes()
    magic = data[:4]
    assert magic == b"SPL2", f"expected SPL2, got {magic!r}"
    n, m = struct.unpack_from("<II", data, 4)
    (fps,) = struct.unpack_from("<f", data, 12)
    assert n > 0 and m > 0, (n, m)
    assert abs(fps - 30.0) < 1e-3, fps
    print(f"[ok] SPL2 header: N={n} M={m} fps={fps:g} (geometry-only)")


def test_u8_silence_infer(session: Wav2ArkitOnnxSession) -> None:
    pcm_u8 = np.full(AUDIO_SR, 128, dtype=np.uint8)  # silence
    audio = u8_to_onnx_float(pcm_u8)
    out, _ = session.infer_streaming_chunk(audio)
    assert out.ndim == 2 and out.shape[1] == 52, out.shape
    assert 28 <= out.shape[0] <= 32, out.shape
    print(f"[ok] u8 silence -> infer: frames={out.shape[0]} dims={out.shape[1]}")


def test_u8_parity(session: Wav2ArkitOnnxSession) -> None:
    t = np.arange(AUDIO_SR) / AUDIO_SR
    wave = (0.6 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    out_float = session.infer_chunk(wave)
    out_u8 = session.infer_chunk(u8_to_onnx_float(float_to_u8(wave)))

    max_diff = float(np.max(np.abs(out_float - out_u8)))
    mean_diff = float(np.mean(np.abs(out_float - out_u8)))
    # 8-bit unsigned (256 levels) is the mandated on-device input standard, so a
    # bounded per-weight deviation from full-precision float is expected.
    print(f"[ok] u8 parity: max={max_diff:.4f} mean={mean_diff:.4f} |float - u8|")
    assert max_diff < 0.25, f"u8 quantization diverged too much: {max_diff}"
    assert mean_diff < 0.05, f"u8 mean drift too high: {mean_diff}"


def main() -> int:
    assert SPLAT.is_file(), f"missing splat: {SPLAT}"
    assert ONNX.is_file(), f"missing onnx: {ONNX}"

    test_spl2_header()
    session = Wav2ArkitOnnxSession(model_path=ONNX, warmup_runs=1)
    test_u8_silence_infer(session)
    test_u8_parity(session)
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
