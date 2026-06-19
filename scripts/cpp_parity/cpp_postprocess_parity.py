"""Parity check: C++ expression_postprocess vs Python postprocess.py.

Generates synthetic streaming chunks, runs the Python reference pipeline
(session._postprocess_chunk equivalent) and the compiled C++ ProcessChunk,
and compares all channels except the RNG-driven eye-blink channels (8, 9).

Usage:
    python cpp_postprocess_parity.py            # build + run + compare
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from postprocess import (  # noqa: E402
    MAX_CONTEXT_FRAMES,
    apply_expression_postprocessing,
    compute_chunk_volume,
)

NUM_CHANNELS = 52
CLANG = r"C:\Program Files\LLVM\bin\clang++.exe"
CPP_SRC = ROOT / "android" / "SplatBench" / "app" / "src" / "main" / "cpp" / "expression_postprocess.cpp"
MAIN_SRC = HERE / "parity_main.cpp"
EXE = HERE / "parity_test.exe"
INPUT_BIN = HERE / "input.bin"
CPP_OUT = HERE / "cpp_out.bin"


class Ctx:
    previous_expression = None
    previous_volume = None


def py_postprocess_chunk(out_exp, volume, ctx):
    if ctx.previous_expression is None:
        out = apply_expression_postprocessing(out_exp.copy(), 0, volume)
        ctx.previous_expression = out.copy()
        ctx.previous_volume = volume.copy()
        return out
    prev = ctx.previous_expression.shape[0]
    combined = np.concatenate([ctx.previous_expression, out_exp], axis=0)
    cvol = np.concatenate([ctx.previous_volume, volume], axis=0)
    processed = apply_expression_postprocessing(combined, prev, cvol)
    out = processed[prev:]
    ctx.previous_expression = np.concatenate(
        [ctx.previous_expression, out], axis=0
    )[-MAX_CONTEXT_FRAMES:]
    ctx.previous_volume = np.concatenate(
        [ctx.previous_volume, volume], axis=0
    )[-MAX_CONTEXT_FRAMES:]
    return out


def make_chunks(rng, num_chunks=6, k=30, audio_len=16000):
    chunks = []
    for c in range(num_chunks):
        frames = rng.random((k, NUM_CHANNELS)).astype(np.float32)
        audio = (rng.random(audio_len).astype(np.float32) * 2.0 - 1.0) * 0.3
        # Inject a silent region (~0.3 s) to exercise smooth_mouth_movements.
        sil0 = int(audio_len * 0.1)
        sil1 = int(audio_len * 0.4)
        audio[sil0:sil1] = 0.0
        chunks.append((frames, audio))
    return chunks


def write_input(chunks):
    with open(INPUT_BIN, "wb") as f:
        f.write(struct.pack("<i", len(chunks)))
        for frames, audio in chunks:
            k = frames.shape[0]
            f.write(struct.pack("<i", k))
            f.write(struct.pack("<i", audio.shape[0]))
            f.write(frames.astype("<f4").tobytes())
            f.write(audio.astype("<f4").tobytes())


def read_cpp_out():
    outs = []
    with open(CPP_OUT, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (k,) = struct.unpack("<i", head)
            data = f.read(k * NUM_CHANNELS * 4)
            arr = np.frombuffer(data, dtype="<f4").reshape(k, NUM_CHANNELS)
            outs.append(arr.copy())
    return outs


def main() -> int:
    if not Path(CLANG).is_file():
        print(f"clang++ not found at {CLANG}")
        return 2

    rng = np.random.default_rng(1234)
    chunks = make_chunks(rng)
    write_input(chunks)

    # Build the C++ harness.
    cmd = [CLANG, "-std=c++17", "-O2", str(MAIN_SRC), str(CPP_SRC), "-o", str(EXE)]
    print("building:", " ".join(cmd))
    build = subprocess.run(cmd, capture_output=True, text=True)
    if build.returncode != 0:
        print(build.stdout)
        print(build.stderr)
        return 3

    run = subprocess.run([str(EXE), str(INPUT_BIN), str(CPP_OUT)],
                         capture_output=True, text=True, cwd=str(HERE))
    print(run.stdout.strip())
    if run.returncode != 0:
        print(run.stderr)
        return 4
    cpp_outs = read_cpp_out()

    # Python reference (fresh context).
    ctx = Ctx()
    py_outs = []
    for frames, audio in chunks:
        vol = compute_chunk_volume(audio)
        py_outs.append(py_postprocess_chunk(frames, vol, ctx))

    # Compare (exclude RNG eye-blink channels 8, 9).
    keep = [i for i in range(NUM_CHANNELS) if i not in (8, 9)]
    worst = 0.0
    for i, (a, b) in enumerate(zip(py_outs, cpp_outs)):
        if a.shape != b.shape:
            print(f"chunk {i}: shape mismatch {a.shape} vs {b.shape}")
            return 5
        diff = np.abs(a[:, keep] - b[:, keep])
        mx = float(diff.max()) if diff.size else 0.0
        mean = float(diff.mean()) if diff.size else 0.0
        worst = max(worst, mx)
        print(f"chunk {i}: k={a.shape[0]} max_diff={mx:.3e} mean_diff={mean:.3e}")

    print(f"\nWORST max_diff (excl. ch 8/9) = {worst:.3e}")
    ok = worst < 1e-4
    print("PARITY OK" if ok else "PARITY FAIL")
    return 0 if ok else 6


if __name__ == "__main__":
    raise SystemExit(main())
