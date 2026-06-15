#!/usr/bin/env python3
"""Dynamic INT8 quantization for wav2arkit_cpu ONNX.

Steps:
  1. Merge external data (.onnx + .onnx.data) into one FP32 file
  2. ORT quant_pre_process (constant folding; skip symbolic shape for this model)
  3. quantize_dynamic (INT8 weights)

Default quantizes MatMul/Gemm only (CPU-compatible). Full quant including Conv
produces ConvInteger nodes that may fail on CPUExecutionProvider.

Usage:
    cd ~/work/wav2arkit_cpu
    python quantize_model.py
    python quantize_model.py --include-conv   # smaller file; CPU infer may fail

Do NOT run: python -m onnxruntime.quantization.quantize_dynamic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "models" / "wav2arkit_cpu.onnx"
DEFAULT_MERGED = ROOT / "models" / "wav2arkit_cpu_merged.onnx"
DEFAULT_PREPROCESSED = ROOT / "models" / "wav2arkit_cpu_preprocessed.onnx"
DEFAULT_OUTPUT = ROOT / "models" / "wav2arkit_cpu_int8.onnx"

CPU_SAFE_OPS = ["MatMul", "Gemm"]


def merge_external_data(src: Path, dst: Path) -> Path:
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    print(f"[1/3] merge external data: {src.name} -> {dst.name}")
    model = onnx.load(str(src), load_external_data=False)
    if src.with_suffix(".onnx.data").is_file():
        load_external_data_for_model(model, str(src.parent))
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst), save_as_external_data=False)
    print(f"      size: {dst.stat().st_size / 1024 / 1024:.1f} MB")
    return dst


def preprocess_for_quant(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization.shape_inference import quant_pre_process

    print(f"[2/3] preprocess (ORT): {src.name} -> {dst.name}")
    quant_pre_process(
        input_model=str(src),
        output_model_path=str(dst),
        skip_symbolic_shape=True,
        save_as_external_data=False,
    )
    print(f"      size: {dst.stat().st_size / 1024 / 1024:.1f} MB")
    return dst


def quantize_int8(src: Path, dst: Path, *, include_conv: bool) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    op_types = None if include_conv else CPU_SAFE_OPS
    label = "all linear/conv ops" if include_conv else "MatMul/Gemm only (CPU-safe)"
    print(f"[3/3] dynamic INT8 ({label}): {src.name} -> {dst.name}")

    kwargs: dict = {
        "model_input": str(src),
        "model_output": str(dst),
        "weight_type": QuantType.QInt8,
        "use_external_data_format": False,
    }
    if op_types is not None:
        kwargs["op_types_to_quantize"] = op_types

    quantize_dynamic(**kwargs)
    print(f"      size: {dst.stat().st_size / 1024 / 1024:.1f} MB")
    return dst


def verify_inference(model_path: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    print(f"verify: {model_path.name}")
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    audio = np.random.randn(1, 16000).astype(np.float32)
    out = session.run(None, {"audio_waveform": audio})[0]
    print(f"      output shape {out.shape}  ok")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge + preprocess + dynamic INT8 quantize wav2arkit_cpu.onnx"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    p.add_argument("--preprocessed", type=Path, default=DEFAULT_PREPROCESSED)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--include-conv",
        action="store_true",
        help="Quantize Conv too (~96MB). May fail on CPU EP (ConvInteger).",
    )
    p.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep merged/preprocessed FP32 files",
    )
    p.add_argument("--no-verify", action="store_true", help="Skip ORT smoke test")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    try:
        merged = merge_external_data(args.input, args.merged)
        preprocessed = preprocess_for_quant(merged, args.preprocessed)
        quantize_int8(preprocessed, args.output, include_conv=args.include_conv)
        if not args.no_verify:
            verify_inference(args.output)
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    if not args.keep_intermediate:
        for path in (args.merged, args.preprocessed):
            if path.is_file() and path != args.output:
                path.unlink()
                print(f"removed: {path.name}")

    print(f"done: {args.output}")
    if args.include_conv:
        print(
            "note: full INT8 with Conv may require NNAPI/GPU EP; "
            "use default (no --include-conv) for CPU."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
