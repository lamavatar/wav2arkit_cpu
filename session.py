"""ONNX Runtime wrapper for myned-ai/wav2arkit_cpu."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from postprocess import (
    DEFAULT_OVERLAP_MS,
    MAX_CONTEXT_FRAMES,
    apply_expression_postprocessing,
    compute_chunk_volume,
    overlap_samples_from_ms,
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_ONNX_PATH = DEFAULT_MODEL_DIR / "wav2arkit_cpu_int8.onnx"
# DEFAULT_ONNX_PATH = DEFAULT_MODEL_DIR / "wav2arkit_cpu.onnx"
AUDIO_SR = 16000
OUTPUT_FPS = 30.0

# ORT CPU thread pinning (reduces run-to-run variance on shared hosts)
DEFAULT_INTRA_OP_THREADS = 4
DEFAULT_INTER_OP_THREADS = 1
DEFAULT_WARMUP_RUNS = 1
DEFAULT_WARMUP_SAMPLES = AUDIO_SR


@dataclass
class StreamingContext:
    """Stateful context for LAM-style streaming (mirrors LAM DEFAULT_CONTEXT)."""

    is_initial_input: bool = True
    previous_audio: np.ndarray | None = None
    previous_expression: np.ndarray | None = None
    previous_volume: np.ndarray | None = None


class Wav2ArkitOnnxSession:
    """End-to-end audio → ARKit blendshapes via ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
        *,
        intra_op_threads: int = DEFAULT_INTRA_OP_THREADS,
        inter_op_threads: int = DEFAULT_INTER_OP_THREADS,
        warmup_runs: int = DEFAULT_WARMUP_RUNS,
        warmup_samples: int = DEFAULT_WARMUP_SAMPLES,
    ) -> None:
        import onnxruntime as ort

        model_path = Path(model_path or DEFAULT_ONNX_PATH)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                "Run: hf download myned-ai/wav2arkit_cpu --local-dir models"
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = inter_op_threads

        self._intra_op_threads = intra_op_threads
        self._inter_op_threads = inter_op_threads
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._warmed_up = False

        if warmup_runs > 0:
            self.warmup(runs=warmup_runs, samples=warmup_samples)

    def warmup(self, *, runs: int = DEFAULT_WARMUP_RUNS, samples: int = DEFAULT_WARMUP_SAMPLES) -> None:
        """Run dummy inferences to stabilize ORT memory pools and CPU caches."""
        dummy = np.zeros(samples, dtype=np.float32)
        for _ in range(runs):
            self.infer_chunk(dummy)
        self._warmed_up = True

    @staticmethod
    def frames_for_samples(num_samples: int) -> int:
        return math.ceil(OUTPUT_FPS * num_samples / AUDIO_SR)

    def infer_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Run ONNX on one audio chunk. Returns [frames, 52]."""
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        audio_input = audio.astype(np.float32, copy=False)[None, :]
        blendshapes = self._session.run(
            None, {self._input_name: audio_input}
        )[0]
        return blendshapes.squeeze(0)

    def _build_overlap_input(
        self,
        in_audio: np.ndarray,
        context: StreamingContext,
        overlap_samples: int,
    ) -> np.ndarray:
        """LAM infer_streaming_audio audio overlap (blank pad or previous tail)."""
        if context.is_initial_input or context.previous_audio is None:
            blank = np.zeros(overlap_samples, dtype=np.float32)
            input_audio = np.concatenate([blank, in_audio])
        else:
            clip_pre = context.previous_audio[-overlap_samples:]
            input_audio = np.concatenate([clip_pre, in_audio])
        context.previous_audio = input_audio.copy()
        return input_audio

    def _slice_new_frames(self, out_exp: np.ndarray, chunk_samples: int) -> np.ndarray:
        chunk_frames = self.frames_for_samples(chunk_samples)
        start_frame = out_exp.shape[0] - chunk_frames
        if start_frame < 0:
            start_frame = 0
        return out_exp[start_frame:]

    def _postprocess_chunk(
        self,
        out_exp: np.ndarray,
        volume: np.ndarray,
        context: StreamingContext,
        *,
        max_context_frames: int = MAX_CONTEXT_FRAMES,
    ) -> np.ndarray:
        """LAM expression post-processing with cross-chunk context."""
        if context.previous_expression is None:
            out_exp = apply_expression_postprocessing(out_exp, audio_volume=volume)
            context.previous_expression = out_exp.copy()
            context.previous_volume = volume.copy()
        else:
            previous_length = context.previous_expression.shape[0]
            combined_exp = np.concatenate([context.previous_expression, out_exp], axis=0)
            combined_vol = np.concatenate([context.previous_volume, volume], axis=0)
            processed = apply_expression_postprocessing(
                combined_exp,
                processed_frames=previous_length,
                audio_volume=combined_vol,
            )
            out_exp = processed[previous_length:]
            context.previous_expression = np.concatenate(
                [context.previous_expression, out_exp], axis=0
            )[-max_context_frames:]
            context.previous_volume = np.concatenate(
                [context.previous_volume, volume], axis=0
            )[-max_context_frames:]

        context.is_initial_input = False
        return out_exp

    def infer_streaming_chunk(
        self,
        audio: np.ndarray,
        context: StreamingContext | None = None,
        *,
        overlap_ms: float = DEFAULT_OVERLAP_MS,
        max_context_frames: int = MAX_CONTEXT_FRAMES,
    ) -> tuple[np.ndarray, StreamingContext]:
        """Process one audio chunk with LAM boundary handling."""
        if context is None:
            context = StreamingContext()

        in_audio = audio.astype(np.float32, copy=False).reshape(-1)
        overlap_samples = overlap_samples_from_ms(overlap_ms, AUDIO_SR)
        volume = compute_chunk_volume(in_audio, sample_rate=AUDIO_SR, fps=OUTPUT_FPS)

        input_audio = self._build_overlap_input(in_audio, context, overlap_samples)
        out_exp = self.infer_chunk(input_audio)
        out_exp = self._slice_new_frames(out_exp, len(in_audio))
        start =  time.perf_counter();
        out_exp = self._postprocess_chunk(
            out_exp, volume, context, max_context_frames=max_context_frames
        )
        end = time.perf_counter();
        print(f"postprocess_chunk time: {(end - start)* 1000:.1f} milliseconds") 
        return out_exp, context

    def infer_streaming(
        self,
        audio: np.ndarray,
        *,
        chunk_samples: int = AUDIO_SR,
        overlap_ms: float = DEFAULT_OVERLAP_MS,
        max_context_frames: int = MAX_CONTEXT_FRAMES,
        verbose: bool = True,
        time_inference: bool = True,
    ) -> tuple[np.ndarray, list[float]]:
        """Process long audio with LAM-style overlap and expression post-processing."""
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        if len(audio) == 0:
            return np.zeros((0, 52), dtype=np.float32), []

        if overlap_ms <= 0:
            return self._infer_streaming_naive(
                audio,
                chunk_samples=chunk_samples,
                verbose=verbose,
                time_inference=time_inference,
            )

        n_chunks = math.ceil(len(audio) / chunk_samples)
        parts: list[np.ndarray] = []
        timings: list[float] = []
        context = StreamingContext()

        for i in range(n_chunks):
            start = i * chunk_samples
            chunk = audio[start : start + chunk_samples]
            if time_inference:
                t0 = time.perf_counter()
            out, context = self.infer_streaming_chunk(
                chunk,
                context,
                overlap_ms=overlap_ms,
                max_context_frames=max_context_frames,
            )
            if time_inference:
                elapsed = time.perf_counter() - t0
                timings.append(elapsed)
                if verbose:
                    print(
                        f"chunk {i + 1}/{n_chunks}  "
                        f"samples={len(chunk)}  frames={out.shape[0]}  "
                        f"overlap={overlap_ms:.0f}ms  "
                        f"time={elapsed * 1000:.1f}ms"
                    )
            elif verbose:
                print(
                    f"chunk {i + 1}/{n_chunks}  "
                    f"samples={len(chunk)}  frames={out.shape[0]}  "
                    f"overlap={overlap_ms:.0f}ms"
                )
            parts.append(out)

        return np.concatenate(parts, axis=0), timings

    def _infer_streaming_naive(
        self,
        audio: np.ndarray,
        *,
        chunk_samples: int,
        verbose: bool,
        time_inference: bool,
    ) -> tuple[np.ndarray, list[float]]:
        """Legacy streaming without overlap or post-processing."""
        n_chunks = math.ceil(len(audio) / chunk_samples)
        parts: list[np.ndarray] = []
        timings: list[float] = []

        for i in range(n_chunks):
            start = i * chunk_samples
            chunk = audio[start : start + chunk_samples]
            if time_inference:
                t0 = time.perf_counter()
            out = self.infer_chunk(chunk)
            if time_inference:
                elapsed = time.perf_counter() - t0
                timings.append(elapsed)
                if verbose:
                    print(
                        f"chunk {i + 1}/{n_chunks}  "
                        f"samples={len(chunk)}  frames={out.shape[0]}  "
                        f"time={elapsed * 1000:.1f}ms"
                    )
            elif verbose:
                print(
                    f"chunk {i + 1}/{n_chunks}  "
                    f"samples={len(chunk)}  frames={out.shape[0]}"
                )
            parts.append(out)

        return np.concatenate(parts, axis=0), timings

    def infer_full(self, audio: np.ndarray) -> np.ndarray:
        """Single forward pass on entire waveform. Returns [frames, 52]."""
        return self.infer_chunk(audio)
