package com.example.splatbench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.ceil

/** Streaming state mirroring session.py StreamingContext (u8 audio domain). */
class StreamingContext {
    var isInitial: Boolean = true
    /** Last overlap-input audio in u8 (for next chunk's overlap tail). */
    var previousAudioU8: ByteArray? = null
    /** Finalized context frames (each length 52), capped to MAX_CONTEXT_FRAMES. */
    var previousExpression: Array<FloatArray>? = null

    fun reset() {
        isInitial = true
        previousAudioU8 = null
        previousExpression = null
    }
}

/**
 * ONNX Runtime wrapper for myned-ai/wav2arkit_cpu. Accepts 8-bit unsigned PCM
 * (0..255) and converts to float32 [-1,1] only at the ONNX boundary, mirroring
 * [session.py] `infer_streaming_chunk` (overlap input + LAM post-processing).
 */
class Wav2ArkitOnnx private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    val activeEp: String,
) {
    private val inputName: String = session.inputNames.iterator().next()

    /** Most recent ORT forward (session.run) duration, ms. */
    @Volatile var lastInferMs: Double = 0.0
        private set

    /** Most recent LAM post-processing duration, ms. */
    @Volatile var lastPostprocessMs: Double = 0.0
        private set

    /** Single forward pass on a float waveform. Returns [frames][52]. */
    private fun inferChunkFloat(audio: FloatArray): Array<FloatArray> {
        val shape = longArrayOf(1, audio.size.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(audio), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val out = result[0].value as Array<Array<FloatArray>> // [1][F][52]
                return out[0]
            }
        }
    }

    /**
     * Process one u8 chunk with LAM boundary handling. Returns the new
     * expression frames (each length 52) produced for this chunk.
     */
    fun inferStreamingChunk(
        pcmU8: ByteArray,
        ctx: StreamingContext,
        overlapBytes: Int = AppConfig.overlapBytes,
        maxContextFrames: Int = MAX_CONTEXT_FRAMES,
    ): Array<FloatArray> {
        // 1) overlap input in u8 (initial = silence-128 pad; else previous tail)
        val inputU8 = buildOverlapInput(pcmU8, ctx, overlapBytes)
        ctx.previousAudioU8 = inputU8.copyOf()

        // 2) u8 → float32 [-1,1]
        val audioF = AudioPcmConverter.u8ToOnnxFloat(inputU8)

        // 3) ORT forward (pure inference time)
        val tInfer = System.nanoTime()
        val outExp = inferChunkFloat(audioF)
        lastInferMs = (System.nanoTime() - tInfer) / 1_000_000.0

        // 4) slice new frames for this chunk's samples
        val newSlice = sliceNewFrames(outExp, pcmU8.size)

        // 5) post-process with cross-chunk context (post-processing time)
        val tPost = System.nanoTime()
        val result = postprocessChunk(newSlice, ctx, maxContextFrames)
        lastPostprocessMs = (System.nanoTime() - tPost) / 1_000_000.0

        ctx.isInitial = false
        return result
    }

    private fun buildOverlapInput(
        inAudio: ByteArray,
        ctx: StreamingContext,
        overlapBytes: Int,
    ): ByteArray {
        val prev = ctx.previousAudioU8
        return if (ctx.isInitial || prev == null) {
            ByteArray(overlapBytes + inAudio.size).also { out ->
                // blank pad = silence 128
                for (i in 0 until overlapBytes) out[i] = AppConfig.AUDIO_SILENCE_U8.toByte()
                System.arraycopy(inAudio, 0, out, overlapBytes, inAudio.size)
            }
        } else {
            val n = minOf(overlapBytes, prev.size)
            ByteArray(n + inAudio.size).also { out ->
                System.arraycopy(prev, prev.size - n, out, 0, n)
                System.arraycopy(inAudio, 0, out, n, inAudio.size)
            }
        }
    }

    private fun sliceNewFrames(outExp: Array<FloatArray>, chunkSamples: Int): Array<FloatArray> {
        val chunkFrames = framesForSamples(chunkSamples)
        var start = outExp.size - chunkFrames
        if (start < 0) start = 0
        return outExp.copyOfRange(start, outExp.size)
    }

    private fun postprocessChunk(
        newSlice: Array<FloatArray>,
        ctx: StreamingContext,
        maxContextFrames: Int,
    ): Array<FloatArray> {
        val prevExp = ctx.previousExpression
        val out: Array<FloatArray>
        if (prevExp == null) {
            ExpressionPostprocess.process(newSlice, 0)
            out = newSlice
            ctx.previousExpression = tailContext(newSlice, maxContextFrames)
        } else {
            val prevLen = prevExp.size
            val combined = Array(prevLen + newSlice.size) { i ->
                if (i < prevLen) prevExp[i].copyOf() else newSlice[i - prevLen].copyOf()
            }
            ExpressionPostprocess.process(combined, prevLen)
            out = combined.copyOfRange(prevLen, combined.size)
            ctx.previousExpression = tailContext(combined, maxContextFrames)
        }
        return out
    }

    private fun tailContext(frames: Array<FloatArray>, max: Int): Array<FloatArray> {
        if (frames.size <= max) return Array(frames.size) { frames[it].copyOf() }
        val start = frames.size - max
        return Array(max) { frames[start + it].copyOf() }
    }

    fun close() {
        try { session.close() } catch (_: Throwable) {}
    }

    companion object {
        const val MAX_CONTEXT_FRAMES = 36

        fun framesForSamples(numSamples: Int): Int =
            ceil(AppConfig.EXPRESSION_FPS * numSamples / AppConfig.AUDIO_SR).toInt()

        fun load(): Wav2ArkitOnnx = load(AppConfig.onnxFile())

        fun load(modelFile: File): Wav2ArkitOnnx {
            require(modelFile.isFile) {
                "ONNX model not found: ${modelFile.absolutePath}\n" +
                    "Copy ${AppConfig.ONNX_FILE_NAME} to ${AppConfig.onnxPathHint()}"
            }
            val env = OrtEnvironment.getEnvironment()
            val choice = OrtEpSelector.createSession(env, modelFile.absolutePath)
            return Wav2ArkitOnnx(env, choice.session, choice.ep)
        }
    }
}
