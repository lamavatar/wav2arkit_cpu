package com.example.splatbench

import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Consumes the shared [AudioRingBuffer] (u8) on a dedicated inference thread,
 * runs [Wav2ArkitOnnx] per chunk and appends results to an [ExpressionBuffer].
 * There is no FILE/MIC branching here — the only difference between sources is
 * the producer that fills the ring and the playback clock.
 */
class OnnxExpressionPipeline(
    private val onnx: Wav2ArkitOnnx,
    private val ring: AudioRingBuffer,
    val expressionBuffer: ExpressionBuffer,
) {
    private val running = AtomicBoolean(false)
    private var thread: Thread? = null
    private val context = StreamingContext()

    val activeEp: String get() = onnx.activeEp
    val lastInferMs: Double get() = onnx.lastInferMs
    val lastPostprocessMs: Double get() = onnx.lastPostprocessMs

    fun start() {
        if (!running.compareAndSet(false, true)) return
        context.reset()
        val t = Thread({
            try {
                loop()
            } catch (e: Exception) {
                Log.w(TAG, "inference loop failed: ${e.message}")
            } finally {
                running.set(false)
            }
        }, "onnx-infer")
        thread = t
        t.start()
    }

    private fun loop() {
        val chunkBytes = AppConfig.chunkBytes
        while (running.get()) {
            val chunk = ring.readChunk(chunkBytes)
            if (chunk != null) {
                process(chunk)
                continue
            }
            if (ring.isEndOfStream()) {
                val rem = ring.availableBytes()
                if (rem > 0) {
                    ring.readChunk(rem)?.let { process(it) }
                }
                break
            }
            try { Thread.sleep(4) } catch (_: InterruptedException) { break }
        }
    }

    private fun process(chunkU8: ByteArray) {
        val frames = onnx.inferStreamingChunk(chunkU8, context)
        expressionBuffer.append(frames)
    }

    fun stop() {
        running.set(false)
        thread?.interrupt()
        thread = null
    }

    fun isRunning(): Boolean = running.get()

    companion object {
        private const val TAG = "OnnxExpressionPipeline"
    }
}

/**
 * [ExpressionSource] backed by streaming ONNX inference. Weights are mapped
 * from the 52-entry ARKit vector onto the pack's baked morphs via
 * [AvatarPack.mapArkitWeights].
 *
 * @param frameCountProvider upper bound on selectable frames: a fixed
 *   duration-derived total for FILE, or a growing value for MIC.
 */
class OnnxExpressionSource(
    private val pack: AvatarPack,
    private val expressionBuffer: ExpressionBuffer,
    private val frameCountProvider: () -> Int,
) : ExpressionSource {
    private val neutral = FloatArray(pack.numMorphs)

    override val fps: Float get() = AppConfig.EXPRESSION_FPS

    override fun frameCount(): Int = frameCountProvider()

    override fun weightsForFrame(frame: Int): FloatArray {
        val arkit = expressionBuffer.get(frame) ?: return neutral
        return pack.mapArkitWeights(arkit)
    }

    override fun hasFrame(frame: Int): Boolean = expressionBuffer.hasFrame(frame)
}
