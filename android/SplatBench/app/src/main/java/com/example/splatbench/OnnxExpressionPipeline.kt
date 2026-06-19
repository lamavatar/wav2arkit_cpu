package com.example.splatbench

import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Polls the shared [AudioBuffer] every [AppConfig.POLL_MS]. When at least
 * [AppConfig.chunkBytes] are available, runs ONNX inference + postprocess and
 * appends blendshapes to [ExpressionBuffer].
 */
class OnnxExpressionPipeline(
    private val onnx: Wav2ArkitOnnx,
    private val buffer: AudioBuffer,
    val expressionBuffer: ExpressionBuffer,
    private val session: PlaybackSession,
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
        while (running.get() && !session.isStopRequested()) {
            val tickStart = System.nanoTime()
            val chunk = buffer.readChunk(chunkBytes)
            if (chunk != null) {
                process(chunk)
            } else if (buffer.isEndOfStream()) {
                finishTail(chunkBytes)
                break
            }
            sleepUntil(tickStart + AppConfig.POLL_MS * 1_000_000L)
        }
    }

    private fun finishTail(chunkBytes: Int) {
        val rem = buffer.readRemaining()
        if (rem == null || rem.isEmpty()) return
        val chunk = if (rem.size < chunkBytes) {
            ByteArray(chunkBytes).also { out ->
                System.arraycopy(rem, 0, out, 0, rem.size)
                val pad = AppConfig.AUDIO_SILENCE_U8.toByte()
                for (i in rem.size until chunkBytes) out[i] = pad
            }
        } else {
            rem
        }
        process(chunk)
    }

    private fun process(chunkU8: ByteArray) {
        val frames = onnx.inferStreamingChunk(chunkU8, context)
        expressionBuffer.append(frames)
    }

    fun stop() {
        running.set(false)
        session.requestStop()
        thread?.interrupt()
        thread = null
    }

    fun resetContext() {
        context.reset()
    }

    fun isRunning(): Boolean = running.get()

    private fun sleepUntil(deadlineNs: Long) {
        val rem = deadlineNs - System.nanoTime()
        if (rem > 0) {
            try {
                Thread.sleep(rem / 1_000_000L, (rem % 1_000_000L).toInt())
            } catch (_: InterruptedException) {
            }
        }
    }

    companion object {
        private const val TAG = "OnnxExpressionPipeline"
    }
}

/**
 * [ExpressionSource] backed by streaming ONNX inference.
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
