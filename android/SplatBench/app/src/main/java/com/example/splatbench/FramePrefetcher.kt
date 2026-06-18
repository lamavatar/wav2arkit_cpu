package com.example.splatbench

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Builds projected splat instances off the GL thread into a [FrameCache].
 *
 * On start: pre-builds [PlaybackController.prebufferSeconds] of frames, then signals
 * [Listener.onPrebufferComplete]. During playback a background loop keeps a lead
 * buffer of the same duration ahead of the audio clock.
 */
class FramePrefetcher(
    private val pack: AvatarPack,
    private val cache: FrameCache,
    private val controller: PlaybackController,
    buildThreads: Int,
) {
    private val threadCount = buildThreads.coerceIn(1, 64)

    interface Listener {
        fun onPrebufferProgress(built: Int, total: Int)
        fun onPrebufferComplete()
        fun onError(message: String)
    }

    private val exec: ExecutorService = Executors.newFixedThreadPool(threadCount)
    val builder: InstanceBuilder = InstanceBuilder(pack)
    private val cancelled = AtomicBoolean(false)
    private val leadRunning = AtomicBoolean(false)
    private val nextBuildIndex = AtomicInteger(0)
    private var leadThread: Thread? = null

    fun cancel() {
        cancelled.set(true)
        leadRunning.set(false)
        leadThread?.interrupt()
        leadThread = null
    }

    fun resetCancel() {
        cancelled.set(false)
    }

    /** Copy the renderer's current view into our builder (call from GL thread before prebuffer). */
    fun syncViewFrom(rot: FloatArray, tv: FloatArray, fy: Float, cx: Float, cy: Float, vw: Float, vh: Float) {
        builder.setView(rot, tv, fy, cx, cy, vw, vh)
    }

    fun buildPreviewFrame(frameIndex: Int, mouthOnly: Boolean, listener: Listener?) {
        Thread({
            try {
                if (cancelled.get()) return@Thread
                buildFrame(frameIndex, mouthOnly)
            } catch (e: Exception) {
                listener?.onError(e.message ?: "preview build failed")
            }
        }, "prefetch-preview").start()
    }

    fun startPrebuffer(mouthOnly: Boolean, listener: Listener) {
        Thread({
            try {
                resetCancel()
                cache.clear()
                nextBuildIndex.set(0)

                val total = controller.frameCountForSeconds(controller.prebufferSeconds, pack.fps)
                    .coerceAtMost(pack.numFrames)
                val indices = renderIndices(mouthOnly)

                for (i in 0 until total) {
                    if (cancelled.get()) return@Thread
                    buildFrame(i, mouthOnly, indices)
                    nextBuildIndex.set(i + 1)
                    listener.onPrebufferProgress(i + 1, total)
                }

                if (cancelled.get()) return@Thread
                listener.onPrebufferComplete()
            } catch (e: Exception) {
                listener.onError(e.message ?: "prebuffer failed")
            }
        }, "prefetch-prebuffer").start()
    }

    fun startLeadPrefetch(mouthOnly: Boolean) {
        if (!leadRunning.compareAndSet(false, true)) return
        val t = Thread({
            try {
                val indices = renderIndices(mouthOnly)
                while (leadRunning.get() && !cancelled.get()) {
                    val state = controller.state
                    if (state != PlaybackState.PLAYING) {
                        Thread.sleep(20)
                        continue
                    }

                    val playbackFrame = controller.currentFrameIndex(pack)
                    val lead = controller.frameCountForSeconds(controller.prebufferSeconds, pack.fps)
                    val target = minOf(playbackFrame + lead, pack.numFrames - 1)
                    var idx = nextBuildIndex.get()

                    while (idx <= target && leadRunning.get() && !cancelled.get()) {
                        if (!cache.has(idx)) {
                            buildFrame(idx, mouthOnly, indices)
                        }
                        idx++
                        nextBuildIndex.set(idx)
                    }

                    if (playbackFrame >= pack.numFrames - 1 &&
                        controller.audioPositionMs >= clipDurationMs() - 50
                    ) {
                        break
                    }
                    Thread.sleep(10)
                }
            } catch (_: InterruptedException) {
                // shutdown
            } finally {
                leadRunning.set(false)
            }
        }, "prefetch-lead")
        leadThread = t
        t.start()
    }

    fun stopLeadPrefetch() {
        leadRunning.set(false)
        leadThread?.interrupt()
        leadThread = null
    }

    fun shutdown() {
        cancel()
        exec.shutdownNow()
    }

    private fun clipDurationMs(): Int =
        ((pack.numFrames / pack.fps) * 1000.0).toInt()

    private fun renderIndices(mouthOnly: Boolean): IntArray =
        if (mouthOnly) pack.dynamicIndices else pack.allIndices

    private fun buildFrame(frameIndex: Int, mouthOnly: Boolean, indices: IntArray? = null) {
        val idx = indices ?: renderIndices(mouthOnly)
        val weights = pack.frameWeights(frameIndex)
        val t0 = System.nanoTime()
        val cached = builder.buildCached(frameIndex, weights, idx, exec, threadCount)
        val buildMs = (System.nanoTime() - t0) / 1_000_000.0
        cache.put(cached.copy(buildMs = buildMs))
    }
}
