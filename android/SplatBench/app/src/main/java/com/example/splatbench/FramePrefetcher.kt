package com.example.splatbench

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Builds projected splat instances off the GL thread into a [FrameCache].
 *
 * On session start: waits for expression frames, builds
 * [AppConfig.GEOMETRY_START_FRAMES] warmup frames, then signals ready for
 * playback. During PLAYING keeps a lead buffer ahead of the audio clock.
 */
class FramePrefetcher(
    private val pack: AvatarPack,
    private val cache: FrameCache,
    private val controller: PlaybackController,
    private val expressionBuffer: ExpressionBuffer,
    private val session: PlaybackSession,
    private val perfStats: PerfStats? = null,
) {
    private val threadCount: Int
        get() = AppConfig.BUILD_THREADS.coerceIn(1, AppConfig.MAX_BUILD_THREADS)

    interface Listener {
        fun onWarmupProgress(built: Int, total: Int) {}
        fun onWarmupComplete() {}
        fun onError(message: String)
    }

    private val exec: ExecutorService = Executors.newFixedThreadPool(AppConfig.MAX_BUILD_THREADS)
    val builder: InstanceBuilder = InstanceBuilder(pack)
    private val cancelled = AtomicBoolean(false)
    private val buildRunning = AtomicBoolean(false)
    private val nextBuildIndex = AtomicInteger(0)
    private var buildThread: Thread? = null

    private val previewExec: ExecutorService = Executors.newSingleThreadExecutor()
    private val previewGen = AtomicInteger(0)

    fun cancel() {
        cancelled.set(true)
        buildRunning.set(false)
        buildThread?.interrupt()
        buildThread = null
        previewGen.incrementAndGet()
    }

    fun resetCancel() {
        cancelled.set(false)
    }

    fun syncViewFrom(rot: FloatArray, tv: FloatArray, fy: Float, cx: Float, cy: Float, vw: Float, vh: Float) {
        builder.setView(rot, tv, fy, cx, cy, vw, vh)
    }

    fun buildNeutralPreview(mouthOnly: Boolean, listener: Listener?) {
        val gen = previewGen.incrementAndGet()
        previewExec.execute {
            try {
                if (gen != previewGen.get()) return@execute
                val idx = renderIndices(mouthOnly)
                val weights = FloatArray(pack.numMorphs)
                val t0 = System.nanoTime()
                val cached = builder.buildCached(0, weights, idx, exec, threadCount)
                val buildMs = (System.nanoTime() - t0) / 1_000_000.0
                if (gen == previewGen.get()) {
                    perfStats?.addBuild(buildMs)
                    cache.put(cached.copy(buildMs = buildMs))
                    listener?.onWarmupComplete()
                }
            } catch (e: Exception) {
                if (gen == previewGen.get()) {
                    listener?.onError(e.message ?: "neutral preview build failed")
                }
            }
        }
    }

    /**
     * Warm up [GEOMETRY_START_FRAMES] geometry frames after expression data is
     * available, then continue building ahead during PLAYING.
     */
    fun startSession(mouthOnly: Boolean, listener: Listener) {
        if (!buildRunning.compareAndSet(false, true)) return
        val t = Thread({
            try {
                resetCancel()
                cache.clear()
                nextBuildIndex.set(0)
                val indices = renderIndices(mouthOnly)
                val warmup = AppConfig.GEOMETRY_START_FRAMES

                for (i in 0 until warmup) {
                    if (cancelled.get() || session.isStopRequested()) return@Thread
                    if (!awaitExpression(i)) return@Thread
                    buildFrame(i, mouthOnly, indices)
                    nextBuildIndex.set(i + 1)
                    listener.onWarmupProgress(i + 1, warmup)
                    pauseBetweenBuilds()
                }

                if (cancelled.get() || session.isStopRequested()) return@Thread
                listener.onWarmupComplete()

                val src = source()
                while (buildRunning.get() && !cancelled.get() && !session.isStopRequested()) {
                    val state = controller.state
                    if (state != PlaybackState.PLAYING) {
                        Thread.sleep(20)
                        continue
                    }

                    val frameCount = src.frameCount()
                    val playbackFrame = controller.currentFrameIndex(pack)
                    val lead = controller.frameCountForSeconds(controller.prebufferSeconds, src.fps)
                    var target = playbackFrame + lead
                    if (frameCount > 0) target = minOf(target, frameCount - 1)
                    var idx = nextBuildIndex.get()

                    var builtAny = false
                    while (idx <= target && buildRunning.get() && !cancelled.get() && !session.isStopRequested()) {
                        if (!src.hasFrame(idx)) break
                        if (!cache.has(idx)) {
                            buildFrame(idx, mouthOnly, indices)
                            builtAny = true
                        }
                        idx++
                        nextBuildIndex.set(idx)
                    }

                    trimPlaybackCache(playbackFrame, lead)

                    val dur = clipDurationMs(src)
                    if (dur > 0 && frameCount > 0 &&
                        playbackFrame >= frameCount - 1 &&
                        controller.audioPositionMs >= dur - 50
                    ) {
                        break
                    }
                    if (!builtAny) Thread.sleep(10) else pauseBetweenBuilds()
                }
            } catch (e: Exception) {
                listener.onError(e.message ?: "geometry build failed")
            } finally {
                buildRunning.set(false)
            }
        }, "geometry-build")
        buildThread = t
        t.start()
    }

    fun stopBuild() {
        buildRunning.set(false)
        buildThread?.interrupt()
        buildThread = null
    }

    fun shutdown() {
        cancel()
        exec.shutdownNow()
        previewExec.shutdownNow()
    }

    fun builtFrameCount(): Int = nextBuildIndex.get()

    private fun source(): ExpressionSource =
        controller.expressionSource ?: BakedExpressionSource(pack)

    private fun awaitExpression(frame: Int): Boolean {
        while (!expressionBuffer.hasFrame(frame)) {
            if (cancelled.get() || session.isStopRequested()) return false
            try {
                Thread.sleep(AppConfig.POLL_MS)
            } catch (_: InterruptedException) {
                return false
            }
        }
        return true
    }

    private fun pauseBetweenBuilds() {
        try {
            Thread.sleep(AppConfig.GEOMETRY_BUILD_PAUSE_MS)
        } catch (_: InterruptedException) {
        }
    }

    private fun clipDurationMs(src: ExpressionSource): Int {
        val fc = src.frameCount()
        if (fc <= 0 || src.fps <= 0f) return 0
        return ((fc / src.fps) * 1000.0).toInt()
    }

    private fun renderIndices(mouthOnly: Boolean): IntArray {
        if (AppConfig.useMouthOnlyIndices(pack)) return pack.dynamicIndices
        if (AppConfig.HEAD_BONE_ENABLED && pack.hasHeadAnimation) return pack.allIndices
        return if (mouthOnly) pack.dynamicIndices else pack.allIndices
    }

    private fun buildFrame(frameIndex: Int, mouthOnly: Boolean, indices: IntArray? = null) {
        if (cache.size() >= AppConfig.FRAME_CACHE_MAX_FRAMES - 1) {
            val src = source()
            val lead = controller.frameCountForSeconds(controller.prebufferSeconds, src.fps)
            trimPlaybackCache(controller.currentFrameIndex(pack), lead)
        }
        val idx = indices ?: renderIndices(mouthOnly)
        val weights = source().weightsForFrame(frameIndex)
        val t0 = System.nanoTime()
        val cached = builder.buildCached(frameIndex, weights, idx, exec, threadCount)
        val buildMs = (System.nanoTime() - t0) / 1_000_000.0
        perfStats?.addBuild(buildMs)
        cache.put(cached.copy(buildMs = buildMs))
    }

    /** Keep only a sliding window around the playhead (full mode frames are ~800 KB). */
    private fun trimPlaybackCache(playbackFrame: Int, leadFrames: Int) {
        val minKeep = (playbackFrame - AppConfig.FRAME_CACHE_BEHIND).coerceAtLeast(0)
        val maxKeep = playbackFrame + leadFrames + AppConfig.FRAME_CACHE_AHEAD_MARGIN
        cache.evictOutside(minKeep, maxKeep, keepFrame = playbackFrame)
        cache.trimToMaxFrames(AppConfig.FRAME_CACHE_MAX_FRAMES)
    }
}
