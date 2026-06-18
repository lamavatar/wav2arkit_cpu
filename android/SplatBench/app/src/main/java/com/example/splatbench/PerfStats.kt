package com.example.splatbench

/**
 * Lightweight performance counters shared across threads.
 *
 * [buildAvgMs], [geomAvgMs] and [drawAvgMs] are averaged over a rolling ~1 second
 * window (the average is refreshed once per window and held until the next one).
 * [inferMs] / [postMs] hold the most recent per-chunk values (chunks are ~1s).
 */
class PerfStats {

    /** A single metric averaged over a ~1 second window. */
    private class Window {
        private var sumMs = 0.0
        private var count = 0
        private var startNs = System.nanoTime()

        @Volatile var avgMs = 0.0
            private set

        @Synchronized fun add(ms: Double) {
            sumMs += ms
            count++
            val now = System.nanoTime()
            if (now - startNs >= WINDOW_NS) {
                avgMs = if (count > 0) sumMs / count else 0.0
                sumMs = 0.0
                count = 0
                startNs = now
            }
        }

        @Synchronized fun reset() {
            sumMs = 0.0
            count = 0
            startNs = System.nanoTime()
            avgMs = 0.0
        }
    }

    private val build = Window()
    private val geom = Window()
    private val draw = Window()

    fun addBuild(ms: Double) = build.add(ms)
    fun addGeom(ms: Double) = geom.add(ms)
    fun addDraw(ms: Double) = draw.add(ms)

    val buildAvgMs: Double get() = build.avgMs
    val geomAvgMs: Double get() = geom.avgMs
    val drawAvgMs: Double get() = draw.avgMs

    /** Most recent ONNX forward (ORT run) duration, ms. */
    @Volatile var inferMs: Double = 0.0
    /** Most recent post-processing duration, ms. */
    @Volatile var postMs: Double = 0.0

    fun reset() {
        build.reset()
        geom.reset()
        draw.reset()
        inferMs = 0.0
        postMs = 0.0
    }

    private companion object {
        const val WINDOW_NS = 1_000_000_000L
    }
}
