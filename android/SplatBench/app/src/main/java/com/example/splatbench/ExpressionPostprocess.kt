package com.example.splatbench

/**
 * First-pass Kotlin port of the LAM expression post-processing
 * ([postprocess.py] `apply_expression_postprocessing`). Implements the parts
 * that matter most for streaming continuity and left/right consistency:
 *
 *  - cross-chunk frame blending ([applyFrameBlending])
 *  - light temporal smoothing (centered moving average, stand-in for the
 *    Savitzky-Golay filter — see Phase 3 for C++ parity)
 *  - left/right symmetrization (average mode)
 *
 * Eye-blink injection and silence-based mouth damping are intentionally left
 * for a later pass; they need RNG / volume parity that is out of scope here.
 *
 * Frames are mutable `FloatArray(52)` rows. Methods operate in place where
 * possible and return the same list for chaining.
 */
object ExpressionPostprocess {

    private val LEFT_RIGHT_PAIRS: Array<Pair<Int, Int>> = run {
        val pairs = listOf(
            "jawLeft" to "jawRight",
            "mouthLeft" to "mouthRight",
            "mouthSmileLeft" to "mouthSmileRight",
            "mouthFrownLeft" to "mouthFrownRight",
            "mouthDimpleLeft" to "mouthDimpleRight",
            "mouthStretchLeft" to "mouthStretchRight",
            "mouthPressLeft" to "mouthPressRight",
            "mouthLowerDownLeft" to "mouthLowerDownRight",
            "mouthUpperUpLeft" to "mouthUpperUpRight",
            "cheekSquintLeft" to "cheekSquintRight",
            "noseSneerLeft" to "noseSneerRight",
            "browDownLeft" to "browDownRight",
            "browOuterUpLeft" to "browOuterUpRight",
            "eyeBlinkLeft" to "eyeBlinkRight",
            "eyeLookDownLeft" to "eyeLookDownRight",
            "eyeLookInLeft" to "eyeLookInRight",
            "eyeLookOutLeft" to "eyeLookOutRight",
            "eyeLookUpLeft" to "eyeLookUpRight",
            "eyeSquintLeft" to "eyeSquintRight",
            "eyeWideLeft" to "eyeWideRight",
        )
        pairs.mapNotNull { (l, r) ->
            val li = ArkitBlendshapes.indexOf(l)
            val ri = ArkitBlendshapes.indexOf(r)
            if (li >= 0 && ri >= 0) li to ri else null
        }.toTypedArray()
    }

    /**
     * Run the post-processing pipeline over [frames] (each row length 52).
     * [processedFrames] is the number of leading rows that are context from a
     * previous chunk (already finalized) — blending references that boundary.
     */
    fun process(frames: Array<FloatArray>, processedFrames: Int) {
        if (frames.isEmpty()) return
        applyFrameBlending(frames, processedFrames)
        smoothMovingAverage(frames, window = 5)
        symmetrize(frames)
    }

    private fun applyFrameBlending(
        frames: Array<FloatArray>,
        processedFrames: Int,
        initialWindow: Int = 3,
        subsequentWindow: Int = 5,
    ) {
        if (processedFrames > 0) {
            blendSegment(frames, processedFrames, subsequentWindow, frames[processedFrames - 1])
        } else {
            blendSegment(frames, 0, initialWindow, FloatArray(52))
        }
    }

    private fun blendSegment(
        frames: Array<FloatArray>,
        start: Int,
        window: Int,
        reference: FloatArray,
    ) {
        val len = minOf(window, frames.size - start)
        for (offset in 0 until len) {
            val idx = start + offset
            val w = (offset + 1f) / (len + 1f)
            val row = frames[idx]
            for (c in 0 until 52) {
                row[c] = reference[c] * (1f - w) + row[c] * w
            }
        }
    }

    /** Centered moving average with mirror edges, clamped to [0,1]. */
    private fun smoothMovingAverage(frames: Array<FloatArray>, window: Int) {
        val n = frames.size
        if (n < window) return
        val half = window / 2
        val out = Array(n) { FloatArray(52) }
        for (i in 0 until n) {
            for (c in 0 until 52) {
                var sum = 0f
                for (k in -half..half) {
                    var j = i + k
                    // mirror padding
                    if (j < 0) j = -j
                    if (j >= n) j = 2 * n - 2 - j
                    j = j.coerceIn(0, n - 1)
                    sum += frames[j][c]
                }
                out[i][c] = (sum / window).coerceIn(0f, 1f)
            }
        }
        for (i in 0 until n) System.arraycopy(out[i], 0, frames[i], 0, 52)
    }

    private fun symmetrize(frames: Array<FloatArray>) {
        for (row in frames) {
            for ((li, ri) in LEFT_RIGHT_PAIRS) {
                val avg = (row[li] + row[ri]) * 0.5f
                row[li] = avg
                row[ri] = avg
            }
        }
    }
}
