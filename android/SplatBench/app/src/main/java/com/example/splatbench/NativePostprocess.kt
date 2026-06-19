package com.example.splatbench

/**
 * JNI bridge to the C++ expression post-processing library (`libexpression_postprocess.so`),
 * an exact port of [postprocess.py] + [session.py] `_postprocess_chunk`. The native side
 * owns the streaming context (previous expression/volume tails, capped to 36 frames).
 */
object NativePostprocess {
    @Volatile var available: Boolean = false
        private set

    init {
        available = try {
            System.loadLibrary("expression_postprocess")
            true
        } catch (t: Throwable) {
            false
        }
    }

    /** Allocates a native StreamingCtx; returns its handle (0 on failure). */
    external fun nativeCreate(): Long

    /** Clears the streaming context (start of a new utterance/session). */
    external fun nativeReset(handle: Long)

    /** Frees the native StreamingCtx. */
    external fun nativeDestroy(handle: Long)

    /**
     * Post-processes one chunk of freshly-sliced model frames.
     *
     * @param handle native context handle
     * @param framesFlat k*52 row-major model output for this chunk
     * @param k number of new frames in this chunk
     * @param audio float waveform [-1,1] of this chunk's samples (for volume)
     * @return k*52 row-major post-processed frames
     */
    external fun nativeProcessChunk(
        handle: Long,
        framesFlat: FloatArray,
        k: Int,
        audio: FloatArray,
    ): FloatArray
}
