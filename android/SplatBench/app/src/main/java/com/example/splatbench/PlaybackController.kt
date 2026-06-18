package com.example.splatbench

import java.util.concurrent.atomic.AtomicReference

enum class PlaybackState {
    IDLE,
    READY,
    PREBUFFERING,
    PLAYING,
    DONE,
}

/** Shared playback state; audio position is the master clock during PLAYING. */
class PlaybackController {
    private val stateRef = AtomicReference(PlaybackState.IDLE)

    @Volatile var audioPositionMs: Int = 0
        private set

    /** Seconds of frames to pre-build before audio starts (and as lead during playback). */
    @Volatile var prebufferSeconds: Float = 1.0f

    /** Active weight source (baked SPL1 or runtime ONNX). Drives fps + frame count. */
    @Volatile var expressionSource: ExpressionSource? = null

    var state: PlaybackState
        get() = stateRef.get()
        set(value) {
            stateRef.set(value)
        }

    fun frameCountForSeconds(seconds: Float, fps: Float): Int =
        (seconds * fps).toInt().coerceAtLeast(1)

    /** Frame index for the current clock position, clamped to the source's frame count. */
    fun currentFrameIndex(pack: AvatarPack): Int {
        val src = expressionSource
        val fps = src?.fps ?: pack.fps
        val f = (audioPositionMs / 1000.0 * fps).toInt()
        val count = src?.frameCount() ?: pack.numFrames
        return if (count <= 0) f.coerceAtLeast(0) else f.coerceIn(0, count - 1)
    }

    fun updateAudioPositionMs(posMs: Int) {
        audioPositionMs = posMs.coerceAtLeast(0)
    }

    fun resetAudioPosition() {
        audioPositionMs = 0
    }
}
