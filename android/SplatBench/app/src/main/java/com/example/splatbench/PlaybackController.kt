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

    var state: PlaybackState
        get() = stateRef.get()
        set(value) {
            stateRef.set(value)
        }

    fun frameCountForSeconds(seconds: Float, fps: Float): Int =
        (seconds * fps).toInt().coerceAtLeast(1)

    fun currentFrameIndex(pack: AvatarPack): Int {
        val f = (audioPositionMs / 1000.0 * pack.fps).toInt()
        return f.coerceIn(0, pack.numFrames - 1)
    }

    fun updateAudioPositionMs(posMs: Int) {
        audioPositionMs = posMs.coerceAtLeast(0)
    }

    fun resetAudioPosition() {
        audioPositionMs = 0
    }
}
