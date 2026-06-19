package com.example.splatbench

import java.util.concurrent.atomic.AtomicBoolean

/** Shared stop/reset signal for audio, inference, and geometry threads. */
class PlaybackSession {
    private val stopRequested = AtomicBoolean(false)

    fun requestStop() {
        stopRequested.set(true)
    }

    fun isStopRequested(): Boolean = stopRequested.get()

    fun reset() {
        stopRequested.set(false)
    }
}
