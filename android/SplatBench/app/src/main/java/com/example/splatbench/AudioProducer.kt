package com.example.splatbench

/**
 * Produces 16kHz mono 8-bit-unsigned PCM into a shared [AudioBuffer] on its
 * own thread. File and mic producers implement this identically so the
 * downstream inference/render path never branches on the audio source.
 */
interface AudioProducer {
    fun start(buffer: AudioBuffer, session: PlaybackSession)
    fun stop()
    fun isRunning(): Boolean
}

enum class ActiveAudioSource { NONE, FILE, MIC }
