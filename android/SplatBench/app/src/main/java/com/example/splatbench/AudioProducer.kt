package com.example.splatbench

/**
 * Produces 16kHz mono 8-bit-unsigned PCM into a shared [AudioRingBuffer] on its
 * own thread. File and mic producers implement this identically so the
 * downstream inference/render path never branches on the audio source.
 */
interface AudioProducer {
    /** Start writing u8 PCM into [ring] on a background thread. */
    fun start(ring: AudioRingBuffer)
    fun stop()
    fun isRunning(): Boolean
}

enum class ActiveAudioSource { NONE, FILE, MIC }
