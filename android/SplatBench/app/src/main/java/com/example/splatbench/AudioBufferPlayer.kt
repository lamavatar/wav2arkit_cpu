package com.example.splatbench

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Build
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Streams u8 PCM from [AudioBuffer] play cursor to [AudioTrack] (16 kHz mono).
 * Invokes [onComplete] on the calling thread when buffer playback reaches EOS.
 */
class AudioBufferPlayer(
    private val buffer: AudioBuffer,
    private val onComplete: () -> Unit,
) {
    private val running = AtomicBoolean(false)
    private var thread: Thread? = null
    private var track: AudioTrack? = null

    @Volatile private var paused = false

    fun start() {
        if (!running.compareAndSet(false, true)) return
        val t = Thread({ playLoop() }, "audio-buffer-play")
        thread = t
        t.start()
    }

    fun stop() {
        running.set(false)
        thread?.interrupt()
        thread = null
        releaseTrack()
    }

    fun pause() {
        paused = true
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            try {
                track?.pause()
            } catch (_: IllegalStateException) {
            }
        }
    }

    fun resume() {
        paused = false
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            try {
                track?.play()
            } catch (_: IllegalStateException) {
            }
        }
    }

    fun positionMs(): Int = buffer.playPositionMs()

    private fun playLoop() {
        val sr = AppConfig.AUDIO_SR
        val channel = AudioFormat.CHANNEL_OUT_MONO
        val encoding = AudioFormat.ENCODING_PCM_16BIT
        val minBuf = AudioTrack.getMinBufferSize(sr, channel, encoding).coerceAtLeast(4096)
        val at = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sr)
                    .setChannelMask(channel)
                    .setEncoding(encoding)
                    .build(),
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(minBuf)
            .build()
        track = at
        at.play()

        val tickU8 = AppConfig.pollTickBytes.coerceAtLeast(320)
        val i16Scratch = ShortArray(tickU8)

        try {
            while (running.get()) {
                if (paused) {
                    sleepBrief()
                    continue
                }

                val u8 = buffer.readForPlayback(tickU8)
                if (u8 != null) {
                    writeU8AsI16(at, u8, i16Scratch)
                    continue
                }

                if (buffer.isEndOfStream()) {
                    val tail = buffer.readRemainingForPlayback()
                    if (tail != null) {
                        writeU8AsI16(at, tail, ShortArray(tail.size))
                    }
                    if (buffer.isPlaybackComplete()) {
                        break
                    }
                }
                sleepBrief()
            }
        } finally {
            running.set(false)
            releaseTrack()
            if (buffer.isPlaybackComplete()) {
                onComplete()
            }
        }
    }

    private fun writeU8AsI16(at: AudioTrack, u8: ByteArray, scratch: ShortArray) {
        val n = u8.size
        val samples = if (scratch.size >= n) scratch else ShortArray(n)
        for (i in 0 until n) {
            samples[i] = (((u8[i].toInt() and 0xFF) - 128) shl 8).toShort()
        }
        var offset = 0
        while (offset < n) {
            val wrote = at.write(samples, offset, n - offset)
            if (wrote <= 0) break
            offset += wrote
        }
    }

    private fun releaseTrack() {
        track?.let { t ->
            try {
                if (t.playState == AudioTrack.PLAYSTATE_PLAYING) t.stop()
            } catch (_: IllegalStateException) {
            }
            t.release()
        }
        track = null
    }

    private fun sleepBrief() {
        try {
            Thread.sleep(5)
        } catch (_: InterruptedException) {
        }
    }
}
