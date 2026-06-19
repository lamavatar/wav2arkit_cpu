package com.example.splatbench

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Captures the microphone at 16kHz mono. Every [AppConfig.POLL_MS] reads
 * [AppConfig.micTickBytes] (200ms) and appends to the shared [AudioBuffer].
 */
class MicAudioProducer : AudioProducer {

    private val running = AtomicBoolean(false)
    private var thread: Thread? = null
    private var record: AudioRecord? = null
    private var buffer: AudioBuffer? = null
    private var session: PlaybackSession? = null

    override fun isRunning(): Boolean = running.get()

    @SuppressLint("MissingPermission")
    override fun start(buffer: AudioBuffer, session: PlaybackSession) {
        check(AppConfig.ENABLE_MIC_INPUT) { "mic input disabled" }
        if (!running.compareAndSet(false, true)) return
        this.buffer = buffer
        this.session = session

        val sr = AppConfig.AUDIO_SR
        val tickBytes = AppConfig.micTickBytes
        val minBuf = AudioRecord.getMinBufferSize(
            sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(tickBytes * 2)

        val rec = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sr,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf,
        )
        record = rec
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            running.set(false)
            rec.release()
            record = null
            throw IllegalStateException("AudioRecord init failed")
        }

        val t = Thread({
            val i16 = ShortArray(tickBytes)
            val u8 = ByteArray(tickBytes)
            try {
                rec.startRecording()
                while (running.get() && !session.isStopRequested()) {
                    val tickStart = System.nanoTime()
                    var got = 0
                    while (got < tickBytes && running.get() && !session.isStopRequested()) {
                        val n = rec.read(i16, got, tickBytes - got)
                        if (n <= 0) break
                        got += n
                    }
                    if (got > 0) {
                        for (i in 0 until got) u8[i] = AudioPcmConverter.i16ToU8(i16[i])
                        buffer.writeBlocking(u8, got) { running.get() && !session.isStopRequested() }
                    }
                    sleepUntil(tickStart + AppConfig.POLL_MS * 1_000_000L)
                }
            } catch (e: Exception) {
                Log.w(TAG, "capture failed: ${e.message}")
            } finally {
                flushLastTick(buffer, session, rec, i16, u8)
                try { rec.stop() } catch (_: Exception) {}
                rec.release()
                record = null
                running.set(false)
            }
        }, "audio-capture-mic")
        thread = t
        t.start()
    }

    private fun flushLastTick(
        buffer: AudioBuffer,
        session: PlaybackSession?,
        rec: AudioRecord,
        i16: ShortArray,
        u8: ByteArray,
    ) {
        val tickBytes = AppConfig.micTickBytes
        var got = 0
        while (got < tickBytes) {
            val n = rec.read(i16, got, tickBytes - got)
            if (n <= 0) break
            got += n
        }
        if (got > 0) {
            for (i in 0 until got) u8[i] = AudioPcmConverter.i16ToU8(i16[i])
            buffer.writeBlocking(u8, got) { true }
        } else {
            buffer.writeSilence(tickBytes)
        }
        buffer.markEndOfStream()
    }

    override fun stop() {
        running.set(false)
        session?.requestStop()
        thread?.interrupt()
        thread = null
    }

    private fun sleepUntil(deadlineNs: Long) {
        val rem = deadlineNs - System.nanoTime()
        if (rem > 0) {
            try {
                Thread.sleep(rem / 1_000_000L, (rem % 1_000_000L).toInt())
            } catch (_: InterruptedException) {
            }
        }
    }

    companion object {
        private const val TAG = "MicAudioProducer"
    }
}
