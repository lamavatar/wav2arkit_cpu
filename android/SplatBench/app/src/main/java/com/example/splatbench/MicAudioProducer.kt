package com.example.splatbench

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Captures the microphone at 16kHz mono and writes 8-bit-unsigned PCM into the
 * shared ring buffer — the same sink the file producer uses. Only instantiated
 * when [AppConfig.ENABLE_MIC_INPUT] is true.
 */
class MicAudioProducer : AudioProducer {

    private val running = AtomicBoolean(false)
    private var thread: Thread? = null
    private var record: AudioRecord? = null

    override fun isRunning(): Boolean = running.get()

    @SuppressLint("MissingPermission")
    override fun start(ring: AudioRingBuffer) {
        check(AppConfig.ENABLE_MIC_INPUT) { "mic input disabled" }
        if (!running.compareAndSet(false, true)) return

        val sr = AppConfig.AUDIO_SR
        val minBuf = AudioRecord.getMinBufferSize(
            sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(sr) // at least ~0.5s of headroom
        val bufBytes = minBuf * 2

        val rec = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sr,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufBytes,
        )
        record = rec
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            running.set(false)
            rec.release()
            record = null
            throw IllegalStateException("AudioRecord init failed")
        }

        val t = Thread({
            val i16 = ShortArray(sr / 4) // ~250ms read blocks
            val u8 = ByteArray(i16.size)
            try {
                rec.startRecording()
                while (running.get()) {
                    val n = rec.read(i16, 0, i16.size)
                    if (n <= 0) continue
                    for (i in 0 until n) u8[i] = AudioPcmConverter.i16ToU8(i16[i])
                    ring.write(u8, n)
                }
            } catch (e: Exception) {
                Log.w(TAG, "capture failed: ${e.message}")
            } finally {
                try { rec.stop() } catch (_: Exception) {}
                rec.release()
                record = null
                running.set(false)
            }
        }, "audio-capture-mic")
        thread = t
        t.start()
    }

    override fun stop() {
        running.set(false)
        thread?.interrupt()
        thread = null
    }

    companion object {
        private const val TAG = "MicAudioProducer"
    }
}
