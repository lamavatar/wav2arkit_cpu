package com.example.splatbench

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri

/**
 * Owns the shared [AudioRingBuffer] and swaps the active [AudioProducer]
 * (file vs mic) without changing the downstream inference/render path. Also
 * provides the mic capture clock; the file clock is driven by MediaPlayer in
 * the activity.
 */
class AudioInputController(private val context: Context) {

    val ring = AudioRingBuffer()

    @Volatile var activeSource: ActiveAudioSource = ActiveAudioSource.NONE
        private set

    private var producer: AudioProducer? = null
    private var micStartNanos: Long = 0L

    /** Start streaming the file into the ring. Returns clip duration in ms. */
    fun startFile(uri: Uri): Int {
        stop()
        ring.reset()
        val p = FileAudioProducer(context, uri)
        producer = p
        p.start(ring)
        activeSource = ActiveAudioSource.FILE
        return durationMs(uri)
    }

    /** Start mic capture into the ring (requires ENABLE_MIC_INPUT + permission). */
    fun startMic() {
        check(AppConfig.ENABLE_MIC_INPUT) { "mic input disabled" }
        stop()
        ring.reset()
        val p = MicAudioProducer()
        producer = p
        p.start(ring)
        micStartNanos = System.nanoTime()
        activeSource = ActiveAudioSource.MIC
    }

    fun stop() {
        producer?.stop()
        producer = null
        activeSource = ActiveAudioSource.NONE
    }

    fun micPositionMs(): Int =
        ((System.nanoTime() - micStartNanos) / 1_000_000L).toInt().coerceAtLeast(0)

    /** Bytes currently buffered, expressed as seconds (for the ring stat line). */
    fun ringSeconds(): Float = ring.availableBytes().toFloat() / AppConfig.AUDIO_SR

    private fun durationMs(uri: Uri): Int {
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(context, uri)
            mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toIntOrNull() ?: 0
        } catch (_: Exception) {
            0
        } finally {
            try { mmr.release() } catch (_: Exception) {}
        }
    }
}
