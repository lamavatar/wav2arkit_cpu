package com.example.splatbench

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlin.math.ceil

/**
 * Owns the shared [AudioBuffer] and swaps the active [AudioProducer]
 * (file vs mic) without changing the downstream inference/render path.
 */
class AudioInputController(private val context: Context) {

    var buffer = AudioBuffer()
        private set

    @Volatile var activeSource: ActiveAudioSource = ActiveAudioSource.NONE
        private set

    private var producer: AudioProducer? = null

    /** Start streaming the file into the buffer. Returns clip duration in ms. */
    fun startFile(uri: Uri, session: PlaybackSession): Int {
        stop()
        val durMs = durationMs(uri)
        val fileBytes = ceil(durMs / 1000.0 * AppConfig.AUDIO_SR).toInt() + AppConfig.chunkBytes * 4
        val cap = maxOf(AppConfig.audioBufferCapacityBytes, fileBytes)
            .coerceAtMost(AppConfig.audioBufferMaxCapacityBytes)
        buffer = if (buffer.capacityBytes == cap) {
            buffer.also { it.reset() }
        } else {
            AudioBuffer(cap)
        }
        val p = FileAudioProducer(context, uri)
        producer = p
        p.start(buffer, session)
        activeSource = ActiveAudioSource.FILE
        return durMs
    }

    /** Start mic capture into the buffer (requires ENABLE_MIC_INPUT + permission). */
    fun startMic(session: PlaybackSession) {
        check(AppConfig.ENABLE_MIC_INPUT) { "mic input disabled" }
        stop()
        buffer = if (buffer.capacityBytes == AppConfig.audioBufferCapacityBytes) {
            buffer.also { it.reset() }
        } else {
            AudioBuffer(AppConfig.audioBufferCapacityBytes)
        }
        val p = MicAudioProducer()
        producer = p
        p.start(buffer, session)
        activeSource = ActiveAudioSource.MIC
    }

    fun stop() {
        producer?.stop()
        producer = null
        activeSource = ActiveAudioSource.NONE
    }

    fun bufferSeconds(): Float = buffer.writtenSeconds()

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
