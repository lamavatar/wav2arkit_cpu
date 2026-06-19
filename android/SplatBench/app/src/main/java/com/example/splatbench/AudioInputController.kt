package com.example.splatbench

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri

/**
 * Owns the shared [AudioBuffer] and swaps the active [AudioProducer]
 * (file vs mic) without changing the downstream inference/render path.
 */
class AudioInputController(private val context: Context) {

    val buffer = AudioBuffer()

    @Volatile var activeSource: ActiveAudioSource = ActiveAudioSource.NONE
        private set

    private var producer: AudioProducer? = null

    /** Start streaming the file into the buffer. Returns clip duration in ms. */
    fun startFile(uri: Uri, session: PlaybackSession): Int {
        stop()
        buffer.reset()
        val p = FileAudioProducer(context, uri)
        producer = p
        p.start(buffer, session)
        activeSource = ActiveAudioSource.FILE
        return durationMs(uri)
    }

    /** Start mic capture into the buffer (requires ENABLE_MIC_INPUT + permission). */
    fun startMic(session: PlaybackSession) {
        check(AppConfig.ENABLE_MIC_INPUT) { "mic input disabled" }
        stop()
        buffer.reset()
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

    fun bufferSeconds(): Float = buffer.availableBytes().toFloat() / AppConfig.AUDIO_SR

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
