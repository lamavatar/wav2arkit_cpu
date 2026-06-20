package com.example.splatbench

/**
 * Fixed-capacity linear PCM buffer (no circular overwrite).
 * Producer appends at [writePos]. ONNX reads [inferReadPos]; speaker playback reads
 * [playReadPos]. When full the producer blocks until capacity is increased for the clip.
 */
class AudioBuffer(
    val capacityBytes: Int = AppConfig.audioBufferCapacityBytes,
) {
    private val buf = ByteArray(capacityBytes)
    private var inferReadPos = 0
    private var playReadPos = 0
    private var writePos = 0
    private val lock = Any()

    @Volatile private var endOfStream = false

    @Volatile var totalWritten: Long = 0L
        private set

    /** Append u8 PCM; blocks until [len] bytes are accepted or returns 0 if interrupted. */
    fun writeBlocking(pcmU8: ByteArray, len: Int = pcmU8.size, shouldContinue: () -> Boolean): Int {
        if (len <= 0) return 0
        var offset = 0
        while (offset < len) {
            if (!shouldContinue()) return offset
            val n = write(pcmU8, offset, len - offset)
            if (n <= 0) {
                try {
                    Thread.sleep(2)
                } catch (_: InterruptedException) {
                    return offset
                }
                continue
            }
            offset += n
        }
        return offset
    }

    /** Non-blocking append; returns bytes written (0 when full). */
    fun write(pcmU8: ByteArray, offset: Int = 0, len: Int = pcmU8.size - offset): Int {
        if (len <= 0) return 0
        synchronized(lock) {
            val free = capacityBytes - writePos
            if (free <= 0) return 0
            val n = minOf(len, free)
            System.arraycopy(pcmU8, offset, buf, writePos, n)
            writePos += n
            totalWritten += n
            return n
        }
    }

    fun writeSilence(len: Int): Int {
        if (len <= 0) return 0
        synchronized(lock) {
            val free = capacityBytes - writePos
            if (free <= 0) return 0
            val n = minOf(len, free)
            val pad = AppConfig.AUDIO_SILENCE_U8.toByte()
            for (i in 0 until n) buf[writePos + i] = pad
            writePos += n
            totalWritten += n
            return n
        }
    }

    /** Bytes available for ONNX inference. */
    fun availableBytes(): Int = synchronized(lock) { writePos - inferReadPos }

    fun freeBytes(): Int = synchronized(lock) { capacityBytes - writePos }

    /** Bytes decoded but not yet played to the speaker. */
    fun availableForPlayback(): Int = synchronized(lock) { writePos - playReadPos }

    fun writtenSeconds(): Float = writePos.toFloat() / AppConfig.AUDIO_SR

    fun playPositionMs(): Int =
        (playReadPos.toLong() * 1000L / AppConfig.AUDIO_SR).toInt()

    fun durationMs(): Int =
        (totalWritten * 1000L / AppConfig.AUDIO_SR).toInt()

    /** ONNX: read exactly [nBytes], or null if not enough data. */
    fun readChunk(nBytes: Int): ByteArray? {
        if (nBytes <= 0) return null
        synchronized(lock) {
            val avail = writePos - inferReadPos
            if (avail < nBytes) return null
            val out = ByteArray(nBytes)
            System.arraycopy(buf, inferReadPos, out, 0, nBytes)
            inferReadPos += nBytes
            return out
        }
    }

    /** Speaker: read up to [maxBytes] from the play cursor, or null when empty. */
    fun readForPlayback(maxBytes: Int): ByteArray? {
        if (maxBytes <= 0) return null
        synchronized(lock) {
            val avail = writePos - playReadPos
            if (avail <= 0) return null
            val n = minOf(maxBytes, avail)
            val out = ByteArray(n)
            System.arraycopy(buf, playReadPos, out, 0, n)
            playReadPos += n
            return out
        }
    }

    /** ONNX tail after EOS. */
    fun readRemaining(): ByteArray? {
        synchronized(lock) {
            val avail = writePos - inferReadPos
            if (avail <= 0) return null
            val out = ByteArray(avail)
            System.arraycopy(buf, inferReadPos, out, 0, avail)
            inferReadPos += avail
            return out
        }
    }

    /** Speaker tail after EOS. */
    fun readRemainingForPlayback(): ByteArray? {
        synchronized(lock) {
            val avail = writePos - playReadPos
            if (avail <= 0) return null
            val out = ByteArray(avail)
            System.arraycopy(buf, playReadPos, out, 0, avail)
            playReadPos += avail
            return out
        }
    }

    fun reset() {
        synchronized(lock) {
            inferReadPos = 0
            playReadPos = 0
            writePos = 0
        }
        endOfStream = false
        totalWritten = 0L
    }

    fun markEndOfStream() {
        endOfStream = true
    }

    fun isEndOfStream(): Boolean = endOfStream

    /** True when EOS is set and every written byte has been sent to the speaker. */
    fun isPlaybackComplete(): Boolean =
        endOfStream && synchronized(lock) { playReadPos >= writePos }
}
