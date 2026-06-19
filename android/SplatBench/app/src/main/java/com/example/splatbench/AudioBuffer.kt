package com.example.splatbench

/**
 * Fixed-capacity linear PCM buffer (no circular overwrite).
 * Producer appends; inference consumes from the front. When full the producer blocks.
 */
class AudioBuffer(
    val capacityBytes: Int = AppConfig.audioBufferCapacityBytes,
) {
    private val buf = ByteArray(capacityBytes)
    private var readPos = 0
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

    fun availableBytes(): Int = synchronized(lock) { writePos - readPos }

    fun freeBytes(): Int = synchronized(lock) { capacityBytes - writePos }

    /** Read exactly [nBytes] from the front, or null if not enough data. */
    fun readChunk(nBytes: Int): ByteArray? {
        if (nBytes <= 0) return null
        synchronized(lock) {
            val avail = writePos - readPos
            if (avail < nBytes) return null
            val out = ByteArray(nBytes)
            System.arraycopy(buf, readPos, out, 0, nBytes)
            readPos += nBytes
            return out
        }
    }

    /** Read all remaining bytes without requiring a fixed size. */
    fun readRemaining(): ByteArray? {
        synchronized(lock) {
            val avail = writePos - readPos
            if (avail <= 0) return null
            val out = ByteArray(avail)
            System.arraycopy(buf, readPos, out, 0, avail)
            readPos += avail
            return out
        }
    }

    fun reset() {
        synchronized(lock) {
            readPos = 0
            writePos = 0
        }
        endOfStream = false
        totalWritten = 0L
    }

    fun markEndOfStream() {
        endOfStream = true
    }

    fun isEndOfStream(): Boolean = endOfStream
}
