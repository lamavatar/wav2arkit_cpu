package com.example.splatbench

/**
 * Single-producer / single-consumer ring buffer of 8-bit unsigned PCM
 * (16kHz mono). Shared by every [AudioProducer] (file or mic) and consumed by
 * the ONNX inference thread, so FILE and MIC use the exact same downstream path.
 *
 * Full policy: **drop oldest** — when the producer outruns the consumer, the
 * oldest samples are overwritten (slides the 10s window forward).
 */
class AudioRingBuffer(
    val capacityBytes: Int = AppConfig.ringBufferCapacityBytes,
) {
    private val buf = ByteArray(capacityBytes)
    private var readPos = 0
    private var writePos = 0
    private var available = 0
    private val lock = Any()

    @Volatile private var endOfStream = false
    /** Total bytes ever written (monotonic), for clock/diagnostics. */
    @Volatile var totalWritten: Long = 0L
        private set

    /** Append u8 PCM; overwrites oldest data when full. Returns bytes accepted. */
    fun write(pcmU8: ByteArray, len: Int = pcmU8.size): Int {
        if (len <= 0) return 0
        synchronized(lock) {
            // If the block is larger than the whole ring, keep only its tail.
            var srcOff = 0
            var n = len
            if (n > capacityBytes) {
                srcOff = n - capacityBytes
                n = capacityBytes
            }
            // Drop oldest if not enough free space.
            val free = capacityBytes - available
            if (n > free) {
                val drop = n - free
                readPos = (readPos + drop) % capacityBytes
                available -= drop
            }
            // Circular copy.
            var w = writePos
            for (i in 0 until n) {
                buf[w] = pcmU8[srcOff + i]
                w++
                if (w == capacityBytes) w = 0
            }
            writePos = w
            available += n
            totalWritten += n
            return n
        }
    }

    fun availableBytes(): Int = synchronized(lock) { available }

    /** Read exactly [nBytes] contiguous bytes, or null if not enough buffered. */
    fun readChunk(nBytes: Int): ByteArray? {
        if (nBytes <= 0) return null
        synchronized(lock) {
            if (available < nBytes) return null
            val out = ByteArray(nBytes)
            var r = readPos
            for (i in 0 until nBytes) {
                out[i] = buf[r]
                r++
                if (r == capacityBytes) r = 0
            }
            readPos = r
            available -= nBytes
            return out
        }
    }

    fun reset() {
        synchronized(lock) {
            readPos = 0
            writePos = 0
            available = 0
        }
        endOfStream = false
        totalWritten = 0L
    }

    fun markEndOfStream() { endOfStream = true }
    fun isEndOfStream(): Boolean = endOfStream
}
