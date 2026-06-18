package com.example.splatbench

/**
 * Audio format conversions for the u8 pipeline. The ring/chunk pipeline stays
 * in 8-bit unsigned (0..255); conversion to ONNX float32 happens only here.
 */
object AudioPcmConverter {

    /** 0..255 u8 → float32 [-1, 1] (silence 128 → 0.0). */
    fun u8ToOnnxFloat(pcmU8: ByteArray, len: Int = pcmU8.size): FloatArray =
        FloatArray(len) { i -> ((pcmU8[i].toInt() and 0xFF) - 128) / 128.0f }

    /** 16-bit signed PCM sample → standard 8-bit unsigned (128 = silence). */
    fun i16ToU8(s: Short): Byte =
        (((s.toInt() shr 8) + 128).coerceIn(0, 255)).toByte()

    /**
     * Convert a 16-bit little-endian PCM block (length [byteLen], must be even)
     * to u8 into [out], returning the number of u8 samples written.
     */
    fun i16BlockToU8(src: ByteArray, byteLen: Int, out: ByteArray): Int {
        val n = byteLen / 2
        var si = 0
        for (i in 0 until n) {
            val lo = src[si].toInt() and 0xFF
            val hi = src[si + 1].toInt()
            out[i] = i16ToU8(((hi shl 8) or lo).toShort())
            si += 2
        }
        return n
    }
}
