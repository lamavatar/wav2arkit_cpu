package com.example.splatbench

/** One projected splat instance buffer ready for GPU upload (10 floats / 40 bytes per splat). */
data class CachedFrame(
    val frameIndex: Int,
    val instanceCount: Int,
    val bytes: ByteArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CachedFrame) return false
        return frameIndex == other.frameIndex &&
            instanceCount == other.instanceCount &&
            bytes.contentEquals(other.bytes)
    }

    override fun hashCode(): Int {
        var result = frameIndex
        result = 31 * result + instanceCount
        result = 31 * result + bytes.contentHashCode()
        return result
    }
}
