package com.example.splatbench

import java.util.concurrent.ConcurrentHashMap

/** Thread-safe store of pre-built per-frame GPU instance buffers. */
class FrameCache {
    private val map = ConcurrentHashMap<Int, CachedFrame>()

    fun put(frame: CachedFrame) {
        map[frame.frameIndex] = frame
    }

    fun get(frameIndex: Int): CachedFrame? = map[frameIndex]

    fun has(frameIndex: Int): Boolean = map.containsKey(frameIndex)

    fun clear() = map.clear()

    fun size(): Int = map.size
}
