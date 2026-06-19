package com.example.splatbench

import java.util.concurrent.ConcurrentHashMap

/**
 * Thread-safe store of pre-built per-frame GPU instance buffers.
 *
 * Full-mode frames are large (~800 KB each); callers must trim via
 * [evictOutside] during playback so the cache stays within
 * [AppConfig.FRAME_CACHE_MAX_FRAMES].
 */
class FrameCache {
    private val map = ConcurrentHashMap<Int, CachedFrame>()

    fun put(frame: CachedFrame) {
        map[frame.frameIndex] = frame
    }

    fun get(frameIndex: Int): CachedFrame? = map[frameIndex]

    fun has(frameIndex: Int): Boolean = map.containsKey(frameIndex)

    fun clear() = map.clear()

    fun size(): Int = map.size

    /** Drop cached frames outside [minInclusive, maxInclusive]. */
    fun evictOutside(minInclusive: Int, maxInclusive: Int) {
        if (minInclusive > maxInclusive) return
        for (key in map.keys) {
            if (key < minInclusive || key > maxInclusive) {
                map.remove(key)
            }
        }
    }

    /** Evict oldest frames until at most [maxFrames] remain. */
    fun trimToMaxFrames(maxFrames: Int) {
        while (map.size > maxFrames) {
            val oldest = map.keys.minOrNull() ?: return
            map.remove(oldest)
        }
    }
}
