package com.example.splatbench

/**
 * Thread-safe append-only store of per-frame ARKit weight vectors (each
 * length 52). Written by the ONNX inference thread, read by the geometry
 * prefetcher / renderer.
 */
class ExpressionBuffer {
    private val frames = ArrayList<FloatArray>(2048)
    private val lock = Any()

    @Volatile var count: Int = 0
        private set

    fun append(rows: Array<FloatArray>) {
        if (rows.isEmpty()) return
        synchronized(lock) {
            for (r in rows) frames.add(r)
            count = frames.size
        }
    }

    /** 52-entry weights for [frame], or null if not yet produced. */
    fun get(frame: Int): FloatArray? {
        if (frame < 0) return null
        synchronized(lock) {
            return if (frame < frames.size) frames[frame] else null
        }
    }

    fun hasFrame(frame: Int): Boolean = frame in 0 until count

    fun clear() {
        synchronized(lock) {
            frames.clear()
            count = 0
        }
    }
}
