package com.example.splatbench

import kotlin.math.max
import kotlin.math.min

/**
 * Screen-space mouth crop for photo composite (matches desktop hybrid crop logic).
 */
object MouthRegion {

    /**
     * Pixel bbox [x0, y0, width, height] in top-left origin, clamped to the square viewport.
     */
    fun screenBBox(
        pack: AvatarPack,
        rot: FloatArray,
        tv: FloatArray,
        fy: Float,
        cx: Float,
        cy: Float,
        vw: Float,
        vh: Float,
        margin: Float = 0.18f,
    ): IntArray {
        val r00 = rot[0]; val r01 = rot[1]; val r02 = rot[2]
        val r10 = rot[3]; val r11 = rot[4]; val r12 = rot[5]
        val r20 = rot[6]; val r21 = rot[7]; val r22 = rot[8]
        val base = pack.base
        var xMin = Float.POSITIVE_INFINITY
        var yMin = Float.POSITIVE_INFINITY
        var xMax = Float.NEGATIVE_INFINITY
        var yMax = Float.NEGATIVE_INFINITY
        var any = false

        for (gi in pack.dynamicIndices) {
            val b3 = gi * 3
            val px = base[b3]; val py = base[b3 + 1]; val pz = base[b3 + 2]
            val tx = r00 * px + r01 * py + r02 * pz + tv[0]
            val ty = r10 * px + r11 * py + r12 * pz + tv[1]
            val tz = r20 * px + r21 * py + r22 * pz + tv[2]
            if (tz >= -1e-4f) continue
            val inv = 1f / -tz
            val sx = cx + fy * tx * inv
            val sy = cy - fy * ty * inv
            any = true
            if (sx < xMin) xMin = sx
            if (sy < yMin) yMin = sy
            if (sx > xMax) xMax = sx
            if (sy > yMax) yMax = sy
        }

        if (!any) {
            val m = (min(vw, vh) * 0.25f).toInt()
            val c = (vw * 0.5f).toInt()
            val cyi = (vh * 0.55f).toInt()
            return intArrayOf(c - m, cyi - m, m * 2, m * 2)
        }

        val mx = (xMax - xMin) * margin
        val my = (yMax - yMin) * margin
        xMin = max(0f, xMin - mx)
        yMin = max(0f, yMin - my)
        xMax = min(vw - 1f, xMax + mx)
        yMax = min(vh - 1f, yMax + my)
        val w = max(2f, xMax - xMin).toInt()
        val h = max(2f, yMax - yMin).toInt()
        return intArrayOf(xMin.toInt(), yMin.toInt(), w, h)
    }
}
