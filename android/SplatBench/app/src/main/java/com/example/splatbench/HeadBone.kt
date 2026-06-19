package com.example.splatbench

import kotlin.math.sqrt

/** Head-bone rotation helpers (pivot rotation, legacy skin-matrix conversion). */
object HeadBone {

    fun centroid(base: FloatArray): FloatArray {
        val n = base.size / 3
        var sx = 0f
        var sy = 0f
        var sz = 0f
        for (i in 0 until n) {
            val j = i * 3
            sx += base[j]
            sy += base[j + 1]
            sz += base[j + 2]
        }
        val inv = 1f / n.coerceAtLeast(1)
        return floatArrayOf(sx * inv, sy * inv, sz * inv)
    }

    /** Rotate [px,py,pz] about [pivot] using the 3x3 block of row-major [headMat]. */
    fun applyPivotRotation(
        px: Float, py: Float, pz: Float,
        pivot: FloatArray,
        headMat: FloatArray,
    ): FloatArray {
        val dx = px - pivot[0]
        val dy = py - pivot[1]
        val dz = pz - pivot[2]
        return floatArrayOf(
            pivot[0] + dx * headMat[0] + dy * headMat[4] + dz * headMat[8],
            pivot[1] + dx * headMat[1] + dy * headMat[5] + dz * headMat[9],
            pivot[2] + dx * headMat[2] + dy * headMat[6] + dz * headMat[10],
        )
    }

    /**
     * Legacy HEAD trailer stored full skin matrices (translation caused pan/zoom).
     * Convert to frame-0 rotation-only deltas.
     */
    fun convertLegacySkinMatrices(mats: FloatArray, frameCount: Int): FloatArray {
        val m0Inv = invert4x4(mats, 0) ?: return mats
        val out = FloatArray(frameCount * 16)
        for (f in 0 until frameCount) {
            val delta = multiply4x4(mats, f * 16, m0Inv)
            val rot = orthonormalize3x3(delta)
            identity4x4(out, f * 16)
            copy3x3(out, f * 16, rot)
        }
        return out
    }

    private fun identity4x4(out: FloatArray, off: Int) {
        out[off] = 1f; out[off + 5] = 1f; out[off + 10] = 1f; out[off + 15] = 1f
    }

    private fun copy3x3(out: FloatArray, off: Int, r: FloatArray) {
        out[off] = r[0]; out[off + 1] = r[1]; out[off + 2] = r[2]
        out[off + 4] = r[3]; out[off + 5] = r[4]; out[off + 6] = r[5]
        out[off + 8] = r[6]; out[off + 9] = r[7]; out[off + 10] = r[8]
    }

    /** Orthonormalize 3x3 (row-major in 4x4 at [off]). */
    private fun orthonormalize3x3(m: FloatArray, off: Int = 0): FloatArray {
        var ax = m[off]; var ay = m[off + 4]; var az = m[off + 8]
        val len0 = sqrt(ax * ax + ay * ay + az * az).coerceAtLeast(1e-8f)
        ax /= len0; ay /= len0; az /= len0

        var bx = m[off + 1]; var by = m[off + 5]; var bz = m[off + 9]
        val dotBa = bx * ax + by * ay + bz * az
        bx -= dotBa * ax; by -= dotBa * ay; bz -= dotBa * az
        val len1 = sqrt(bx * bx + by * by + bz * bz).coerceAtLeast(1e-8f)
        bx /= len1; by /= len1; bz /= len1

        val cx = ay * bz - az * by
        val cy = az * bx - ax * bz
        val cz = ax * by - ay * bx
        return floatArrayOf(ax, bx, cx, ay, by, cy, az, bz, cz)
    }

    private fun multiply4x4(a: FloatArray, aOff: Int, b: FloatArray): FloatArray {
        val out = FloatArray(16)
        for (row in 0 until 4) {
            for (col in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) {
                    sum += a[aOff + row * 4 + k] * b[k * 4 + col]
                }
                out[row * 4 + col] = sum
            }
        }
        return out
    }

    private fun invert4x4(m: FloatArray, off: Int): FloatArray? {
        val a = DoubleArray(16)
        for (i in 0 until 16) a[i] = m[off + i].toDouble()
        val inv = DoubleArray(16)
        inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] +
            a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10]
        inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] -
            a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10]
        inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] +
            a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9]
        inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] -
            a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9]
        inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] -
            a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10]
        inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] +
            a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10]
        inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] -
            a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9]
        inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] +
            a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9]
        inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] +
            a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6]
        inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] -
            a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6]
        inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] +
            a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5]
        inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] -
            a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5]
        inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] -
            a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6]
        inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] +
            a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6]
        inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] -
            a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5]
        inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] +
            a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5]
        val det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12]
        if (kotlin.math.abs(det) < 1e-12) return null
        val invDet = 1.0 / det
        val out = FloatArray(16)
        for (i in 0 until 16) out[i] = (inv[i] * invDet).toFloat()
        return out
    }
}
