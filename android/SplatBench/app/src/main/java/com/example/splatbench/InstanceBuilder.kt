package com.example.splatbench

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Arrays
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService

/**
 * Turns ARKit weights into sorted, projected splat instances on the CPU.
 *
 * This is the stage we parallelize for the benchmark: per-Gaussian morph
 * deform + view transform + EWA 2D-covariance projection + frustum cull, split
 * across worker threads, followed by a back-to-front depth sort and packing
 * into the GPU instance buffer. The math mirrors `gaussian_splat.py`
 * (`_project_instances`) exactly.
 *
 * Instance float layout (10 floats): center.xy, radius, conic.xyz, color.rgb, alpha.
 */
class InstanceBuilder(private val pack: AvatarPack) {

    private val n = pack.numGaussians
    private val m = pack.numMorphs
    private val n3 = n * 3

    /** When true and the pack has a HEAD trailer, apply baked head-bone matrices. */
    @Volatile var headBoneEnabled: Boolean = false

    // Per-slot scratch (slot j == position in the index list being rendered).
    private val sSx = FloatArray(n)
    private val sSy = FloatArray(n)
    private val sR = FloatArray(n)        // radius, or < 0 when culled
    private val sCa = FloatArray(n)
    private val sCb = FloatArray(n)
    private val sCc = FloatArray(n)
    private val sZ = FloatArray(n)        // view-space depth (tz), used for sorting

    /** Native-order instance bytes; the renderer uploads [instanceCount]*40 bytes. */
    val instanceBytes: ByteBuffer =
        ByteBuffer.allocateDirect(n * 10 * 4).order(ByteOrder.nativeOrder())
    private val instanceFloats: FloatBuffer = instanceBytes.asFloatBuffer()

    var instanceCount: Int = 0
        private set

    // View state (set per resize / camera change).
    private val r = FloatArray(9)
    private var tvx = 0f; private var tvy = 0f; private var tvz = 0f
    private var fy = 0f
    private var cx = 0f; private var cy = 0f
    private var vw = 0f; private var vh = 0f

    fun setView(rot: FloatArray, tv: FloatArray, fy: Float, cx: Float, cy: Float, vw: Float, vh: Float) {
        System.arraycopy(rot, 0, r, 0, 9)
        tvx = tv[0]; tvy = tv[1]; tvz = tv[2]
        this.fy = fy; this.cx = cx; this.cy = cy; this.vw = vw; this.vh = vh
    }

    /**
     * Build sorted instances for [indices] under [weights]. Returns the number
     * of visible instances; data is in [instanceBytes]. CPU work is split over
     * [threads] worker tasks on [exec].
     */
    fun build(weights: FloatArray, indices: IntArray, exec: ExecutorService, threads: Int, frameIndex: Int = 0): Int {
        // Active morphs this frame (skip the zero-weight majority).
        var nz = 0
        val mi = IntArray(m)
        val mw = FloatArray(m)
        for (k in 0 until m) {
            val w = weights[k]
            if (w > 1e-6f || w < -1e-6f) { mi[nz] = k; mw[nz] = w; nz++ }
        }

        val len = indices.size
        val headMat = if (headBoneEnabled && pack.hasHeadAnimation) pack.headMatrixForFrame(frameIndex) else null
        val t = threads.coerceIn(1, 64)
        if (t == 1) {
            processRange(0, len, indices, mi, mw, nz, headMat)
        } else {
            val chunk = (len + t - 1) / t
            val tasks = ArrayList<Callable<Unit>>(t)
            var start = 0
            while (start < len) {
                val s = start
                val e = minOf(start + chunk, len)
                tasks.add(Callable { processRange(s, e, indices, mi, mw, nz, headMat); Unit })
                start = e
            }
            exec.invokeAll(tasks)
        }

        return compactSortPack(indices, len)
    }

    /** Build and copy instance bytes into an immutable [CachedFrame]. */
    fun buildCached(
        frameIndex: Int,
        weights: FloatArray,
        indices: IntArray,
        exec: ExecutorService,
        threads: Int,
    ): CachedFrame {
        val count = build(weights, indices, exec, threads, frameIndex)
        val size = count * 40
        val bytes = ByteArray(size)
        instanceBytes.position(0)
        if (size > 0) {
            instanceBytes.get(bytes, 0, size)
        }
        return CachedFrame(frameIndex, count, bytes)
    }

    private fun processRange(
        from: Int, to: Int, indices: IntArray, mi: IntArray, mw: FloatArray, nz: Int,
        headMat: FloatArray?,
    ) {
        val r00 = r[0]; val r01 = r[1]; val r02 = r[2]
        val r10 = r[3]; val r11 = r[4]; val r12 = r[5]
        val r20 = r[6]; val r21 = r[7]; val r22 = r[8]
        val base = pack.base
        val cov = pack.cov6
        val deltas = pack.deltas
        val fy = this.fy
        val vw = this.vw; val vh = this.vh
        val cx = this.cx; val cy = this.cy

        for (j in from until to) {
            val gi = indices[j]
            val b3 = gi * 3
            var px = base[b3]; var py = base[b3 + 1]; var pz = base[b3 + 2]

            // Morph deform: p += sum_k w_k * delta_k
            var q = 0
            while (q < nz) {
                val w = mw[q]
                val off = mi[q] * n3 + b3
                px += w * deltas[off]
                py += w * deltas[off + 1]
                pz += w * deltas[off + 2]
                q++
            }

            // Rigid head-bone skinning after morphs (gaussian_splat.py positions()).
            if (headMat != null) {
                val nx = px * headMat[0] + py * headMat[4] + pz * headMat[8] + headMat[3]
                val ny = px * headMat[1] + py * headMat[5] + pz * headMat[9] + headMat[7]
                val nz = px * headMat[2] + py * headMat[6] + pz * headMat[10] + headMat[11]
                px = nx; py = ny; pz = nz
            }

            // View transform t = R p + tv
            val tx = r00 * px + r01 * py + r02 * pz + tvx
            val ty = r10 * px + r11 * py + r12 * pz + tvy
            val tz = r20 * px + r21 * py + r22 * pz + tvz
            if (tz >= -1e-4f) { sR[j] = -1f; continue }

            val zp = -tz
            val inv = 1f / zp
            val sx = cx + fy * tx * inv
            val sy = cy - fy * ty * inv

            // Jacobian-projected 2x3 M = J @ R
            val k1 = fy * inv
            val k2 = fy * tx * inv * inv
            val m00 = k1 * r00 + k2 * r20
            val m01 = k1 * r01 + k2 * r21
            val m02 = k1 * r02 + k2 * r22
            val k3 = -fy * inv
            val k4 = -fy * ty * inv * inv
            val m10 = k3 * r10 + k4 * r20
            val m11 = k3 * r11 + k4 * r21
            val m12 = k3 * r12 + k4 * r22

            // Symmetric covariance
            val c6 = gi * 6
            val s0 = cov[c6]; val s1 = cov[c6 + 1]; val s2 = cov[c6 + 2]
            val s3 = cov[c6 + 3]; val s4 = cov[c6 + 4]; val s5 = cov[c6 + 5]

            // MS0 = M0 . Cov ; MS1 = M1 . Cov
            val ms00 = m00 * s0 + m01 * s1 + m02 * s2
            val ms01 = m00 * s1 + m01 * s3 + m02 * s4
            val ms02 = m00 * s2 + m01 * s4 + m02 * s5
            val ms10 = m10 * s0 + m11 * s1 + m12 * s2
            val ms11 = m10 * s1 + m11 * s3 + m12 * s4
            val ms12 = m10 * s2 + m11 * s4 + m12 * s5

            val a = ms00 * m00 + ms01 * m01 + ms02 * m02 + 0.3f
            val bb = ms00 * m10 + ms01 * m11 + ms02 * m12
            val c = ms10 * m10 + ms11 * m11 + ms12 * m12 + 0.3f

            val det = a * c - bb * bb
            if (det <= 1e-9f) { sR[j] = -1f; continue }

            val invDet = 1f / det
            val mid = 0.5f * (a + c)
            val lam = mid + Math.sqrt(Math.max(mid * mid - det, 0f).toDouble()).toFloat()
            val radius = 3f * Math.sqrt(Math.max(lam, 1e-8f).toDouble()).toFloat()

            if (sx + radius < 0f || sx - radius > vw || sy + radius < 0f || sy - radius > vh) {
                sR[j] = -1f; continue
            }

            sSx[j] = sx; sSy[j] = sy; sR[j] = radius
            sCa[j] = c * invDet
            sCb[j] = -bb * invDet
            sCc[j] = a * invDet
            sZ[j] = tz
        }
    }

    private fun compactSortPack(indices: IntArray, len: Int): Int {
        // Collect visible slots, keyed by depth for a back-to-front sort.
        var count = 0
        val keys = scratchKeys(len)
        for (j in 0 until len) {
            if (sR[j] < 0f) continue
            val bits = java.lang.Float.floatToRawIntBits(sZ[j])
            val sortable = bits xor ((bits shr 31) or Int.MIN_VALUE)
            keys[count++] = ((sortable.toLong() and 0xFFFFFFFFL) shl 32) or (j.toLong() and 0xFFFFFFFFL)
        }
        Arrays.sort(keys, 0, count)

        val color = pack.color
        val opacity = pack.opacity
        instanceFloats.clear()
        for (idx in 0 until count) {
            val j = (keys[idx] and 0xFFFFFFFFL).toInt()
            val gi = indices[j]
            val c3 = gi * 3
            instanceFloats.put(sSx[j]); instanceFloats.put(sSy[j]); instanceFloats.put(sR[j])
            instanceFloats.put(sCa[j]); instanceFloats.put(sCb[j]); instanceFloats.put(sCc[j])
            instanceFloats.put(color[c3]); instanceFloats.put(color[c3 + 1]); instanceFloats.put(color[c3 + 2])
            instanceFloats.put(opacity[gi])
        }
        instanceBytes.position(0)
        instanceBytes.limit(count * 10 * 4)
        instanceCount = count
        return count
    }

    private var keyScratch: LongArray = LongArray(0)
    private fun scratchKeys(len: Int): LongArray {
        if (keyScratch.size < len) keyScratch = LongArray(len)
        return keyScratch
    }
}
