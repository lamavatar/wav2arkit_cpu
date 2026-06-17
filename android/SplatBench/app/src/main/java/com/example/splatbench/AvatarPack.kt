package com.example.splatbench

import android.content.res.AssetManager
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Flat avatar data baked by `avatar_registry/bake_android.py` ("SPL1").
 *
 * Holds canonical Gaussian attributes plus per-morph position deltas and the
 * per-frame ARKit weight sequence, ready for the CPU instance builder.
 */
class AvatarPack private constructor(
    val numGaussians: Int,
    val numMorphs: Int,
    val numFrames: Int,
    val fps: Float,
    val morphNames: Array<String>,
    /** eye[3], center[3], up[3], fovyRad */
    val camera: FloatArray,
    /** neutral + offset, length N*3 */
    val base: FloatArray,
    /** symmetric 3x3 covariance per Gaussian: xx,xy,xz,yy,yz,zz; length N*6 */
    val cov6: FloatArray,
    /** RGB per Gaussian, length N*3 */
    val color: FloatArray,
    /** opacity per Gaussian, length N */
    val opacity: FloatArray,
    /** 1 if the Gaussian moves during the clip (mouth/jaw set), length N */
    val dynamic: ByteArray,
    /** per-morph deltas, layout [m * N*3 + i*3 + c], length M*N*3 */
    val deltas: FloatArray,
    /** per-frame weights aligned to morphNames, layout [f*M + m], length F*M */
    val weights: FloatArray,
) {
    /** Indices of Gaussians flagged dynamic (mouth region). */
    val dynamicIndices: IntArray = run {
        var c = 0
        for (b in dynamic) if (b.toInt() != 0) c++
        val out = IntArray(c)
        var k = 0
        for (i in dynamic.indices) if (dynamic[i].toInt() != 0) out[k++] = i
        out
    }

    /** Identity index list 0..N-1 (full set). */
    val allIndices: IntArray = IntArray(numGaussians) { it }

    fun frameWeights(frame: Int): FloatArray {
        val f = ((frame % numFrames) + numFrames) % numFrames
        return weights.copyOfRange(f * numMorphs, f * numMorphs + numMorphs)
    }

    companion object {
        fun load(assets: AssetManager, name: String): AvatarPack {
            val bytes = assets.open(name).use { input ->
                val bos = ByteArrayOutputStream(16 shl 20)
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    bos.write(buf, 0, n)
                }
                bos.toByteArray()
            }
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

            val magic = ByteArray(4)
            bb.get(magic)
            require(String(magic, Charsets.US_ASCII) == "SPL1") { "bad magic" }

            val n = bb.int
            val m = bb.int
            val f = bb.int
            val fps = bb.float
            val cam = readFloats(bb, 10)

            val names = Array(m) {
                val len = bb.get().toInt() and 0xFF
                val nb = ByteArray(len)
                bb.get(nb)
                String(nb, Charsets.US_ASCII)
            }

            val base = readFloats(bb, n * 3)
            val cov6 = readFloats(bb, n * 6)
            val color = readFloats(bb, n * 3)
            val opacity = readFloats(bb, n)
            val dyn = ByteArray(n)
            bb.get(dyn)
            val deltas = readFloats(bb, m * n * 3)
            val weights = readFloats(bb, f * m)

            return AvatarPack(n, m, f, fps, names, cam, base, cov6, color, opacity, dyn, deltas, weights)
        }

        private fun readFloats(bb: ByteBuffer, count: Int): FloatArray {
            val out = FloatArray(count)
            bb.asFloatBuffer().get(out)
            bb.position(bb.position() + count * 4)
            return out
        }
    }
}
