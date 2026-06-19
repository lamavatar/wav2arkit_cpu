package com.example.splatbench

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Flat avatar data baked by `avatar_registry/bake_android.py`.
 *
 * Supports two formats:
 *  - **SPL1**: geometry + a baked per-frame ARKit weight sequence ([weights]).
 *  - **SPL2**: geometry only ([weights] empty, [numFrames] == 0). Per-frame
 *    weights are produced at runtime by ONNX inference and mapped onto the
 *    baked morphs via [mapArkitWeights].
 */
class AvatarPack private constructor(
    val numGaussians: Int,
    val numMorphs: Int,
    val numFrames: Int,
    val fps: Float,
    val morphNames: Array<String>,
    /** true = geometry-only pack (runtime ONNX weights); false = baked SPL1. */
    val isGeometryOnly: Boolean,
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
    /** per-frame weights aligned to morphNames, layout [f*M + m], length F*M (SPL1 only) */
    val weights: FloatArray,
    /** Optional baked head-bone clip (SPL2 HEAD trailer). */
    val headAnimName: String? = null,
    val headClipDurationSec: Float = 0f,
    /** row-major 4x4 per keyframe, layout [k*16 + e], length H*16 */
    val headMatrices: FloatArray? = null,
) {
    val hasHeadAnimation: Boolean
        get() = headMatrices != null && headFrameCount > 0

    val headFrameCount: Int
        get() = headMatrices?.size?.div(16) ?: 0

    /** Row-major 4x4 head skin matrix for [frameIndex] (loops the baked clip). */
    fun headMatrixForFrame(frameIndex: Int): FloatArray? {
        val mats = headMatrices ?: return null
        val h = headFrameCount
        if (h <= 0) return null
        val f = ((frameIndex % h) + h) % h
        return mats.copyOfRange(f * 16, f * 16 + 16)
    }
    /** For each baked morph, its index in the 52-entry ARKit vector (-1 if unknown). */
    val morphIndexInArkit52: IntArray = IntArray(numMorphs) { ArkitBlendshapes.indexOf(morphNames[it]) }

    /** Map a full 52-entry ARKit weight vector onto this pack's baked morphs. */
    fun mapArkitWeights(arkit52: FloatArray): FloatArray {
        val out = FloatArray(numMorphs)
        for (m in 0 until numMorphs) {
            val idx = morphIndexInArkit52[m]
            out[m] = if (idx in 0 until arkit52.size) arkit52[idx] else 0f
        }
        return out
    }
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
        if (numFrames <= 0) return FloatArray(numMorphs)
        val f = ((frame % numFrames) + numFrames) % numFrames
        return weights.copyOfRange(f * numMorphs, f * numMorphs + numMorphs)
    }

    companion object {
        fun load(): AvatarPack = loadFromFile(AppConfig.splatFile())

        fun loadFromFile(file: File): AvatarPack {
            require(file.isFile) {
                "Splat avatar not found: ${file.absolutePath}\n" +
                    "Copy ${AppConfig.SPLAT_FILE_NAME} to ${AppConfig.splatPathHint()}"
            }
            val len = file.length()
            require(len in 1..Int.MAX_VALUE.toLong()) { "invalid splat size: $len" }
            val bytes = ByteArray(len.toInt())
            file.inputStream().use { input ->
                var off = 0
                while (off < bytes.size) {
                    val n = input.read(bytes, off, bytes.size - off)
                    if (n < 0) break
                    off += n
                }
                require(off == bytes.size) { "short read: $off of ${bytes.size}" }
            }
            return parse(bytes)
        }

        private fun parse(bytes: ByteArray): AvatarPack {
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

            val magic = ByteArray(4)
            bb.get(magic)
            val tag = String(magic, Charsets.US_ASCII)
            val geometryOnly = when (tag) {
                "SPL2" -> true
                "SPL1" -> false
                else -> throw IllegalArgumentException("bad magic: $tag")
            }

            val n = bb.int
            val m = bb.int
            // SPL1 stores F before fps; SPL2 has no frame count.
            val f = if (geometryOnly) 0 else bb.int
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
            val weights = if (geometryOnly) FloatArray(0) else readFloats(bb, f * m)
            val head = if (geometryOnly) parseHeadTrailer(bb) else HeadTrailer(null, 0f, null)

            return AvatarPack(
                n, m, f, fps, names, geometryOnly, cam, base, cov6, color, opacity, dyn, deltas, weights,
                head.name, head.clipDurationSec, head.matrices,
            )
        }

        private data class HeadTrailer(
            val name: String?,
            val clipDurationSec: Float,
            val matrices: FloatArray?,
        )

        private fun parseHeadTrailer(bb: ByteBuffer): HeadTrailer {
            if (bb.remaining() < 4) return HeadTrailer(null, 0f, null)
            val pos = bb.position()
            val magic = ByteArray(4)
            bb.get(magic)
            if (String(magic, Charsets.US_ASCII) != "HEAD") {
                bb.position(pos)
                return HeadTrailer(null, 0f, null)
            }
            val h = bb.int
            val clipDur = bb.float
            val nameLen = bb.get().toInt() and 0xFF
            require(bb.remaining() >= nameLen + h * 16 * 4) { "truncated HEAD trailer" }
            val nameBytes = ByteArray(nameLen)
            bb.get(nameBytes)
            val name = String(nameBytes, Charsets.US_ASCII)
            val matrices = readFloats(bb, h * 16)
            return HeadTrailer(name, clipDur, matrices)

        private fun readFloats(bb: ByteBuffer, count: Int): FloatArray {
            val out = FloatArray(count)
            bb.asFloatBuffer().get(out)
            bb.position(bb.position() + count * 4)
            return out
        }
    }
}
