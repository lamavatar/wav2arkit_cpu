package com.example.splatbench

/**
 * Supplies per-frame morph weights to the geometry builder, decoupling the
 * renderer from where weights come from (neutral, baked SPL1, or runtime ONNX).
 *
 * [weightsForFrame] returns weights already aligned to the pack's baked morph
 * order (length == [AvatarPack.numMorphs]), so it can be fed straight to the
 * [InstanceBuilder].
 */
interface ExpressionSource {
    val fps: Float

    /** Upper bound on selectable frames; may grow over time for live sources. */
    fun frameCount(): Int

    /** Pack-morph weights for [frame] (length == pack.numMorphs). */
    fun weightsForFrame(frame: Int): FloatArray

    /** Whether weights for [frame] are ready (always true for static sources). */
    fun hasFrame(frame: Int): Boolean
}

/** All-zero weights — renders the neutral pose. */
class NeutralExpressionSource(
    private val pack: AvatarPack,
    private val frames: Int = 1,
) : ExpressionSource {
    override val fps: Float get() = pack.fps
    override fun frameCount(): Int = frames
    override fun weightsForFrame(frame: Int): FloatArray = FloatArray(pack.numMorphs)
    override fun hasFrame(frame: Int): Boolean = true
}

/** Weights from the SPL1 baked per-frame sequence. */
class BakedExpressionSource(private val pack: AvatarPack) : ExpressionSource {
    override val fps: Float get() = pack.fps
    override fun frameCount(): Int = pack.numFrames
    override fun weightsForFrame(frame: Int): FloatArray = pack.frameWeights(frame)
    override fun hasFrame(frame: Int): Boolean = pack.numFrames > 0
}
