package com.example.splatbench

/**
 * Manual mouth crop rectangle (top-left origin, normalized 0..1).
 * Used by [RenderMode.MOUTH_CROP_ONLY] display and as the extract region for
 * [RenderMode.MOUTH_ON_PHOTO] compositing.
 */
object MouthCropConfig {

  /** Left edge (0 = left, 1 = right). */
  @Volatile var X = 0.32f

  /** Top edge (0 = top, 1 = bottom). */
  @Volatile var Y = 0.52f

  /** Width as fraction of viewport side. */
  @Volatile var W = 0.36f

  /** Height as fraction of viewport side. */
  @Volatile var H = 0.28f

  /**
   * When true: show full frame + guide overlay so the crop rect can be aligned.
   * When false: apply crop (in-place blit or photo scissor).
   */
  @Volatile var GUIDE_ENABLED = true

  fun ratios(): FloatArray = floatArrayOf(X, Y, W, H)

  fun setFromRatios(r: FloatArray) {
    if (r.size < 4) return
    X = r[0].coerceIn(0f, 1f)
    Y = r[1].coerceIn(0f, 1f)
    W = r[2].coerceIn(0.02f, 1f)
    H = r[3].coerceIn(0.02f, 1f)
    clampInBounds()
  }

  fun clampInBounds() {
    W = W.coerceIn(0.02f, 1f - X)
    H = H.coerceIn(0.02f, 1f - Y)
    if (X + W > 1f) W = 1f - X
    if (Y + H > 1f) H = 1f - Y
  }

  /** Pixel bbox `[x, y, w, h]` (top-left) for a square viewport of side [viewportSq]. */
  fun toPixels(viewportSq: Int): IntArray {
    clampInBounds()
    val s = viewportSq.coerceAtLeast(1)
    val x = (X * s).toInt().coerceIn(0, s - 1)
    val y = (Y * s).toInt().coerceIn(0, s - 1)
    val w = (W * s).toInt().coerceIn(2, s - x)
    val h = (H * s).toInt().coerceIn(2, s - y)
    return intArrayOf(x, y, w, h)
  }

  fun formatRatios(): String =
      String.format("crop:%.3f,%.3f,%.3f,%.3f", X, Y, W, H)
}
