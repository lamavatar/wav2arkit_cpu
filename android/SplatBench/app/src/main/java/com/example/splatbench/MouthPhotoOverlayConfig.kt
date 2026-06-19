package com.example.splatbench

/**
 * Manual 2D transform for the cropped mouth patch composited over [AppConfig.PHOTO_FILE_NAME].
 * Crop rect comes from [MouthCropConfig]; this only positions/scales the patch on the photo.
 */
object MouthPhotoOverlayConfig {

  /** Horizontal shift as fraction of viewport width (-0.5 .. 0.5). */
  @Volatile var OFFSET_X = 0.011f

  /** Vertical shift as fraction of viewport height (-0.5 .. 0.5). */
  @Volatile var OFFSET_Y = -0.036f

  /** Uniform scale around [PIVOT_X]/[PIVOT_Y] (0.25 .. 3.0). */
  @Volatile var SCALE = 1.266f

  /** Scale pivot X (0 = left, 1 = right). */
  @Volatile var PIVOT_X = 0.513f

  /** Scale pivot Y (0 = top, 1 = bottom). */
  @Volatile var PIVOT_Y = 0.58f

  /** Draw alignment guide on the photo viewport. */
  @Volatile var GUIDE_ENABLED = true

  fun offsetXPx(viewportSq: Int): Float = OFFSET_X * viewportSq
  fun offsetYPx(viewportSq: Int): Float = OFFSET_Y * viewportSq
  fun pivotXPx(viewportSq: Int): Float = PIVOT_X * viewportSq
  fun pivotYPx(viewportSq: Int): Float = PIVOT_Y * viewportSq

  fun format(): String =
      String.format("mouth:%.3f,%.3f scale:%.3f pivot:%.3f,%.3f", OFFSET_X, OFFSET_Y, SCALE, PIVOT_X, PIVOT_Y)

  fun offsetXFromSeek(progress: Int): Float = (progress - 500) / 1000f

  fun offsetYFromSeek(progress: Int): Float = (progress - 500) / 1000f

  fun scaleFromSeek(progress: Int): Float = 0.5f + progress / 1000f * 1.5f

  fun pivotFromSeek(progress: Int): Float = progress / 1000f

  fun seekFromOffset(v: Float): Int = (v * 1000f + 500f).toInt().coerceIn(0, 1000)

  fun seekFromScale(v: Float): Int = ((v - 0.5f) / 1.5f * 1000f).toInt().coerceIn(0, 1000)

  fun seekFromPivot(v: Float): Int = (v * 1000f).toInt().coerceIn(0, 1000)
}
