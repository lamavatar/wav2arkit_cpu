package com.example.splatbench

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/** Draws a manual crop rectangle and alignment guide lines over the GL viewport. */
class CropGuideOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

  private val dimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.argb(100, 0, 0, 0)
    style = Paint.Style.FILL
  }
  private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.argb(230, 255, 193, 7)
    style = Paint.Style.STROKE
    strokeWidth = 3f
  }
  private val guidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.argb(180, 0, 229, 255)
    style = Paint.Style.STROKE
    strokeWidth = 1.5f
    pathEffect = DashPathEffect(floatArrayOf(12f, 8f), 0f)
  }
  private val crossPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.argb(200, 255, 64, 129)
    style = Paint.Style.STROKE
    strokeWidth = 1.5f
  }
  private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.argb(220, 255, 255, 255)
    textSize = 28f
  }

  private val cropRect = RectF()
  private val holePath = Path()

  var guideEnabled: Boolean = false
    set(value) {
      field = value
      visibility = if (value) VISIBLE else GONE
      invalidate()
    }

  fun syncFromConfig() {
    invalidate()
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    if (!guideEnabled) return

    val vw = width.toFloat()
    val vh = height.toFloat()
    if (vw <= 0f || vh <= 0f) return

    val left = MouthCropConfig.X * vw
    val top = MouthCropConfig.Y * vh
    val right = left + MouthCropConfig.W * vw
    val bottom = top + MouthCropConfig.H * vh
    cropRect.set(left, top, right, bottom)

    holePath.reset()
    holePath.fillType = Path.FillType.EVEN_ODD
    holePath.addRect(0f, 0f, vw, vh, Path.Direction.CW)
    holePath.addRect(cropRect, Path.Direction.CW)
    canvas.drawPath(holePath, dimPaint)

    canvas.drawRect(cropRect, borderPaint)

    val cx = (left + right) * 0.5f
    val cy = (top + bottom) * 0.5f
    for (t in listOf(0.25f, 0.5f, 0.75f)) {
      val gy = top + (bottom - top) * t
      canvas.drawLine(left, gy, right, gy, guidePaint)
      val gx = left + (right - left) * t
      canvas.drawLine(gx, top, gx, bottom, guidePaint)
    }
    canvas.drawLine(cx, top, cx, bottom, crossPaint)
    canvas.drawLine(left, cy, right, cy, crossPaint)

    canvas.drawText(
        MouthCropConfig.formatRatios(),
        left + 6f,
        (top - 8f).coerceAtLeast(labelPaint.textSize),
        labelPaint,
    )
  }
}
