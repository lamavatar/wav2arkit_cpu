package com.example.splatbench

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/** Alignment guide for photo mouth overlay (pivot + transformed reference rect). */
class PhotoOverlayGuide @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

  private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.argb(220, 0, 229, 255)
    style = Paint.Style.STROKE
    strokeWidth = 2.5f
  }
  private val pivotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.argb(230, 255, 64, 129)
    style = Paint.Style.STROKE
    strokeWidth = 2f
  }
  private val guidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.argb(160, 255, 193, 7)
    style = Paint.Style.STROKE
    strokeWidth = 1.5f
    pathEffect = DashPathEffect(floatArrayOf(10f, 8f), 0f)
  }
  private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.argb(220, 255, 255, 255)
    textSize = 26f
  }

  var guideEnabled: Boolean = false
    set(value) {
      field = value
      visibility = if (value) VISIBLE else GONE
      invalidate()
    }

  fun syncFromConfig() {
    invalidate()
  }

  private fun transform(px: Float, py: Float, vw: Float, vh: Float): Pair<Float, Float> {
    val pivotX = MouthPhotoOverlayConfig.PIVOT_X * vw
    val pivotY = MouthPhotoOverlayConfig.PIVOT_Y * vh
    val offX = MouthPhotoOverlayConfig.OFFSET_X * vw
    val offY = MouthPhotoOverlayConfig.OFFSET_Y * vh
    val s = MouthPhotoOverlayConfig.SCALE
    val tx = (px - pivotX) * s + pivotX + offX
    val ty = (py - pivotY) * s + pivotY + offY
    return tx to ty
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    if (!guideEnabled) return

    val vw = width.toFloat()
    val vh = height.toFloat()
    if (vw <= 0f || vh <= 0f) return

    val refL = 0f
    val refT = 0f
    val refR = vw
    val refB = vh

    val tl = transform(refL, refT, vw, vh)
    val tr = transform(refR, refT, vw, vh)
    val br = transform(refR, refB, vw, vh)
    val bl = transform(refL, refB, vw, vh)

    canvas.drawLine(tl.first, tl.second, tr.first, tr.second, borderPaint)
    canvas.drawLine(tr.first, tr.second, br.first, br.second, borderPaint)
    canvas.drawLine(br.first, br.second, bl.first, bl.second, borderPaint)
    canvas.drawLine(bl.first, bl.second, tl.first, tl.second, borderPaint)

    val cx = (tl.first + tr.first + br.first + bl.first) * 0.25f
    val cy = (tl.second + tr.second + br.second + bl.second) * 0.25f
    val half = minOf(vw, vh) * 0.04f
    canvas.drawLine(cx - half, cy, cx + half, cy, pivotPaint)
    canvas.drawLine(cx, cy - half, cx, cy + half, pivotPaint)

    val pivotX = MouthPhotoOverlayConfig.PIVOT_X * vw
    val pivotY = MouthPhotoOverlayConfig.PIVOT_Y * vh
    canvas.drawLine(pivotX - half * 0.7f, pivotY, pivotX + half * 0.7f, pivotY, guidePaint)
    canvas.drawLine(pivotX, pivotY - half * 0.7f, pivotX, pivotY + half * 0.7f, guidePaint)

    canvas.drawText(MouthPhotoOverlayConfig.format(), 8f, vh - 12f, labelPaint)
  }
}
