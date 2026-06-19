package com.example.splatbench

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.os.SystemClock
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.tan

/** GLES 3.x EWA Gaussian-splat renderer for the LAM avatar. */
class SplatRenderer(
    private val pack: AvatarPack,
    private val callbacks: Callbacks,
    private val playback: PlaybackController,
    private val frameCache: FrameCache,
    private val prefetchBuilder: InstanceBuilder,
    private val perfStats: PerfStats? = null,
) : GLSurfaceView.Renderer {

    interface Callbacks {
        fun onStatsTick(splats: Int)

        /** Refresh [PlaybackController.audioPositionMs] from the audio master clock. */
        fun onUpdateAudioClock()

        fun onSurfaceReady()

        /** Schedule the next 30fps draw ([deadlineUptimeMs] = uptime millis). */
        fun scheduleNextFrame(deadlineUptimeMs: Long)
    }

    @Volatile var mouthOnly = AppConfig.MOUTH_ONLY
        private set

    @Volatile var fixedFpsEnabled = false
        private set

    private var nextFrameUptimeMs = 0L

    fun setMouthOnly(enabled: Boolean) {
        mouthOnly = enabled
    }

    @Volatile var headBoneEnabled = AppConfig.HEAD_BONE_ENABLED
        private set

    fun setHeadBoneEnabled(enabled: Boolean) {
        headBoneEnabled = enabled
        builder.headBoneEnabled = enabled
        prefetchBuilder.headBoneEnabled = enabled
        if (enabled && pack.hasHeadAnimation) {
            baseReady = false
        }
    }

    @Volatile var photoComposite = AppConfig.PHOTO_COMPOSITE
        private set

    fun setPhotoComposite(enabled: Boolean) {
        photoComposite = enabled
        if (!enabled) {
            mouthScissor = null
        }
    }

    /** Upload [file] to [photoTex] on the GL thread. */
    fun loadPhotoTexture(file: File) {
        val bmp = BitmapFactory.decodeFile(file.absolutePath) ?: return
        val square = cropCenterSquare(bmp)
        if (square !== bmp) bmp.recycle()
        if (photoTex == 0) {
            val tex = IntArray(1)
            GLES30.glGenTextures(1, tex, 0)
            photoTex = tex[0]
        }
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, photoTex)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, square, 0)
        square.recycle()
        photoReady = true
        updateMouthScissor()
    }

    fun reloadPhotoTexture() {
        val file = AppConfig.photoFile()
        if (file.isFile) loadPhotoTexture(file)
    }

    private fun cropCenterSquare(src: Bitmap): Bitmap {
        val w = src.width
        val h = src.height
        if (w == h) return src
        val side = min(w, h)
        val x = (w - side) / 2
        val y = (h - side) / 2
        return Bitmap.createBitmap(src, x, y, side, side)
    }

    private fun updateMouthScissor() {
        if (!photoComposite || !photoReady) {
            mouthScissor = null
            return
        }
        mouthScissor = MouthRegion.screenBBox(
            pack, rot, tv, fy, sq * 0.5f, sq * 0.5f, sq.toFloat(), sq.toFloat(),
        )
    }

    private fun compositeMouthOnly(): Boolean =
        AppConfig.useCompositeMouthOnly(pack)

    private fun usePhotoBackground(): Boolean =
        photoComposite && photoReady

    @Volatile var surfaceReady = false
        private set

    private val buildThreads: Int
        get() = AppConfig.BUILD_THREADS.coerceIn(1, AppConfig.MAX_BUILD_THREADS)

    private var splatProg = 0
    private var quadProg = 0
    private var vao = 0
    private var cornerVbo = 0
    private var instVbo = 0
    private var staticVbo = 0
    private var quadVbo = 0
    private var quadVao = 0
    private var baseFbo = 0
    private var baseTex = 0
    private var photoTex = 0
    private var photoReady = false
    private var mouthScissor: IntArray? = null
    private var uViewport = 0
    private var uTex = 0

    private var surfW = 1
    private var surfH = 1
    private var sq = 1
    private var ox = 0
    private var oy = 0
    private val rot = FloatArray(9)
    private val tv = FloatArray(3)
    private var fy = 0f
    private var fovy = 0.5f

    private val builder = InstanceBuilder(pack)
    private var staticCount = 0
    private var baseReady = false

    private var liveExec: ExecutorService = Executors.newFixedThreadPool(AppConfig.MAX_BUILD_THREADS)
    private val bgR = 0.063f; private val bgG = 0.063f; private val bgB = 0.078f

    private val uploadScratch: ByteBuffer =
        ByteBuffer.allocateDirect(pack.numGaussians * 10 * 4).order(ByteOrder.nativeOrder())

    fun startFixedFps() {
        fixedFpsEnabled = true
        nextFrameUptimeMs = 0L
    }

    fun stopFixedFps() {
        fixedFpsEnabled = false
        nextFrameUptimeMs = 0L
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        splatProg = link(SPLAT_VS, SPLAT_FS)
        quadProg = link(QUAD_VS, QUAD_FS)
        uViewport = GLES30.glGetUniformLocation(splatProg, "viewport")
        uTex = GLES30.glGetUniformLocation(quadProg, "tex")

        val ids = IntArray(4)
        GLES30.glGenBuffers(4, ids, 0)
        cornerVbo = ids[0]; instVbo = ids[1]; staticVbo = ids[2]; quadVbo = ids[3]
        val vaos = IntArray(2)
        GLES30.glGenVertexArrays(2, vaos, 0)
        vao = vaos[0]; quadVao = vaos[1]

        val corners = floatArrayOf(-1f, -1f, 1f, -1f, 1f, 1f, -1f, -1f, 1f, 1f, -1f, 1f)
        uploadStatic(cornerVbo, corners)
        val fs = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        uploadStatic(quadVbo, fs)

        GLES30.glBindVertexArray(quadVao)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, quadVbo)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 8, 0)
        GLES30.glBindVertexArray(0)

        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDisable(GLES30.GL_CULL_FACE)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        surfW = width; surfH = height
        sq = min(width, height)
        ox = (width - sq) / 2
        oy = 0
        setupCamera()
        setupBaseFbo()
        baseReady = false
        if (photoComposite) reloadPhotoTexture()
        updateMouthScissor()
        surfaceReady = true
        callbacks.onSurfaceReady()
    }

    private fun setupCamera() {
        val cam = pack.camera
        val ex = cam[0]; val ey = cam[1]; val ez = cam[2]
        val cxw = cam[3]; val cyw = cam[4]; val czw = cam[5]
        val ux = cam[6]; val uy = cam[7]; val uz = cam[8]
        fovy = cam[9]

        var fxd = (cxw - ex); var fyd = (cyw - ey); var fzd = (czw - ez)
        var nf = sqrt(fxd * fxd + fyd * fyd + fzd * fzd)
        fxd /= nf; fyd /= nf; fzd /= nf
        var sxv = fyd * uz - fzd * uy
        var syv = fzd * ux - fxd * uz
        var szv = fxd * uy - fyd * ux
        val ns = sqrt(sxv * sxv + syv * syv + szv * szv)
        sxv /= ns; syv /= ns; szv /= ns
        val uxv = syv * fzd - szv * fyd
        val uyv = szv * fxd - sxv * fzd
        val uzv = sxv * fyd - syv * fxd

        rot[0] = sxv; rot[1] = syv; rot[2] = szv
        rot[3] = uxv; rot[4] = uyv; rot[5] = uzv
        rot[6] = -fxd; rot[7] = -fyd; rot[8] = -fzd
        tv[0] = -(sxv * ex + syv * ey + szv * ez)
        tv[1] = -(uxv * ex + uyv * ey + uzv * ez)
        tv[2] = (fxd * ex + fyd * ey + fzd * ez)

        fy = (sq * 0.5f) / tan(fovy * 0.5f)
        builder.setView(rot, tv, fy, sq * 0.5f, sq * 0.5f, sq.toFloat(), sq.toFloat())
        syncPrefetchView()
    }

    fun syncPrefetchView() {
        prefetchBuilder.setView(rot, tv, fy, sq * 0.5f, sq * 0.5f, sq.toFloat(), sq.toFloat())
    }

    fun ensureStaticBase() {
        if (!AppConfig.needsStaticGaussianBase(pack)) return
        baseReady = false
        ensureBase()
    }

    fun invalidateStaticBase() {
        baseReady = false
    }

    private fun setupBaseFbo() {
        if (baseFbo != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(baseFbo), 0)
            GLES30.glDeleteTextures(1, intArrayOf(baseTex), 0)
        }
        val tex = IntArray(1); GLES30.glGenTextures(1, tex, 0); baseTex = tex[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, baseTex)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, sq, sq, 0,
            GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        val fbo = IntArray(1); GLES30.glGenFramebuffers(1, fbo, 0); baseFbo = fbo[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, baseFbo)
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, baseTex, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    private fun ensureBase() {
        if (baseReady) return
        val zero = FloatArray(pack.numMorphs)
        staticCount = build(zero, staticIndices(), 0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, staticVbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, staticCount * 40, builder.instanceBytes, GLES30.GL_STATIC_DRAW)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, baseFbo)
        GLES30.glViewport(0, 0, sq, sq)
        GLES30.glClearColor(bgR, bgG, bgB, 1f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        drawSplats(staticVbo, staticCount)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        baseReady = true
    }

    private var staticIdxCache: IntArray? = null
    private fun staticIndices(): IntArray {
        staticIdxCache?.let { return it }
        val n = pack.numGaussians
        val cnt = n - pack.dynamicIndices.size
        val out = IntArray(cnt); var k = 0
        for (i in 0 until n) if (pack.dynamic[i].toInt() == 0) out[k++] = i
        staticIdxCache = out
        return out
    }

    private fun build(weights: FloatArray, indices: IntArray, frameIndex: Int = 0): Int =
        builder.build(weights, indices, liveExec, buildThreads, frameIndex)

    override fun onDrawFrame(gl: GL10?) {
        val frameStartUptime = SystemClock.uptimeMillis()
        val state = playback.state
        val lipSyncActive = state == PlaybackState.PLAYING

        if (lipSyncActive) {
            callbacks.onUpdateAudioClock()
        }

        val frameIdx = when (state) {
            PlaybackState.PLAYING -> playback.currentFrameIndex(pack)
            PlaybackState.DONE -> playback.currentFrameIndex(pack)
            else -> 0
        }

        val cached = when {
            lipSyncActive -> frameCache.get(frameIdx)
            state == PlaybackState.DONE -> frameCache.get(frameIdx) ?: frameCache.get(0)
            else -> frameCache.get(0)
        }
        if (cached != null && AppConfig.needsStaticGaussianBase(pack)) ensureBase()

        val tGeom = System.nanoTime()
        val instSize = if (cached != null) uploadInstance(cached) else 0
        perfStats?.addGeom((System.nanoTime() - tGeom) / 1_000_000.0)

        val tDraw = System.nanoTime()
        if (cached != null) {
            drawScene(cached, instSize)
        } else if (usePhotoBackground()) {
            drawPhotoOnly()
        } else {
            renderBackgroundOnly()
        }
        GLES30.glFinish()
        perfStats?.addDraw((System.nanoTime() - tDraw) / 1_000_000.0)

        callbacks.onStatsTick(cached?.instanceCount ?: 0)

        if (fixedFpsEnabled && lipSyncActive) {
            scheduleNextFrameFromStart(frameStartUptime)
        }
    }

    /** Next frame deadline = previous deadline + 1/30s (not after draw completes). */
    private fun scheduleNextFrameFromStart(frameStartUptimeMs: Long) {
        val periodMs = AppConfig.framePeriodNs / 1_000_000L
        if (nextFrameUptimeMs == 0L) {
            nextFrameUptimeMs = frameStartUptimeMs + periodMs
        } else {
            nextFrameUptimeMs += periodMs
        }
        val now = SystemClock.uptimeMillis()
        if (nextFrameUptimeMs < now) {
            nextFrameUptimeMs = now
        }
        callbacks.scheduleNextFrame(nextFrameUptimeMs)
    }

    private fun renderBackgroundOnly() {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, surfW, surfH)
        GLES30.glClearColor(bgR, bgG, bgB, 1f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
    }

    private fun uploadInstance(cached: CachedFrame): Int {
        val size = cached.instanceCount * 40
        if (size > 0) {
            uploadScratch.clear()
            uploadScratch.put(cached.bytes, 0, size)
            uploadScratch.position(0)
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, instVbo)
            GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, size, uploadScratch, GLES30.GL_DYNAMIC_DRAW)
        }
        return size
    }

    private fun drawPhotoOnly() {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, surfW, surfH)
        GLES30.glClearColor(bgR, bgG, bgB, 1f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glViewport(ox, oy, sq, sq)
        drawQuad(photoTex)
    }

    private fun drawScene(cached: CachedFrame, instSize: Int) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, surfW, surfH)
        GLES30.glClearColor(bgR, bgG, bgB, 1f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glViewport(ox, oy, sq, sq)
        when {
            usePhotoBackground() -> drawQuad(photoTex)
            compositeMouthOnly() -> {
                ensureBase()
                drawQuad(baseTex)
            }
        }
        if (instSize > 0) {
            val sc = if (usePhotoBackground()) mouthScissor else null
            drawSplats(instVbo, cached.instanceCount, sc)
        }
    }

    private fun drawSplats(vbo: Int, count: Int, scissorTopLeft: IntArray? = null) {
        if (count <= 0) return
        if (scissorTopLeft != null) {
            val x = ox + scissorTopLeft[0]
            val yTop = scissorTopLeft[1]
            val w = scissorTopLeft[2]
            val h = scissorTopLeft[3]
            val glY = surfH - oy - yTop - h
            GLES30.glEnable(GLES30.GL_SCISSOR_TEST)
            GLES30.glScissor(x, glY, w, h)
        }
        GLES30.glUseProgram(splatProg)
        GLES30.glUniform2f(uViewport, sq.toFloat(), sq.toFloat())
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA)

        GLES30.glBindVertexArray(vao)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, cornerVbo)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 8, 0)
        GLES30.glVertexAttribDivisor(0, 0)

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)
        bindInstanceAttr(1, 2, 0)
        bindInstanceAttr(2, 1, 8)
        bindInstanceAttr(3, 3, 12)
        bindInstanceAttr(4, 3, 24)
        bindInstanceAttr(5, 1, 36)

        GLES30.glDrawArraysInstanced(GLES30.GL_TRIANGLES, 0, 6, count)
        GLES30.glBindVertexArray(0)
        GLES30.glDisable(GLES30.GL_BLEND)
        if (scissorTopLeft != null) {
            GLES30.glDisable(GLES30.GL_SCISSOR_TEST)
        }
    }

    private fun bindInstanceAttr(loc: Int, size: Int, offset: Int) {
        GLES30.glEnableVertexAttribArray(loc)
        GLES30.glVertexAttribPointer(loc, size, GLES30.GL_FLOAT, false, 40, offset)
        GLES30.glVertexAttribDivisor(loc, 1)
    }

    private fun drawQuad(tex: Int) {
        GLES30.glUseProgram(quadProg)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, tex)
        GLES30.glUniform1i(uTex, 0)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glBindVertexArray(quadVao)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glBindVertexArray(0)
    }

    private fun uploadStatic(vbo: Int, data: FloatArray) {
        val bb = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
        bb.asFloatBuffer().put(data); bb.position(0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, data.size * 4, bb, GLES30.GL_STATIC_DRAW)
    }

    private fun link(vs: String, fs: String): Int {
        val v = compile(GLES30.GL_VERTEX_SHADER, vs)
        val f = compile(GLES30.GL_FRAGMENT_SHADER, fs)
        val p = GLES30.glCreateProgram()
        GLES30.glAttachShader(p, v); GLES30.glAttachShader(p, f)
        GLES30.glLinkProgram(p)
        val ok = IntArray(1); GLES30.glGetProgramiv(p, GLES30.GL_LINK_STATUS, ok, 0)
        check(ok[0] != 0) { "link failed: " + GLES30.glGetProgramInfoLog(p) }
        GLES30.glDeleteShader(v); GLES30.glDeleteShader(f)
        return p
    }

    private fun compile(type: Int, src: String): Int {
        val s = GLES30.glCreateShader(type)
        GLES30.glShaderSource(s, src)
        GLES30.glCompileShader(s)
        val ok = IntArray(1); GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
        check(ok[0] != 0) { "compile failed: " + GLES30.glGetShaderInfoLog(s) }
        return s
    }

    companion object {
        private const val SPLAT_VS = """#version 300 es
            precision highp float;
            layout(location=0) in vec2 corner;
            layout(location=1) in vec2 i_center;
            layout(location=2) in float i_radius;
            layout(location=3) in vec3 i_conic;
            layout(location=4) in vec3 i_color;
            layout(location=5) in float i_alpha;
            uniform vec2 viewport;
            out vec2 v_d;
            out vec3 v_conic;
            out vec3 v_color;
            out float v_alpha;
            void main() {
                vec2 off = corner * i_radius;
                vec2 pix = i_center + off;
                vec2 ndc = vec2(pix.x / viewport.x * 2.0 - 1.0, 1.0 - pix.y / viewport.y * 2.0);
                gl_Position = vec4(ndc, 0.0, 1.0);
                v_d = off;
                v_conic = i_conic;
                v_color = i_color;
                v_alpha = i_alpha;
            }"""

        private const val SPLAT_FS = """#version 300 es
            precision highp float;
            in vec2 v_d;
            in vec3 v_conic;
            in vec3 v_color;
            in float v_alpha;
            out vec4 f_color;
            void main() {
                float power = -0.5 * (v_conic.x * v_d.x * v_d.x
                                    + 2.0 * v_conic.y * v_d.x * v_d.y
                                    + v_conic.z * v_d.y * v_d.y);
                if (power > 0.0) discard;
                float a = v_alpha * exp(power);
                if (a < 0.00392) discard;
                f_color = vec4(v_color * a, a);
            }"""

        private const val QUAD_VS = """#version 300 es
            precision highp float;
            layout(location=0) in vec2 pos;
            out vec2 uv;
            void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }"""

        private const val QUAD_FS = """#version 300 es
            precision mediump float;
            in vec2 uv;
            uniform sampler2D tex;
            out vec4 c;
            void main() { c = vec4(texture(tex, uv).rgb, 1.0); }"""
    }
}
