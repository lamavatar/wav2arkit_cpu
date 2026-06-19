package com.example.splatbench

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import com.example.splatbench.databinding.ActivityMainBinding
import kotlin.concurrent.thread
import kotlin.math.ceil

class MainActivity : AppCompatActivity(), SplatRenderer.Callbacks {

    private lateinit var binding: ActivityMainBinding
    private var glView: GLSurfaceView? = null
    private var renderer: SplatRenderer? = null
    private var pack: AvatarPack? = null

    private val playback = PlaybackController()
    private val frameCache = FrameCache()
    private var prefetcher: FramePrefetcher? = null
    private val perfStats = PerfStats()
    private var lastSplats = 0

    /** Uptime when audio ingest starts ([startPlayback]). */
    private var playbackPipelineStartUptimeMs = 0L

    /** Audio-read start → first audio output (locked when playback begins). */
    private var measuredStartupLatencyMs: Long? = null

    private val session = PlaybackSession()
    private val expressionBuffer = ExpressionBuffer()
    private lateinit var audioInput: AudioInputController

    @Volatile private var onnx: Wav2ArkitOnnx? = null
    @Volatile private var onnxError: String? = null
    private var pipeline: OnnxExpressionPipeline? = null

    private var mediaPlayer: MediaPlayer? = null
    private var audioUri: Uri? = null

    private var currentMode = AppConfig.AudioInputMode.FILE
    private var fileDurationMs = 0
    private var micClockStartNanos = 0L
    private var micGranted = false

    private val mainHandler = Handler(Looper.getMainLooper())
    private val frameRenderRunnable = Runnable { glView?.requestRender() }
    private val statRefreshRunnable = object : Runnable {
        override fun run() {
            refreshStats(lastSplats)
            mainHandler.postDelayed(this, AppConfig.STAT_REFRESH_MS)
        }
    }

    private val pickAudio = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        try {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION,
            )
        } catch (_: SecurityException) {
        }
        releaseMediaPlayer()
        audioUri = uri
        playback.state = PlaybackState.READY
        binding.audioLabel.text = "Audio: ${uri.lastPathSegment ?: uri}"
        updateControls()
        rebuildPreview()
    }

    private val requestMicPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            micGranted = granted
            if (!granted) {
                Toast.makeText(this, "Microphone permission denied", Toast.LENGTH_SHORT).show()
            }
            updateControls()
        }

    private val requestStoragePermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) beginDataLoad() else showStorageAccessHint()
        }

    private val requestAllFilesAccess =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
            if (hasAvatarTalkAccess()) beginDataLoad() else showStorageAccessHint()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, true)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        audioInput = AudioInputController(applicationContext)
        currentMode = AppConfig.effectiveAudioMode()

        binding.pickAudioButton.isEnabled = false
        setupModeAndThreadControls()
        updateControls()

        binding.pickAudioButton.setOnClickListener {
            pickAudio.launch(arrayOf("audio/*"))
        }
        binding.playButton.setOnClickListener { togglePlayback() }

        if (currentMode == AppConfig.AudioInputMode.MIC) {
            binding.pickAudioButton.visibility = View.GONE
            micGranted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.RECORD_AUDIO,
            ) == PackageManager.PERMISSION_GRANTED
            if (!micGranted) requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
        }

        ensureStorageAccessAndLoad()
        mainHandler.post(statRefreshRunnable)
    }

    private fun setupModeAndThreadControls() {
        setupRenderModeSpinner()

        binding.headBoneSwitch.isChecked = AppConfig.HEAD_BONE_ENABLED
        binding.headBoneSwitch.setOnCheckedChangeListener { _, checked ->
            if (checked == AppConfig.HEAD_BONE_ENABLED) return@setOnCheckedChangeListener
            if (checked && pack?.hasHeadAnimation != true) {
                binding.headBoneSwitch.isChecked = false
                Toast.makeText(this, "No head animation in .splat (re-bake with animation.glb)", Toast.LENGTH_LONG).show()
                return@setOnCheckedChangeListener
            }
            if (checked && !AppConfig.allowsHeadBone()) {
                binding.headBoneSwitch.isChecked = false
                Toast.makeText(this, "Head bone is not available in this render mode", Toast.LENGTH_SHORT).show()
                return@setOnCheckedChangeListener
            }
            AppConfig.HEAD_BONE_ENABLED = checked
            syncHeadBoneToBuilders()
            perfStats.reset()
            rebuildPreview()
        }

        val options = AppConfig.THREAD_OPTIONS
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            options.map { "$it" },
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.threadSpinner.adapter = adapter
        val sel = options.indexOf(AppConfig.BUILD_THREADS).coerceAtLeast(0)
        binding.threadSpinner.setSelection(sel)
        binding.threadSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val n = options[position]
                if (n == AppConfig.BUILD_THREADS) return
                AppConfig.BUILD_THREADS = n
                perfStats.reset()
                if (playback.state != PlaybackState.WARMING_UP &&
                    playback.state != PlaybackState.PLAYING
                ) {
                    rebuildPreview()
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private var renderModeSpinnerBusy = false
    private var previewRebuildPending = false

    private fun setupRenderModeSpinner() {
        val modes = RenderMode.entries
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            modes.map { it.label },
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.renderModeSpinner.adapter = adapter
        val sel = modes.indexOf(AppConfig.RENDER_MODE).coerceAtLeast(0)
        renderModeSpinnerBusy = true
        binding.renderModeSpinner.setSelection(sel)
        renderModeSpinnerBusy = false

        binding.renderModeSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (renderModeSpinnerBusy) return
                val mode = modes[position]
                if (mode == AppConfig.RENDER_MODE) return

                if (mode == RenderMode.MOUTH_ON_PHOTO && !AppConfig.photoFile().isFile) {
                    renderModeSpinnerBusy = true
                    binding.renderModeSpinner.setSelection(modes.indexOf(AppConfig.RENDER_MODE).coerceAtLeast(0))
                    renderModeSpinnerBusy = false
                    Toast.makeText(
                        this@MainActivity,
                        "Photo not found: ${AppConfig.photoPathHint()}",
                        Toast.LENGTH_LONG,
                    ).show()
                    return
                }

                applyRenderMode(mode)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun applyRenderMode(mode: RenderMode) {
        AppConfig.RENDER_MODE = mode
        if (!mode.allowsHeadBone() && AppConfig.HEAD_BONE_ENABLED) {
            AppConfig.HEAD_BONE_ENABLED = false
            binding.headBoneSwitch.isChecked = false
            syncHeadBoneToBuilders()
        }
        renderer?.setRenderMode(mode)
        updateHeadBoneSwitchState(pack)
        glView?.queueEvent {
            if (AppConfig.PHOTO_COMPOSITE) renderer?.reloadPhotoTexture()
                runOnUiThread {
                    perfStats.reset()
                    rebuildPreview()
                }
        }
    }

    private fun hasAvatarTalkAccess(): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            return Environment.isExternalStorageManager()
        }
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.READ_EXTERNAL_STORAGE,
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun ensureStorageAccessAndLoad() {
        if (hasAvatarTalkAccess()) {
            beginDataLoad()
            return
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                data = Uri.parse("package:$packageName")
            }
            try {
                requestAllFilesAccess.launch(intent)
            } catch (_: Exception) {
                requestAllFilesAccess.launch(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
            }
        } else {
            requestStoragePermission.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    private fun showStorageAccessHint() {
        Toast.makeText(
            this,
            "Allow file access, then copy model, splat & photo to ${AppConfig.avatarTalkDir()}",
            Toast.LENGTH_LONG,
        ).show()
    }

    private fun beginDataLoad() {
        AppConfig.avatarTalkDir().mkdirs()
        thread(name = "data-load") {
            try {
                val p = AvatarPack.load()
                runOnUiThread { setupGl(p) }
                try {
                    onnx = Wav2ArkitOnnx.load()
                } catch (e: Exception) {
                    onnxError = e.message ?: "onnx load failed"
                    runOnUiThread {
                        Toast.makeText(this, onnxError, Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(this, e.message ?: "avatar load failed", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun setupGl(p: AvatarPack) {
        pack = p
        playback.prebufferSeconds = AppConfig.PREBUFFER_SECONDS
        val pf = FramePrefetcher(
            pack = p,
            cache = frameCache,
            controller = playback,
            expressionBuffer = expressionBuffer,
            session = session,
            perfStats = perfStats,
        )
        prefetcher = pf
        syncHeadBoneToBuilders()
        updateHeadBoneSwitchState(p)
        updateRenderModeSpinnerState()

        val view = binding.glView
        view.setEGLContextClientVersion(3)
        val r = SplatRenderer(p, this, playback, frameCache, pf.builder, perfStats)
        renderer = r
        r.setRenderMode(AppConfig.RENDER_MODE)
        view.setRenderer(r)
        view.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        glView = view
        view.queueEvent {
            if (AppConfig.PHOTO_COMPOSITE) r.reloadPhotoTexture()
        }
        view.onResume()

        binding.pickAudioButton.isEnabled = currentMode == AppConfig.AudioInputMode.FILE
        playback.state = PlaybackState.IDLE
        maybeRebuildIdlePreview()
        updateControls()
    }

    private fun syncHeadBoneToBuilders() {
        val enabled = AppConfig.HEAD_BONE_ENABLED
        prefetcher?.builder?.headBoneEnabled = enabled
        renderer?.setHeadBoneEnabled(enabled)
    }

    private fun updateHeadBoneSwitchState(p: AvatarPack?) {
        val has = p?.hasHeadAnimation == true
        if (!has) {
            AppConfig.HEAD_BONE_ENABLED = false
            binding.headBoneSwitch.isChecked = false
        }
        val busy = playback.state == PlaybackState.WARMING_UP || playback.state == PlaybackState.PLAYING
        binding.headBoneSwitch.isEnabled = has && !busy && AppConfig.allowsHeadBone()
        binding.headBoneSwitch.alpha = if (has && AppConfig.allowsHeadBone()) 1f else 0.45f
    }

    private fun updateRenderModeSpinnerState() {
        val modes = RenderMode.entries
        if (!AppConfig.photoFile().isFile && AppConfig.RENDER_MODE == RenderMode.MOUTH_ON_PHOTO) {
            AppConfig.RENDER_MODE = RenderMode.MOUTH_ON_STATIC
            renderModeSpinnerBusy = true
            binding.renderModeSpinner.setSelection(modes.indexOf(AppConfig.RENDER_MODE).coerceAtLeast(0))
            renderModeSpinnerBusy = false
            renderer?.setRenderMode(AppConfig.RENDER_MODE)
        }
    }

    /**
     * ready. Safe to call from setupGl, onSurfaceReady, or onResume — whichever
     * runs last will succeed (fixes first-launch race).
     */
    private fun maybeRebuildIdlePreview() {
        if (prefetcher == null || pack == null) return
        if (renderer?.surfaceReady != true) return
        val state = playback.state
        if (state != PlaybackState.IDLE && state != PlaybackState.READY) return

        if (frameCache.has(0)) {
            glView?.queueEvent {
                if (AppConfig.needsStaticGaussianBase(pack ?: return@queueEvent)) renderer?.ensureStaticBase()
                runOnUiThread { glView?.requestRender() }
            }
            return
        }
        if (previewRebuildPending) return
        previewRebuildPending = true
        rebuildPreview()
    }

    private fun togglePlayback() {
        when (playback.state) {
            PlaybackState.WARMING_UP, PlaybackState.PLAYING -> stopPlayback()
            else -> startPlayback()
        }
    }

    private fun rebuildPreview() {
        val pf = prefetcher ?: return
        val mouthOnly = AppConfig.useMouthOnlyIndices(pack ?: return)
        previewRebuildPending = true
        pf.cancel()
        frameCache.clear()
        pf.resetCancel()
        glView?.queueEvent {
            renderer?.syncPrefetchView()
            if (AppConfig.needsStaticGaussianBase(pack ?: return@queueEvent)) renderer?.ensureStaticBase()
            runOnUiThread {
                pf.buildNeutralPreview(mouthOnly, previewListener)
            }
        }
    }

    override fun onSurfaceReady() {
        runOnUiThread { maybeRebuildIdlePreview() }
    }

    private val previewListener = object : FramePrefetcher.Listener {
        override fun onWarmupComplete() {
            previewRebuildPending = false
            runOnUiThread {
                val cached = frameCache.get(0)
                lastSplats = cached?.instanceCount ?: splatCount(pack)
                glView?.queueEvent {
                    if (AppConfig.needsStaticGaussianBase(pack ?: return@queueEvent)) renderer?.ensureStaticBase()
                    runOnUiThread { glView?.requestRender() }
                }
            }
        }

        override fun onError(message: String) {
            previewRebuildPending = false
            runOnUiThread { Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show() }
        }
    }

    private fun startPlayback() {
        val pf = prefetcher ?: return
        val pk = pack ?: return
        val model = onnx
        if (model == null) {
            Toast.makeText(this, onnxError ?: "Model still loading…", Toast.LENGTH_SHORT).show()
            return
        }
        if (currentMode == AppConfig.AudioInputMode.FILE && audioUri == null) {
            Toast.makeText(this, "Pick an audio file first", Toast.LENGTH_SHORT).show()
            return
        }
        if (currentMode == AppConfig.AudioInputMode.MIC && !micGranted) {
            requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
            return
        }

        val mouthOnly = AppConfig.useMouthOnlyIndices(pk)
        perfStats.reset()
        session.reset()
        measuredStartupLatencyMs = null
        playback.prebufferSeconds = AppConfig.PREBUFFER_SECONDS
        playback.resetAudioPosition()
        playback.state = PlaybackState.WARMING_UP

        pf.cancel()
        pf.stopBuild()
        frameCache.clear()
        expressionBuffer.clear()
        audioInput.buffer.reset()

        playbackPipelineStartUptimeMs = SystemClock.uptimeMillis()
        when (currentMode) {
            AppConfig.AudioInputMode.FILE ->
                fileDurationMs = audioInput.startFile(audioUri!!, session)
            AppConfig.AudioInputMode.MIC -> audioInput.startMic(session)
        }

        val pipe = OnnxExpressionPipeline(model, audioInput.buffer, expressionBuffer, session)
        pipeline = pipe
        pipe.start()

        playback.expressionSource = OnnxExpressionSource(pk, expressionBuffer) { liveFrameCount() }
        updateControls()

        glView?.queueEvent {
            renderer?.syncPrefetchView()
            if (AppConfig.needsStaticGaussianBase(pk)) renderer?.ensureStaticBase()
            runOnUiThread {
                pf.startSession(mouthOnly, sessionListener)
            }
        }
    }

    private fun liveFrameCount(): Int =
        when (currentMode) {
            AppConfig.AudioInputMode.FILE -> {
                val inferred = expressionBuffer.count
                val fromDur = ceil(fileDurationMs * AppConfig.EXPRESSION_FPS / 1000.0).toInt()
                maxOf(inferred, fromDur).coerceAtLeast(1)
            }
            AppConfig.AudioInputMode.MIC -> expressionBuffer.count.coerceAtLeast(1)
        }

    private val sessionListener = object : FramePrefetcher.Listener {
        override fun onWarmupProgress(built: Int, total: Int) {
            val last = frameCache.get(built - 1)
            runOnUiThread {
                lastSplats = last?.instanceCount ?: splatCount(pack)
                glView?.requestRender()
            }
        }

        override fun onWarmupComplete() {
            runOnUiThread { beginAudioAndPlayback() }
        }

        override fun onError(message: String) {
            runOnUiThread {
                finishSession(resetToReady = true)
                Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun beginAudioAndPlayback() {
        if (playback.state != PlaybackState.WARMING_UP) return

        playback.resetAudioPosition()

        if (currentMode == AppConfig.AudioInputMode.MIC) {
            recordStartupLatency()
            micClockStartNanos = System.nanoTime()
            playback.state = PlaybackState.PLAYING
            renderer?.startFixedFps()
            updateControls()
            glView?.requestRender()
            return
        }

        val uri = audioUri ?: return
        releaseMediaPlayer()
        val mp = MediaPlayer()
        mediaPlayer = mp
        try {
            mp.setDataSource(this, uri)
            mp.setOnCompletionListener {
                runOnUiThread { onPlaybackComplete() }
            }
            mp.setOnPreparedListener {
                playback.resetAudioPosition()
                recordStartupLatency()
                mp.start()
                playback.state = PlaybackState.PLAYING
                renderer?.startFixedFps()
                updateControls()
                glView?.requestRender()
            }
            mp.prepareAsync()
        } catch (e: Exception) {
            finishSession(resetToReady = true)
            Toast.makeText(this, e.message ?: "audio failed", Toast.LENGTH_SHORT).show()
        }
    }

    private fun onPlaybackComplete() {
        finishSession(resetToReady = true)
    }

  override fun onUpdateAudioClock() {
        when (currentMode) {
            AppConfig.AudioInputMode.FILE -> {
                val mp = mediaPlayer
                if (mp != null) {
                    try {
                        playback.updateAudioPositionMs(mp.currentPosition)
                    } catch (_: IllegalStateException) {
                    }
                }
            }
            AppConfig.AudioInputMode.MIC -> {
                val ms = ((System.nanoTime() - micClockStartNanos) / 1_000_000L).toInt()
                playback.updateAudioPositionMs(ms)
            }
        }
    }

    override fun scheduleNextFrame(deadlineUptimeMs: Long) {
        mainHandler.removeCallbacks(frameRenderRunnable)
        val delay = deadlineUptimeMs - SystemClock.uptimeMillis()
        if (delay <= 0L) {
            mainHandler.post(frameRenderRunnable)
        } else {
            mainHandler.postAtTime(frameRenderRunnable, deadlineUptimeMs)
        }
    }

    private fun stopPlayback() {
        finishSession(resetToReady = true)
    }

    /** Stop all worker threads and reset buffers; show neutral preview. */
    private fun finishSession(resetToReady: Boolean) {
        session.requestStop()
        prefetcher?.cancel()
        prefetcher?.stopBuild()
        pipeline?.stop()
        pipeline?.resetContext()
        pipeline = null
        audioInput.stop()
        releaseMediaPlayer()

        playback.expressionSource = null
        playback.resetAudioPosition()
        playbackPipelineStartUptimeMs = 0L
        renderer?.stopFixedFps()
        mainHandler.removeCallbacks(frameRenderRunnable)

        expressionBuffer.clear()
        frameCache.clear()
        audioInput.buffer.reset()

        playback.state = when {
            resetToReady && currentMode == AppConfig.AudioInputMode.FILE && audioUri != null ->
                PlaybackState.READY
            resetToReady -> PlaybackState.IDLE
            else -> PlaybackState.IDLE
        }

        perfStats.reset()
        updateControls()
        rebuildPreview()
    }

    private fun releaseMediaPlayer() {
        mediaPlayer?.let {
            try {
                if (it.isPlaying) it.stop()
            } catch (_: IllegalStateException) {
            }
            it.release()
        }
        mediaPlayer = null
    }

    private fun updateControls() {
        val state = playback.state
        val busy = state == PlaybackState.WARMING_UP || state == PlaybackState.PLAYING
        binding.playButton.text = if (busy) "Stop" else "Start"
        val canStart = when (currentMode) {
            AppConfig.AudioInputMode.FILE -> audioUri != null
            AppConfig.AudioInputMode.MIC -> true
        }
        binding.playButton.isEnabled = busy || canStart
        binding.pickAudioButton.isEnabled = !busy && currentMode == AppConfig.AudioInputMode.FILE
        binding.renderModeSpinner.isEnabled = !busy
        binding.renderModeSpinner.alpha = if (busy) 0.45f else 1f
        binding.headBoneSwitch.isEnabled = !busy && pack?.hasHeadAnimation == true && AppConfig.allowsHeadBone()
        binding.threadSpinner.isEnabled = !busy
    }

    override fun onStatsTick(splats: Int) {
        lastSplats = splats
    }

    private fun recordStartupLatency() {
        if (measuredStartupLatencyMs != null) return
        val start = playbackPipelineStartUptimeMs
        if (start <= 0L) return
        measuredStartupLatencyMs = (SystemClock.uptimeMillis() - start).coerceAtLeast(0L)
    }

    private fun startupLatencyText(): String {
        measuredStartupLatencyMs?.let { return "${it}ms" }
        if (playback.state == PlaybackState.WARMING_UP && playbackPipelineStartUptimeMs > 0L) {
            val elapsed = SystemClock.uptimeMillis() - playbackPipelineStartUptimeMs
            return "${elapsed}ms…"
        }
        return "—"
    }

    private fun refreshStats(splats: Int) {
        lastSplats = splats
        val p = pack
        val modeLabel = AppConfig.RENDER_MODE.label
        val mode = if (AppConfig.HEAD_BONE_ENABLED && p?.hasHeadAnimation == true && AppConfig.allowsHeadBone()) {
            "$modeLabel+head"
        } else {
            modeLabel
        }
        val head = if (p?.hasHeadAnimation == true) " head:${p.headAnimName}" else ""
        val state = playback.state.name
        val renderSq = renderer?.renderSquarePx ?: 0
        val renderRes = if (renderSq > 0) "${renderSq}x${renderSq}" else "—"
        binding.statLine1.text =
            "$mode | $state | threads=${AppConfig.BUILD_THREADS} | splats=$splats$head | $renderRes"
        binding.statLine2.text = String.format(
            "infer: %.1f ms | post: %.1f ms | build: %.2f | geom: %.2f | draw: %.2f ms",
            pipeline?.lastInferMs ?: 0.0,
            pipeline?.lastPostprocessMs ?: 0.0,
            perfStats.buildAvgMs, perfStats.geomAvgMs, perfStats.drawAvgMs,
        )
        binding.statLine3.text = String.format(
            "latency: %s | audioBuf: %.1fs | expr: %d | geomCache: %d",
            startupLatencyText(),
            audioInput.bufferSeconds(),
            expressionBuffer.count,
            frameCache.size(),
        )
        binding.statLine4.text = statLine4()
    }

    private fun statLine4(): String {
        val src = currentMode.name
        val ep = onnx?.activeEp ?: "—"
        return String.format(
            "src:%s ep:%s audioPos:%dms",
            src, ep, playback.audioPositionMs,
        )
    }

    private fun splatCount(p: AvatarPack?): Int {
        if (p == null) return 0
        return if (AppConfig.useMouthOnlyIndices(p)) p.dynamicIndices.size else p.numGaussians
    }

    override fun onResume() {
        super.onResume()
        glView?.onResume()
        when (playback.state) {
            PlaybackState.PLAYING -> {
                if (currentMode == AppConfig.AudioInputMode.FILE) mediaPlayer?.start()
                renderer?.startFixedFps()
                glView?.requestRender()
            }
            PlaybackState.IDLE, PlaybackState.READY -> maybeRebuildIdlePreview()
            else -> {}
        }
    }

    override fun onPause() {
        super.onPause()
        if (playback.state == PlaybackState.PLAYING) {
            if (currentMode == AppConfig.AudioInputMode.FILE) mediaPlayer?.pause()
            renderer?.stopFixedFps()
            mainHandler.removeCallbacks(frameRenderRunnable)
        }
        glView?.onPause()
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(statRefreshRunnable)
        finishSession(resetToReady = false)
        prefetcher?.shutdown()
        onnx?.close()
        super.onDestroy()
    }
}
