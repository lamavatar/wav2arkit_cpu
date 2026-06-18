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
    private var lastStatUpdateMs = 0L

    private var mediaPlayer: MediaPlayer? = null
    private var audioUri: Uri? = null

    // ONNX lip-sync pipeline
    private lateinit var audioInput: AudioInputController
    private val expressionBuffer = ExpressionBuffer()
    @Volatile private var onnx: Wav2ArkitOnnx? = null
    @Volatile private var onnxError: String? = null
    private var pipeline: OnnxExpressionPipeline? = null

    private var currentMode = AppConfig.AudioInputMode.FILE
    private var fileDurationMs = 0
    private var micClockStartNanos = 0L
    private var micGranted = false

    private val mainHandler = Handler(Looper.getMainLooper())
    private val positionRunnable = object : Runnable {
        override fun run() {
            if (playback.state == PlaybackState.PLAYING) {
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
                mainHandler.postDelayed(this, 16L)
            }
        }
    }

    private val pickAudio = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        try {
            contentResolver.takePersistableUriPermission(
                uri,
                android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION,
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
        refreshStats(splats = 0)
        updateControls()

        binding.pickAudioButton.setOnClickListener {
            pickAudio.launch(arrayOf("audio/*"))
        }
        binding.playButton.setOnClickListener { togglePlayback() }

        // Mic UI / permission only when enabled.
        if (currentMode == AppConfig.AudioInputMode.MIC) {
            binding.pickAudioButton.visibility = View.GONE
            micGranted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.RECORD_AUDIO,
            ) == PackageManager.PERMISSION_GRANTED
            if (!micGranted) requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
        }

        ensureStorageAccessAndLoad()
    }

    private fun setupModeAndThreadControls() {
        binding.mouthOnlySwitch.isChecked = AppConfig.MOUTH_ONLY
        binding.mouthOnlySwitch.setOnCheckedChangeListener { _, checked ->
            if (checked == AppConfig.MOUTH_ONLY) return@setOnCheckedChangeListener
            AppConfig.MOUTH_ONLY = checked
            renderer?.setMouthOnly(checked)
            perfStats.reset()
            rebuildPreview()
            refreshStats(splatCount(pack))
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
                if (playback.state != PlaybackState.PREBUFFERING &&
                    playback.state != PlaybackState.PLAYING
                ) {
                    rebuildPreview()
                }
                refreshStats(splatCount(pack))
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
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
            "Allow file access, then copy model & splat to ${AppConfig.avatarTalkDir()}",
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
            perfStats = perfStats,
        )
        prefetcher = pf

        val view = binding.glView
        view.setEGLContextClientVersion(3)
        val r = SplatRenderer(p, this, playback, frameCache, pf.builder, perfStats)
        renderer = r
        view.setRenderer(r)
        view.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        glView = view
        view.onResume()

        binding.pickAudioButton.isEnabled = currentMode == AppConfig.AudioInputMode.FILE
        playback.state = PlaybackState.IDLE
        refreshStats(splats = splatCount(p))
        rebuildPreview()
        updateControls()
    }

    private fun togglePlayback() {
        when (playback.state) {
            PlaybackState.PREBUFFERING, PlaybackState.PLAYING -> stopPlayback()
            else -> startPlayback()
        }
    }

    private fun rebuildPreview() {
        val pf = prefetcher ?: return
        val mouthOnly = AppConfig.MOUTH_ONLY
        pf.cancel()
        frameCache.clear()
        pf.resetCancel()
        glView?.queueEvent {
            renderer?.syncPrefetchView()
            if (mouthOnly) renderer?.ensureStaticBase()
            runOnUiThread {
                pf.buildPreviewFrame(0, mouthOnly, previewListener)
            }
        }
    }

    private val previewListener = object : FramePrefetcher.Listener {
        override fun onPrebufferProgress(built: Int, total: Int) {}
        override fun onPrebufferComplete() {}
        override fun onError(message: String) {
            runOnUiThread { Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show() }
        }
    }

    private fun startPlayback() {
        val pf = prefetcher ?: return
        val pk = pack ?: return
        val model = onnx
        if (model == null) {
            val msg = onnxError ?: "Model still loading…"
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
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

        val mouthOnly = AppConfig.MOUTH_ONLY
        perfStats.reset()
        playback.prebufferSeconds = AppConfig.PREBUFFER_SECONDS
        playback.resetAudioPosition()
        playback.state = PlaybackState.PREBUFFERING
        pf.cancel()
        pf.stopLeadPrefetch()

        // Reset + start the audio producer feeding the shared ring.
        expressionBuffer.clear()
        when (currentMode) {
            AppConfig.AudioInputMode.FILE -> fileDurationMs = audioInput.startFile(audioUri!!)
            AppConfig.AudioInputMode.MIC -> audioInput.startMic()
        }

        // Inference pipeline reads the ring (FILE/MIC identical path).
        val pipe = OnnxExpressionPipeline(model, audioInput.ring, expressionBuffer)
        pipeline = pipe
        pipe.start()

        // Route geometry weights through the ONNX expression source.
        playback.expressionSource = OnnxExpressionSource(pk, expressionBuffer) { frameCountProvider() }

        updateControls()

        glView?.queueEvent {
            renderer?.syncPrefetchView()
            if (mouthOnly) renderer?.ensureStaticBase()
            runOnUiThread {
                pf.startPrebuffer(mouthOnly, prebufferListener)
            }
        }
    }

    private fun frameCountProvider(): Int =
        when (currentMode) {
            AppConfig.AudioInputMode.FILE ->
                ceil(fileDurationMs * AppConfig.EXPRESSION_FPS / 1000.0).toInt().coerceAtLeast(1)
            AppConfig.AudioInputMode.MIC -> Int.MAX_VALUE / 4
        }

    private val prebufferListener = object : FramePrefetcher.Listener {
        override fun onPrebufferProgress(built: Int, total: Int) {
            val last = frameCache.get(built - 1)
            runOnUiThread { refreshStats(last?.instanceCount ?: splatCount(pack)) }
        }

        override fun onPrebufferComplete() {
            runOnUiThread { beginAudioAndPlayback() }
        }

        override fun onError(message: String) {
            runOnUiThread {
                playback.state = PlaybackState.READY
                updateControls()
                Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun beginAudioAndPlayback() {
        val pf = prefetcher ?: return
        val mouthOnly = AppConfig.MOUTH_ONLY

        if (currentMode == AppConfig.AudioInputMode.MIC) {
            micClockStartNanos = System.nanoTime()
            playback.resetAudioPosition()
            playback.state = PlaybackState.PLAYING
            updateControls()
            startPositionUpdates()
            pf.startLeadPrefetch(mouthOnly)
            return
        }

        val uri = audioUri ?: return
        releaseMediaPlayer()
        val mp = MediaPlayer()
        mediaPlayer = mp
        try {
            mp.setDataSource(this, uri)
            mp.setOnCompletionListener {
                playback.state = PlaybackState.DONE
                pf.stopLeadPrefetch()
                stopPositionUpdates()
                updateControls()
            }
            mp.setOnPreparedListener {
                playback.resetAudioPosition()
                mp.start()
                playback.state = PlaybackState.PLAYING
                updateControls()
                startPositionUpdates()
                pf.startLeadPrefetch(mouthOnly)
            }
            mp.prepareAsync()
        } catch (e: Exception) {
            playback.state = PlaybackState.READY
            updateControls()
            Toast.makeText(this, e.message ?: "audio failed", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startPositionUpdates() {
        mainHandler.removeCallbacks(positionRunnable)
        mainHandler.post(positionRunnable)
    }

    private fun stopPositionUpdates() {
        mainHandler.removeCallbacks(positionRunnable)
    }

    private fun stopPlayback() {
        prefetcher?.cancel()
        prefetcher?.stopLeadPrefetch()
        stopPositionUpdates()
        releaseMediaPlayer()
        pipeline?.stop()
        pipeline = null
        audioInput.stop()
        playback.expressionSource = null
        playback.resetAudioPosition()
        playback.state = when {
            currentMode == AppConfig.AudioInputMode.MIC -> PlaybackState.IDLE
            audioUri != null -> PlaybackState.READY
            else -> PlaybackState.IDLE
        }
        frameCache.clear()
        expressionBuffer.clear()
        perfStats.reset()
        refreshStats(splats = splatCount(pack))
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
        val busy = state == PlaybackState.PREBUFFERING || state == PlaybackState.PLAYING
        binding.playButton.text = if (busy) "Stop" else "Start"
        val canStart = when (currentMode) {
            AppConfig.AudioInputMode.FILE -> audioUri != null
            AppConfig.AudioInputMode.MIC -> true
        }
        binding.playButton.isEnabled = busy || canStart
        binding.pickAudioButton.isEnabled = !busy && currentMode == AppConfig.AudioInputMode.FILE
        binding.mouthOnlySwitch.isEnabled = !busy
        binding.threadSpinner.isEnabled = !busy
    }

    /** GL-thread tick: throttle UI text updates to ~5/s. */
    override fun onStatsTick(splats: Int) {
        val now = System.currentTimeMillis()
        if (now - lastStatUpdateMs < 200) return
        lastStatUpdateMs = now
        runOnUiThread { refreshStats(splats) }
    }

    private fun refreshStats(splats: Int) {
        val mode = if (AppConfig.MOUTH_ONLY) "Mouth-only" else "Full"
        binding.statLine1.text =
            "$mode | threads=${AppConfig.BUILD_THREADS} | splats=$splats"
        binding.statLine2.text = String.format(
            "infer: %.1f ms | post: %.1f ms",
            pipeline?.lastInferMs ?: 0.0,
            pipeline?.lastPostprocessMs ?: 0.0,
        )
        binding.statLine3.text = String.format(
            "build: %.2f | geom: %.2f | draw: %.2f ms (1s avg)",
            perfStats.buildAvgMs, perfStats.geomAvgMs, perfStats.drawAvgMs,
        )
        binding.statLine4.text = statLine4()
    }

    private fun statLine4(): String {
        val src = currentMode.name
        val ep = onnx?.activeEp ?: "—"
        val expr = expressionBuffer.count
        val ringS = audioInput.ringSeconds()
        return String.format(
            "src:%s ep:%s expr:%d ring:%.1fs audio:%dms",
            src, ep, expr, ringS, playback.audioPositionMs,
        )
    }

    private fun splatCount(p: AvatarPack?): Int {
        if (p == null) return 0
        return if (AppConfig.MOUTH_ONLY) p.dynamicIndices.size else p.numGaussians
    }

    override fun onResume() {
        super.onResume()
        glView?.onResume()
        if (playback.state == PlaybackState.PLAYING) {
            if (currentMode == AppConfig.AudioInputMode.FILE) mediaPlayer?.start()
            startPositionUpdates()
        }
    }

    override fun onPause() {
        super.onPause()
        if (playback.state == PlaybackState.PLAYING) {
            if (currentMode == AppConfig.AudioInputMode.FILE) mediaPlayer?.pause()
            stopPositionUpdates()
        }
        glView?.onPause()
    }

    override fun onDestroy() {
        stopPlayback()
        prefetcher?.shutdown()
        onnx?.close()
        super.onDestroy()
    }
}
