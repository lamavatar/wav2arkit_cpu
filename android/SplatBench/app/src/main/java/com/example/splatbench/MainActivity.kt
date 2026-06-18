package com.example.splatbench

import android.media.MediaPlayer
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.WindowCompat
import com.example.splatbench.databinding.ActivityMainBinding
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity(), SplatRenderer.Callbacks {

    private lateinit var binding: ActivityMainBinding
    private var glView: GLSurfaceView? = null
    private var renderer: SplatRenderer? = null
    private var pack: AvatarPack? = null

    private val playback = PlaybackController()
    private val frameCache = FrameCache()
    private var prefetcher: FramePrefetcher? = null

    private var mediaPlayer: MediaPlayer? = null
    private var audioUri: Uri? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    private val positionRunnable = object : Runnable {
        override fun run() {
            val mp = mediaPlayer
            if (playback.state == PlaybackState.PLAYING && mp != null) {
                try {
                    playback.updateAudioPositionMs(mp.currentPosition)
                } catch (_: IllegalStateException) {
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, true)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.pickAudioButton.isEnabled = false
        updateStatLines(splats = null, buildMs = null, audioMs = 0)
        updateControls()

        binding.pickAudioButton.setOnClickListener {
            pickAudio.launch(arrayOf("audio/*"))
        }
        binding.playButton.setOnClickListener { togglePlayback() }

        thread(name = "avatar-load") {
            try {
                val p = AvatarPack.load(assets, AppConfig.SPLAT_ASSET)
                runOnUiThread { setupGl(p) }
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
            buildThreads = AppConfig.BUILD_THREADS,
        )
        prefetcher = pf

        val view = binding.glView
        view.setEGLContextClientVersion(3)
        val r = SplatRenderer(p, this, playback, frameCache, pf.builder)
        renderer = r
        view.setRenderer(r)
        view.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        glView = view
        view.onResume()

        binding.pickAudioButton.isEnabled = true
        playback.state = PlaybackState.IDLE
        updateStatLines(splats = splatCount(p), buildMs = null, audioMs = 0)
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
        if (audioUri == null) {
            Toast.makeText(this, "Pick an audio file first", Toast.LENGTH_SHORT).show()
            return
        }
        val pf = prefetcher ?: return
        val mouthOnly = AppConfig.MOUTH_ONLY
        playback.prebufferSeconds = AppConfig.PREBUFFER_SECONDS
        playback.resetAudioPosition()
        playback.state = PlaybackState.PREBUFFERING
        pf.cancel()
        pf.stopLeadPrefetch()
        updateControls()

        glView?.queueEvent {
            renderer?.syncPrefetchView()
            if (mouthOnly) renderer?.ensureStaticBase()
            runOnUiThread {
                pf.startPrebuffer(mouthOnly, prebufferListener)
            }
        }
    }

    private val prebufferListener = object : FramePrefetcher.Listener {
        override fun onPrebufferProgress(built: Int, total: Int) {
            val last = frameCache.get(built - 1)
            runOnUiThread {
                updateStatLines(
                    splats = last?.instanceCount ?: splatCount(pack),
                    buildMs = last?.buildMs,
                    audioMs = 0,
                )
            }
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
        val uri = audioUri ?: return
        val pf = prefetcher ?: return
        val mouthOnly = AppConfig.MOUTH_ONLY

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
        playback.resetAudioPosition()
        playback.state = if (audioUri != null) PlaybackState.READY else PlaybackState.IDLE
        frameCache.clear()
        updateStatLines(splats = splatCount(pack), buildMs = null, audioMs = 0)
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
        binding.playButton.isEnabled = busy || audioUri != null
        binding.pickAudioButton.isEnabled = !busy
    }

    override fun onStats(line1: String, line2: String, line3: String) {
        runOnUiThread {
            binding.statLine1.text = line1
            binding.statLine2.text = line2
            if (playback.state != PlaybackState.PREBUFFERING) {
                binding.statLine3.text = line3
            }
        }
    }

    private fun updateStatLines(splats: Int?, buildMs: Double?, audioMs: Int) {
        val p = pack
        val mode = if (AppConfig.MOUTH_ONLY) "Mouth-only" else "Full"
        val splatStr = (splats ?: p?.let { splatCount(it) } ?: 0).toString()
        binding.statLine1.text = "$mode | threads=${AppConfig.BUILD_THREADS} | splats=$splatStr"
        binding.statLine2.text = if (buildMs != null) {
            String.format("build: %.2f ms", buildMs)
        } else {
            "build: — ms"
        }
        binding.statLine3.text = String.format("audio: %d ms", audioMs)
    }

    private fun splatCount(p: AvatarPack?): Int {
        if (p == null) return 0
        return if (AppConfig.MOUTH_ONLY) p.dynamicIndices.size else p.numGaussians
    }

    override fun onResume() {
        super.onResume()
        glView?.onResume()
        if (playback.state == PlaybackState.PLAYING) {
            mediaPlayer?.start()
            startPositionUpdates()
        }
    }

    override fun onPause() {
        super.onPause()
        if (playback.state == PlaybackState.PLAYING) {
            mediaPlayer?.pause()
            stopPositionUpdates()
        }
        glView?.onPause()
    }

    override fun onDestroy() {
        stopPlayback()
        prefetcher?.shutdown()
        super.onDestroy()
    }
}
