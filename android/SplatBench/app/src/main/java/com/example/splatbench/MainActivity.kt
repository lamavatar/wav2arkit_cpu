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
                    if (mp.isPlaying) {
                        playback.updateAudioPositionMs(mp.currentPosition)
                    }
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
            // Some providers do not allow persistable permission; read still works this session.
        }
        releaseMediaPlayer()
        audioUri = uri
        playback.state = PlaybackState.READY
        binding.audioLabel.text = "Audio: ${uri.lastPathSegment ?: uri}"
        updateControls()
        binding.results.append("audio selected: $uri\n")
        rebuildPreview()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.results.text = "loading avatar asset...\n"
        binding.benchButton.isEnabled = false
        binding.pickAudioButton.isEnabled = false
        updateControls()

        binding.pickAudioButton.setOnClickListener {
            pickAudio.launch(arrayOf("audio/*"))
        }
        binding.startButton.setOnClickListener { startPlayback() }
        binding.stopButton.setOnClickListener { stopPlayback(userInitiated = true) }

        thread {
            val p = AvatarPack.load(assets, ASSET_NAME)
            runOnUiThread { setupGl(p) }
        }
    }

    private fun setupGl(p: AvatarPack) {
        pack = p
        val nproc = Runtime.getRuntime().availableProcessors().coerceAtMost(8)
        val pf = FramePrefetcher(
            pack = p,
            cache = frameCache,
            controller = playback,
            buildThreads = nproc,
        )
        prefetcher = pf

        val view = binding.glView
        view.setEGLContextClientVersion(3)
        val r = SplatRenderer(p, this, playback, frameCache, pf.builder)
        renderer = r
        view.setRenderer(r)
        view.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        glView = view

        binding.mouthSwitch.setOnCheckedChangeListener { _, _ ->
            if (playback.state == PlaybackState.IDLE || playback.state == PlaybackState.READY) {
                r.invalidateStaticBase()
                rebuildPreview()
            }
        }

        val benchThreads = sortedSetOf(1, 2, 4).apply {
            add(Runtime.getRuntime().availableProcessors())
            if (Runtime.getRuntime().availableProcessors() >= 8) add(8)
        }.filter { it <= Runtime.getRuntime().availableProcessors() }.toIntArray()
        val framesPerSetting = minOf(p.numFrames, 120)

        binding.benchButton.setOnClickListener {
            if (playback.state == PlaybackState.PLAYING ||
                playback.state == PlaybackState.PREBUFFERING
            ) {
                return@setOnClickListener
            }
            binding.benchButton.isEnabled = false
            val mouthOnly = binding.mouthSwitch.isChecked
            binding.results.append(
                "running benchmark (${if (mouthOnly) "mouth-only" else "full"}), " +
                    "threads=${benchThreads.joinToString(",")} ...\n"
            )
            r.requestBenchmark(SplatRenderer.BenchRequest(mouthOnly, benchThreads, framesPerSetting))
        }

        binding.benchButton.isEnabled = true
        binding.pickAudioButton.isEnabled = true
        playback.state = PlaybackState.IDLE
        binding.results.append(
            "ready: ${p.numGaussians} gaussians, ${p.numFrames} frames @ ${p.fps}fps, " +
                "${p.dynamicIndices.size} dynamic. cores=$nproc\n\n"
        )
        rebuildPreview()
        updateControls()
    }

    private fun rebuildPreview() {
        val pf = prefetcher ?: return
        val mouthOnly = binding.mouthSwitch.isChecked
        glView?.queueEvent {
            renderer?.syncPrefetchView()
            if (mouthOnly) renderer?.ensureStaticBase()
        }
        pf.cancel()
        frameCache.clear()
        pf.resetCancel()
        pf.buildPreviewFrame(0, mouthOnly, previewListener)
    }

    private val previewListener = object : FramePrefetcher.Listener {
        override fun onPrebufferProgress(built: Int, total: Int) {}
        override fun onPrebufferComplete() {}
        override fun onError(message: String) {
            runOnUiThread { Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show() }
        }
    }

    private fun readPrebufferSeconds(): Float {
        val raw = binding.prebufferInput.text?.toString()?.trim() ?: "1.0"
        return raw.toFloatOrNull()?.coerceIn(0.1f, 30f) ?: 1.0f
    }

    private fun startPlayback() {
        if (audioUri == null) {
            Toast.makeText(this, "Pick an audio file first", Toast.LENGTH_SHORT).show()
            return
        }
        val pf = prefetcher ?: return
        val p = pack ?: return
        val mouthOnly = binding.mouthSwitch.isChecked
        playback.prebufferSeconds = readPrebufferSeconds()
        playback.resetAudioPosition()
        playback.state = PlaybackState.PREBUFFERING
        pf.cancel()
        pf.stopLeadPrefetch()
        updateControls()
        val prebufferFrames = playback.frameCountForSeconds(playback.prebufferSeconds, p.fps)
        binding.results.append("prebuffering ${playback.prebufferSeconds}s ($prebufferFrames frames)...\n")

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
            runOnUiThread {
                binding.liveStats.text = "Preparing: $built / $total frames"
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
                binding.results.append("prebuffer error: $message\n")
            }
        }
    }

    private fun beginAudioAndPlayback() {
        val uri = audioUri ?: return
        val pf = prefetcher ?: return
        val mouthOnly = binding.mouthSwitch.isChecked

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
                binding.results.append("playback finished\n")
            }
            mp.setOnPreparedListener {
                playback.resetAudioPosition()
                mp.start()
                playback.state = PlaybackState.PLAYING
                updateControls()
                startPositionUpdates()
                pf.startLeadPrefetch(mouthOnly)
                binding.results.append("playing (audio + avatar synced)\n")
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

    private fun stopPlayback(userInitiated: Boolean) {
        prefetcher?.cancel()
        prefetcher?.stopLeadPrefetch()
        stopPositionUpdates()
        releaseMediaPlayer()
        playback.resetAudioPosition()
        playback.state = if (audioUri != null) PlaybackState.READY else PlaybackState.IDLE
        frameCache.clear()
        updateControls()
        if (userInitiated) {
            binding.results.append("stopped\n")
        }
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
        binding.startButton.isEnabled = audioUri != null && !busy
        binding.stopButton.isEnabled = busy || state == PlaybackState.DONE
        binding.pickAudioButton.isEnabled = !busy
        binding.prebufferInput.isEnabled = !busy
        binding.mouthSwitch.isEnabled = !busy
        binding.benchButton.isEnabled = !busy
    }

    override fun onLiveStats(text: String) {
        if (playback.state != PlaybackState.PREBUFFERING) {
            runOnUiThread { binding.liveStats.text = text }
        }
    }

    override fun onBenchmarkDone(report: String) {
        runOnUiThread {
            binding.results.append(report)
            updateControls()
        }
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
        stopPlayback(userInitiated = false)
        prefetcher?.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val ASSET_NAME = "vfhq_case1.splat"
    }
}
