package com.example.splatbench

import android.opengl.GLSurfaceView
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.splatbench.databinding.ActivityMainBinding
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity(), SplatRenderer.Callbacks {

    private lateinit var binding: ActivityMainBinding
    private var glView: GLSurfaceView? = null
    private var renderer: SplatRenderer? = null
    private var pack: AvatarPack? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.results.text = "loading avatar asset...\n"
        binding.benchButton.isEnabled = false

        // 13 MB asset: load off the UI thread, then build the GL renderer.
        thread {
            val p = AvatarPack.load(assets, ASSET_NAME)
            runOnUiThread { setupGl(p) }
        }
    }

    private fun setupGl(p: AvatarPack) {
        pack = p
        val view = binding.glView
        view.setEGLContextClientVersion(3)
        val r = SplatRenderer(p, this)
        renderer = r
        view.setRenderer(r)
        view.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        glView = view

        binding.mouthSwitch.setOnCheckedChangeListener { _, checked ->
            r.liveMouthOnly = checked
        }

        val nproc = Runtime.getRuntime().availableProcessors()
        val threads = sortedSetOf(1, 2, 4).apply {
            add(nproc)
            if (nproc >= 8) add(8)
        }.filter { it <= nproc }.toIntArray()
        val framesPerSetting = minOf(p.numFrames, 120)

        binding.benchButton.setOnClickListener {
            binding.benchButton.isEnabled = false
            val mouthOnly = binding.mouthSwitch.isChecked
            binding.results.append(
                "running benchmark (${if (mouthOnly) "mouth-only" else "full"}), " +
                    "threads=${threads.joinToString(",")} ...\n"
            )
            r.requestBenchmark(SplatRenderer.BenchRequest(mouthOnly, threads, framesPerSetting))
        }

        binding.benchButton.isEnabled = true
        binding.results.append(
            "ready: ${p.numGaussians} gaussians, ${p.numFrames} frames, " +
                "${p.dynamicIndices.size} dynamic. cores=$nproc\n\n"
        )
    }

    override fun onLiveStats(text: String) {
        runOnUiThread { binding.liveStats.text = text }
    }

    override fun onBenchmarkDone(report: String) {
        runOnUiThread {
            binding.results.append(report)
            binding.benchButton.isEnabled = true
        }
    }

    override fun onResume() {
        super.onResume()
        glView?.onResume()
    }

    override fun onPause() {
        super.onPause()
        glView?.onPause()
    }

    companion object {
        private const val ASSET_NAME = "vfhq_case1.splat"
    }
}
