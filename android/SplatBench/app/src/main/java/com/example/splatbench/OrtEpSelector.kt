package com.example.splatbench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.util.Log
import java.nio.FloatBuffer
import java.util.EnumSet

/**
 * Probes the device's ONNX Runtime execution providers and builds the session
 * from a **file path** (native mmap — no 256MB Java heap copy).
 */
object OrtEpSelector {
    private const val TAG = "OrtEpSelector"

    data class Choice(val session: OrtSession, val ep: String)

    fun createSession(env: OrtEnvironment, modelPath: String): Choice {
        return when (AppConfig.ORT_EP_MODE) {
            AppConfig.OrtEpMode.CPU -> Choice(buildCpuSession(env, modelPath), "CPU")
            AppConfig.OrtEpMode.NNAPI -> buildNnapiOrFallback(env, modelPath)
            AppConfig.OrtEpMode.AUTO -> autoSelect(env, modelPath)
        }
    }

    private fun cpuOptions(): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(AppConfig.ORT_INTRA_THREADS)
            setInterOpNumThreads(1)
        }

    private fun nnapiOptions(): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
            setIntraOpNumThreads(1)
        }

    private fun buildCpuSession(env: OrtEnvironment, modelPath: String): OrtSession =
        env.createSession(modelPath, cpuOptions())

    private fun buildNnapiOrFallback(env: OrtEnvironment, modelPath: String): Choice {
        return try {
            val s = env.createSession(modelPath, nnapiOptions())
            benchmark(env, s, runs = 1)
            Choice(s, "NNAPI")
        } catch (e: Throwable) {
            Log.w(TAG, "NNAPI session failed, falling back to CPU: ${e.message}")
            Choice(buildCpuSession(env, modelPath), "CPU")
        }
    }

    private fun nnapiAvailable(): Boolean =
        try {
            OrtEnvironment.getAvailableProviders().contains(OrtProvider.NNAPI)
        } catch (_: Throwable) {
            false
        }

    /** Benchmark CPU and NNAPI sequentially; only one probe session alive at a time. */
    private fun autoSelect(env: OrtEnvironment, modelPath: String): Choice {
        if (!nnapiAvailable()) {
            Log.i(TAG, "AUTO: NNAPI not available, using CPU")
            return Choice(buildCpuSession(env, modelPath), "CPU")
        }

        val cpuMs = try {
            buildCpuSession(env, modelPath).use { benchmark(env, it, AppConfig.ORT_EP_BENCHMARK_RUNS) }
        } catch (e: Throwable) {
            Log.w(TAG, "AUTO: CPU benchmark failed: ${e.message}")
            return Choice(buildCpuSession(env, modelPath), "CPU")
        }

        val nnapiMs = try {
            env.createSession(modelPath, nnapiOptions()).use {
                benchmark(env, it, AppConfig.ORT_EP_BENCHMARK_RUNS)
            }
        } catch (e: Throwable) {
            Log.w(TAG, "AUTO: NNAPI benchmark failed: ${e.message}")
            return Choice(buildCpuSession(env, modelPath), "CPU")
        }

        val gain = (cpuMs - nnapiMs) / cpuMs
        Log.i(TAG, "AUTO: cpu=${"%.1f".format(cpuMs)}ms nnapi=${"%.1f".format(nnapiMs)}ms gain=${"%.2f".format(gain)}")
        return if (gain >= AppConfig.ORT_EP_NNAPI_MIN_GAIN) {
            Choice(env.createSession(modelPath, nnapiOptions()), "NNAPI")
        } else {
            Choice(buildCpuSession(env, modelPath), "CPU")
        }
    }

    private inline fun <T> OrtSession.use(block: (OrtSession) -> T): T {
        try {
            return block(this)
        } finally {
            try { close() } catch (_: Throwable) {}
        }
    }

    /** Median wall-clock ms over [runs] of a 1s silence chunk. */
    private fun benchmark(env: OrtEnvironment, session: OrtSession, runs: Int): Double {
        if (!AppConfig.ORT_EP_BENCHMARK_WARMUP) return Double.MAX_VALUE
        val inputName = session.inputNames.iterator().next()
        val data = FloatArray(AppConfig.AUDIO_SR)
        val times = ArrayList<Double>(runs)
        runOnce(env, session, inputName, data)
        repeat(runs) {
            val t0 = System.nanoTime()
            runOnce(env, session, inputName, data)
            times.add((System.nanoTime() - t0) / 1_000_000.0)
        }
        times.sort()
        return times[times.size / 2]
    }

    private fun runOnce(env: OrtEnvironment, session: OrtSession, inputName: String, data: FloatArray) {
        val shape = longArrayOf(1, data.size.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { /* discard */ }
        }
    }
}
