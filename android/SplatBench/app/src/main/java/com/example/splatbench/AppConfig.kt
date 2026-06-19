package com.example.splatbench



import android.os.Environment

import java.io.File



/** App settings (change here, not in the UI). */

object AppConfig {

    /** Folder on shared internal storage (visible in any file manager). */

    const val AVATAR_TALK_DIR_NAME = "AvatarTalk"



    /** Splat filename inside [avatarTalkDir]. */

    const val SPLAT_FILE_NAME = "third.splat"



    /** ONNX filename inside [avatarTalkDir]. */

    const val ONNX_FILE_NAME = "wav2arkit_cpu_int8.onnx"



    /** e.g. /storage/emulated/0/AvatarTalk */

    fun avatarTalkDir(): File = File(Environment.getExternalStorageDirectory(), AVATAR_TALK_DIR_NAME)



    fun splatFile(): File = File(avatarTalkDir(), SPLAT_FILE_NAME)



    fun onnxFile(): File = File(avatarTalkDir(), ONNX_FILE_NAME)



    /** Human-readable path for error messages (e.g. /sdcard/AvatarTalk/...). */

    fun splatPathHint(): String = "/sdcard/$AVATAR_TALK_DIR_NAME/$SPLAT_FILE_NAME"



    fun onnxPathHint(): String = "/sdcard/$AVATAR_TALK_DIR_NAME/$ONNX_FILE_NAME"



    /** CPU worker threads for instance build (runtime-selectable from the UI). */

    @Volatile var BUILD_THREADS = 1



    /** Upper bound for selectable build threads (device cores). */

    val MAX_BUILD_THREADS: Int = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)



    /** Thread-count options offered in the UI spinner (clamped to device cores). */

    val THREAD_OPTIONS: List<Int> = run {

        val base = listOf(1, 2, 3, 4, 6, 8).filter { it <= MAX_BUILD_THREADS }

        val opts = if (base.isEmpty()) listOf(1) else base

        if (opts.contains(MAX_BUILD_THREADS)) opts else opts + MAX_BUILD_THREADS

    }



    /** true = mouth-only (dynamic Gaussians over static base); false = full avatar.
        Runtime-toggleable from the UI. */

    @Volatile var MOUTH_ONLY = true

    /** Apply baked head-bone matrices from the SPL2 HEAD trailer (runtime toggle). */
    @Volatile var HEAD_BONE_ENABLED = false

    fun useCompositeMouthOnly(pack: AvatarPack): Boolean =
        MOUTH_ONLY && !(HEAD_BONE_ENABLED && pack.hasHeadAnimation)



    /** Geometry lead kept ahead of the audio clock during PLAYING (seconds). */

    const val PREBUFFER_SECONDS = 1.0f

    /** [FrameCache] sliding window: frames to keep behind the playhead. */
    const val FRAME_CACHE_BEHIND = 2

    /** Extra frames beyond the prebuffer lead to retain when trimming. */
    const val FRAME_CACHE_AHEAD_MARGIN = 8

    /** Hard cap on cached geometry frames (full mode ~800 KB/frame). */
    const val FRAME_CACHE_MAX_FRAMES = 48

    /** Frames built before audio playback starts (0.5s @ 30fps). */

    const val GEOMETRY_START_FRAMES = 15

    /** Pause between successive geometry builds (ms). */

    const val GEOMETRY_BUILD_PAUSE_MS = 8L



    // --- Audio standard: 16kHz, mono, 8-bit unsigned (0..255) -------------- //

    const val AUDIO_SR = 16000

    const val AUDIO_BYTES_PER_SAMPLE = 1    // 8bit unsigned (u8)

    const val AUDIO_SILENCE_U8 = 128        // silence = 128 (overlap blank pad)

    const val EXPRESSION_FPS = 30.0f        // wav2arkit output fps

    const val RENDER_FPS = 30



    // --- Audio buffer / polling / chunking -------------------------------- //

    const val AUDIO_BUFFER_SECONDS = 20.0f  // fixed linear buffer (no overwrite)

    const val POLL_MS = 200L                // producer + inference poll interval

    const val FILE_TICK_MS = 500L           // file producer: bytes per tick

    const val MIC_TICK_MS = 200L            // mic producer: bytes per tick

    const val CHUNK_SECONDS = 1.0f          // ONNX chunk consumed per infer

    const val OVERLAP_MS = 200f             // postprocess.py DEFAULT_OVERLAP_MS



    val audioBufferCapacityBytes: Int

        get() = (AUDIO_BUFFER_SECONDS * AUDIO_SR).toInt()

    val fileTickBytes: Int

        get() = (FILE_TICK_MS * AUDIO_SR / 1000L).toInt()

    val micTickBytes: Int

        get() = (MIC_TICK_MS * AUDIO_SR / 1000L).toInt()

    val pollTickBytes: Int

        get() = (POLL_MS * AUDIO_SR / 1000L).toInt()

    val chunkBytes: Int

        get() = (CHUNK_SECONDS * AUDIO_SR).toInt()

    val overlapBytes: Int

        get() = (OVERLAP_MS * AUDIO_SR / 1000f).toInt()

    val framePeriodNs: Long

        get() = 1_000_000_000L / RENDER_FPS



    // --- ONNX Runtime execution provider ---------------------------------- //

    const val ORT_INTRA_THREADS = 2         // SD855: tune 2~4



    /** ORT execution provider selection (code only, no UI). */

    enum class OrtEpMode { AUTO, CPU, NNAPI }

    val ORT_EP_MODE = OrtEpMode.AUTO        // AUTO = probe device, pick fastest

    const val ORT_EP_BENCHMARK_WARMUP = true

    const val ORT_EP_BENCHMARK_RUNS = 3

    /** NNAPI must beat CPU by at least this fraction to be chosen in AUTO mode. */

    const val ORT_EP_NNAPI_MIN_GAIN = 0.10f



    // --- Audio input mode (code only, no UI switch) ----------------------- //

    enum class AudioInputMode { FILE, MIC }



    /** false = mic path/permission/UI all disabled, file Pick only. */

    const val ENABLE_MIC_INPUT = true



    /** Default input at start; forced to FILE when ENABLE_MIC_INPUT == false. */

    val DEFAULT_AUDIO_INPUT = AudioInputMode.FILE



    fun effectiveAudioMode(): AudioInputMode =

        if (!ENABLE_MIC_INPUT) AudioInputMode.FILE else DEFAULT_AUDIO_INPUT

}


