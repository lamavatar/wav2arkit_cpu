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



    /** CPU worker threads for instance build. */

    const val BUILD_THREADS = 1



    /** true = mouth-only (dynamic Gaussians over static base); false = full avatar. */

    const val MOUTH_ONLY = true



    /** Seconds of frames to pre-build before audio starts (and as lead during playback). */

    const val PREBUFFER_SECONDS = 1.0f



    // --- Audio standard: 16kHz, mono, 8-bit unsigned (0..255) -------------- //

    const val AUDIO_SR = 16000

    const val AUDIO_BYTES_PER_SAMPLE = 1    // 8bit unsigned (u8)

    const val AUDIO_SILENCE_U8 = 128        // silence = 128 (overlap blank pad)

    const val EXPRESSION_FPS = 30.0f        // wav2arkit output fps



    // --- Ring buffer / chunking (configurable here) ----------------------- //

    const val RING_BUFFER_SECONDS = 10.0f   // default 10s ring; change here

    const val CHUNK_SECONDS = 1.0f          // ONNX chunk consumed per infer

    const val OVERLAP_MS = 200f             // postprocess.py DEFAULT_OVERLAP_MS



    /** Ring capacity in bytes (u8 → 1 byte/sample). */

    val ringBufferCapacityBytes: Int

        get() = (RING_BUFFER_SECONDS * AUDIO_SR).toInt()

    val chunkBytes: Int

        get() = (CHUNK_SECONDS * AUDIO_SR).toInt()

    val overlapBytes: Int

        get() = (OVERLAP_MS * AUDIO_SR / 1000f).toInt()



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


