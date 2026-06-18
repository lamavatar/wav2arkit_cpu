package com.example.splatbench

/** App settings (change here, not in the UI). */
object AppConfig {
    const val SPLAT_ASSET = "vfhq_case1.splat"

    /** CPU worker threads for instance build. */
    const val BUILD_THREADS = 1

    /** true = mouth-only (dynamic Gaussians over static base); false = full avatar. */
    const val MOUTH_ONLY = true

    /** Seconds of frames to pre-build before audio starts (and as lead during playback). */
    const val PREBUFFER_SECONDS = 1.0f
}
