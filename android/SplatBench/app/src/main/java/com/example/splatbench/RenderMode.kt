package com.example.splatbench

/** Mutually exclusive avatar display presets (UI spinner). */
enum class RenderMode(val label: String) {
    /** ④ Full Gaussian avatar. */
    FULL("Full avatar"),

    /** ② Dynamic mouth splats on pre-rendered static 3D face. */
    MOUTH_ON_STATIC("Mouth on 3D face"),

    /** ③ Dynamic mouth splats on [AppConfig.PHOTO_FILE_NAME]. */
    MOUTH_ON_PHOTO("Mouth on photo"),

    /** ① Dynamic mouth splats only (no photo / static base). */
    MOUTH_CROP_ONLY("Mouth crop only"),
    ;

    fun allowsHeadBone(): Boolean = this == FULL || this == MOUTH_ON_STATIC
}
