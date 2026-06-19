package com.example.splatbench

/** Mutually exclusive avatar display modes (4 presets). */
enum class RenderMode(val label: String) {
    /** ④ Full Gaussian avatar, no background layer. */
    FULL("Full avatar"),

    /** ② Dynamic mouth splats composited on pre-rendered static 3D face. */
    MOUTH_ON_STATIC("Mouth on 3D face"),

    /** ③ Dynamic mouth splats composited on [AppConfig.PHOTO_FILE_NAME]. */
    MOUTH_ON_PHOTO("Mouth on photo"),

    /** ① Cropped mouth region only, no photo or static base. */
    MOUTH_CROP_ONLY("Mouth crop only"),
    ;

    fun allowsHeadBone(): Boolean = this == FULL || this == MOUTH_ON_STATIC

    fun usesPhotoBackground(): Boolean = this == MOUTH_ON_PHOTO

    fun usesMouthScissor(): Boolean = this == MOUTH_CROP_ONLY || this == MOUTH_ON_PHOTO
}
