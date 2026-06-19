// JNI entry points for com.example.splatbench.NativePostprocess.
#include <jni.h>

#include <vector>

#include "expression_postprocess.h"

namespace {
inline ep::StreamingCtx* AsCtx(jlong handle) {
    return reinterpret_cast<ep::StreamingCtx*>(handle);
}
}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_splatbench_NativePostprocess_nativeCreate(JNIEnv*, jclass) {
    return reinterpret_cast<jlong>(new ep::StreamingCtx());
}

JNIEXPORT void JNICALL
Java_com_example_splatbench_NativePostprocess_nativeReset(JNIEnv*, jclass,
                                                          jlong handle) {
    ep::StreamingCtx* ctx = AsCtx(handle);
    if (ctx != nullptr) ctx->Reset();
}

JNIEXPORT void JNICALL
Java_com_example_splatbench_NativePostprocess_nativeDestroy(JNIEnv*, jclass,
                                                            jlong handle) {
    delete AsCtx(handle);
}

// framesFlat: k * 52 row-major; audio: float waveform [-1,1] for volume.
// Returns a new float[] of size k * 52 (post-processed new frames).
JNIEXPORT jfloatArray JNICALL
Java_com_example_splatbench_NativePostprocess_nativeProcessChunk(
    JNIEnv* env, jclass, jlong handle, jfloatArray framesFlat, jint k,
    jfloatArray audio) {
    ep::StreamingCtx* ctx = AsCtx(handle);
    if (ctx == nullptr || k <= 0) {
        return env->NewFloatArray(0);
    }

    const jsize frames_len = env->GetArrayLength(framesFlat);
    const jsize audio_len = (audio != nullptr) ? env->GetArrayLength(audio) : 0;

    jfloat* frames_ptr = env->GetFloatArrayElements(framesFlat, nullptr);
    jfloat* audio_ptr =
        (audio_len > 0) ? env->GetFloatArrayElements(audio, nullptr) : nullptr;

    std::vector<float> out;
    if (frames_len >= k * ep::kNumChannels) {
        out = ep::ProcessChunk(ctx, frames_ptr, k, audio_ptr,
                               static_cast<int>(audio_len));
    }

    if (frames_ptr != nullptr)
        env->ReleaseFloatArrayElements(framesFlat, frames_ptr, JNI_ABORT);
    if (audio_ptr != nullptr)
        env->ReleaseFloatArrayElements(audio, audio_ptr, JNI_ABORT);

    jfloatArray result = env->NewFloatArray(static_cast<jsize>(out.size()));
    if (result != nullptr && !out.empty()) {
        env->SetFloatArrayRegion(result, 0, static_cast<jsize>(out.size()),
                                 out.data());
    }
    return result;
}

}  // extern "C"
