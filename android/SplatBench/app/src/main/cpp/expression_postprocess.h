// Exact C++ port of postprocess.py + session.py `_postprocess_chunk`.
// Operates on 52-channel ARKit blendshape frames (row-major, float32).
#ifndef EXPRESSION_POSTPROCESS_H
#define EXPRESSION_POSTPROCESS_H

#include <cstdint>
#include <random>
#include <vector>

namespace ep {

constexpr int kNumChannels = 52;          // ARKit blendshapes
constexpr int kMaxContextFrames = 36;     // postprocess.MAX_CONTEXT_FRAMES
constexpr int kAudioSampleRate = 16000;   // session.AUDIO_SR
constexpr float kOutputFps = 30.0f;       // session.OUTPUT_FPS

// Per-frame RMS volume (postprocess.compute_chunk_volume).
std::vector<float> ComputeChunkVolume(const float* audio, int audio_len,
                                      int sample_rate = kAudioSampleRate,
                                      float fps = kOutputFps);

// In-place full pipeline (postprocess.apply_expression_postprocessing).
// `frames` is frame_count * 52 row-major; `volume` length == frame_count
// (may be empty to skip smooth_mouth_movements, matching audio_volume=None).
void ApplyExpressionPostprocessing(float* frames, int frame_count,
                                   int processed_frames,
                                   const float* volume, int volume_len);

// Streaming context holding the previous (finalized) expression + volume tails,
// mirroring session.StreamingContext for the expression branch.
struct StreamingCtx {
    std::vector<float> prev_expr;   // prev_len * 52
    int prev_len = 0;
    std::vector<float> prev_volume; // prev_len
    bool is_initial = true;
    std::mt19937 rng{std::random_device{}()};

    void Reset() {
        prev_expr.clear();
        prev_volume.clear();
        prev_len = 0;
        is_initial = true;
    }
};

// Eye-blink stage needs the per-context RNG; the full pipeline is wrapped so the
// RNG threads through. This overload is used by ProcessChunk.
void ApplyExpressionPostprocessing(float* frames, int frame_count,
                                   int processed_frames,
                                   const float* volume, int volume_len,
                                   std::mt19937* rng);

// Process one chunk of freshly-sliced model frames (new_frames: k * 52) with the
// new chunk's audio (for volume). Returns the post-processed new frames (k * 52)
// and updates the rolling context. Mirrors session._postprocess_chunk.
std::vector<float> ProcessChunk(StreamingCtx* ctx,
                                const float* new_frames, int k,
                                const float* audio, int audio_len);

}  // namespace ep

#endif  // EXPRESSION_POSTPROCESS_H
