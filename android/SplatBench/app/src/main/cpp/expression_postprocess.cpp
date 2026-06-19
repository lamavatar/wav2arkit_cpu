// Exact C++ port of postprocess.py + session.py `_postprocess_chunk`.
#include "expression_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ep {
namespace {

// ---- ARKit 52-channel index constants (match ArkitBlendshapes.kt order) ----
// Left/right pairs used by symmetrize_blendshapes (ARKIT_LEFT_RIGHT_PAIRS).
constexpr int kPairs[][2] = {
    {23, 25},  // jawLeft / jawRight
    {32, 38},  // mouthLeft / mouthRight
    {43, 44},  // mouthSmileLeft / mouthSmileRight
    {29, 30},  // mouthFrownLeft / mouthFrownRight
    {27, 28},  // mouthDimpleLeft / mouthDimpleRight
    {45, 46},  // mouthStretchLeft / mouthStretchRight
    {35, 36},  // mouthPressLeft / mouthPressRight
    {33, 34},  // mouthLowerDownLeft / mouthLowerDownRight
    {47, 48},  // mouthUpperUpLeft / mouthUpperUpRight
    {6, 7},    // cheekSquintLeft / cheekSquintRight
    {49, 50},  // noseSneerLeft / noseSneerRight
    {0, 1},    // browDownLeft / browDownRight
    {3, 4},    // browOuterUpLeft / browOuterUpRight
    {8, 9},    // eyeBlinkLeft / eyeBlinkRight
    {10, 11},  // eyeLookDownLeft / eyeLookDownRight
    {12, 13},  // eyeLookInLeft / eyeLookInRight
    {14, 15},  // eyeLookOutLeft / eyeLookOutRight
    {16, 17},  // eyeLookUpLeft / eyeLookUpRight
    {18, 19},  // eyeSquintLeft / eyeSquintRight
    {20, 21},  // eyeWideLeft / eyeWideRight
};
constexpr int kNumPairs = sizeof(kPairs) / sizeof(kPairs[0]);

// Mouth blendshapes damped during silence (MOUTH_BLENDSHAPES, excludes mouthClose=26).
constexpr int kMouth[] = {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                          41, 42, 43, 44, 45, 46, 47, 48, 22, 23, 24, 25, 49, 50, 5};
constexpr int kNumMouth = sizeof(kMouth) / sizeof(kMouth[0]);

constexpr int kEyeBlinkL = 8;
constexpr int kEyeBlinkR = 9;

// BLINK_PATTERNS (4 x 7) from postprocess.py.
constexpr float kBlinkPatterns[4][7] = {
    {0.365f, 0.950f, 0.956f, 0.917f, 0.367f, 0.119f, 0.025f},
    {0.235f, 0.910f, 0.945f, 0.778f, 0.191f, 0.235f, 0.089f},
    {0.870f, 0.950f, 0.949f, 0.696f, 0.191f, 0.073f, 0.007f},
    {0.000f, 0.557f, 0.953f, 0.942f, 0.426f, 0.148f, 0.018f},
};
constexpr int kBlinkLen = 7;

inline float* Row(float* f, int i) { return f + static_cast<long>(i) * kNumChannels; }

// ---- Stage 1: smooth_mouth_movements -------------------------------------
// find_low_value_regions: contiguous runs of (volume < threshold) with length
// >= min_region_length. Returns [start, end] inclusive index pairs.
std::vector<std::pair<int, int>> FindLowValueRegions(const float* volume, int n,
                                                     float threshold,
                                                     int min_region_length) {
    std::vector<std::pair<int, int>> regions;
    int i = 0;
    while (i < n) {
        if (volume[i] < threshold) {
            int start = i;
            while (i < n && volume[i] < threshold) ++i;
            int end = i - 1;  // inclusive
            if (end - start + 1 >= min_region_length) regions.emplace_back(start, end);
        } else {
            ++i;
        }
    }
    return regions;
}

void SmoothMouthMovements(float* frames, int n, int processed_frames,
                          const float* volume, int volume_len) {
    if (volume == nullptr || volume_len <= 0) return;  // matches audio_volume=None
    const float threshold = 0.001f;
    const int min_len = 7;
    const int blend_window = 3;
    auto regions = FindLowValueRegions(volume, volume_len, threshold, min_len);
    for (const auto& region : regions) {
        const int r0 = region.first;
        const int r1 = region.second;
        // Damp mouth channels across the silent region.
        for (int idx = r0; idx <= r1 && idx < n; ++idx) {
            float* row = Row(frames, idx);
            for (int m = 0; m < kNumMouth; ++m) row[kMouth[m]] *= 0.1f;
        }
        // _blend_region_start: blend in from the frame before the region.
        {
            int blend_length = std::min(blend_window, r0 - processed_frames);
            if (blend_length > 0 && r0 - 1 >= 0 && r0 - 1 < n) {
                const float* pre = Row(frames, r0 - 1);
                for (int i = 0; i < blend_length; ++i) {
                    int idx = r0 + i;
                    if (idx >= n) break;
                    float w = static_cast<float>(i + 1) / (blend_length + 1);
                    float* row = Row(frames, idx);
                    for (int c = 0; c < kNumChannels; ++c)
                        row[c] = pre[c] * (1.0f - w) + row[c] * w;
                }
            }
        }
        // _blend_region_end: blend out toward the frame after the region.
        {
            int blend_length = std::min(blend_window, n - r1 - 1);
            if (blend_length > 0 && r1 + 1 < n) {
                const float* post = Row(frames, r1 + 1);
                for (int i = 0; i < blend_length; ++i) {
                    int idx = r1 - i;
                    if (idx < 0) break;
                    float w = static_cast<float>(i + 1) / (blend_length + 1);
                    float* row = Row(frames, idx);
                    for (int c = 0; c < kNumChannels; ++c)
                        row[c] = post[c] * (1.0f - w) + row[c] * w;
                }
            }
        }
    }
}

// ---- Stage 2: apply_frame_blending ---------------------------------------
void BlendAnimationSegment(float* frames, int n, int transition_start,
                           int blend_window, const float* reference) {
    int actual = std::min(blend_window, n - transition_start);
    if (actual <= 0) return;
    for (int offset = 0; offset < actual; ++offset) {
        int idx = transition_start + offset;
        float w = static_cast<float>(offset + 1) / (actual + 1);
        float* row = Row(frames, idx);
        for (int c = 0; c < kNumChannels; ++c)
            row[c] = reference[c] * (1.0f - w) + row[c] * w;
    }
}

void ApplyFrameBlending(float* frames, int n, int processed_frames) {
    if (n <= 0) return;
    if (processed_frames > 0) {
        if (processed_frames - 1 < 0 || processed_frames >= n) return;
        std::vector<float> ref(Row(frames, processed_frames - 1),
                               Row(frames, processed_frames - 1) + kNumChannels);
        BlendAnimationSegment(frames, n, processed_frames, 5, ref.data());
    } else {
        std::vector<float> zeros(kNumChannels, 0.0f);
        BlendAnimationSegment(frames, n, 0, 3, zeros.data());
    }
}

// ---- Stage 3: apply_savitzky_golay_smoothing -----------------------------
// window_length=5, polyorder=2, deriv=0 -> coeffs [-3,12,17,12,-3]/35, mode mirror.
void ApplySavitzkyGolay(float* frames, int n) {
    if (n < 5) return;  // matches len < window_length -> return input
    static const double kCoeff[5] = {-3.0 / 35.0, 12.0 / 35.0, 17.0 / 35.0,
                                     12.0 / 35.0, -3.0 / 35.0};
    const int half = 2;
    std::vector<float> out(static_cast<size_t>(n) * kNumChannels);
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < kNumChannels; ++c) {
            double acc = 0.0;
            for (int k = -half; k <= half; ++k) {
                int idx = i + k;
                // ndimage 'mirror': reflect about edge sample, edge excluded.
                if (idx < 0) idx = -idx;
                else if (idx >= n) idx = 2 * (n - 1) - idx;
                acc += kCoeff[k + half] * static_cast<double>(Row(frames, idx)[c]);
            }
            if (acc < 0.0) acc = 0.0;
            else if (acc > 1.0) acc = 1.0;
            out[static_cast<size_t>(i) * kNumChannels + c] = static_cast<float>(acc);
        }
    }
    std::memcpy(frames, out.data(), out.size() * sizeof(float));
}

// ---- Stage 4: symmetrize_blendshapes (mode=average) ----------------------
void SymmetrizeBlendshapes(float* frames, int n) {
    for (int i = 0; i < n; ++i) {
        float* row = Row(frames, i);
        for (int p = 0; p < kNumPairs; ++p) {
            int l = kPairs[p][0];
            int r = kPairs[p][1];
            float avg = (row[l] + row[r]) * 0.5f;
            row[l] = avg;
            row[r] = avg;
        }
    }
}

// ---- Stage 5: apply_random_eye_blinks_context ----------------------------
void ApplyRandomEyeBlinks(float* frames, int n, int processed_frames,
                          std::mt19937* rng) {
    int remaining = n - processed_frames;
    if (remaining <= 7) return;
    // last processed blink (eyeBlinkL > 0.5) in already-finalized frames.
    int last_processed_blink = processed_frames;
    for (int i = 0; i < processed_frames && i < n; ++i) {
        if (Row(frames, i)[kEyeBlinkL] > 0.5f) last_processed_blink = i - 7;
    }
    std::uniform_int_distribution<int> interval_dist(40, 99);  // randint(40,100)
    std::uniform_int_distribution<int> pattern_dist(0, 3);     // randint(0,4)
    std::uniform_real_distribution<float> intensity_dist(0.8f, 1.0f);

    int blink_interval = interval_dist(*rng);
    int first_blink_start = std::max(0, blink_interval - last_processed_blink);
    if (first_blink_start <= remaining - 7) {
        const float* pattern = kBlinkPatterns[pattern_dist(*rng)];
        float intensity = intensity_dist(*rng);
        int blink_start = processed_frames + first_blink_start;
        int blink_end = blink_start + 7;
        for (int j = 0; j < kBlinkLen && blink_start + j < n; ++j) {
            float v = pattern[j] * intensity;
            Row(frames, blink_start + j)[kEyeBlinkL] = v;
            Row(frames, blink_start + j)[kEyeBlinkR] = v;
        }
        int remaining_after = n - blink_end;
        if (remaining_after > 40) {
            float second_intensity = intensity_dist(*rng);
            int second_interval = interval_dist(*rng);
            if (remaining_after - 7 > second_interval) {
                const float* second_pattern = kBlinkPatterns[pattern_dist(*rng)];
                int second_blink_start = blink_end + second_interval;
                for (int j = 0; j < kBlinkLen && second_blink_start + j < n; ++j) {
                    float v = second_pattern[j] * second_intensity;
                    Row(frames, second_blink_start + j)[kEyeBlinkL] = v;
                    Row(frames, second_blink_start + j)[kEyeBlinkR] = v;
                }
            }
        }
    }
}

}  // namespace

std::vector<float> ComputeChunkVolume(const float* audio, int audio_len,
                                      int sample_rate, float fps) {
    std::vector<float> volume;
    if (audio_len <= 0) return volume;
    int hop = std::max(1, static_cast<int>(static_cast<double>(sample_rate) / fps));
    int frame_len = std::max(1, std::min(hop, audio_len));
    int n_frames = static_cast<int>(std::ceil(static_cast<double>(audio_len) / hop));
    volume.resize(n_frames, 0.0f);
    for (int i = 0; i < n_frames; ++i) {
        int start = i * hop;
        int end = std::min(start + frame_len, audio_len);
        if (end <= start) continue;
        double sum_sq = 0.0;
        for (int s = start; s < end; ++s) {
            double v = static_cast<double>(audio[s]);
            sum_sq += v * v;
        }
        volume[i] = static_cast<float>(std::sqrt(sum_sq / (end - start)));
    }
    int expected = static_cast<int>(
        std::ceil(static_cast<double>(audio_len) / sample_rate * fps));
    if (expected < 0) expected = 0;
    if (expected < static_cast<int>(volume.size())) volume.resize(expected);
    return volume;
}

void ApplyExpressionPostprocessing(float* frames, int frame_count,
                                   int processed_frames, const float* volume,
                                   int volume_len, std::mt19937* rng) {
    SmoothMouthMovements(frames, frame_count, processed_frames, volume, volume_len);
    ApplyFrameBlending(frames, frame_count, processed_frames);
    ApplySavitzkyGolay(frames, frame_count);
    SymmetrizeBlendshapes(frames, frame_count);
    if (rng != nullptr)
        ApplyRandomEyeBlinks(frames, frame_count, processed_frames, rng);
}

void ApplyExpressionPostprocessing(float* frames, int frame_count,
                                   int processed_frames, const float* volume,
                                   int volume_len) {
    static thread_local std::mt19937 fallback(std::random_device{}());
    ApplyExpressionPostprocessing(frames, frame_count, processed_frames, volume,
                                  volume_len, &fallback);
}

std::vector<float> ProcessChunk(StreamingCtx* ctx, const float* new_frames, int k,
                                const float* audio, int audio_len) {
    std::vector<float> volume = ComputeChunkVolume(audio, audio_len);
    const int vol_len = static_cast<int>(volume.size());

    if (ctx->prev_len == 0) {
        // First chunk: process in place, no context cap (matches None branch).
        std::vector<float> out(new_frames, new_frames + static_cast<size_t>(k) * kNumChannels);
        ApplyExpressionPostprocessing(out.data(), k, 0, volume.data(), vol_len,
                                      &ctx->rng);
        ctx->prev_expr = out;  // copy of finalized new frames
        ctx->prev_len = k;
        ctx->prev_volume = volume;
        ctx->is_initial = false;
        return out;
    }

    const int prev_len = ctx->prev_len;
    // combined_exp = prev_expr ++ new_frames
    std::vector<float> combined(static_cast<size_t>(prev_len + k) * kNumChannels);
    std::memcpy(combined.data(), ctx->prev_expr.data(),
                static_cast<size_t>(prev_len) * kNumChannels * sizeof(float));
    std::memcpy(combined.data() + static_cast<size_t>(prev_len) * kNumChannels,
                new_frames, static_cast<size_t>(k) * kNumChannels * sizeof(float));
    // combined_vol = prev_volume ++ volume
    std::vector<float> combined_vol;
    combined_vol.reserve(ctx->prev_volume.size() + volume.size());
    combined_vol.insert(combined_vol.end(), ctx->prev_volume.begin(),
                        ctx->prev_volume.end());
    combined_vol.insert(combined_vol.end(), volume.begin(), volume.end());

    ApplyExpressionPostprocessing(combined.data(), prev_len + k, prev_len,
                                  combined_vol.data(),
                                  static_cast<int>(combined_vol.size()), &ctx->rng);

    // out_exp = combined[prev_len:]
    std::vector<float> out(static_cast<size_t>(k) * kNumChannels);
    std::memcpy(out.data(),
                combined.data() + static_cast<size_t>(prev_len) * kNumChannels,
                static_cast<size_t>(k) * kNumChannels * sizeof(float));

    // new context = concat(old_prev_expr, out_exp)[-MAX:]
    {
        std::vector<float> ctx_expr;
        ctx_expr.reserve(static_cast<size_t>(prev_len + k) * kNumChannels);
        ctx_expr.insert(ctx_expr.end(), ctx->prev_expr.begin(), ctx->prev_expr.end());
        ctx_expr.insert(ctx_expr.end(), out.begin(), out.end());
        int total = prev_len + k;
        if (total > kMaxContextFrames) {
            int drop = total - kMaxContextFrames;
            ctx_expr.erase(ctx_expr.begin(),
                           ctx_expr.begin() + static_cast<size_t>(drop) * kNumChannels);
            total = kMaxContextFrames;
        }
        ctx->prev_expr = std::move(ctx_expr);
        ctx->prev_len = total;
    }
    // new prev_volume = concat(old_prev_volume, volume)[-MAX:]
    {
        std::vector<float> ctx_vol;
        ctx_vol.reserve(ctx->prev_volume.size() + volume.size());
        ctx_vol.insert(ctx_vol.end(), ctx->prev_volume.begin(), ctx->prev_volume.end());
        ctx_vol.insert(ctx_vol.end(), volume.begin(), volume.end());
        if (static_cast<int>(ctx_vol.size()) > kMaxContextFrames) {
            int drop = static_cast<int>(ctx_vol.size()) - kMaxContextFrames;
            ctx_vol.erase(ctx_vol.begin(), ctx_vol.begin() + drop);
        }
        ctx->prev_volume = std::move(ctx_vol);
    }
    ctx->is_initial = false;
    return out;
}

}  // namespace ep
