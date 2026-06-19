// Host parity harness for expression_postprocess.cpp (no JNI).
// Reads input.bin (synthetic chunks), runs ProcessChunk, writes cpp_out.bin.
//
// input.bin layout (little-endian):
//   int32 num_chunks
//   repeat num_chunks:
//     int32 k            (frames this chunk)
//     int32 audio_len    (samples this chunk)
//     float32[k*52]      model frames (row-major)
//     float32[audio_len] audio waveform
//
// cpp_out.bin layout:
//   repeat num_chunks:
//     int32 k
//     float32[k*52]
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../android/SplatBench/app/src/main/cpp/expression_postprocess.h"

namespace {
template <typename T>
bool ReadPod(FILE* f, T* v) {
    return std::fread(v, sizeof(T), 1, f) == 1;
}
}  // namespace

int main(int argc, char** argv) {
    const char* in_path = (argc > 1) ? argv[1] : "input.bin";
    const char* out_path = (argc > 2) ? argv[2] : "cpp_out.bin";

    FILE* in = std::fopen(in_path, "rb");
    if (in == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", in_path);
        return 1;
    }
    FILE* out = std::fopen(out_path, "wb");
    if (out == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", out_path);
        std::fclose(in);
        return 1;
    }

    int32_t num_chunks = 0;
    if (!ReadPod(in, &num_chunks)) {
        std::fclose(in);
        std::fclose(out);
        return 1;
    }

    ep::StreamingCtx ctx;  // mirrors a fresh session/utterance
    for (int c = 0; c < num_chunks; ++c) {
        int32_t k = 0, audio_len = 0;
        ReadPod(in, &k);
        ReadPod(in, &audio_len);
        std::vector<float> frames(static_cast<size_t>(k) * ep::kNumChannels);
        std::fread(frames.data(), sizeof(float), frames.size(), in);
        std::vector<float> audio(audio_len);
        if (audio_len > 0) std::fread(audio.data(), sizeof(float), audio.size(), in);

        std::vector<float> result =
            ep::ProcessChunk(&ctx, frames.data(), k, audio.data(), audio_len);

        int32_t kk = k;
        std::fwrite(&kk, sizeof(int32_t), 1, out);
        std::fwrite(result.data(), sizeof(float), result.size(), out);
    }

    std::fclose(in);
    std::fclose(out);
    std::printf("processed %d chunks\n", num_chunks);
    return 0;
}
