// omnivoice-codec.cpp : codec CLI for OmniVoice.
//
// Encode a 24 kHz mono WAV into RVQ codes (.rvq), or decode RVQ codes
// back into a 24 kHz mono float32 WAV. Mode is inferred from the input file
// extension : .wav in -> encode, .rvq in -> decode. Output is auto-named
// next to the input file by swapping the extension.
//
// File format (.rvq) : flat code stream packed at 11 bits per code, LSB-first,
// no header. Layout is [K, T] row-major. K is fixed by the codec config in
// the GGUF (8 codebooks). T = (filesize * 8) / (K * 11).

#include "audio-io.h"
#include "backend.h"
#include "pipeline-codec.h"
#include "version.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// 11 bits per code (V <= 2048).
static const int      RVQ_CODE_BITS = 11;
static const uint32_t RVQ_CODE_MASK = (1u << RVQ_CODE_BITS) - 1u;

static void print_usage(const char * prog) {
    fprintf(stderr, "omnivoice.cpp %s\n\n", OMNIVOICE_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> -i <input>\n\n"
            "Required:\n"
            "  --model <gguf>          Codec GGUF (omnivoice-tokenizer-*.gguf)\n"
            "  -i <path>               Input. WAV -> encode, .rvq -> decode\n\n"
            "Optional:\n"
            "  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)\n\n"
            "Output is auto-named next to input : clip.wav -> clip.rvq, clip.rvq -> clip.wav.\n",
            prog);
}

// Pack a flat code stream into 11-bit-per-code, LSB-first. Output size is
// ceil(N * 11 / 8) bytes.
static std::vector<uint8_t> pack_codes(const std::vector<int32_t> & codes) {
    const size_t         total_bits = codes.size() * (size_t) RVQ_CODE_BITS;
    std::vector<uint8_t> out((total_bits + 7) / 8, 0);

    uint64_t acc         = 0;
    int      bits_in_acc = 0;
    size_t   out_pos     = 0;
    for (size_t i = 0; i < codes.size(); i++) {
        acc |= ((uint64_t) ((uint32_t) codes[i] & RVQ_CODE_MASK)) << bits_in_acc;
        bits_in_acc += RVQ_CODE_BITS;
        while (bits_in_acc >= 8) {
            out[out_pos++] = (uint8_t) (acc & 0xFF);
            acc >>= 8;
            bits_in_acc -= 8;
        }
    }
    if (bits_in_acc > 0) {
        out[out_pos++] = (uint8_t) (acc & 0xFF);
    }
    return out;
}

// Symmetric unpack : reads N codes from packed bytes.
static std::vector<int32_t> unpack_codes(const std::vector<uint8_t> & in, size_t n_codes) {
    std::vector<int32_t> out(n_codes);

    uint64_t acc         = 0;
    int      bits_in_acc = 0;
    size_t   in_pos      = 0;
    for (size_t i = 0; i < n_codes; i++) {
        while (bits_in_acc < RVQ_CODE_BITS && in_pos < in.size()) {
            acc |= ((uint64_t) in[in_pos++]) << bits_in_acc;
            bits_in_acc += 8;
        }
        out[i] = (int32_t) (acc & RVQ_CODE_MASK);
        acc >>= RVQ_CODE_BITS;
        bits_in_acc -= RVQ_CODE_BITS;
    }
    return out;
}

// Read a .rvq file and unpack it into K*T codes. T is inferred from the file
// size : T = (filesize * 8) / (K * RVQ_CODE_BITS).
static bool read_rvq(const char * path, int K, std::vector<int32_t> & codes, int * n_frames) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[RVQ] FATAL: cannot open %s\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fprintf(stderr, "[RVQ] FATAL: %s is empty\n", path);
        fclose(f);
        return false;
    }
    std::vector<uint8_t> buf((size_t) sz);
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        fprintf(stderr, "[RVQ] FATAL: short read on %s\n", path);
        fclose(f);
        return false;
    }
    fclose(f);

    const size_t total_bits = (size_t) sz * 8;
    const size_t n_codes    = total_bits / (size_t) RVQ_CODE_BITS;
    if (n_codes == 0 || (n_codes % (size_t) K) != 0) {
        fprintf(stderr, "[RVQ] FATAL: %s yields %zu codes, not a multiple of K=%d\n", path, n_codes, K);
        return false;
    }
    codes     = unpack_codes(buf, n_codes);
    *n_frames = (int) (n_codes / (size_t) K);
    return true;
}

// Pack and write a .rvq file.
static bool write_rvq(const char * path, const std::vector<int32_t> & codes) {
    std::vector<uint8_t> packed = pack_codes(codes);
    FILE *               f      = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[RVQ] FATAL: cannot open %s for write\n", path);
        return false;
    }
    if (fwrite(packed.data(), 1, packed.size(), f) != packed.size()) {
        fprintf(stderr, "[RVQ] FATAL: short write on %s\n", path);
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

// Replace or append extension on a path string.
static std::string swap_ext(const std::string & path, const char * ext) {
    size_t dot = path.find_last_of('.');
    size_t sep = path.find_last_of("/\\");
    if (dot != std::string::npos && (sep == std::string::npos || dot > sep)) {
        return path.substr(0, dot) + ext;
    }
    return path + ext;
}

// Mode 1 = encode (audio in), 2 = decode (.rvq in). Inferred from extension.
static int infer_mode(const char * path) {
    if (audio_io_ends_with(path, ".rvq")) {
        return 2;
    }
    if (audio_io_ends_with(path, ".wav")) {
        return 1;
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc <= 1) {
        print_usage(argv[0]);
        return 0;
    }

    const char * model_path = NULL;
    const char * input_path = NULL;
    WavFormat    wav_fmt    = WAV_S16;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            if (!audio_parse_format(argv[++i], wav_fmt)) {
                fprintf(stderr, "[CLI] ERROR: unknown format: %s\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !input_path) {
        print_usage(argv[0]);
        return 1;
    }

    const int mode = infer_mode(input_path);
    if (mode == 0) {
        fprintf(stderr, "[CLI] ERROR: %s: unsupported extension (expect .wav or .rvq)\n", input_path);
        return 1;
    }

    const std::string out_str = (mode == 1) ? swap_ext(input_path, ".rvq") : swap_ext(input_path, ".wav");

    BackendPair bp = backend_init("Codec");

    PipelineCodec pc = {};
    if (!pipeline_codec_load(&pc, model_path, bp)) {
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    int rc = 0;

    if (mode == 1) {
        int     n_samples = 0;
        float * audio     = audio_read_mono(input_path, 24000, &n_samples);
        if (!audio || n_samples <= 0) {
            fprintf(stderr, "[OmniVoice-Codec] FATAL: failed to load %s\n", input_path);
            rc = 1;
        } else {
            fprintf(stderr, "[OmniVoice-Codec] Encode: %s, %d samples @ 24 kHz mono (%.2f s)\n", input_path, n_samples,
                    (double) n_samples / 24000.0);
            std::vector<int32_t> codes = pipeline_codec_encode(&pc, audio, n_samples);
            free(audio);
            if (codes.empty()) {
                fprintf(stderr, "[OmniVoice-Codec] FATAL: encode failed\n");
                rc = 1;
            } else if (!write_rvq(out_str.c_str(), codes)) {
                rc = 1;
            } else {
                const int    K           = pc.rvq.num_codebooks;
                const int    T           = (int) codes.size() / K;
                const size_t packed_size = (codes.size() * (size_t) RVQ_CODE_BITS + 7) / 8;
                fprintf(stderr, "[OmniVoice-Codec] Wrote %s: K=%d T=%d (%zu bytes)\n", out_str.c_str(), K, T,
                        packed_size);
            }
        }
    } else {
        const int            K = pc.rvq.num_codebooks;
        std::vector<int32_t> codes;
        int                  T = 0;
        if (!read_rvq(input_path, K, codes, &T)) {
            rc = 1;
        } else {
            fprintf(stderr, "[OmniVoice-Codec] Decode: %s, K=%d T=%d\n", input_path, K, T);
            std::vector<float> audio = pipeline_codec_decode(&pc, codes.data(), K, T);
            if (audio.empty()) {
                fprintf(stderr, "[OmniVoice-Codec] FATAL: decode failed\n");
                rc = 1;
            } else if (!audio_write_wav(out_str.c_str(), audio.data(), (int) audio.size(), pc.sample_rate, wav_fmt)) {
                rc = 1;
            } else {
                fprintf(stderr, "[OmniVoice-Codec] Wrote %s: %d samples @ %d Hz, %.2f s\n", out_str.c_str(),
                        (int) audio.size(), pc.sample_rate, (double) audio.size() / (double) pc.sample_rate);
            }
        }
    }

    pipeline_codec_free(&pc);
    backend_release(bp.backend, bp.cpu_backend);
    return rc;
}
