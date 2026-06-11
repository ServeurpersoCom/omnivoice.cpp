// omnivoice-codec.cpp: codec CLI for OmniVoice.
//
// Encode a 24 kHz mono WAV into RVQ codes (.rvq), or decode RVQ codes
// back into a 24 kHz mono float32 WAV. Mode is inferred from the input file
// extension: .wav in -> encode, .rvq in -> decode. Output is auto-named
// next to the input file by swapping the extension.
//
// Encode applies the exact TTS reference preprocessing (RMS auto-gain,
// silence trim, hop truncation), so a .rvq produced here is bit-identical
// to what the --ref-wav path of omnivoice-tts encodes internally and can
// be fed back via --ref-rvq.
//
// File format (.rvq): flat code stream packed at 11 bits per code, LSB-first,
// no header. Layout is [K, T] row-major. K is fixed by the codec config in
// the GGUF (8 codebooks). T = (filesize * 8) / (K * 11).

#include "audio-io.h"
#include "audio-postproc.h"
#include "backend.h"
#include "pipeline-codec.h"
#include "rvq-file.h"
#include "utf8.h"
#include "version.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// 11 bits per code (V <= 2048).
static const int RVQ_CODE_BITS = 11;

static void print_usage(const char * prog) {
    fprintf(stderr, "omnivoice.cpp %s\n\n", OMNIVOICE_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> -i <input>\n\n"
            "Required:\n"
            "  --model <gguf>          Codec GGUF (omnivoice-tokenizer-*.gguf)\n"
            "  -i <path>               Input. WAV -> encode, .rvq -> decode\n\n"
            "Optional:\n"
            "  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)\n\n"
            "Output is auto-named next to input : clip.wav -> clip.rvq, clip.rvq -> clip.wav.\n"
            "Encode applies the TTS reference preprocessing (RMS auto-gain, silence trim,\n"
            "hop truncation); the resulting .rvq feeds omnivoice-tts --ref-rvq directly.\n",
            prog);
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

int main_impl(int argc, char ** argv) {
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
    if (!bp.backend) {
        return 1;
    }

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
            free(audio);
            rc = 1;
        } else {
            fprintf(stderr, "[OmniVoice-Codec] Encode: %s, %d samples @ 24 kHz mono (%.2f s)\n", input_path, n_samples,
                    (double) n_samples / 24000.0);

            // Strict conformance with the TTS --ref-wav path: RMS auto-gain,
            // silence trim, then truncation to the hop boundary.
            std::vector<float> buf(audio, audio + n_samples);
            free(audio);
            ref_preprocess_audio(buf, 24000, true);

            int n_aligned = ((int) buf.size() / pc.hop_length) * pc.hop_length;
            if (n_aligned <= 0) {
                fprintf(stderr, "[OmniVoice-Codec] FATAL: input too short after preprocessing (%zu samples, hop %d)\n",
                        buf.size(), pc.hop_length);
                rc = 1;
            } else {
                std::vector<int32_t> codes = pipeline_codec_encode(&pc, buf.data(), n_aligned);
                if (codes.empty()) {
                    fprintf(stderr, "[OmniVoice-Codec] FATAL: encode failed\n");
                    rc = 1;
                } else if (!rvq_write_file(out_str.c_str(), codes, RVQ_CODE_BITS)) {
                    rc = 1;
                } else {
                    const int    K           = pc.rvq.num_codebooks;
                    const int    T           = (int) codes.size() / K;
                    const size_t packed_size = (codes.size() * (size_t) RVQ_CODE_BITS + 7) / 8;
                    fprintf(stderr, "[OmniVoice-Codec] Wrote %s: K=%d T=%d (%zu bytes)\n", out_str.c_str(), K, T,
                            packed_size);
                }
            }
        }
    } else {
        const int            K = pc.rvq.num_codebooks;
        std::vector<int32_t> codes;
        int                  T = 0;
        if (!rvq_read_file(input_path, K, RVQ_CODE_BITS, codes, &T)) {
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

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    // Top-level boundary: the codec load chain signals fatal errors via
    // exceptions instead of exit(1). Catching here turns std::terminate
    // into a clean error line.
    try {
        return main_impl(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "[OmniVoice-Codec] FATAL: %s\n", e.what());
        return 1;
    }
}
