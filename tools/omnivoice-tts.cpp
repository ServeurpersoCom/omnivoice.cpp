// omnivoice-tts.cpp : TTS CLI for OmniVoice.
//
// Default mode synthesises an audio WAV from the target text read on stdin.
// Voice cloning is enabled by passing --ref-wav <path> and --ref-text <path>
// (the transcript is read from a file, never from the command line, to keep
// shell escaping out of the critical path). Debug modes dump intermediate
// tensors and bypass the codec decode.

#include "audio-io.h"
#include "backend.h"
#include "bpe.h"
#include "duration-estimator.h"
#include "maskgit-tts.h"
#include "pipeline-codec.h"
#include "pipeline-tts.h"
#include "version.h"
#include "voice-design.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "omnivoice.cpp %s\n\n", OMNIVOICE_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options] -o <out.wav> < text.txt\n\n"
            "Required:\n"
            "  --model <gguf>          LLM GGUF (F32 / BF16 / Q8_0)\n"
            "  --codec <gguf>          Codec GGUF (omnivoice-tokenizer-*.gguf)\n"
            "  -o <path>               Output WAV (24 kHz mono)\n\n"
            "Input:\n"
            "  stdin                   Target text to synthesise\n\n"
            "Optional:\n"
            "  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)\n"
            "  --lang <str>            Language label (default 'None')\n"
            "  --instruct <str>        Style instruction (default 'None')\n"
            "  --duration <sec>        Output duration in seconds (default: estimate from text)\n"
            "  --no-denoise            Omit the <|denoise|> prefix\n"
            "  --ref-wav <path>        Reference WAV for voice cloning\n"
            "  --ref-text <path>       Transcript file for the reference (required with --ref-wav)\n"
            "  --seed <int>            Sampling seed (default: -1 for random)\n"
            "  --no-preprocess-prompt  Skip ref-wav silence trim and ref-text terminal punctuation\n"
            "  --chunk-duration <sec>  Long-form chunk duration (default: 15.0, <= 0 disables chunking)\n"
            "  --chunk-threshold <sec> Activate chunking above this estimated duration (default: 30.0)\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention (matches Python eager attention)\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --dump <dir>            Dump intermediate tensors (f32) to <dir>\n"
            "  --llm-test <input.bin>  Full LLM forward, dump audio_logits\n"
            "  --maskgit-test          Greedy MaskGIT decoder, dump audio_tokens [K, T]\n"
            "                          (no codec decode, reads target text from stdin)\n",
            prog);
}

// Read all of stdin into a string. Trims trailing newlines so the prompt
// matches what a user typed without invisible suffix tokens.
static std::string read_stdin_text() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

// Read a small text file (transcript) into a string. Trims trailing newlines.
static bool read_text_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[OmniVoice-TTS] FATAL: cannot open %s\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        return false;
    }
    out.resize((size_t) sz);
    if (sz > 0 && fread(&out[0], 1, (size_t) sz, f) != (size_t) sz) {
        fclose(f);
        return false;
    }
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return true;
}

// Read [i32 K, i32 S, K*S i32 input_ids, S i32 audio_mask] for --llm-test.
static bool read_embed_input_dump(const char *           path,
                                  int *                  K_out,
                                  int *                  S_out,
                                  std::vector<int32_t> & input_ids,
                                  std::vector<int32_t> & audio_mask) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[Dump] FATAL: cannot open %s\n", path);
        return false;
    }
    int32_t k_le = 0, s_le = 0;
    if (fread(&k_le, sizeof(int32_t), 1, f) != 1 || fread(&s_le, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "[Dump] FATAL: truncated header in %s\n", path);
        fclose(f);
        return false;
    }
    if (k_le <= 0 || s_le <= 0) {
        fprintf(stderr, "[Dump] FATAL: invalid header K=%d S=%d in %s\n", (int) k_le, (int) s_le, path);
        fclose(f);
        return false;
    }
    *K_out = (int) k_le;
    *S_out = (int) s_le;
    input_ids.resize((size_t) k_le * (size_t) s_le);
    audio_mask.resize((size_t) s_le);
    if (fread(input_ids.data(), sizeof(int32_t), input_ids.size(), f) != input_ids.size() ||
        fread(audio_mask.data(), sizeof(int32_t), audio_mask.size(), f) != audio_mask.size()) {
        fprintf(stderr, "[Dump] FATAL: truncated payload in %s\n", path);
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

// Write [i32 V, i32 K, i32 S, V*K*S f32 audio_logits] (--llm-test out).
static bool write_logits_dump(const char * path, int V, int K, int n_frames, const float * data) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Dump] FATAL: cannot open %s for write\n", path);
        return false;
    }
    int32_t hdr[3] = { (int32_t) V, (int32_t) K, (int32_t) n_frames };
    if (fwrite(hdr, sizeof(int32_t), 3, f) != 3) {
        fprintf(stderr, "[Dump] FATAL: header write failed for %s\n", path);
        fclose(f);
        return false;
    }
    const size_t n = (size_t) V * (size_t) K * (size_t) n_frames;
    if (fwrite(data, sizeof(float), n, f) != n) {
        fprintf(stderr, "[Dump] FATAL: payload write failed for %s\n", path);
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

// Write raw audio_tokens [K, T] i32 row-major (--maskgit-test out).
static bool write_audio_tokens_dump(const char * path, int K, int T, const std::vector<int32_t> & tokens) {
    if ((size_t) K * (size_t) T != tokens.size()) {
        fprintf(stderr, "[Dump] FATAL: token vector size %zu does not match K*T=%d*%d\n", tokens.size(), K, T);
        return false;
    }
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Dump] FATAL: cannot open %s for write\n", path);
        return false;
    }
    if (fwrite(tokens.data(), sizeof(int32_t), tokens.size(), f) != tokens.size()) {
        fprintf(stderr, "[Dump] FATAL: payload write failed for %s\n", path);
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

// Load BPE tokenizer with OmniVoice specials. Combines the base BPE load
// and the special-token load shared by every synthesis mode.
static bool load_omnivoice_tokenizer(BPETokenizer * tok, const char * gguf_path) {
    return load_bpe_from_gguf(tok, gguf_path) && bpe_load_omnivoice_specials(tok, gguf_path);
}

int main(int argc, char ** argv) {
    if (argc <= 1) {
        print_usage(argv[0]);
        return 0;
    }

    VoiceDesign vd;
    voice_design_init(&vd);

    const char * model_path             = NULL;
    const char * codec_path             = NULL;
    const char * llm_test_in            = NULL;
    bool         maskgit_test_mode      = false;
    const char * prompt_lang            = NULL;
    const char * prompt_instruct        = NULL;
    int          prompt_duration_tokens = 0;
    float        prompt_duration_sec    = 0.0f;
    bool         prompt_denoise         = true;
    bool         preprocess_prompt      = true;
    float        chunk_duration_sec     = 15.0f;
    float        chunk_threshold_sec    = 30.0f;
    const char * ref_wav_path           = NULL;
    const char * ref_text_path          = NULL;
    const char * output_path            = NULL;
    bool         use_fa                 = true;
    bool         clamp_fp16             = false;
    int          seed_arg               = -1;
    const char * dump_dir               = NULL;
    WavFormat    wav_fmt                = WAV_S16;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (strcmp(argv[i], "--no-fa") == 0) {
            use_fa = false;
        } else if (strcmp(argv[i], "--clamp-fp16") == 0) {
            clamp_fp16 = true;
        } else if (strcmp(argv[i], "--llm-test") == 0 && i + 1 < argc) {
            llm_test_in = argv[++i];
        } else if (strcmp(argv[i], "--maskgit-test") == 0) {
            maskgit_test_mode = true;
        } else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
            prompt_lang = argv[++i];
        } else if (strcmp(argv[i], "--instruct") == 0 && i + 1 < argc) {
            prompt_instruct = argv[++i];
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            prompt_duration_sec = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--no-denoise") == 0) {
            prompt_denoise = false;
        } else if (strcmp(argv[i], "--no-preprocess-prompt") == 0) {
            preprocess_prompt = false;
        } else if (strcmp(argv[i], "--chunk-duration") == 0 && i + 1 < argc) {
            chunk_duration_sec = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--chunk-threshold") == 0 && i + 1 < argc) {
            chunk_threshold_sec = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--ref-wav") == 0 && i + 1 < argc) {
            ref_wav_path = argv[++i];
        } else if (strcmp(argv[i], "--ref-text") == 0 && i + 1 < argc) {
            ref_text_path = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed_arg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
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

    // Mode resolution : llm_test_in OR maskgit_test_mode are debug, the
    // default is full TTS synthesis. Modes are mutually exclusive.
    int n_debug = (llm_test_in ? 1 : 0) + (maskgit_test_mode ? 1 : 0);
    if (n_debug > 1) {
        fprintf(stderr, "[CLI] ERROR: --llm-test and --maskgit-test are mutually exclusive\n");
        return 1;
    }
    const bool tts_mode = (n_debug == 0);

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }
    if (!output_path) {
        print_usage(argv[0]);
        return 1;
    }
    if (tts_mode && !codec_path) {
        fprintf(stderr, "[CLI] ERROR: synthesis requires --codec\n");
        return 1;
    }
    if (ref_wav_path && !ref_text_path) {
        fprintf(stderr, "[CLI] ERROR: --ref-wav requires --ref-text <path>\n");
        return 1;
    }
    if (ref_wav_path && !tts_mode) {
        fprintf(stderr, "[CLI] ERROR: --ref-wav is only supported in synthesis mode\n");
        return 1;
    }

    // Resolve sampling seed : -1 picks a fresh random seed from std::random_device,
    // any other value is used verbatim for reproducible runs across the maskgit
    // RNG.
    uint64_t seed_resolved = (seed_arg < 0) ? (uint64_t) std::random_device{}() : (uint64_t) seed_arg;
    fprintf(stderr, "[CLI] Seed: %llu%s\n", (unsigned long long) seed_resolved, (seed_arg < 0) ? " (random)" : "");

    BackendPair bp = backend_init("LM");

    PipelineTTS pt = {};
    if (!pipeline_tts_load(&pt, model_path, bp, use_fa, clamp_fp16)) {
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    int rc = 0;

    if (llm_test_in) {
        int                  K = 0, S = 0;
        std::vector<int32_t> input_ids, audio_mask;
        if (!read_embed_input_dump(llm_test_in, &K, &S, input_ids, audio_mask)) {
            rc = 1;
        } else {
            fprintf(stderr, "[OmniVoice-TTS] LM forward: K=%d S=%d\n", K, S);
            std::vector<float> out = pipeline_tts_llm_forward(&pt, input_ids.data(), audio_mask.data(), NULL, K, S);
            const int          V   = pt.lm.audio_vocab_size;
            if (out.empty()) {
                rc = 1;
            } else if (!write_logits_dump(output_path, V, K, S, out.data())) {
                rc = 1;
            } else {
                fprintf(stderr, "[OmniVoice-TTS] LM forward: wrote %s (V=%d K=%d S=%d f32)\n", output_path, V, K, S);
            }
        }
    } else if (maskgit_test_mode) {
        BPETokenizer tok = {};
        if (!load_omnivoice_tokenizer(&tok, model_path)) {
            rc = 1;
        } else {
            // Force fully greedy run for bytewise reproducibility against the
            // reference dump. Both temperatures at zero collapse the gumbel
            // paths, so the CLI seed has no effect here but is wired in for
            // consistency with the synthesis path.
            MaskgitConfig mg_cfg        = {};
            mg_cfg.class_temperature    = 0.0f;
            mg_cfg.position_temperature = 0.0f;
            mg_cfg.seed                 = seed_resolved;

            std::string text         = read_stdin_text();
            std::string lang         = prompt_lang ? prompt_lang : "";
            std::string raw_instruct = prompt_instruct ? prompt_instruct : "";
            std::string instruct;
            if (!pipeline_tts_resolve_instruct(&vd, text, raw_instruct, &instruct)) {
                rc = 1;
            } else {
                // Resolve target frame count : explicit --duration in seconds
                // (OmniVoice runs at a fixed 25 fps : 24000 / 960), otherwise
                // estimate from text via the byte-perfect RuleDurationEstimator
                // mirror. The codec is not loaded in this debug mode, so the
                // 25 fps frame rate is hardcoded here rather than read from
                // PipelineCodec.
                if (prompt_duration_sec > 0.0f) {
                    prompt_duration_tokens = (int) (prompt_duration_sec * 25.0f);
                    if (prompt_duration_tokens < 1) {
                        prompt_duration_tokens = 1;
                    }
                } else {
                    prompt_duration_tokens = duration_estimate_tokens(text, "", 0);
                }

                std::vector<int32_t> tokens =
                    pipeline_tts_generate(&pt, &tok, text, lang, instruct, prompt_duration_tokens, prompt_denoise,
                                          mg_cfg, "", NULL, 0, dump_dir);
                if (tokens.empty()) {
                    rc = 1;
                } else if (!write_audio_tokens_dump(output_path, pt.lm.num_audio_codebook, prompt_duration_tokens,
                                                    tokens)) {
                    rc = 1;
                } else {
                    fprintf(stderr, "[OmniVoice-TTS] MaskGIT test: wrote %s (K=%d T=%d i32)\n", output_path,
                            pt.lm.num_audio_codebook, prompt_duration_tokens);
                }
            }
        }
    } else {
        BPETokenizer  tok          = {};
        PipelineCodec pc           = {};
        bool          codec_loaded = false;

        if (!load_omnivoice_tokenizer(&tok, model_path)) {
            rc = 1;
        } else if (!pipeline_codec_load(&pc, codec_path, bp)) {
            rc = 1;
        } else {
            codec_loaded = true;

            // Optional reference WAV : the tool only reads the raw mono
            // 24 kHz buffer here. All preprocessing (RMS, auto-gain,
            // add_punctuation, silence-trim, hop alignment, codec encode)
            // runs inside pipeline_tts_synthesize_long_with_ref.
            std::vector<float> ref_audio;
            std::string        ref_text;
            if (ref_wav_path) {
                fprintf(stderr, "[CLI] Reference WAV: %s\n", ref_wav_path);
                if (!read_text_file(ref_text_path, ref_text)) {
                    rc = 1;
                } else {
                    int     n_samples = 0;
                    float * raw       = audio_read_mono(ref_wav_path, 24000, &n_samples);
                    if (!raw || n_samples <= 0) {
                        fprintf(stderr, "[OmniVoice-TTS] FATAL: failed to load %s\n", ref_wav_path);
                        rc = 1;
                    } else {
                        ref_audio.assign(raw, raw + n_samples);
                        free(raw);
                    }
                }
            }

            if (rc == 0) {
                // Defaults mirror OmniVoiceGenerationConfig (Python) :
                // num_step=32, guidance_scale=2.0, t_shift=0.1,
                // layer_penalty_factor=5.0, position_temperature=5.0,
                // class_temperature=0.0. The seed is plumbed from the CLI.
                MaskgitConfig mg_cfg = {};
                mg_cfg.seed          = seed_resolved;

                std::string text         = read_stdin_text();
                std::string lang         = prompt_lang ? prompt_lang : "";
                std::string raw_instruct = prompt_instruct ? prompt_instruct : "";
                std::string instruct;
                if (!pipeline_tts_resolve_instruct(&vd, text, raw_instruct, &instruct)) {
                    rc = 1;
                } else {
                    // Resolve target frame count override from --duration. When
                    // unset, pipeline_tts_synthesize_long estimates internally
                    // and may activate long-form chunking. An explicit value
                    // forces the single-shot path with that exact frame count.
                    int T_override = 0;
                    if (prompt_duration_sec > 0.0f) {
                        T_override = pipeline_tts_duration_sec_to_tokens(&pc, prompt_duration_sec);
                    }

                    std::vector<float> audio = pipeline_tts_synthesize_long_with_ref(
                        &pt, &pc, &tok, text, lang, instruct, T_override, chunk_duration_sec, chunk_threshold_sec,
                        prompt_denoise, preprocess_prompt, mg_cfg, ref_audio.empty() ? NULL : ref_audio.data(),
                        (int) ref_audio.size(), ref_text, dump_dir);
                    if (audio.empty()) {
                        rc = 1;
                    } else if (!audio_write_wav(output_path, audio.data(), (int) audio.size(), pc.sample_rate,
                                                wav_fmt)) {
                        rc = 1;
                    } else {
                        fprintf(stderr, "[OmniVoice-TTS] TTS: wrote %s (%d samples @ %d Hz, %.2f s)\n", output_path,
                                (int) audio.size(), pc.sample_rate, (double) audio.size() / (double) pc.sample_rate);
                    }
                }
            }
        }
        if (codec_loaded) {
            pipeline_codec_free(&pc);
        }
    }

    pipeline_tts_free(&pt);
    backend_release(bp.backend, bp.cpu_backend);
    return rc;
}
