// omnivoice-tts.cpp: TTS CLI for OmniVoice.
//
// Default mode synthesises an audio WAV from the target text read on stdin.
// Voice cloning is enabled by passing --ref-wav <path> and --ref-text <path>
// (the transcript is read from a file, never from the command line, to keep
// shell escaping out of the critical path). --ref-rvq <path> replaces
// --ref-wav with a pre-encoded reference produced by omnivoice-codec,
// skipping the codec encode entirely. Debug modes dump intermediate
// tensors and bypass the codec decode.

#include "audio-io.h"
#include "backend.h"
#include "bpe.h"
#include "duration-estimator.h"
#include "maskgit-tts.h"
#include "omnivoice.h"
#include "pipeline-codec.h"
#include "pipeline-tts.h"
#include "rvq-file.h"
#include "srt.h"
#include "text-chunker-stream.h"
#include "text-chunker.h"
#include "utf8.h"
#include "version.h"
#include "voice-design.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// 11 bits per code (V <= 2048), matching omnivoice-codec.
static const int RVQ_CODE_BITS = 11;

#if defined(_WIN32)
#    include <fcntl.h>
#    include <io.h>
#endif

static void print_usage(const char * prog) {
    fprintf(stderr, "omnivoice.cpp %s\n\n", OMNIVOICE_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options] -o <out.wav> < text.txt\n\n"
            "Required:\n"
            "  --model <gguf>          LLM GGUF (F32 / BF16 / Q8_0)\n"
            "  --codec <gguf>          Codec GGUF (omnivoice-tokenizer-*.gguf)\n"
            "  -o <path>               Output WAV (24 kHz mono). '-' streams to stdout (pipe friendly).\n\n"
            "Input:\n"
            "  stdin                   Target text to synthesise. With -o '-', stdin is read\n"
            "                          incrementally and synthesis starts as soon as the first\n"
            "                          sentence boundary is reached. With -o file.wav, stdin is\n"
            "                          read fully then synthesised in one shot.\n"
            "  --srt <path>            Dub an SRT: synth each cue into its time slot, write one\n"
            "                          timeline WAV ready to mux. Pairs with --ref-wav / --ref-rvq\n"
            "                          for a cloned voice. Per cue duration comes from the SRT.\n\n"
            "Optional:\n"
            "  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)\n"
            "  --lang <str>            Language label (default 'None')\n"
            "  --instruct <str>        Style instruction (default 'None')\n"
            "  --duration <sec>        Output duration in seconds (default: estimate from text)\n"
            "  --no-denoise            Omit the <|denoise|> prefix\n"
            "  --ref-wav <path>        Reference WAV for voice cloning\n"
            "  --ref-text <path>       Transcript file for the reference (required with --ref-wav / --ref-rvq)\n"
            "  --ref-rvq <path>        Pre-encoded reference codes from omnivoice-codec (replaces --ref-wav)\n"
            "  --seed <int>            Sampling seed (default: -1 for random)\n"
            "  --steps <int>           MaskGIT decode steps (default: 32, fewer is faster)\n"
            "  --no-preprocess-prompt  Skip ref-wav silence trim and ref-text terminal punctuation\n"
            "  --chunk-duration <sec>  Long-form chunk duration (default: 15.0, <= 0 disables chunking)\n"
            "  --chunk-threshold <sec> Activate chunking above this estimated duration (default: 30.0)\n"
            "  --stream-by-line        Flush synthesis at each newline, one WAV header per line (-o '-')\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --dump <dir>            Dump intermediate tensors (f32) to <dir>\n"
            "  --llm-test <input.bin>  Full LLM forward, dump audio_logits\n"
            "  --maskgit-test          Greedy MaskGIT decoder, dump audio_tokens [K, T]\n"
            "                          (no codec decode, reads target text from stdin)\n",
            prog);
}

// Read all of stdin into a string. Binary mode on Windows so UTF-16 input
// survives CRLF translation, then normalised to UTF-8. Trims trailing
// newlines so the prompt matches what a user typed without invisible
// suffix tokens. Used by the non-streaming code paths (debug dumps).
static std::string read_stdin_text() {
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    std::string s = ss.str();
    utf8_normalize(s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

// Read a small text file (transcript) into a string, normalised to UTF-8.
// Trims trailing newlines.
static bool read_text_file(const char * path, std::string & out) {
    FILE * f = utf8_fopen(path, "rb");
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
    utf8_normalize(out);
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
    FILE * f = utf8_fopen(path, "rb");
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
    FILE * f = utf8_fopen(path, "wb");
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
    FILE * f = utf8_fopen(path, "wb");
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

// SRT dubbing path. Reads an SRT, synthesises each cue into its own time
// slot, and assembles one WAV on an absolute timeline so the result muxes
// straight onto the source video. Each cue runs single shot with
// T_override set to its slot and postproc off, so the raw decode lands at
// exactly the slot length (the floor rounding is the only undershoot, at
// most one frame). The reference voice, when given, clones across every
// cue. Silences between cues fall out of the zero initialised timeline.
static int run_srt_dub(ov_context *                 ov,
                       const char *                 srt_path,
                       const std::vector<float> &   ref_audio,
                       const std::vector<int32_t> & ref_tokens,
                       int                          ref_T,
                       const std::string &          ref_text,
                       const std::string &          lang,
                       const char *                 prompt_instruct,
                       bool                         prompt_denoise,
                       bool                         preprocess_prompt,
                       int                          mg_steps,
                       uint64_t                     seed_resolved,
                       const char *                 dump_dir,
                       const char *                 output_path,
                       WavFormat                    wav_fmt) {
    std::string raw;
    if (!read_text_file(srt_path, raw)) {
        return 1;
    }

    std::vector<SrtCue> cues;
    srt_parse(raw, cues);
    if (cues.empty()) {
        fprintf(stderr, "[CLI] ERROR: no usable cues in %s\n", srt_path);
        return 1;
    }

    // Absolute timeline: sort by start so placement and overlap clipping are
    // monotonic whatever the source ordering.
    std::sort(cues.begin(), cues.end(), [](const SrtCue & a, const SrtCue & b) { return a.t0 < b.t0; });

    const int sr = 24000;  // OmniVoice codec sample rate, matches ov_audio.sample_rate

    double max_t1 = 0.0;
    for (const auto & c : cues) {
        if (c.t1 > max_t1) {
            max_t1 = c.t1;
        }
    }

    size_t             n_total = (size_t) llround(max_t1 * (double) sr);
    std::vector<float> timeline(n_total, 0.0f);

    // Short raised cosine fade on each placed segment edge, 5 ms at sr, to
    // kill the click the raw decode leaves at its boundaries.
    const int  fade_n  = sr / 200;
    const bool has_ref = !ref_audio.empty() || !ref_tokens.empty();

    for (size_t i = 0; i < cues.size(); i++) {
        const SrtCue & c    = cues[i];
        double         slot = c.t1 - c.t0;
        if (slot <= 0.0 || c.text.empty()) {
            continue;
        }

        ov_tts_params p;
        ov_tts_default_params(&p);
        p.text              = c.text.c_str();
        p.lang              = lang.c_str();
        p.instruct          = prompt_instruct ? prompt_instruct : "";
        p.T_override        = ov_duration_sec_to_tokens(ov, (float) slot);
        p.denoise           = prompt_denoise;
        p.preprocess_prompt = preprocess_prompt;
        p.postproc          = false;
        p.mg_seed           = seed_resolved;
        if (mg_steps > 0) {
            p.mg_num_step = mg_steps;
        }
        p.ref_audio_24k    = ref_audio.empty() ? nullptr : ref_audio.data();
        p.ref_n_samples    = (int) ref_audio.size();
        p.ref_audio_tokens = ref_tokens.empty() ? nullptr : ref_tokens.data();
        p.ref_T            = ref_T;
        p.ref_text         = ref_text.c_str();
        p.dump_dir         = dump_dir;

        ov_audio seg = {};
        if (ov_synthesize(ov, &p, &seg) != OV_STATUS_OK) {
            fprintf(stderr, "[CLI] ERROR: cue %d synth failed: %s\n", c.index, ov_last_error());
            ov_audio_free(&seg);
            return 1;
        }

        // Place at the cue start. Clip the tail to the next cue start (or
        // the timeline end) so an overlapping source stamp never bleeds into
        // the following line.
        size_t off   = (size_t) llround(c.t0 * (double) sr);
        size_t limit = n_total;
        if (i + 1 < cues.size()) {
            size_t next_off = (size_t) llround(cues[i + 1].t0 * (double) sr);
            if (next_off < limit) {
                limit = next_off;
            }
        }

        int n = seg.n_samples;
        if (off >= limit) {
            n = 0;
        } else if (off + (size_t) n > limit) {
            n = (int) (limit - off);
        }

        for (int k = 0; k < n; k++) {
            float w = 1.0f;
            if (fade_n > 0 && k < fade_n) {
                w = (float) k / (float) fade_n;
            } else if (fade_n > 0 && k >= n - fade_n) {
                w = (float) (n - 1 - k) / (float) fade_n;
            }
            timeline[off + (size_t) k] += seg.samples[k] * w;
        }

        fprintf(stderr, "[OmniVoice-TTS] dub cue %d: t0=%.3f slot=%.3f placed=%d samples\n", c.index, c.t0, slot, n);
        ov_audio_free(&seg);
    }

    // Without a reference the per cue peak normalisation was skipped
    // (postproc off), so normalise the whole timeline once to a 0.5 peak,
    // keeping levels consistent across the dub. With a reference the ref_rms
    // scaling already ran per cue and stays untouched.
    if (!has_ref) {
        float peak = 0.0f;
        for (float s : timeline) {
            float a = std::fabs(s);
            if (a > peak) {
                peak = a;
            }
        }
        if (peak > 1e-6f) {
            float g = 0.5f / peak;
            for (float & s : timeline) {
                s *= g;
            }
        }
    }

    if (!audio_write_wav(output_path, timeline.data(), (int) timeline.size(), sr, wav_fmt)) {
        return 1;
    }
    fprintf(stderr, "[OmniVoice-TTS] dub: wrote %s (%zu samples @ %d Hz, %.2f s, %zu cues)\n", output_path,
            timeline.size(), sr, (double) timeline.size() / (double) sr, cues.size());
    return 0;
}

// Full TTS synthesis path via the OmniVoice handle. Lives outside main so the
// debug paths (--llm-test, --maskgit-test) keep their lower-level init flow
// completely untouched.
static int run_tts_via_ov(const char * model_path,
                          const char * codec_path,
                          bool         use_fa,
                          bool         clamp_fp16,
                          const char * ref_wav_path,
                          const char * ref_rvq_path,
                          const char * ref_text_path,
                          const char * prompt_lang,
                          const char * prompt_instruct,
                          float        prompt_duration_sec,
                          bool         prompt_denoise,
                          bool         preprocess_prompt,
                          float        chunk_duration_sec,
                          float        chunk_threshold_sec,
                          bool         stream_by_line,
                          const char * srt_path,
                          int          mg_steps,
                          uint64_t     seed_resolved,
                          const char * dump_dir,
                          const char * output_path,
                          WavFormat    wav_fmt) {
    ov_init_params iparams;
    ov_init_default_params(&iparams);
    iparams.model_path = model_path;
    iparams.codec_path = codec_path;
    iparams.use_fa     = use_fa;
    iparams.clamp_fp16 = clamp_fp16;

    ov_context * ov = ov_init(&iparams);
    if (!ov) {
        return 1;
    }

    int rc = 0;

    // Optional reference, raw WAV or pre-encoded .rvq tokens. The raw
    // buffer goes through every preprocessing step (RMS, auto-gain,
    // add_punctuation, silence trim, hop alignment, codec encode) inside
    // ov_synthesize; pre-encoded tokens skip straight to the prompt. The
    // transcript file is common to both reference formats.
    std::vector<float>   ref_audio;
    std::vector<int32_t> ref_tokens;
    int                  ref_T = 0;
    std::string          ref_text;
    if (ref_text_path) {
        if (!read_text_file(ref_text_path, ref_text)) {
            ov_free(ov);
            return 1;
        }
    }
    if (ref_wav_path) {
        fprintf(stderr, "[CLI] Reference WAV: %s\n", ref_wav_path);
        int     n_samples = 0;
        float * raw       = audio_read_mono(ref_wav_path, 24000, &n_samples);
        if (!raw || n_samples <= 0) {
            fprintf(stderr, "[OmniVoice-TTS] FATAL: failed to load %s\n", ref_wav_path);
            free(raw);
            ov_free(ov);
            return 1;
        }
        ref_audio.assign(raw, raw + n_samples);
        free(raw);
    }
    if (ref_rvq_path) {
        const int K = ov_num_codebooks(ov);
        if (!rvq_read_file(ref_rvq_path, K, RVQ_CODE_BITS, ref_tokens, &ref_T)) {
            ov_free(ov);
            return 1;
        }
        fprintf(stderr, "[CLI] Reference RVQ: %s, K=%d T=%d\n", ref_rvq_path, K, ref_T);
    }

    std::string lang = prompt_lang ? prompt_lang : "";

    // SRT dubbing: synthesise every cue onto an absolute timeline and write
    // one WAV. Uses the reference and language resolved above, owns its own
    // duration and post filtering per cue. Returns before the single text
    // streaming and buffered paths below.
    if (srt_path) {
        rc = run_srt_dub(ov, srt_path, ref_audio, ref_tokens, ref_T, ref_text, lang, prompt_instruct, prompt_denoise,
                         preprocess_prompt, mg_steps, seed_resolved, dump_dir, output_path, wav_fmt);
        ov_free(ov);
        return rc;
    }

    // Resolve target frame count override from --duration. When unset, the
    // synthesis pipeline estimates internally and may activate long-form
    // chunking. An explicit value forces the single-shot path with that
    // exact frame count.
    int T_override = 0;
    if (prompt_duration_sec > 0.0f) {
        T_override = ov_duration_sec_to_tokens(ov, prompt_duration_sec);
    }

    // Streaming detection: -o '-' writes a wide RIFF header to stdout up
    // front and pipes encoded samples as the synthesis emits them. Any
    // other path uses the buffered route so the file gets accurate sizes
    // in its header.
    bool       stream_to_stdout = (output_path[0] == '-' && output_path[1] == '\0');
    wav_stream ws               = {};
    if (stream_to_stdout) {
        if (!wav_stream_open_stdout(&ws, 24000, wav_fmt)) {
            ov_free(ov);
            return 1;
        }
    }

    if (stream_to_stdout) {
        // Streaming stdin -> streaming stdout. Bytes arrive as the upstream
        // produces them; the incremental text chunker drives synthesis as
        // soon as a chunk of text is ready. Each chunk goes through a full
        // ov_synthesize call with on_chunk forwarding samples to the wav
        // stream sink. The text chunker is bit-perfect equivalent to the
        // offline chunk_text_punctuation with min_chunk_len = 0.
        //
        // chunk_len is computed from chunk_duration_sec assuming a typical
        // 1 frame per codepoint ratio (English speech). Languages with a
        // higher token-per-char ratio (CJK) produce shorter audio per
        // chunk; the upstream long-form path measures this ratio from the
        // full text but the streaming path cannot. The observed audio
        // chunks therefore stay bounded above by chunk_duration_sec but
        // may run shorter, which is the safe direction for prosody.
        const int frame_rate     = 24000 / 480;  // codec hop length, see codec
        const int chunk_len_text = (int) ((float) frame_rate * chunk_duration_sec);

        text_chunker_stream chunker;
        chunker.init(chunk_len_text, OMNIVOICE_MIN_CHUNK_LEN);

        int    n_emitted = 0;
        size_t bytes_in  = 0;

        // Line oriented streaming opens every utterance after the first
        // with a fresh RIFF header, so a client can split the stream into
        // one standalone WAV per line on the RIFF magic. The flag is armed
        // when a line finishes and consumed lazily at the next audio, so a
        // trailing or empty line never emits an orphan header.
        bool need_header = false;

        auto synth_one = [&](const std::string & chunk_text) -> int {
            if (need_header) {
                if (!wav_stream_write_header(&ws)) {
                    return 1;
                }
                need_header = false;
            }
            ov_tts_params params;
            ov_tts_default_params(&params);
            params.text                = chunk_text.c_str();
            params.lang                = lang.c_str();
            params.instruct            = prompt_instruct ? prompt_instruct : "";
            params.T_override          = T_override;
            params.chunk_duration_sec  = chunk_duration_sec;
            params.chunk_threshold_sec = chunk_threshold_sec;
            params.denoise             = prompt_denoise;
            params.preprocess_prompt   = preprocess_prompt;
            params.mg_seed             = seed_resolved;
            if (mg_steps > 0) {
                params.mg_num_step = mg_steps;
            }
            params.ref_audio_24k    = ref_audio.empty() ? nullptr : ref_audio.data();
            params.ref_n_samples    = (int) ref_audio.size();
            params.ref_audio_tokens = ref_tokens.empty() ? nullptr : ref_tokens.data();
            params.ref_T            = ref_T;
            params.ref_text         = ref_text.c_str();
            params.dump_dir         = dump_dir;
            params.on_chunk         = [](const float * s, int n, void * ud) -> bool {
                return wav_stream_write((wav_stream *) ud, s, n);
            };
            params.on_chunk_user_data = &ws;

            ov_status status = ov_synthesize(ov, &params, nullptr);
            if (status != OV_STATUS_OK) {
                fprintf(stderr, "[OmniVoice-TTS] streaming synth failed on chunk %d: %s\n", n_emitted, ov_last_error());
                return 1;
            }
            n_emitted++;
            return 0;
        };

        // Read loop: 4 KiB chunks, push to the incremental chunker, drain
        // ready chunks, synth each. Block on stdin between reads, no
        // polling. Suitable for piped LLM output that produces bytes at
        // its own pace. With --stream-by-line the read is line oriented:
        // every newline drains the chunker so the line synthesises now,
        // and the next line opens with a fresh RIFF header. Lines are
        // text, an embedded NUL truncates the read at strlen.
        char   buf[4096];
        FILE * in = stdin;
#if defined(_WIN32)
        _setmode(_fileno(stdin), _O_BINARY);
#endif

        while (true) {
            bool   flush = false;
            size_t r     = 0;
            if (stream_by_line) {
                if (fgets(buf, sizeof(buf), in) != nullptr) {
                    r     = strlen(buf);
                    flush = (r > 0 && buf[r - 1] == '\n');
                }
            } else {
                r = fread(buf, 1, sizeof(buf), in);
            }

            if (r > 0) {
                bytes_in += r;
                std::vector<std::string> ready = chunker.push_bytes(buf, r);

                if (flush) {
                    std::vector<std::string> tail = chunker.flush_eof();
                    ready.insert(ready.end(), std::make_move_iterator(tail.begin()),
                                 std::make_move_iterator(tail.end()));
                }

                for (const auto & ct : ready) {
                    if (synth_one(ct) != 0) {
                        wav_stream_close(&ws);
                        ov_free(ov);
                        return 1;
                    }
                }

                if (flush) {
                    need_header = true;
                }
            }
            if (feof(in) || ferror(in)) {
                break;
            }
        }

        std::vector<std::string> tail = chunker.flush_eof();
        for (const auto & ct : tail) {
            if (synth_one(ct) != 0) {
                wav_stream_close(&ws);
                ov_free(ov);
                return 1;
            }
        }

        wav_stream_close(&ws);
        ov_free(ov);
        fprintf(stderr, "[OmniVoice-TTS] streamed %d chunks (%zu bytes input) to stdout\n", n_emitted, bytes_in);
        return 0;
    }

    // Buffered path: read full stdin, single ov_synthesize, write WAV file.
    std::string text = read_stdin_text();

    // Defaults mirror OmniVoiceGenerationConfig (Python): num_step=32,
    // guidance_scale=2.0, t_shift=0.1, layer_penalty_factor=5.0,
    // position_temperature=5.0, class_temperature=0.0. ov_tts_default_params
    // sets the lot; the CLI seed lands on mg_seed below.
    ov_tts_params params;
    ov_tts_default_params(&params);
    params.text                = text.c_str();
    params.lang                = lang.c_str();
    params.instruct            = prompt_instruct ? prompt_instruct : "";
    params.T_override          = T_override;
    params.chunk_duration_sec  = chunk_duration_sec;
    params.chunk_threshold_sec = chunk_threshold_sec;
    params.denoise             = prompt_denoise;
    params.preprocess_prompt   = preprocess_prompt;
    params.mg_seed             = seed_resolved;
    if (mg_steps > 0) {
        params.mg_num_step = mg_steps;
    }
    params.ref_audio_24k    = ref_audio.empty() ? nullptr : ref_audio.data();
    params.ref_n_samples    = (int) ref_audio.size();
    params.ref_audio_tokens = ref_tokens.empty() ? nullptr : ref_tokens.data();
    params.ref_T            = ref_T;
    params.ref_text         = ref_text.c_str();
    params.dump_dir         = dump_dir;

    ov_audio audio = {};
    if (ov_synthesize(ov, &params, &audio) != OV_STATUS_OK) {
        rc = 1;
    } else if (!audio_write_wav(output_path, audio.samples, audio.n_samples, audio.sample_rate, wav_fmt)) {
        rc = 1;
    } else {
        fprintf(stderr, "[OmniVoice-TTS] TTS: wrote %s (%d samples @ %d Hz, %.2f s)\n", output_path, audio.n_samples,
                audio.sample_rate, (double) audio.n_samples / (double) audio.sample_rate);
    }
    ov_audio_free(&audio);
    ov_free(ov);
    return rc;
}

static int main_impl(int argc, char ** argv) {
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
    bool         stream_by_line         = false;
    const char * srt_path               = NULL;
    const char * ref_wav_path           = NULL;
    const char * ref_rvq_path           = NULL;
    const char * ref_text_path          = NULL;
    const char * output_path            = NULL;
    bool         use_fa                 = true;
    bool         clamp_fp16             = false;
    int          seed_arg               = -1;
    int          mg_steps               = 0;
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
        } else if (strcmp(argv[i], "--stream-by-line") == 0) {
            stream_by_line = true;
        } else if (strcmp(argv[i], "--srt") == 0 && i + 1 < argc) {
            srt_path = argv[++i];
        } else if (strcmp(argv[i], "--ref-wav") == 0 && i + 1 < argc) {
            ref_wav_path = argv[++i];
        } else if (strcmp(argv[i], "--ref-rvq") == 0 && i + 1 < argc) {
            ref_rvq_path = argv[++i];
        } else if (strcmp(argv[i], "--ref-text") == 0 && i + 1 < argc) {
            ref_text_path = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed_arg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            mg_steps = atoi(argv[++i]);
            if (mg_steps < 1) {
                fprintf(stderr, "[CLI] ERROR: --steps must be >= 1\n");
                return 1;
            }
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

    // Mode resolution: llm_test_in OR maskgit_test_mode are debug, the
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
    if (ref_wav_path && ref_rvq_path) {
        fprintf(stderr, "[CLI] ERROR: --ref-wav and --ref-rvq are mutually exclusive\n");
        return 1;
    }
    if ((ref_wav_path || ref_rvq_path) && !ref_text_path) {
        fprintf(stderr, "[CLI] ERROR: --ref-wav / --ref-rvq requires --ref-text <path>\n");
        return 1;
    }
    if ((ref_wav_path || ref_rvq_path) && !tts_mode) {
        fprintf(stderr, "[CLI] ERROR: --ref-wav / --ref-rvq is only supported in synthesis mode\n");
        return 1;
    }
    if (srt_path && !tts_mode) {
        fprintf(stderr, "[CLI] ERROR: --srt is only supported in synthesis mode\n");
        return 1;
    }
    if (srt_path && output_path[0] == '-' && output_path[1] == '\0') {
        fprintf(stderr, "[CLI] ERROR: --srt writes a timeline WAV, incompatible with streaming -o '-'\n");
        return 1;
    }
    if (srt_path && stream_by_line) {
        fprintf(stderr, "[CLI] ERROR: --srt is incompatible with --stream-by-line\n");
        return 1;
    }
    if (srt_path && prompt_duration_sec > 0.0f) {
        fprintf(stderr, "[CLI] ERROR: --srt derives per cue duration from the SRT, drop --duration\n");
        return 1;
    }

    // Resolve sampling seed: -1 picks a fresh random seed from std::random_device,
    // any other value is used verbatim for reproducible runs across the maskgit
    // RNG.
    uint64_t seed_resolved = (seed_arg < 0) ? (uint64_t) std::random_device{}() : (uint64_t) seed_arg;
    fprintf(stderr, "[CLI] Seed: %llu%s\n", (unsigned long long) seed_resolved, (seed_arg < 0) ? " (random)" : "");

    // TTS mode runs through the OmniVoice handle. Debug modes (--llm-test,
    // --maskgit-test) keep their lower-level init flow below.
    if (tts_mode) {
        return run_tts_via_ov(model_path, codec_path, use_fa, clamp_fp16, ref_wav_path, ref_rvq_path, ref_text_path,
                              prompt_lang, prompt_instruct, prompt_duration_sec, prompt_denoise, preprocess_prompt,
                              chunk_duration_sec, chunk_threshold_sec, stream_by_line, srt_path, mg_steps,
                              seed_resolved, dump_dir, output_path, wav_fmt);
    }

    BackendPair bp = backend_init("LM");
    if (!bp.backend) {
        return 1;
    }

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
            if (mg_steps > 0) {
                mg_cfg.num_step = mg_steps;
            }

            std::string text         = read_stdin_text();
            std::string lang         = prompt_lang ? prompt_lang : "";
            std::string raw_instruct = prompt_instruct ? prompt_instruct : "";
            std::string instruct;
            if (!pipeline_tts_resolve_instruct(&vd, text, raw_instruct, &instruct)) {
                rc = 1;
            } else {
                // Resolve target frame count: explicit --duration in seconds
                // (OmniVoice runs at a fixed 25 fps: 24000 / 960), otherwise
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
    }

    // Shared cleanup for both debug paths (--llm-test and --maskgit-test).
    // The TTS path returns earlier through run_tts_via_ov, which manages
    // its own ov_free / backend_release pair.
    pipeline_tts_free(&pt);
    backend_release(bp.backend, bp.cpu_backend);
    return rc;
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    // Top-level boundary: the lib now signals fatal load errors via
    // exceptions instead of exit(1). The TTS path goes through ov_init
    // which catches them internally, but the lower-level debug paths
    // (--llm-test, --maskgit-test) call pipeline_tts_load directly and
    // need an explicit guard so the user sees a clean error line instead
    // of a std::terminate trace.
    try {
        return main_impl(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "[OmniVoice-TTS] FATAL: %s\n", e.what());
        return 1;
    }
}
