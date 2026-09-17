// tts-server.cpp: OpenAI-compatible HTTP server backed by the
// omnivoice ABI. Loads an LM + codec once, GPU resident, and serves
// synthesis over POST /v1/audio/speech. The shared core lives in
// src/tts-server.h ; this file only wires the ov_* ABI into the adapter.
//
// OmniVoice has no named speaker table: every voice comes from the
// registry, filled over POST /v1/audio/voices. A request without a voice
// runs voice design from the instructions field.

#include "tts-server.h"

#include "omnivoice.h"
#include "rvq-file.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Packed .rvq code width, fixed by the OmniVoice codec (2048 entries per
// codebook).
static const int RVQ_CODE_BITS = 11;

// One registered cloned voice: the RVQ codes in the ABI ownership contract
// (malloc-owned, released by ov_voice_ref_free) plus the reference
// transcript that anchors the clone.
struct voice_entry {
    struct ov_voice_ref ref;
    std::string         ref_text;
};

// Registered voices, name keyed. Every access happens under
// g_voices_mutex ; the synthesize lookup copies the codes out so a
// concurrent replace or delete never frees a buffer a running synthesis
// still reads.
static std::mutex                                   g_voices_mutex;
static std::unordered_map<std::string, voice_entry> g_voices;

static void print_usage(const char * prog) {
    fprintf(stderr, "omnivoice.cpp %s\n\n", OMNIVOICE_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          LLM GGUF (F32 / BF16 / Q8_0)\n"
            "  --codec <gguf>          Codec GGUF (omnivoice-tokenizer-*.gguf)\n\n"
            "Optional:\n"
            "  --host <ip>             Listen address (default: 127.0.0.1)\n"
            "  --port <n>              Listen port (default: 8080)\n"
            "  --lang <str>            Language label when a request omits one (default 'None')\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n",
            prog);
}

// Trim a path down to its file name for the reported model id.
static std::string basename_of(const char * path) {
    std::string s = path;
    size_t      p = s.find_last_of("/\\");
    return p == std::string::npos ? s : s.substr(p + 1);
}

int main(int argc, char ** argv) {
    const char *  model_path = NULL;
    const char *  codec_path = NULL;
    std::string   lang       = "None";
    server_config cfg;
    bool          use_fa     = true;
    bool          clamp_fp16 = false;

    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (!std::strcmp(arg, "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!std::strcmp(arg, "--codec") && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (!std::strcmp(arg, "--host") && i + 1 < argc) {
            cfg.host = argv[++i];
        } else if (!std::strcmp(arg, "--port") && i + 1 < argc) {
            cfg.port = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--lang") && i + 1 < argc) {
            lang = argv[++i];
        } else if (!std::strcmp(arg, "--no-fa")) {
            use_fa = false;
        } else if (!std::strcmp(arg, "--clamp-fp16")) {
            clamp_fp16 = true;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !codec_path) {
        print_usage(argv[0]);
        return 1;
    }

    struct ov_init_params iparams;
    ov_init_default_params(&iparams);
    iparams.model_path = model_path;
    iparams.codec_path = codec_path;
    iparams.use_fa     = use_fa;
    iparams.clamp_fp16 = clamp_fp16;

    struct ov_context * ov = ov_init(&iparams);
    if (!ov) {
        fprintf(stderr, "[Server] FATAL: %s\n", ov_last_error());
        return 1;
    }

    tts_backend be;
    be.model_id = basename_of(model_path);

    // A WAV payload is encoded server side through the single synthesis
    // context, so the extraction takes g_synth_mutex. An .rvq payload is
    // unpacked in place and needs no GPU. Re-registering a name replaces
    // the previous entry.
    be.register_voice = [ov](const tts_voice_upload & up, std::string & err) -> bool {
        voice_entry entry;
        entry.ref      = {};
        entry.ref_text = up.ref_text;

        if (!up.wav.empty()) {
            int     T   = 0;
            float * pcm = audio_read_mono_buf((const uint8_t *) up.wav.data(), up.wav.size(), 24000, &T);
            if (!pcm) {
                err = "cannot decode the WAV payload";
                return false;
            }
            enum ov_status rc;
            {
                std::lock_guard<std::mutex> lock(g_synth_mutex);
                rc = ov_extract_voice_ref(ov, pcm, T, &entry.ref);
            }
            free(pcm);
            if (rc != OV_STATUS_OK) {
                err = ov_last_error();
                return false;
            }
        } else {
            std::vector<int32_t> codes;
            int                  ref_T = 0;
            const int            K     = ov_num_codebooks(ov);
            if (!rvq_read_buf((const uint8_t *) up.rvq.data(), up.rvq.size(), K, RVQ_CODE_BITS, codes, &ref_T)) {
                err = "'rvq_b64' does not decode to a valid packed code stream";
                return false;
            }
            const size_t codes_bytes = codes.size() * sizeof(int32_t);
            entry.ref.ref_codes      = (int32_t *) malloc(codes_bytes);
            if (!entry.ref.ref_codes) {
                err = "out of memory";
                return false;
            }
            std::memcpy(entry.ref.ref_codes, codes.data(), codes_bytes);
            entry.ref.ref_T         = ref_T;
            entry.ref.num_codebooks = K;
        }

        std::lock_guard<std::mutex> lock(g_voices_mutex);
        auto                        it = g_voices.find(up.name);
        if (it != g_voices.end()) {
            ov_voice_ref_free(&it->second.ref);
            g_voices.erase(it);
        }
        fprintf(stderr, "[Server] voice '%s' registered (K=%d T=%d)\n", up.name.c_str(), entry.ref.num_codebooks,
                entry.ref.ref_T);
        g_voices.emplace(up.name, std::move(entry));
        return true;
    };

    be.remove_voice = [](const std::string & name) -> bool {
        std::lock_guard<std::mutex> lock(g_voices_mutex);
        auto                        it = g_voices.find(name);
        if (it == g_voices.end()) {
            return false;
        }
        ov_voice_ref_free(&it->second.ref);
        g_voices.erase(it);
        return true;
    };

    be.registered_voices = []() -> std::vector<std::string> {
        std::lock_guard<std::mutex> lock(g_voices_mutex);
        std::vector<std::string>    names;
        names.reserve(g_voices.size());
        for (const auto & kv : g_voices) {
            names.push_back(kv.first);
        }
        return names;
    };

    // The adapter always drives the streaming pipeline : on_chunk routes to
    // the shared sink, which either streams to the socket (pcm) or fills a
    // one-shot buffer (wav). OmniVoice streams at chunk_duration_sec
    // granularity, the same path either way. A named voice injects its
    // pre-encoded reference ; an unknown name is rejected instead of
    // silently falling back to voice design.
    be.synthesize = [ov, &lang](const tts_request & req, const tts_sink & sink, std::string & err) -> int {
        struct ov_tts_params p;
        ov_tts_default_params(&p);
        p.text = req.input.c_str();
        p.lang = req.lang.empty() ? lang.c_str() : req.lang.c_str();

        // Copy the registered codes out under the lock: the synthesis runs
        // for seconds, during which another connection may replace or
        // delete the entry.
        std::vector<int32_t> voice_codes;
        std::string          voice_ref_text;
        int                  voice_ref_T = 0;
        if (!req.voice.empty()) {
            std::lock_guard<std::mutex> lock(g_voices_mutex);
            auto                        vit = g_voices.find(req.voice);
            if (vit == g_voices.end()) {
                err = "unknown voice '" + req.voice + "'";
                return (int) OV_STATUS_INVALID_PARAMS;
            }
            const voice_entry & v = vit->second;
            voice_codes.assign(v.ref.ref_codes, v.ref.ref_codes + (size_t) v.ref.num_codebooks * (size_t) v.ref.ref_T);
            voice_ref_T    = v.ref.ref_T;
            voice_ref_text = v.ref_text;
        }
        if (!voice_codes.empty()) {
            p.ref_audio_tokens = voice_codes.data();
            p.ref_T            = voice_ref_T;
            p.ref_text         = voice_ref_text.c_str();
        }
        if (!req.instructions.empty()) {
            p.instruct = req.instructions.c_str();
        }

        // Seed resolution mirrors the CLI: negative or absent draws a
        // fresh hardware random seed, anything else lands verbatim on
        // the MaskGIT sampler for reproducible output.
        p.mg_seed = (req.seed < 0) ? (uint64_t) std::random_device{}() : (uint64_t) req.seed;

        // Trampoline : the C ABI on_chunk forwards to the C++ sink.
        const tts_sink * sink_ptr = &sink;
        p.on_chunk                = [](const float * s, int ns, void * u) -> bool {
            return (*static_cast<const tts_sink *>(u))(s, ns);
        };
        p.on_chunk_user_data = (void *) sink_ptr;

        struct ov_audio out = {};
        enum ov_status  rc  = ov_synthesize(ov, &p, &out);
        ov_audio_free(&out);
        if (rc != OV_STATUS_OK) {
            err = ov_last_error();
            return (int) rc;
        }
        return 0;
    };

    int rc = tts_server_run(be, cfg);
    ov_free(ov);
    return rc;
}
