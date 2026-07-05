// tts-server.cpp: OpenAI-compatible HTTP server backed by the
// omnivoice ABI. Loads an LM + codec once, GPU resident, and serves
// synthesis over POST /v1/audio/speech. The shared core lives in
// src/tts-server.h ; this file only wires the ov_* ABI into the adapter.
//
// OmniVoice has no named speaker table, so GET /v1/voices is empty. The
// OAI voice field is ignored ; the instructions field drives voice design.

#include "tts-server.h"

#include "omnivoice.h"
#include "version.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>

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
            "  --lang <str>            Language label (default 'None')\n"
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
        } else if (!std::strcmp(arg, "--help") || !std::strcmp(arg, "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", arg);
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
    // OmniVoice carries no named speaker table : voices stays empty.

    // The adapter always drives the streaming pipeline : on_chunk routes to
    // the shared sink, which either streams to the socket (pcm) or fills a
    // one-shot buffer (wav). OmniVoice streams at chunk_duration_sec
    // granularity, the same path either way.
    be.synthesize = [ov, &lang](const tts_request & req, const tts_sink & sink, std::string & err) -> int {
        struct ov_tts_params p;
        ov_tts_default_params(&p);
        p.text = req.input.c_str();
        p.lang = lang.c_str();
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
