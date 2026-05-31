// omnivoice-tts-server.cpp : HTTP TTS server for OmniVoice.
//
// Exposes a REST API for text-to-speech generation with voice cloning support.
// Uses cpp-httplib for the webserver, JSON for request/response handling.
//
// Endpoints:
//   POST /v1/audio/speech - Generate speech from text (OpenAI API compliant)
//
// Request body (JSON):
//   {
//     "input": string, required, max 4096 chars
//     "instructions": string, optional, max 4096 chars
//     "response_format": string, optional, default "wav16" (wav16|wav24|wav32)
//     "speed": number, optional, default 1.0 (0.25-4.0)
//     "lang": string, optional, default "None"
//     "model": string, optional, ignored
//     "voice": string, optional, ignored
//     "stream_format": string, optional, ignored
//   }
//
// Response body (binary):
//   Raw WAV audio data with chunked transfer encoding

#include "audio-io.h"
#include "omnivoice.h"
#include "version.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>

using namespace std::literals::chrono_literals;

#include <httplib.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Global variables
ov_context * ov = nullptr;

float        chunk_duration_sec  = 15.0f;
float        chunk_threshold_sec = 30.0f;
bool         prompt_denoise      = true;
bool         preprocess_prompt   = true;
uint64_t     seed_resolved;
const char * default_wav_fmt_name = "wav16";
WavFormat    default_wav_fmt      = WAV_S16;
const char * default_lang         = nullptr;
const char * default_instruct     = nullptr;
float        prompt_duration_sec  = 0.0f;

// Mutex
std::mutex generation_mutex;

// Webserver
httplib::Server server;

// Clean up
void clean_up() {
    if (server.is_running()) {
        server.stop();
    }

    if (ov) {
        ov_free(ov);
        ov = nullptr;
    }
}

// Global signal handler
void signal_handler(int signal) {
    if (signal == SIGINT) {
        clean_up();
        std::cout << "[Server] Terminated\n";
    }
}

// Generate audio
void generate_audio_task(const std::string &         input,
                         const std::string &         lang,
                         const std::string &         instructions,
                         WavFormat                   wav_fmt,
                         std::promise<std::string> & result) {
    // Lock the mutex
    std::lock_guard lock(generation_mutex);

    try {
        // Resolve target frame count override from --duration. When
        // unset, the synthesis pipeline estimates internally and may
        // activate long-form chunking. An explicit value forces the
        // single-shot path with that exact frame count.
        int T_override = 0;
        if (prompt_duration_sec > 0.0f) {
            T_override = ov_duration_sec_to_tokens(ov, prompt_duration_sec);
        }

        // Defaults mirror OmniVoiceGenerationConfig (Python) :
        // num_step=32, guidance_scale=2.0, t_shift=0.1,
        // layer_penalty_factor=5.0, position_temperature=5.0,
        // class_temperature=0.0. The seed is plumbed from the CLI.
        ov_tts_params params;
        ov_tts_default_params(&params);
        params.text                = input.c_str();
        params.lang                = !lang.empty() ? lang.c_str() : nullptr;
        params.instruct            = !instructions.empty() ? instructions.c_str() : nullptr;
        params.T_override          = T_override;
        params.chunk_duration_sec  = chunk_duration_sec;
        params.chunk_threshold_sec = chunk_threshold_sec;
        params.denoise             = prompt_denoise;
        params.preprocess_prompt   = preprocess_prompt;
        params.mg_seed             = seed_resolved;
        params.ref_audio_24k       = nullptr;  // TODO: voice cloning support
        params.ref_n_samples       = 0;        // TODO: voice cloning support
        params.ref_text            = nullptr;  // TODO: voice cloning support
        params.dump_dir            = nullptr;

        ov_audio audio = {};
        if (ov_synthesize(ov, &params, &audio) != OV_STATUS_OK) {
            throw std::runtime_error("TTS generation failed");
        }

        // Convert to WAV
        std::string audio_data = audio_encode_wav(audio.samples, audio.n_samples, audio.sample_rate, wav_fmt);
        fprintf(stderr, "[Server] TTS: wrote %d samples @ %d Hz, %.2f s\n", audio.n_samples, audio.sample_rate,
                static_cast<double>(audio.n_samples) / static_cast<double>(audio.sample_rate));
        ov_audio_free(&audio);

        if (audio_data.empty()) {
            throw std::runtime_error("Failed to encode WAV");
        }

        result.set_value(std::move(audio_data));
    } catch (...) {
        result.set_exception(std::current_exception());
    }
}

// Build error response
static std::string build_error_response(const std::string & message) {
    return json{
        { "message", message }
    }.dump();
}

// Print usage information
static void print_usage(const char * prog) {
    fprintf(stderr, "omnivoice-tts-server %s\n\n", OMNIVOICE_VERSION);
    fprintf(stderr, "Usage: %s --model <gguf> --codec <gguf> [options]\n\n", prog);
    fprintf(stderr, "OmniVoice TTS Server - HTTP API for text-to-speech generation\n\n");
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  --model <path>          LLM GGUF model file (F32 / BF16 / Q8_0)\n");
    fprintf(stderr, "  --codec <path>          Codec GGUF file (omnivoice-tokenizer-*.gguf)\n\n");
    fprintf(stderr, "Server Configuration:\n");
    fprintf(stderr, "  -l, --listen-ip <ip>    Server listen IP (default: 127.0.0.1)\n");
    fprintf(stderr, "  --listen-port <port>    Server listen port (default: 1234)\n\n");
    fprintf(stderr, "TTS Generation:\n");
    fprintf(stderr, "  --format <fmt>          Output format: wav16, wav24, wav32 (default: wav16)\n");
    fprintf(stderr, "  --lang <str>            Default language label (default: 'None')\n");
    fprintf(stderr, "  --instruct <str>        Default style instruction (default: 'None')\n");
    fprintf(stderr, "  --duration <sec>        Default output duration in seconds (default: auto-estimate)\n");
    fprintf(stderr, "  --chunk-duration <sec>  Long-form chunk duration (default: 15.0, <= 0 disables chunking)\n");
    fprintf(stderr, "  --chunk-threshold <sec> Activate chunking above this estimated duration (default: 30.0)\n");
    fprintf(stderr, "  --no-denoise            Omit the <|denoise|> prefix in generation\n");
    // TODO: voice cloning support
    // fprintf(stderr,
    //         "  --no-preprocess-prompt  Skip reference WAV silence trim and ref-text terminal punctuation\n");
    fprintf(stderr, "  --no-flash-attention    Disable flash attention (matches Python eager attention)\n");
    fprintf(stderr, "  --clamp-fp16            Clamp hidden states to FP16 range\n");
    fprintf(stderr, "  --seed <int>            Sampling seed (default: -1 for random)\n\n");
    fprintf(stderr, "Endpoints:\n");
    fprintf(stderr, "  POST /v1/audio/speech    Generate speech from text (OpenAPI-compliant)\n\n");
    fprintf(stderr, "Request body (JSON):\n");
    fprintf(stderr, "  {\n");
    fprintf(stderr, "    \"input\": \"text to synthesize\",\n");
    fprintf(stderr, "    \"instructions\": \"friendly tone\",\n");
    fprintf(stderr, "    \"response_format\": \"wav16\",\n");
    fprintf(stderr, "    \"speed\": 1.0,\n");
    fprintf(stderr, "    \"lang\": \"English\",\n");
    fprintf(stderr, "    \"model\": \"ignored\",\n");
    fprintf(stderr, "    \"voice\": \"ignored\",\n");
    fprintf(stderr, "    \"stream_format\": \"ignored\"\n");
    fprintf(stderr, "  }\n");
}

int main(int argc, char ** argv) {
    // Install the signal handler
    std::signal(SIGINT, signal_handler);

    // Command line arguments
    const char * model_path  = nullptr;
    const char * codec_path  = nullptr;
    bool         use_fa      = true;
    bool         clamp_fp16  = false;
    int          seed_arg    = -1;
    const char * listen_ip   = "127.0.0.1";
    int          listen_port = 1234;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--listen-ip") == 0) {
            if (i + 1 < argc) {
                listen_ip = argv[++i];
            }
        } else if (strcmp(argv[i], "--listen-port") == 0) {
            if (i + 1 < argc) {
                listen_port = std::atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            default_wav_fmt_name = argv[++i];
            if (!audio_parse_format(default_wav_fmt_name, default_wav_fmt)) {
                fprintf(stderr, "[Server] ERROR: unknown format: %s\n", default_wav_fmt_name);
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
            default_lang = argv[++i];
        } else if (strcmp(argv[i], "--instruct") == 0 && i + 1 < argc) {
            default_instruct = argv[++i];
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            prompt_duration_sec = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--chunk-duration") == 0 && i + 1 < argc) {
            chunk_duration_sec = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--chunk-threshold") == 0 && i + 1 < argc) {
            chunk_threshold_sec = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--no-denoise") == 0) {
            prompt_denoise = false;
            // TODO Ref WAV
            // } else if (strcmp(argv[i], "--no-preprocess-prompt") == 0) {
            //     preprocess_prompt = false;
        } else if (strcmp(argv[i], "--no-flash-attention") == 0) {
            use_fa = false;
        } else if (strcmp(argv[i], "--clamp-fp16") == 0) {
            clamp_fp16 = true;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed_arg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[Server] ERROR: unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "[Server] ERROR: --model is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!codec_path) {
        fprintf(stderr, "[Server] ERROR: --codec is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Resolve sampling seed : -1 picks a fresh random seed from std::random_device,
    // any other value is used verbatim for reproducible runs across the maskgit
    // RNG.
    seed_resolved = (seed_arg < 0) ? (uint64_t) std::random_device{}() : (uint64_t) seed_arg;
    fprintf(stderr, "[Server] Seed: %llu%s\n", static_cast<unsigned long long>(seed_resolved),
            (seed_arg < 0) ? " (random)" : "");

    // Initialize OmniVoice handle
    ov_init_params iparams;
    ov_init_default_params(&iparams);
    iparams.model_path = model_path;
    iparams.codec_path = codec_path;
    iparams.use_fa     = use_fa;
    iparams.clamp_fp16 = clamp_fp16;

    ov = ov_init(&iparams);
    if (!ov) {
        std::cerr << "[Server] ERROR: Failed to initialize OmniVoice handle\n";
        return 1;
    }

    /**
     * HTTP Webserver
     **/

    // Server URL
    std::string server_url;
    {
        std::stringstream ss;
        ss << "http://" << listen_ip << ":" << listen_port << "/";
        server_url = ss.str();
    };

    // CORS headers for all routes
    server.Options("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("Allow: GET, POST, OPTIONS", "text/plain");
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.set_header("Access-Control-Max-Age", "86400");
    });

    // API Documentation endpoint
    server.Get("/v1/api-docs", [&server_url](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Content-Type", "application/json");

        json api_docs = {
            { "openapi",    "3.1.0"                                                                },
            { "info",       { { "title", "OmniVoice TTS API" }, { "version", OMNIVOICE_VERSION } } },
            { "servers",    { { "url", server_url + "v1" } }                                       },
            { "paths",
             { { "/audio/speech",
                  { { "post",
                      { { "summary", "Generates audio from the input text" },
                        { "tags", { "Audio" } },
                        { "operationId", "createSpeech" },
                        { "requestBody",
                          { { "required", true },
                            { "content",
                              { { "application/json",
                                  { { "schema", { { "$ref", "#/components/schemas/CreateSpeechRequest" } } } } } } },
                            { "description", "The text to generate audio for" } } },
                        { "responses",
                          { { "200",
                              { { "description", "OK" },
                                { "headers",
                                  { { "Transfer-Encoding",
                                      { { "schema", { "type", "string" } }, { "description", "chunked" } } } } },
                                { "content",
                                  { { "application/octet-stream",
                                      { { "schema",
                                          { { "type", "string" },
                                            { "format", "binary" } } } } } } } } } } } } } } } }   },
            { "components",
             { { "schemas",
                  { { "CreateSpeechRequest",
                      { { "type", "object" },
                        { "additionalProperties", false },
                        { "properties",
                          { { "model",
                              { { "type", "string" },
                                { "description",
                                  "Dummy, ignored. The model specified on the command line is always used." } } },
                            { "input",
                              { { "type", "string" },
                                { "description",
                                  "The text to generate audio for. The maximum length is 4096 characters." },
                                { "maxLength", 4096 } } },
                            { "instructions",
                              { { "type", "string" },
                                { "description",
                                  "Control the voice of your generated audio with additional instructions. Valid "
                                  "English items: american accent, australian accent, british accent, canadian accent, "
                                  "child, chinese accent, elderly, female, high pitch, indian accent, japanese accent, "
                                  "korean accent, low pitch, male, middle-aged, moderate pitch, portuguese accent, "
                                  "russian accent, teenager, very high pitch, very low pitch, whisper, young adult. "
                                  "Valid Chinese items: "
                                  "东北话，中年，中音调，云南话，低音调，儿童，四川话，女，宁夏话，少年，极低音调，极高"
                                  "音调，桂林话，河南话，济南话，甘肃话，男，石家庄话，老年，耳语，贵州话，陕西话，青岛"
                                  "话，青年，高音调. Tip: Use only English or only Chinese instructs. English "
                                  "instructs should use comma + space (e.g. 'male, indian accent'), Chinese instructs "
                                  "should use full-width comma (e.g. '男，河南话')." },
                                { "maxLength", 4096 } } },
                            { "voice", { { "type", "string" }, { "description", "Dummy, ignored" } } },
                            { "response_format",
                              { { "type", "string" },
                                { "description",
                                  "The format to audio in. Supported formats are `wav16`, `wav24`, `wav32" },
                                { "default", "wav16" },
                                { "enum", { "wav16", "wav24", "wav32" } } } },
                            { "speed",
                              { { "type", "number" },
                                { "description",
                                  "Dummy, ignored. The speed of the generated audio. Select a value from `0.25` to "
                                  "`4.0`. `1.0` is the default." },
                                { "default", 1.0 },
                                { "minimum", 0.25 },
                                { "maximum", 4.0 } } },
                            { "stream_format",
                              { { "type", "string" },
                                { "description", "Dummy, ignored. The audio is transported as a BLOB." },
                                { "default", "audio" },
                                { "enum", { "sse", "audio" } } } } } },
                        { "required", { "input" } } } } } } }                                      }
        };

        res.set_content(api_docs.dump(2), "application/json");
    });

    // Main TTS endpoint (OpenAPI-compliant /v1/audio/speech)
    server.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) mutable {
        // Parse request
        std::string error_msg;
        json        req_json;

        if (!req.has_header("Content-Type") || req.get_header_value("Content-Type") != "application/json") {
            error_msg  = "Content-Type must be application/json";
            res.status = 400;
            res.set_content(build_error_response(error_msg), "application/json");
            return;
        }

        try {
            req_json = json::parse(req.body);
        } catch (const json::parse_error & e) {
            error_msg  = "Invalid JSON: " + std::string(e.what());
            res.status = 400;
            res.set_content(build_error_response(error_msg), "application/json");
            return;
        }

        // Validate required field: input (max 4096 chars)
        if (!req_json.contains("input") || !req_json["input"].is_string() || req_json["input"].empty()) {
            error_msg  = "Missing or invalid required field: input";
            res.status = 400;
            res.set_content(build_error_response(error_msg), "application/json");
            return;
        }

        std::string input = req_json["input"].get<std::string>();
        if (input.length() > 4096) {
            error_msg  = "'input' exceeds maximum length of 4096 characters";
            res.status = 400;
            res.set_content(build_error_response(error_msg), "application/json");
            return;
        }

        // Extract optional parameters with defaults
        std::string lang = req_json.contains("lang") && req_json["lang"].is_string() ?
                               req_json["lang"].get<std::string>() :
                               (default_lang ? default_lang : "");

        std::string instructions = req_json.contains("instructions") && req_json["instructions"].is_string() ?
                                       req_json["instructions"].get<std::string>() :
                                       (default_instruct ? default_instruct : "");

        std::string response_format = default_wav_fmt_name;
        WavFormat   wav_fmt         = default_wav_fmt;
        if (req_json.contains("response_format") && req_json["response_format"].is_string()) {
            response_format = req_json["response_format"].get<std::string>();
            if (!audio_parse_format(response_format.c_str(), wav_fmt)) {
                error_msg  = "Invalid response_format. Must be one of: wav16, wav24, wav32";
                res.status = 400;
                res.set_content(build_error_response(error_msg), "application/json");
                return;
            }
        }

        // Validate speed (dummy field but must be in range 0.25-4.0)
        // if (req_json.contains("speed") && req_json["speed"].is_number()) {
        //     double speed = req_json["speed"].get<double>();
        //     if (speed < 0.25 || speed > 4.0) {
        //         error_msg = "Invalid speed. Must be between 0.25 and 4.0";
        //         res.status = 400;
        //         res.set_content(build_error_response(error_msg), "application/json");
        //         return;
        //     }
        // }

        // Dummy fields (accepted but ignored for API compatibility)
        const char * dummy_fields[] = { "model", "voice", "speed", "stream_format" };
        for (uint8_t i = 0; i < 4; i++) {
            const char * field_name = dummy_fields[i];
            if (req_json.contains(field_name)) {
                std::cerr << "WARNING: value " << std::quoted(req_json[field_name].get<std::string>())
                          << " ignored for dummy field " << std::quoted(field_name) << "\n";
            }
        }

        // Generate audio with global backend in an async worker
        std::promise<std::string> audio_promise;
        std::future<std::string>  audio_future = audio_promise.get_future();

        // create an independent worker thread
        std::thread worker_thread(generate_audio_task, input, lang, instructions, wav_fmt, std::ref(audio_promise));
        worker_thread.detach();

        // track the future status
        std::future_status status;
        do {
            status = audio_future.wait_for(1s);
            if (req.is_connection_closed()) {
                // no reason to wait in case the client disconnected
                break;
            }
        } while (status != std::future_status::ready);

        if (status == std::future_status::ready) {
            // extract the data
            std::string data;
            try {
                data = audio_future.get();
            } catch (const std::exception & e) {
                error_msg = e.what();
                std::cerr << "Got an exception " << std::quoted(error_msg) << std::endl;
                res.status = 500;
                res.set_content(build_error_response(error_msg), "application/json");
                return;
            }

            // Send binary WAV data
            res.status = 200;
            res.set_content(data, "application/octet-stream");
        }
    });

    // Exception handler
    server.set_exception_handler([](const httplib::Request & req, httplib::Response & res, std::exception_ptr eptr) {
        try {
            if (eptr) {
                std::rethrow_exception(eptr);
            }
        } catch (std::exception & e) {
            std::cerr << "Error: " << e.what() << std::endl;
            res.status = 500;
            res.set_content(build_error_response("unknown error"), "application/json");
        }
    });

    // Logger
    server.set_logger([](const auto & req, const auto & res) {
        auto now = std::time(nullptr);
        char timebuf[32];
        std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        std::cout << timebuf << " " << req.remote_addr << " "
                  << "\"" << req.method << " " << req.path << "\" " << res.status << " " << res.body.size() << "B"
                  << std::endl;
    });

    // Server info
    std::cout << "Starting OmniVoice TTS Server on " << listen_ip << ":" << listen_port << "\n";
    std::cout << "  Model: " << model_path << "\n";
    std::cout << "  Codec: " << codec_path << "\n";
    std::cout << "  Default format: "
              << (default_wav_fmt == WAV_S16 ? "wav16" : (default_wav_fmt == WAV_S24 ? "wav24" : "wav32")) << "\n";
    std::cout << "  Default lang: " << (default_lang ? default_lang : "None") << "\n";
    std::cout << "  Default instruct: " << (default_instruct ? default_instruct : "None") << "\n";
    std::cout << "  Chunk duration: " << chunk_duration_sec << "s\n";
    std::cout << "  Chunk threshold: " << chunk_threshold_sec << "s\n";
    std::cout << "  Denoise: " << (prompt_denoise ? "enabled" : "disabled") << "\n";
    std::cout << "  Flash attention: " << (use_fa ? "enabled" : "disabled") << "\n";
    std::cerr << "  FP16 clamp  " << (clamp_fp16 ? "enabled" : "disabled") << "\n";
    std::cout << "  Press Ctrl+C to stop\n\n";

    // Start server
    server.listen(listen_ip, listen_port);

    // clean up
    clean_up();

    return 0;
}
