#pragma once
// tts-server.h: OpenAI-compatible TTS HTTP core for omnivoice.cpp.
//
// One synthesis context lives GPU resident for the process lifetime. The
// tool fills a tts_backend adapter that wires ov_synthesize and the voice
// registry into the generic sink, then calls tts_server_run. This file
// holds the HTTP layer, tuning, OAI parsing and audio framing ; the ABI
// stays entirely on the adapter side.
//
// Endpoints:
//   POST   /v1/audio/speech         OAI text-to-speech
//   GET    /v1/audio/voices         registered cloned voices
//   POST   /v1/audio/voices         register a cloned voice: {name, ref_text,
//                                   wav_b64} encodes server side, {name,
//                                   ref_text, rvq_b64} takes pre-encoded codes
//   DELETE /v1/audio/voices/{name}  drop a registered voice
//   GET    /v1/models               single loaded model
//   GET    /health                  liveness probe
//
// Audio out: response_format "pcm" streams s16le 24 kHz mono chunked as it
// is generated (real time), "wav" returns a one-shot RIFF file. pcm is the
// default so streaming is on unless the client asks for a file.

#include "../vendor/cpp-httplib/httplib.h"
#include "audio-io.h"
#include "yyjson.h"

#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

// One synthesis request parsed from the OAI JSON body.
struct tts_request {
    std::string input;         // text to speak
    std::string lang;          // language label, empty keeps the server default
    std::string voice;         // registered voice name, empty runs voice design
    std::string instructions;  // OAI instructions, mapped to the ABI instruct field
    std::string format;        // "pcm" (stream) or "wav" (one-shot)

    // Optional sampling seed. Negative or absent draws a hardware
    // random seed per request, matching the CLI convention; anything
    // else forwards verbatim for reproducible output.
    int64_t seed;

    // Optional MaskGIT step count. Absent keeps the library default
    // (32), anything else must be positive, matching the CLI --steps.
    int steps;
};

// One voice registration parsed from the POST /v1/audio/voices JSON body.
// Exactly one payload form is present: wav holds decoded base64 WAV bytes
// for server side encoding, or rvq holds the raw contents of a pre-encoded
// .rvq file. ref_text carries the reference transcript.
struct tts_voice_upload {
    std::string name;
    std::string ref_text;
    std::string wav;  // WAV file bytes
    std::string rvq;  // .rvq file bytes, packed codes
};

// The adapter pushes mono f32 24 kHz audio here. Returns false to abort the
// synthesis (client gone or cancellation), which propagates into the ABI
// on_chunk and stops generation.
using tts_sink = std::function<bool(const float * samples, int n_samples)>;

// Adapter implemented by each project tool.
struct tts_backend {
    std::string model_id;  // reported by GET /v1/models
    // Run synthesis. When the request streams, the adapter routes the ABI
    // on_chunk to sink ; otherwise it pushes the whole buffer once. Returns
    // the ABI status (0 on success), and fills err with the ABI message on
    // failure. The shared layer maps the status to an HTTP code.
    std::function<int(const tts_request & req, const tts_sink & sink, std::string & err)> synthesize;

    // Voice registry: register_voice stores or replaces a cloned voice,
    // remove_voice drops one (false when absent), registered_voices lists
    // the current names for GET /v1/audio/voices.
    std::function<bool(const tts_voice_upload & up, std::string & err)> register_voice;
    std::function<bool(const std::string & name)>                       remove_voice;
    std::function<std::vector<std::string>()>                           registered_voices;
};

struct server_config {
    std::string host = "127.0.0.1";
    int         port = 8080;
};

// Single GPU context : synthesis is serialised FIFO across connections.
static std::mutex        g_synth_mutex;
static httplib::Server * g_svr = nullptr;

static void tts_on_signal(int) {
    if (g_svr) {
        g_svr->stop();
    }
}

// Clamp to [-1, 1] and scale to s16. lrintf rounds to nearest, ties to even.
static inline int16_t tts_f32_to_s16(float x) {
    float v = x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
    return (int16_t) lrintf(v * 32767.0f);
}

// Append a mono f32 block as s16le bytes onto out.
static void tts_append_s16le(std::string & out, const float * samples, int n_samples) {
    size_t base = out.size();
    out.resize(base + (size_t) n_samples * 2);
    char * p = &out[base];
    for (int i = 0; i < n_samples; i++) {
        int16_t s = tts_f32_to_s16(samples[i]);
        *p++      = (char) ((uint16_t) s & 0xff);
        *p++      = (char) (((uint16_t) s >> 8) & 0xff);
    }
}

// Write a JSON error body in the OAI error envelope and set the status.
static void tts_json_error(httplib::Response & res, int status, const char * type, const char * message) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val * err = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, err, "message", message);
    yyjson_mut_obj_add_str(doc, err, "type", type);
    yyjson_mut_obj_add_val(doc, root, "error", err);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.status  = status;
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

// Parse the OAI body into req. Returns false and fills err on bad input.
static bool tts_parse_request(const std::string & body, tts_request & req, std::string & err) {
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (!doc) {
        err = "request body is not valid JSON";
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        err = "request body must be a JSON object";
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_val * input = yyjson_obj_get(root, "input");
    if (!yyjson_is_str(input) || yyjson_get_len(input) == 0) {
        err = "'input' must be a non-empty string";
        yyjson_doc_free(doc);
        return false;
    }
    req.input = yyjson_get_str(input);

    yyjson_val * language = yyjson_obj_get(root, "language");
    req.lang              = yyjson_is_str(language) ? yyjson_get_str(language) : "";

    yyjson_val * voice = yyjson_obj_get(root, "voice");
    req.voice          = yyjson_is_str(voice) ? yyjson_get_str(voice) : "";

    yyjson_val * instructions = yyjson_obj_get(root, "instructions");
    req.instructions          = yyjson_is_str(instructions) ? yyjson_get_str(instructions) : "";

    yyjson_val * fmt = yyjson_obj_get(root, "response_format");
    req.format       = yyjson_is_str(fmt) ? yyjson_get_str(fmt) : "pcm";

    yyjson_val * seed = yyjson_obj_get(root, "seed");
    if (seed && !yyjson_is_int(seed)) {
        err = "'seed' must be an integer";
        yyjson_doc_free(doc);
        return false;
    }
    req.seed = seed ? yyjson_get_sint(seed) : -1;

    yyjson_val * steps = yyjson_obj_get(root, "steps");
    if (steps && (!yyjson_is_int(steps) || yyjson_get_sint(steps) < 1)) {
        err = "'steps' must be a positive integer";
        yyjson_doc_free(doc);
        return false;
    }
    req.steps = steps ? (int) yyjson_get_sint(steps) : 0;

    yyjson_doc_free(doc);

    if (req.format != "pcm" && req.format != "wav") {
        err = "response_format must be 'pcm' or 'wav'";
        return false;
    }
    return true;
}

// Map an ABI status to an HTTP code. The two ABIs share numeric values:
// -1 invalid params, -2 mode/instruct invalid -> client error ; the rest
// are server side failures.
static int tts_status_to_http(int rc) {
    if (rc == 0) {
        return 200;
    }
    if (rc == -1 || rc == -2) {
        return 400;
    }
    return 502;
}

static void tts_handle_speech(const tts_backend & be, const httplib::Request & http_req, httplib::Response & res) {
    tts_request req;
    std::string err;
    if (!tts_parse_request(http_req.body, req, err)) {
        tts_json_error(res, 400, "invalid_request_error", err.c_str());
        return;
    }

    if (req.format == "wav") {
        // One-shot : collect the whole utterance, then emit a RIFF file.
        std::vector<float> buf;
        tts_sink           sink = [&buf](const float * s, int n) {
            buf.insert(buf.end(), s, s + n);
            return true;
        };
        std::string synth_err;
        int         rc;
        {
            std::lock_guard<std::mutex> lock(g_synth_mutex);
            rc = be.synthesize(req, sink, synth_err);
        }
        if (rc != 0) {
            tts_json_error(res, tts_status_to_http(rc), "server_error",
                           synth_err.empty() ? "synthesis failed" : synth_err.c_str());
            return;
        }
        std::string wav = audio_encode_wav(buf.data(), (int) buf.size(), 24000, WAV_S16);
        res.set_content(std::move(wav), "audio/wav");
        return;
    }

    // Streaming : run synthesis inside the chunked provider on the connection
    // thread, pushing s16le frames as the codec produces them. A failed
    // sink.write means the client disconnected, which aborts generation and
    // frees the GPU instead of finishing a stream nobody reads.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider("audio/pcm", [&be, req](size_t, httplib::DataSink & sink) mutable -> bool {
        tts_sink push = [&sink](const float * s, int n) {
            std::string bytes;
            tts_append_s16le(bytes, s, n);
            return sink.write(bytes.data(), bytes.size());
        };
        std::string synth_err;
        {
            std::lock_guard<std::mutex> lock(g_synth_mutex);
            be.synthesize(req, push, synth_err);
        }
        sink.done();
        return true;
    });
}

// Decode standard base64 (with optional padding) into out. Returns false
// on any character outside the alphabet.
static bool tts_b64_decode(const std::string & in, std::string & out) {
    static const std::vector<int8_t> table = [] {
        std::vector<int8_t> t(256, -1);
        const char *        alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) {
            t[(uint8_t) alpha[i]] = (int8_t) i;
        }
        return t;
    }();

    out.clear();
    out.reserve(in.size() / 4 * 3);
    uint32_t acc  = 0;
    int      bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r') {
            continue;
        }
        int8_t v = table[(uint8_t) c];
        if (v < 0) {
            return false;
        }
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char) ((acc >> bits) & 0xff));
        }
    }
    return true;
}

// Parse the POST /v1/audio/voices body: name, ref_text, and exactly one of
// wav_b64 or rvq_b64. The transcript anchors the clone, so it is required.
static bool tts_parse_voice_upload(const std::string & body, tts_voice_upload & up, std::string & err) {
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (!doc) {
        err = "request body is not valid JSON";
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        err = "request body must be a JSON object";
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_val * name = yyjson_obj_get(root, "name");
    if (!yyjson_is_str(name) || yyjson_get_len(name) == 0) {
        err = "'name' must be a non-empty string";
        yyjson_doc_free(doc);
        return false;
    }
    up.name = yyjson_get_str(name);

    yyjson_val * ref_text = yyjson_obj_get(root, "ref_text");
    if (!yyjson_is_str(ref_text) || yyjson_get_len(ref_text) == 0) {
        err = "'ref_text' must be a non-empty string";
        yyjson_doc_free(doc);
        return false;
    }
    up.ref_text = yyjson_get_str(ref_text);

    yyjson_val * wav = yyjson_obj_get(root, "wav_b64");
    yyjson_val * rvq = yyjson_obj_get(root, "rvq_b64");

    const bool has_wav = yyjson_is_str(wav) && yyjson_get_len(wav) > 0;
    const bool has_rvq = yyjson_is_str(rvq) && yyjson_get_len(rvq) > 0;
    if (has_wav == has_rvq) {
        err = "provide either 'wav_b64' or 'rvq_b64'";
        yyjson_doc_free(doc);
        return false;
    }
    if ((has_wav && !tts_b64_decode(yyjson_get_str(wav), up.wav)) ||
        (has_rvq && !tts_b64_decode(yyjson_get_str(rvq), up.rvq))) {
        err = "invalid base64 payload";
        yyjson_doc_free(doc);
        return false;
    }
    yyjson_doc_free(doc);
    return true;
}

static void tts_handle_voice_register(const tts_backend &      be,
                                      const httplib::Request & http_req,
                                      httplib::Response &      res) {
    tts_voice_upload up;
    std::string      err;
    if (!tts_parse_voice_upload(http_req.body, up, err)) {
        tts_json_error(res, 400, "invalid_request_error", err.c_str());
        return;
    }
    if (!be.register_voice(up, err)) {
        tts_json_error(res, 400, "invalid_request_error", err.empty() ? "voice registration failed" : err.c_str());
        return;
    }
    std::string body = "{\"name\":\"" + up.name + "\",\"status\":\"registered\"}";
    res.set_content(body, "application/json");
}

static void tts_handle_voice_delete(const tts_backend &      be,
                                    const httplib::Request & http_req,
                                    httplib::Response &      res) {
    const std::string name = http_req.matches[1];
    if (!be.remove_voice(name)) {
        tts_json_error(res, 404, "not_found_error", "no registered voice with this name");
        return;
    }
    res.set_content("{\"status\":\"deleted\"}", "application/json");
}

static void tts_handle_models(const tts_backend & be, const httplib::Request &, httplib::Response & res) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "object", "list");
    yyjson_mut_val * data = yyjson_mut_arr(doc);
    yyjson_mut_val * one  = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, one, "id", be.model_id.c_str());
    yyjson_mut_obj_add_str(doc, one, "object", "model");
    yyjson_mut_obj_add_str(doc, one, "owned_by", "local");
    yyjson_mut_arr_add_val(data, one);
    yyjson_mut_obj_add_val(doc, root, "data", data);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

static void tts_handle_voices(const tts_backend & be, const httplib::Request &, httplib::Response & res) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val * arr = yyjson_mut_arr(doc);
    for (const std::string & v : be.registered_voices()) {
        yyjson_mut_val * one = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, one, "name", yyjson_mut_strcpy(doc, v.c_str()));
        yyjson_mut_arr_add_val(arr, one);
    }
    yyjson_mut_obj_add_val(doc, root, "voices", arr);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

static void tts_handle_health(const httplib::Request &, httplib::Response & res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
}

static int tts_server_run(const tts_backend & be, const server_config & cfg) {
    httplib::Server svr;
    g_svr = &svr;

    // per-operation socket idle timeouts. read is small (text in), write is
    // generous to cover a long streamed utterance without tripping on a slow
    // client.
    svr.set_read_timeout(60);
    svr.set_write_timeout(120);

    // reject oversized bodies. text plus an optional reference clip stays
    // well under this.
    svr.set_payload_max_length(32 * 1024 * 1024);

    // Nagle coalescing holds small packets back for tens of ms ; streamed
    // PCM chunks must leave the socket the moment they are written.
    svr.set_tcp_nodelay(true);

    // SO_REUSEADDR lets us rebind a port still in TIME_WAIT after a restart.
    // SO_REUSEPORT is deliberately not set : a second instance on the same
    // port then fails with EADDRINUSE instead of silently sharing the socket
    // and splitting traffic between two daemons.
    svr.set_socket_options([](socket_t sock) {
        int one = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    });

    // permissive CORS so a browser client can call the API directly.
    svr.set_default_headers({
        { "Access-Control-Allow-Origin", "*" }
    });
    svr.Options("/.*", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    svr.Post("/v1/audio/speech",
             [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_speech(be, req, res); });
    svr.Get("/v1/models",
            [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_models(be, req, res); });
    svr.Get("/v1/audio/voices",
            [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voices(be, req, res); });
    svr.Post("/v1/audio/voices",
             [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voice_register(be, req, res); });
    svr.Delete(R"(/v1/audio/voices/(.+))",
               [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voice_delete(be, req, res); });
    svr.Get("/health", tts_handle_health);

    signal(SIGINT, tts_on_signal);
    signal(SIGTERM, tts_on_signal);

    fprintf(stderr, "[Server] model %s\n", be.model_id.c_str());
    fprintf(stderr, "[Server] listening on %s:%d\n", cfg.host.c_str(), cfg.port);
    if (!svr.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "[Server] FATAL: cannot bind %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}
