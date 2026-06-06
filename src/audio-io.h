#pragma once
// audio-io.h: WAV read/write for omnivoice.cpp.
// Reads any WAV (PCM16 / PCM24 / float32, mono or stereo, any rate).
// Writes mono WAV in S16, S24 or F32 at the source sample rate.
// Internal pipelines: planar stereo float [L:T][R:T] for reads,
// flat mono float [T] for writes (omnivoice is mono only).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <fcntl.h>
#    include <io.h>
#endif

// wav.h: WAV reader (returns interleaved, we deinterleave below)
#include "wav.h"

// audio-resample.h: sample rate conversion
#include "audio-resample.h"

// utf8.h: utf8_fopen, the path-UTF-8-aware fopen used below.
#include "utf8.h"

// case-insensitive extension check
static bool audio_io_ends_with(const char * str, const char * suffix) {
    int slen = (int) strlen(str);
    int xlen = (int) strlen(suffix);
    if (slen < xlen) {
        return false;
    }
    for (int i = 0; i < xlen; i++) {
        char a = str[slen - xlen + i];
        char b = suffix[i];
        if (a >= 'A' && a <= 'Z') {
            a += 32;
        }
        if (b >= 'A' && b <= 'Z') {
            b += 32;
        }
        if (a != b) {
            return false;
        }
    }
    return true;
}

// Load entire file into memory. Caller frees the returned pointer.
static uint8_t * audio_io_load_file(const char * path, size_t * size_out) {
    *size_out = 0;
    FILE * fp = utf8_fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[Audio] Cannot open %s\n", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t * buf = (uint8_t *) malloc((size_t) fsize);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t nr = fread(buf, 1, (size_t) fsize, fp);
    fclose(fp);
    if (nr != (size_t) fsize) {
        free(buf);
        return NULL;
    }

    *size_out = (size_t) fsize;
    return buf;
}

// Decode WAV from memory buffer. Returns planar stereo float [L:T][R:T].
static float * audio_io_read_wav_buf(const uint8_t * data, size_t size, int * T_out, int * sr_out) {
    *T_out  = 0;
    *sr_out = 0;

    int     T = 0, sr = 0;
    float * interleaved = read_wav_buf(data, size, &T, &sr);
    if (!interleaved) {
        return NULL;
    }

    float * planar = (float *) malloc((size_t) T * 2 * sizeof(float));
    if (!planar) {
        free(interleaved);
        return NULL;
    }
    for (int t = 0; t < T; t++) {
        planar[t]     = interleaved[t * 2 + 0];
        planar[T + t] = interleaved[t * 2 + 1];
    }
    free(interleaved);

    *T_out  = T;
    *sr_out = sr;
    return planar;
}

// Read WAV file. Returns planar stereo float [L: T][R: T]. Caller frees.
static float * audio_read(const char * path, int * T_out, int * sr_out) {
    size_t    size = 0;
    uint8_t * buf  = audio_io_load_file(path, &size);
    if (!buf) {
        *T_out  = 0;
        *sr_out = 0;
        return NULL;
    }
    float * result = audio_io_read_wav_buf(buf, size, T_out, sr_out);
    free(buf);
    return result;
}

// Read WAV, resample to target_sr, downmix to mono.
// Returns a flat buffer of T floats at target_sr mono. Caller frees.
static float * audio_read_mono(const char * path, int target_sr, int * T_out) {
    int     T   = 0;
    int     sr  = 0;
    float * raw = audio_read(path, &T, &sr);
    if (!raw) {
        *T_out = 0;
        return NULL;
    }

    // Resample planar stereo to target_sr first to keep both channels
    // coherent when the source rate differs.
    float * stereo_rs = raw;
    int     T_rs      = T;
    if (sr != target_sr) {
        fprintf(stderr, "[Audio-Resample] %d Hz -> %d Hz, %d samples...\n", sr, target_sr, T);
        int     T_new     = 0;
        float * resampled = audio_resample(raw, T, sr, target_sr, 2, &T_new);
        free(raw);
        if (!resampled) {
            fprintf(stderr, "[Audio-Resample] Resample failed\n");
            *T_out = 0;
            return NULL;
        }
        fprintf(stderr, "[Audio-Resample] Done: %d -> %d samples\n", T, T_new);
        stereo_rs = resampled;
        T_rs      = T_new;
    }

    // Downmix planar [L:T][R:T] to mono = 0.5 * (L + R).
    float * mono = (float *) malloc((size_t) T_rs * sizeof(float));
    if (!mono) {
        free(stereo_rs);
        *T_out = 0;
        return NULL;
    }
    const float * left  = stereo_rs;
    const float * right = stereo_rs + (size_t) T_rs;
    for (int i = 0; i < T_rs; i++) {
        mono[i] = 0.5f * (left[i] + right[i]);
    }
    free(stereo_rs);

    *T_out = T_rs;
    return mono;
}

// WAV output format
enum WavFormat {
    WAV_S16,  // 16-bit signed integer PCM (classic RIFF, default)
    WAV_S24,  // 24-bit signed integer PCM (classic RIFF)
    WAV_F32,  // 32-bit IEEE 754 float (classic RIFF, fmt_tag=3)
};

// Parse a CLI format string into a WavFormat. Accepts: wav16, wav24, wav32.
// Returns false on unknown format.
static bool audio_parse_format(const char * s, WavFormat & wav_fmt) {
    if (!s) {
        return false;
    }
    if (!strcmp(s, "wav16")) {
        wav_fmt = WAV_S16;
        return true;
    }
    if (!strcmp(s, "wav24")) {
        wav_fmt = WAV_S24;
        return true;
    }
    if (!strcmp(s, "wav32")) {
        wav_fmt = WAV_F32;
        return true;
    }
    return false;
}

// Byte-level write helpers (endian-safe)

static void wav_write_u16le(char *& p, uint16_t x) {
    *p++ = (char) (x & 0xff);
    *p++ = (char) ((x >> 8) & 0xff);
}

static void wav_write_u24le(char *& p, uint32_t x) {
    *p++ = (char) (x & 0xff);
    *p++ = (char) ((x >> 8) & 0xff);
    *p++ = (char) ((x >> 16) & 0xff);
}

static void wav_write_u32le(char *& p, uint32_t x) {
    *p++ = (char) (x & 0xff);
    *p++ = (char) ((x >> 8) & 0xff);
    *p++ = (char) ((x >> 16) & 0xff);
    *p++ = (char) ((x >> 24) & 0xff);
}

static float wav_clamp1(float x) {
    return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
}

static float wav_sanitize(float x) {
    return std::isfinite(x) ? x : 0.0f;
}

// Classic RIFF header: fmt_tag 1 (PCM int) or 3 (IEEE float), 16-byte fmt chunk
static void wav_write_header_basic(char *& p, int T_audio, int sr, int n_channels, int bits, uint16_t fmt_tag) {
    uint32_t bytes_per_sample = (uint32_t) bits / 8;
    uint32_t byte_rate        = (uint32_t) sr * (uint32_t) n_channels * bytes_per_sample;
    uint16_t block_align      = (uint16_t) (n_channels * (int) bytes_per_sample);
    uint32_t data_size        = (uint32_t) T_audio * (uint32_t) n_channels * bytes_per_sample;
    uint32_t file_size        = 36 + data_size;

    memcpy(p, "RIFF", 4);
    p += 4;
    wav_write_u32le(p, file_size);
    memcpy(p, "WAVE", 4);
    p += 4;

    memcpy(p, "fmt ", 4);
    p += 4;
    wav_write_u32le(p, 16);
    wav_write_u16le(p, fmt_tag);
    wav_write_u16le(p, (uint16_t) n_channels);
    wav_write_u32le(p, (uint32_t) sr);
    wav_write_u32le(p, byte_rate);
    wav_write_u16le(p, block_align);
    wav_write_u16le(p, (uint16_t) bits);

    memcpy(p, "data", 4);
    p += 4;
    wav_write_u32le(p, data_size);
}

// Encode mono float to WAV 16-bit signed integer PCM in memory.
// 44-byte classic RIFF header (fmt_tag=1) + int16 samples.
// Clamps to [-1, +1], coerces NaN/Inf to zero.
static std::string audio_encode_wav_s16(const float * audio, int T_audio, int sr) {
    int n_channels = 1;
    int data_size  = T_audio * n_channels * 2;

    std::string out;
    out.resize(44 + (size_t) data_size);
    char * p = &out[0];

    wav_write_header_basic(p, T_audio, sr, n_channels, 16, 1);

    for (int t = 0; t < T_audio; t++) {
        int16_t s = (int16_t) (wav_clamp1(wav_sanitize(audio[t])) * 32767.0f);
        wav_write_u16le(p, (uint16_t) s);
    }

    return out;
}

// Encode mono float to WAV 24-bit signed integer PCM in memory.
// 44-byte classic RIFF header (fmt_tag=1) + int24 samples.
// Clamps to [-1, +1], coerces NaN/Inf to zero.
static std::string audio_encode_wav_s24(const float * audio, int T_audio, int sr) {
    int n_channels = 1;
    int data_size  = T_audio * n_channels * 3;

    std::string out;
    out.resize(44 + (size_t) data_size);
    char * p = &out[0];

    wav_write_header_basic(p, T_audio, sr, n_channels, 24, 1);

    for (int t = 0; t < T_audio; t++) {
        int32_t s = (int32_t) (wav_clamp1(wav_sanitize(audio[t])) * 8388607.0f);
        wav_write_u24le(p, (uint32_t) s);
    }

    return out;
}

// Encode mono float to WAV 32-bit IEEE 754 float in memory.
// 44-byte classic RIFF header (fmt_tag=3) + float32 samples.
// Coerces NaN/Inf to zero. No clamping: output may exceed [-1, +1].
static std::string audio_encode_wav_f32(const float * audio, int T_audio, int sr) {
    int n_channels = 1;
    int data_size  = T_audio * n_channels * 4;

    std::string out;
    out.resize(44 + (size_t) data_size);
    char * p = &out[0];

    wav_write_header_basic(p, T_audio, sr, n_channels, 32, 3);

    for (int t = 0; t < T_audio; t++) {
        float    f = wav_sanitize(audio[t]);
        uint32_t u;
        memcpy(&u, &f, 4);
        wav_write_u32le(p, u);
    }

    return out;
}

// Encode mono float to WAV in memory in the requested format.
// audio is flat mono [T], pre-normalized by caller.
// NaN and Inf are coerced to zero. S16/S24 clamp to [-1, +1].
static std::string audio_encode_wav(const float * audio, int T_audio, int sr, WavFormat fmt = WAV_S16) {
    switch (fmt) {
        case WAV_S16:
            return audio_encode_wav_s16(audio, T_audio, sr);
        case WAV_S24:
            return audio_encode_wav_s24(audio, T_audio, sr);
        case WAV_F32:
            return audio_encode_wav_f32(audio, T_audio, sr);
    }
    fprintf(stderr, "[WAV] unknown format %d\n", (int) fmt);
    return {};
}

// Write mono float audio to WAV file in the requested format.
// S16/S24 hard clip to [-1, +1], F32 preserves the full range.
static bool audio_write_wav(const char * path, const float * audio, int T_audio, int sr, WavFormat fmt = WAV_S16) {
    std::string wav = audio_encode_wav(audio, T_audio, sr, fmt);
    if (wav.empty()) {
        return false;
    }

    FILE * fp = utf8_fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[WAV] Cannot open %s for writing\n", path);
        return false;
    }
    if (fwrite(wav.data(), 1, wav.size(), fp) != wav.size()) {
        fprintf(stderr, "[WAV] Failed to write %s\n", path);
        fclose(fp);
        return false;
    }
    fclose(fp);

    const char * fmt_name = (fmt == WAV_S16) ? "S16" : (fmt == WAV_S24) ? "S24" : "F32";
    fprintf(stderr, "[WAV] Wrote %s: %d samples, %d Hz, mono %s\n", path, T_audio, sr, fmt_name);
    return true;
}

// Minimal streaming WAV sink. Writes a wide RIFF / data size at open and
// never updates them: the stream is one shot, non seekable, suitable for
// stdout pipes where the player reads until EOF. Use audio_write_wav for
// seekable file output (the file there has accurate sizes in headers).
struct wav_stream {
    FILE *    fp;
    WavFormat fmt;
    int       sr;
};

// Write a fresh 44 byte RIFF header on the sink. Both the RIFF chunk size
// and the data chunk size advertise 0x7FFFFFFF, the conventional
// "unknown / live" marker that aplay, ffmpeg and most players accept by
// reading until EOF. Called once at open, and again at every utterance
// boundary by line oriented streaming so a client can split the stream
// into standalone WAV clips on the RIFF magic.
static bool wav_stream_write_header(wav_stream * ws) {
    int      bits             = (ws->fmt == WAV_S16) ? 16 : (ws->fmt == WAV_S24) ? 24 : 32;
    uint16_t fmt_tag          = (ws->fmt == WAV_F32) ? 3 : 1;
    int      n_channels       = 1;
    uint32_t bytes_per_sample = (uint32_t) bits / 8;
    uint32_t byte_rate        = (uint32_t) ws->sr * (uint32_t) n_channels * bytes_per_sample;
    uint16_t block_align      = (uint16_t) (n_channels * (int) bytes_per_sample);
    uint32_t data_size        = 0x7FFFFFFFu;
    uint32_t file_size        = 0x7FFFFFFFu;

    char   header[44];
    char * p = header;

    memcpy(p, "RIFF", 4);
    p += 4;
    wav_write_u32le(p, file_size);
    memcpy(p, "WAVE", 4);
    p += 4;

    memcpy(p, "fmt ", 4);
    p += 4;
    wav_write_u32le(p, 16);
    wav_write_u16le(p, fmt_tag);
    wav_write_u16le(p, (uint16_t) n_channels);
    wav_write_u32le(p, (uint32_t) ws->sr);
    wav_write_u32le(p, byte_rate);
    wav_write_u16le(p, block_align);
    wav_write_u16le(p, (uint16_t) bits);

    memcpy(p, "data", 4);
    p += 4;
    wav_write_u32le(p, data_size);

    if (fwrite(header, 1, 44, ws->fp) != 44) {
        fprintf(stderr, "[WAV-Stream] header write failed\n");
        return false;
    }
    fflush(ws->fp);
    return true;
}

// Open a streaming WAV sink on stdout. Switches stdout to binary mode on
// Windows and writes the initial live header.
static bool wav_stream_open_stdout(wav_stream * ws, int sr, WavFormat fmt) {
    ws->fp  = stdout;
    ws->fmt = fmt;
    ws->sr  = sr;

#if defined(_WIN32)
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    if (!wav_stream_write_header(ws)) {
        return false;
    }

    const char * fmt_name = (fmt == WAV_S16) ? "S16" : (fmt == WAV_S24) ? "S24" : "F32";
    fprintf(stderr, "[WAV-Stream] stdout: %d Hz, mono %s\n", sr, fmt_name);
    return true;
}

// Encode and write n mono samples to the streaming sink. NaN and Inf coerce
// to zero. S16 / S24 clamp to [-1, +1] before quantisation. Flushes after
// every write so a downstream pipe sees the bytes immediately.
static bool wav_stream_write(wav_stream * ws, const float * audio, int n) {
    if (n <= 0) {
        return true;
    }

    if (ws->fmt == WAV_S16) {
        std::vector<uint8_t> out((size_t) n * 2);
        char *               p = (char *) out.data();
        for (int t = 0; t < n; t++) {
            int16_t s = (int16_t) (wav_clamp1(wav_sanitize(audio[t])) * 32767.0f);
            wav_write_u16le(p, (uint16_t) s);
        }
        if (fwrite(out.data(), 1, out.size(), ws->fp) != out.size()) {
            return false;
        }
    } else if (ws->fmt == WAV_S24) {
        std::vector<uint8_t> out((size_t) n * 3);
        char *               p = (char *) out.data();
        for (int t = 0; t < n; t++) {
            int32_t s = (int32_t) (wav_clamp1(wav_sanitize(audio[t])) * 8388607.0f);
            wav_write_u24le(p, (uint32_t) s);
        }
        if (fwrite(out.data(), 1, out.size(), ws->fp) != out.size()) {
            return false;
        }
    } else {
        std::vector<uint8_t> out((size_t) n * 4);
        char *               p = (char *) out.data();
        for (int t = 0; t < n; t++) {
            float    f = wav_sanitize(audio[t]);
            uint32_t u;
            memcpy(&u, &f, 4);
            wav_write_u32le(p, u);
        }
        if (fwrite(out.data(), 1, out.size(), ws->fp) != out.size()) {
            return false;
        }
    }

    fflush(ws->fp);
    return true;
}

// Final flush. The sink does not own the stdout FILE so no fclose is issued.
static void wav_stream_close(wav_stream * ws) {
    fflush(ws->fp);
}
