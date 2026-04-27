#pragma once
// wav.h: minimal WAV reader
//
// read_wav_buf: PCM16 / PCM24 / float32, classic or WAVE_FORMAT_EXTENSIBLE,
//               mono or stereo, any rate -> interleaved [T, 2] float

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint16_t wav_read_u16le(const uint8_t * p) {
    return (uint16_t) (p[0] | (p[1] << 8));
}

static uint32_t wav_read_u32le(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

static int32_t wav_read_s24le(const uint8_t * p) {
    uint32_t u = (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16);

    if (u & 0x00800000u) {
        u |= 0xff000000u;
    }

    return (int32_t) u;
}

static float wav_read_f32le(const uint8_t * p) {
    uint32_t u = wav_read_u32le(p);
    float    f;
    memcpy(&f, &u, 4);
    return f;
}

// Read WAV from memory buffer.
// Returns interleaved float [T, 2]. Sets *T_audio, *sr. Caller frees.
static float * read_wav_buf(const uint8_t * data, size_t size, int * T_audio, int * sr) {
    *T_audio = 0;
    *sr      = 0;

    if (size < 12 || memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "[WAV] Not a valid WAV buffer\n");
        return NULL;
    }

    int      n_channels           = 0;
    int      sample_rate          = 0;
    int      bits_per_sample      = 0;
    uint16_t audio_format         = 0;
    uint16_t extensible_subformat = 0;
    float *  audio                = NULL;
    int      n_samples            = 0;
    size_t   pos                  = 12;

    while (pos + 8 <= size) {
        const uint8_t * chunk_id   = data + pos;
        uint32_t        chunk_size = wav_read_u32le(data + pos + 4);
        pos += 8;

        if (pos + (size_t) chunk_size > size) {
            chunk_size = (uint32_t) (size - pos);
        }

        if (memcmp(chunk_id, "fmt ", 4) == 0 && chunk_size >= 16) {
            audio_format    = wav_read_u16le(data + pos + 0);
            n_channels      = (int) wav_read_u16le(data + pos + 2);
            sample_rate     = (int) wav_read_u32le(data + pos + 4);
            bits_per_sample = (int) wav_read_u16le(data + pos + 14);

            extensible_subformat = 0;
            if (audio_format == 0xfffe && chunk_size >= 40) {
                extensible_subformat = wav_read_u16le(data + pos + 24);

                // collapse extensible to its effective sample format
                // 1 -> PCM int, 3 -> IEEE float
                if (extensible_subformat == 1 || extensible_subformat == 3) {
                    audio_format = extensible_subformat;
                }
            }

            pos += (size_t) chunk_size;

        } else if (memcmp(chunk_id, "data", 4) == 0 && n_channels > 0) {
            size_t data_bytes = (size_t) chunk_size;

            if (audio_format == 1 && bits_per_sample == 16) {
                n_samples = (int) (data_bytes / ((size_t) n_channels * 2));
                audio     = (float *) malloc((size_t) n_samples * 2 * sizeof(float));
                if (!audio) {
                    fprintf(stderr, "[WAV] OOM allocating PCM16 buffer for %d samples\n", n_samples);
                    return NULL;
                }
                const uint8_t * p = data + pos;

                for (int t = 0; t < n_samples; t++) {
                    if (n_channels == 1) {
                        int16_t s        = (int16_t) wav_read_u16le(p + t * 2);
                        float   f        = (float) s / 32768.0f;
                        audio[t * 2 + 0] = f;
                        audio[t * 2 + 1] = f;
                    } else {
                        const uint8_t * frame = p + (size_t) t * n_channels * 2;
                        int16_t         l     = (int16_t) wav_read_u16le(frame + 0);
                        int16_t         r     = (int16_t) wav_read_u16le(frame + 2);
                        audio[t * 2 + 0]      = (float) l / 32768.0f;
                        audio[t * 2 + 1]      = (float) r / 32768.0f;
                    }
                }
            } else if (audio_format == 1 && bits_per_sample == 24) {
                n_samples = (int) (data_bytes / ((size_t) n_channels * 3));
                audio     = (float *) malloc((size_t) n_samples * 2 * sizeof(float));
                if (!audio) {
                    fprintf(stderr, "[WAV] OOM allocating PCM24 buffer for %d samples\n", n_samples);
                    return NULL;
                }
                const uint8_t * p = data + pos;

                for (int t = 0; t < n_samples; t++) {
                    if (n_channels == 1) {
                        int32_t s        = wav_read_s24le(p + t * 3);
                        float   f        = (float) s / 8388608.0f;
                        audio[t * 2 + 0] = f;
                        audio[t * 2 + 1] = f;
                    } else {
                        const uint8_t * frame = p + (size_t) t * n_channels * 3;
                        int32_t         l     = wav_read_s24le(frame + 0);
                        int32_t         r     = wav_read_s24le(frame + 3);
                        audio[t * 2 + 0]      = (float) l / 8388608.0f;
                        audio[t * 2 + 1]      = (float) r / 8388608.0f;
                    }
                }
            } else if (audio_format == 3 && bits_per_sample == 32) {
                n_samples = (int) (data_bytes / ((size_t) n_channels * 4));
                audio     = (float *) malloc((size_t) n_samples * 2 * sizeof(float));
                if (!audio) {
                    fprintf(stderr, "[WAV] OOM allocating F32 buffer for %d samples\n", n_samples);
                    return NULL;
                }
                const uint8_t * p = data + pos;

                for (int t = 0; t < n_samples; t++) {
                    if (n_channels == 1) {
                        float s          = wav_read_f32le(p + t * 4);
                        audio[t * 2 + 0] = s;
                        audio[t * 2 + 1] = s;
                    } else {
                        const uint8_t * frame = p + (size_t) t * n_channels * 4;
                        float           l     = wav_read_f32le(frame + 0);
                        float           r     = wav_read_f32le(frame + 4);
                        audio[t * 2 + 0]      = l;
                        audio[t * 2 + 1]      = r;
                    }
                }
            } else {
                fprintf(stderr, "[WAV] Unsupported: format=%u bits=%d subformat=%u\n", (unsigned) audio_format,
                        bits_per_sample, (unsigned) extensible_subformat);
                return NULL;
            }

            break;
        } else {
            pos += (size_t) chunk_size;
        }

        if (chunk_size & 1) {
            pos += 1;
        }
    }

    if (!audio) {
        fprintf(stderr, "[WAV] No audio data in buffer\n");
        return NULL;
    }

    *T_audio = n_samples;
    *sr      = sample_rate;
    fprintf(stderr, "[WAV] Read buffer: %d samples, %d Hz, %d ch, %d bit\n", n_samples, sample_rate, n_channels,
            bits_per_sample);
    return audio;
}
