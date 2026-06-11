#pragma once
// audio-postproc.h: TTS waveform post-processing
//
// Strict math port of omnivoice/utils/audio.py and pydub.silence.
// All public functions take and return float32 mono PCM in [-1, 1] at the
// pipeline sample rate (24 kHz). Internal silence detection runs on int16
// samples to match pydub bit-for-bit.

#include "ov-error.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

// RMS of an int16 slice [start, start + n) clamped to s16.size(). A slice that
// extends past the end shrinks accordingly. Empty slices return 0.0, matching
// pydub's AudioSegment.rms on empty segments.
static double postproc_slice_rms_s16(const std::vector<int16_t> & s16, size_t start, size_t n) {
    size_t end = start + n;
    if (end > s16.size()) {
        end = s16.size();
    }

    if (start >= end) {
        return 0.0;
    }

    int64_t ssq = 0;
    for (size_t i = start; i < end; i++) {
        int32_t s = s16[i];
        ssq += (int64_t) s * (int64_t) s;
    }

    size_t cnt = end - start;
    return std::sqrt((double) ssq / (double) cnt);
}

// Converts float32 [-1, 1] to int16 with the exact pydub recipe:
// (audio * 32768.0).clip(-32768, 32767).astype(int16). Truncation toward 0,
// matching numpy's astype(int16).
static std::vector<int16_t> postproc_f32_to_s16(const std::vector<float> & a) {
    std::vector<int16_t> out(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        double v = (double) a[i] * 32768.0;
        if (v > 32767.0) {
            v = 32767.0;
        }

        if (v < -32768.0) {
            v = -32768.0;
        }

        out[i] = (int16_t) v;
    }

    return out;
}

// Inverse of postproc_f32_to_s16: int16 -> float32 via division by 32768.0.
static std::vector<float> postproc_s16_to_f32(const std::vector<int16_t> & s16) {
    std::vector<float> out(s16.size());
    for (size_t i = 0; i < s16.size(); i++) {
        out[i] = (float) ((double) s16[i] / 32768.0);
    }

    return out;
}

// pydub.silence.detect_silence ported to int16 samples. seek_step and
// min_silence_len are in samples. Returns inclusive ranges [start, end] in
// samples where end = start + min_silence_len of the last hit, exactly as
// pydub builds them.
static std::vector<std::pair<int, int>> postproc_detect_silence(const std::vector<int16_t> & s16,
                                                                int                          min_silence_len,
                                                                double                       thresh_lin,
                                                                int                          seek_step) {
    std::vector<std::pair<int, int>> ranges;
    int                              seg_len = (int) s16.size();

    if (seg_len < min_silence_len) {
        return ranges;
    }

    int last_slice_start = seg_len - min_silence_len;

    std::vector<int> starts;
    for (int i = 0; i <= last_slice_start; i += seek_step) {
        starts.push_back(i);
    }

    if ((last_slice_start % seek_step) != 0) {
        starts.push_back(last_slice_start);
    }

    std::vector<int> silence_starts;
    for (int i : starts) {
        double r = postproc_slice_rms_s16(s16, (size_t) i, (size_t) min_silence_len);
        if (r <= thresh_lin) {
            silence_starts.push_back(i);
        }
    }

    if (silence_starts.empty()) {
        return ranges;
    }

    int prev_i      = silence_starts[0];
    int range_start = prev_i;

    for (size_t k = 1; k < silence_starts.size(); k++) {
        int  si         = silence_starts[k];
        bool continuous = (si == prev_i + seek_step);
        bool has_gap    = (si > prev_i + min_silence_len);

        if (!continuous && has_gap) {
            ranges.push_back({ range_start, prev_i + min_silence_len });
            range_start = si;
        }

        prev_i = si;
    }

    ranges.push_back({ range_start, prev_i + min_silence_len });
    return ranges;
}

// pydub.silence.detect_nonsilent: invert detect_silence over [0, seg_len].
static std::vector<std::pair<int, int>> postproc_detect_nonsilent(const std::vector<int16_t> & s16,
                                                                  int                          min_silence_len,
                                                                  double                       thresh_lin,
                                                                  int                          seek_step) {
    std::vector<std::pair<int, int>> nonsilent;
    int                              seg_len = (int) s16.size();
    auto                             silent  = postproc_detect_silence(s16, min_silence_len, thresh_lin, seek_step);

    if (silent.empty()) {
        nonsilent.push_back({ 0, seg_len });
        return nonsilent;
    }

    if (silent.front().first == 0 && silent.front().second == seg_len) {
        return nonsilent;
    }

    int prev_end = 0;
    int last_end = 0;

    for (const auto & r : silent) {
        nonsilent.push_back({ prev_end, r.first });
        prev_end = r.second;
        last_end = r.second;
    }

    if (last_end != seg_len) {
        nonsilent.push_back({ prev_end, seg_len });
    }

    if (!nonsilent.empty() && nonsilent.front().first == 0 && nonsilent.front().second == 0) {
        nonsilent.erase(nonsilent.begin());
    }

    return nonsilent;
}

// pydub.silence.detect_leading_silence ported to int16. chunk_n is in samples.
// Returns the sample index where the leading silence ends (clamped to len).
static int postproc_detect_leading_silence(const std::vector<int16_t> & s16, double thresh_lin, int chunk_n) {
    int trim    = 0;
    int seg_len = (int) s16.size();

    while (trim < seg_len) {
        int    slice_end = std::min(trim + chunk_n, seg_len);
        int    n         = slice_end - trim;
        double r         = postproc_slice_rms_s16(s16, (size_t) trim, (size_t) n);

        // pydub compares dBFS < threshold; in linear amplitude that is
        // r < thresh_lin (strict), since dBFS is monotonic in r and r=0
        // gives -inf which is always below any finite threshold.
        if (r >= thresh_lin) {
            break;
        }

        trim += chunk_n;
    }

    if (trim > seg_len) {
        trim = seg_len;
    }

    return trim;
}

// remove_silence: strict 1:1 port of omnivoice/utils/audio.py:remove_silence.
// Removes mid silences longer than mid_sil_ms (kept down to mid_sil_ms via
// pydub split_on_silence with keep_silence == mid_sil_ms), then trims the
// leading and trailing silences leaving lead_sil_ms / trail_sil_ms intact.
// thresh_db is the dBFS threshold (default -50 dBFS in upstream).
static void remove_silence(std::vector<float> & a,
                           int                  sr,
                           int                  mid_sil_ms,
                           int                  lead_sil_ms,
                           int                  trail_sil_ms,
                           double               thresh_db) {
    if (a.empty()) {
        return;
    }

    std::vector<int16_t> s16        = postproc_f32_to_s16(a);
    double               thresh_lin = 32768.0 * std::pow(10.0, thresh_db / 20.0);
    int                  seek_step  = sr / 100;  // 10 ms

    // Mid silence removal via split_on_silence + concat.
    if (mid_sil_ms > 0) {
        int min_sil_n = sr * mid_sil_ms / 1000;
        int keep_n    = min_sil_n;

        auto nonsilent = postproc_detect_nonsilent(s16, min_sil_n, thresh_lin, seek_step);

        std::vector<std::pair<int, int>> output_ranges;
        output_ranges.reserve(nonsilent.size());
        for (const auto & r : nonsilent) {
            output_ranges.push_back({ r.first - keep_n, r.second + keep_n });
        }

        // pydub pairwise overlap dedup: split overlap at the midpoint.
        for (size_t i = 0; i + 1 < output_ranges.size(); i++) {
            int last_end   = output_ranges[i].second;
            int next_start = output_ranges[i + 1].first;
            if (next_start < last_end) {
                int mid                    = (last_end + next_start) / 2;
                output_ranges[i].second    = mid;
                output_ranges[i + 1].first = mid;
            }
        }

        // Concat clipped slices. Empty slices contribute nothing, matching
        // AudioSegment.silent(0) += seg semantics.
        std::vector<int16_t> out;
        out.reserve(s16.size());
        int seg_len = (int) s16.size();

        for (const auto & r : output_ranges) {
            int cs = std::max(0, r.first);
            int ce = std::min(seg_len, r.second);
            if (cs < ce) {
                out.insert(out.end(), s16.begin() + cs, s16.begin() + ce);
            }
        }

        s16 = std::move(out);
    }

    // Edge trimming: leading then trailing via reverse trick.
    int chunk_n = sr / 100;  // 10 ms

    int trim_lead = postproc_detect_leading_silence(s16, thresh_lin, chunk_n);
    trim_lead     = std::max(0, trim_lead - sr * lead_sil_ms / 1000);
    if (trim_lead > 0) {
        s16.erase(s16.begin(), s16.begin() + std::min(trim_lead, (int) s16.size()));
    }

    std::reverse(s16.begin(), s16.end());

    int trim_trail = postproc_detect_leading_silence(s16, thresh_lin, chunk_n);
    trim_trail     = std::max(0, trim_trail - sr * trail_sil_ms / 1000);
    if (trim_trail > 0) {
        s16.erase(s16.begin(), s16.begin() + std::min(trim_trail, (int) s16.size()));
    }

    std::reverse(s16.begin(), s16.end());

    a = postproc_s16_to_f32(s16);
}

// ref_preprocess_audio: reference waveform preprocessing shared by the TTS
// reference path and the codec CLI encode path. Mirrors the upstream Python
// chain: RMS measurement, auto-gain to RMS 0.1 when the original RMS sits
// in (0, 0.1), then silence-trim with mid=200ms / lead=100ms / trail=200ms
// at -50 dBFS when trim_silence is set. Returns the ORIGINAL RMS so the
// TTS post-proc can rescale generated audio back to the reference
// loudness; an empty buffer returns -1.
static float ref_preprocess_audio(std::vector<float> & a, int sr, bool trim_silence) {
    if (a.empty()) {
        return -1.0f;
    }

    double sumsq = 0.0;
    for (float v : a) {
        sumsq += (double) v * (double) v;
    }
    double ref_rms = std::sqrt(sumsq / (double) a.size());

    if (ref_rms > 0.0 && ref_rms < 0.1) {
        float gain = (float) (0.1 / ref_rms);
        for (float & v : a) {
            v *= gain;
        }
        ov_log(OV_LOG_INFO, "[RefPrep] RMS %.4f -> 0.1 gain %.4f", ref_rms, gain);
    }

    if (trim_silence) {
        size_t before = a.size();
        remove_silence(a, sr, 200, 100, 200, -50.0);
        ov_log(OV_LOG_INFO, "[RefPrep] silence-trim %zu -> %zu samples", before, a.size());
    }

    return (float) ref_rms;
}

// peak_normalize_half: rescale so peak amplitude becomes 0.5 (-6 dBFS).
// Mirrors the no-ref branch of _post_process_audio in omnivoice.py.
static void peak_normalize_half(std::vector<float> & a) {
    if (a.empty()) {
        return;
    }

    float peak = 0.0f;
    for (float s : a) {
        float v = std::fabs(s);
        if (v > peak) {
            peak = v;
        }
    }

    if (peak > 1e-6f) {
        float k = 0.5f / peak;
        for (float & s : a) {
            s *= k;
        }
    }
}

// fade_and_pad: linear fade-in / fade-out on the first and last fade_dur
// seconds, then pad pad_dur seconds of silence on each side. 1:1 port of
// fade_and_pad_audio in omnivoice/utils/audio.py.
static void fade_and_pad(std::vector<float> & a, int sr, double fade_dur, double pad_dur) {
    if (a.empty()) {
        return;
    }

    int fade_n = (int) (fade_dur * (double) sr);
    int pad_n  = (int) (pad_dur * (double) sr);

    if (fade_n > 0) {
        int k = std::min(fade_n, (int) a.size() / 2);
        if (k > 0) {
            int denom = std::max(k - 1, 1);

            for (int i = 0; i < k; i++) {
                float w = (float) i / (float) denom;
                a[(size_t) i] *= w;
            }

            for (int i = 0; i < k; i++) {
                float w = 1.0f - (float) i / (float) denom;
                a[a.size() - (size_t) k + (size_t) i] *= w;
            }
        }
    }

    if (pad_n > 0) {
        std::vector<float> padded((size_t) pad_n + a.size() + (size_t) pad_n, 0.0f);
        std::copy(a.begin(), a.end(), padded.begin() + pad_n);
        a = std::move(padded);
    }
}

// cross_fade_chunks: concatenate audio chunks with a silence_dur gap split
// into fade_out, pure silence, fade_in. 1:1 port of cross_fade_chunks in
// omnivoice/utils/audio.py.
static std::vector<float> cross_fade_chunks(const std::vector<std::vector<float>> & chunks,
                                            int                                     sr,
                                            double                                  silence_dur) {
    if (chunks.empty()) {
        return std::vector<float>();
    }

    if (chunks.size() == 1) {
        return chunks[0];
    }

    int total_n   = (int) (silence_dur * (double) sr);
    int fade_n    = total_n / 3;
    int silence_n = fade_n;

    std::vector<float> merged = chunks[0];

    for (size_t i = 1; i < chunks.size(); i++) {
        const auto & chunk = chunks[i];

        // Fade-out tail of merged.
        int fout_n = std::min(fade_n, (int) merged.size());
        if (fout_n > 0) {
            int denom = std::max(fout_n - 1, 1);
            for (int j = 0; j < fout_n; j++) {
                float w = 1.0f - (float) j / (float) denom;
                merged[merged.size() - (size_t) fout_n + (size_t) j] *= w;
            }
        }

        // Silence gap.
        if (silence_n > 0) {
            merged.insert(merged.end(), (size_t) silence_n, 0.0f);
        }

        // Fade-in head of next chunk (worked on a copy to keep input const).
        std::vector<float> head  = chunk;
        int                fin_n = std::min(fade_n, (int) head.size());
        if (fin_n > 0) {
            int denom = std::max(fin_n - 1, 1);
            for (int j = 0; j < fin_n; j++) {
                float w = (float) j / (float) denom;
                head[(size_t) j] *= w;
            }
        }

        merged.insert(merged.end(), head.begin(), head.end());
    }

    return merged;
}
