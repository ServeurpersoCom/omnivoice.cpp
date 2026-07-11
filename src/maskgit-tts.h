#pragma once
// maskgit-tts.h: iterative non autoregressive decoder for OmniVoice TTS.
//
// Implements the reference _generate_iterative loop: N stateless forwards of
// the LLM with cond + uncond CFG batching, top-k confidence sampling on the
// audio slots, and a cosine timestep schedule. Greedy decoding (class
// temperature 0, position temperature 0) is fully deterministic and is used
// for bytewise validation against the reference. Higher temperatures rely
// on a seedable PRNG and stay reproducible per seed.

#include "debug.h"
#include "philox.h"
#include "pipeline-tts.h"
#include "prompt-tts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

struct MaskgitConfig {
    int      num_step             = 32;
    float    guidance_scale       = 2.0f;
    float    t_shift              = 0.1f;
    float    layer_penalty_factor = 5.0f;
    float    position_temperature = 5.0f;
    float    class_temperature    = 0.0f;
    uint64_t seed                 = 42;  // only consulted when temperatures > 0
};

// Build the cosine timesteps: t_shift * t / (1 + (t_shift - 1) * t) on
// linspace(0, 1, num_step + 1). Returns a vector of size num_step + 1.
static std::vector<float> maskgit_timesteps(int num_step, float t_shift) {
    std::vector<float> ts(num_step + 1);
    for (int i = 0; i <= num_step; i++) {
        float t = (float) i / (float) num_step;
        ts[i]   = t_shift * t / (1.0f + (t_shift - 1.0f) * t);
    }
    return ts;
}

// Build the per-step demask schedule: how many slots to fill at each step.
// total = T * K, last step takes the remainder. Mirrors the reference
// rounding (ceil) and clamping (rem).
static std::vector<int> maskgit_schedule(int num_step, int total_mask, const std::vector<float> & timesteps) {
    std::vector<int> sched(num_step);
    int              rem = total_mask;
    for (int step = 0; step < num_step; step++) {
        int num;
        if (step == num_step - 1) {
            num = rem;
        } else {
            double frac   = (double) (timesteps[step + 1] - timesteps[step]);
            int    target = (int) std::ceil((double) total_mask * frac);
            num           = std::min(target, rem);
        }
        sched[step] = num;
        rem -= num;
    }
    return sched;
}

// log_softmax(x) along the last dim of length V. log_probs[v] = x[v] - log(sum exp(x - max)).
// FP32 accumulation matches PyTorch CUDA log_softmax. Going to FP64 here is
// more accurate but diverges from the reference at positions where top-1 and
// top-2 are within a few ULPs.
static void maskgit_log_softmax_inplace(float * x, int V) {
    float m = x[0];
    for (int v = 1; v < V; v++) {
        if (x[v] > m) {
            m = x[v];
        }
    }
    float sum = 0.0f;
    for (int v = 0; v < V; v++) {
        sum += std::exp(x[v] - m);
    }
    float lse = m + std::log(sum);
    for (int v = 0; v < V; v++) {
        x[v] = x[v] - lse;
    }
}

// Top-k keep filter on a length-V row: keep the top ratio*V values, set
// the rest to -INF. Mirrors _filter_top_k.
static void maskgit_top_k_filter_inplace(float * x, int V, float ratio) {
    int k = (int) std::ceil((double) ratio * (double) V);
    if (k <= 0 || k >= V) {
        return;
    }
    std::vector<int> idx(V);
    for (int i = 0; i < V; i++) {
        idx[i] = i;
    }
    std::nth_element(idx.begin(), idx.begin() + (V - k), idx.end(), [&](int a, int b) { return x[a] < x[b]; });
    float threshold = x[idx[V - k]];
    for (int v = 0; v < V; v++) {
        if (x[v] < threshold) {
            x[v] = -INFINITY;
        }
    }
}

// Gumbel augmented sampling: x[v] = x[v] / temperature + gumbel(0, 1).
// Mirrors the reference _gumbel_sample: single uniform draw per slot,
// then noise = log(log(u + eps) + eps) negated twice. Uniforms come from
// our Philox4x32-10 helper with PyTorch CUDA conventions :
//   key   = seed
//   subseq = element index (0 .. n-1)
//   ctr   = (ctr_lo, 0, subseq_lo, subseq_hi)
// The caller maintains ctr_lo across successive kernels so successive
// torch.rand_like calls stay aligned with the Python reference.
static void maskgit_gumbel_inplace(float * x, int n, float temperature, int64_t seed, uint32_t & ctr_lo) {
    const float        inv_t = 1.0f / temperature;
    std::vector<float> u((size_t) n);
    philox_uniform_fill(seed, 0, ctr_lo, u.data(), n);
    for (int i = 0; i < n; i++) {
        float g = -std::log(-std::log(u[i] + 1e-10f) + 1e-10f);
        x[i]    = x[i] * inv_t + g;
    }
    // PyTorch advances philox_offset_per_thread by ceil(numel / slab) blocks
    // between kernels (slab = block_size * grid * unroll, ~1.15M elements on
    // any GPU >= 80 SM). With OmniVoice's numel <= K * V ~ 8200, this ratio
    // is always 1, so we just bump ctr_lo by one block.
    ctr_lo += 1;
}

// Run the iterative decoder. Returns flat audio_tokens of size K * T (k slow,
// t fast). The prompt input_ids buffer is mutated in place during decoding.
// ctr_lo_inout, when non-NULL, threads the Philox counter across successive
// MaskGIT calls so a chunked pipeline matches the PyTorch reference (which
// keeps a single global RNG state across model.generate() boundaries).
static std::vector<int32_t> maskgit_generate(PipelineTTS *         pt,
                                             PromptTTS *           prompt,
                                             const MaskgitConfig & cfg,
                                             int                   T,
                                             const char *          dump_dir     = nullptr,
                                             uint32_t *            ctr_lo_inout = nullptr) {
    const int K       = prompt->K;
    const int B_prime = prompt->B_prime;
    const int S       = prompt->S_max;
    const int V       = pt->lm.audio_vocab_size;
    const int mask_id = pt->lm.audio_mask_id;
    if (T <= 0 || K <= 0 || S <= 0) {
        return {};
    }
    if (T > S) {
        fprintf(stderr, "[MaskGIT] FATAL: T=%d exceeds S=%d\n", T, S);
        return {};
    }

    const int audio_start_cond = S - T;  // cond row: audio starts at c_len - T

    std::vector<int32_t> tokens((size_t) K * (size_t) T, mask_id);

    std::vector<float> timesteps = maskgit_timesteps(cfg.num_step, cfg.t_shift);
    std::vector<int>   sched     = maskgit_schedule(cfg.num_step, T * K, timesteps);

    uint32_t ctr_lo = (ctr_lo_inout != nullptr) ? *ctr_lo_inout : 0;

    fprintf(
        stderr,
        "[MaskGIT] Start: T=%d K=%d S=%d num_step=%d guidance=%.2f t_shift=%.3f layer_pen=%.2f cls_t=%.2f pos_t=%.2f\n",
        T, K, S, cfg.num_step, (double) cfg.guidance_scale, (double) cfg.t_shift, (double) cfg.layer_penalty_factor,
        (double) cfg.class_temperature, (double) cfg.position_temperature);

    double fwd_total_ms = 0.0;

    // Cache the inputs that stay constant across the 32 MaskGIT steps. The
    // attention mask in particular is heavy to convert to F16 (~9 ms on the
    // typical voice cloning shape), so building it once amortizes the cost.
    MaskgitBatchedCtx batched_ctx;
    pipeline_tts_llm_batched_ctx_init(&batched_ctx, prompt->audio_mask.data(), prompt->attention_mask.data(), B_prime,
                                      S);

    for (int step = 0; step < cfg.num_step; step++) {
        int k_demask = sched[step];
        if (k_demask <= 0) {
            continue;
        }

        const char *                                hidden_dump = (step == 0 && dump_dir) ? dump_dir : nullptr;
        const std::chrono::steady_clock::time_point fwd_t0      = std::chrono::steady_clock::now();
        std::vector<float>                          logits_full =
            pipeline_tts_llm_forward_batched(pt, prompt->input_ids.data(), &batched_ctx, K, T, hidden_dump);
        const std::chrono::steady_clock::time_point fwd_t1 = std::chrono::steady_clock::now();
        fwd_total_ms += std::chrono::duration<double, std::milli>(fwd_t1 - fwd_t0).count();

        // The forward returns either the full B'*V*K*S logits (debug path,
        // hidden_dump set) or just the audio window 2*V*K*T (production path).
        // Adapt the offsets accordingly.
        const bool   audio_only    = (hidden_dump == nullptr);
        const size_t per_audio     = (size_t) V * (size_t) K * (size_t) T;
        const size_t per_full_item = (size_t) V * (size_t) K * (size_t) S;
        const size_t expected      = audio_only ? 2 * per_audio : (size_t) B_prime * per_full_item;
        if (logits_full.size() != expected) {
            fprintf(stderr, "[MaskGIT] FATAL: forward returned %zu f32 (expected %zu)\n", logits_full.size(), expected);
            pipeline_tts_llm_batched_ctx_free(pt, &batched_ctx);
            return {};
        }

        // Extract cond and uncond logits at audio positions, layout [K, T, V].
        // Audio-only path: cond and uncond are already compact [V, K, T] per
        // row, source index = (t * K + k) * V on each row.
        // Full path: cond row 0 audio is at S range [audio_start_cond, S),
        // uncond row 1 audio is at [0, T).
        std::vector<float> c_log((size_t) K * (size_t) T * (size_t) V);
        std::vector<float> u_log((size_t) K * (size_t) T * (size_t) V);
        const float *      src_cond = audio_only ? logits_full.data() : logits_full.data() + 0 * per_full_item;
        const float * src_uncond = audio_only ? logits_full.data() + per_audio : logits_full.data() + 1 * per_full_item;
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < T; t++) {
                size_t cond_off = audio_only ? ((size_t) t * K + k) * V : ((size_t) (audio_start_cond + t) * K + k) * V;
                size_t uncond_off = ((size_t) t * K + k) * V;
                size_t dst_off    = ((size_t) k * T + t) * V;
                std::copy(src_cond + cond_off, src_cond + cond_off + V, c_log.begin() + dst_off);
                std::copy(src_uncond + uncond_off, src_uncond + uncond_off + V, u_log.begin() + dst_off);
            }
        }

        // Dump LM logits at step 0 only, layout [K, T, V] for both cond and
        // uncond rows. The Python side mirrors this layout via a hook on
        // _predict_tokens_with_scoring.
        if (step == 0 && dump_dir) {
            DebugDumper dbg;
            debug_init(&dbg, dump_dir);
            debug_dump_3d(&dbg, "lm-logits-step0-cond", c_log.data(), K, T, V);
            debug_dump_3d(&dbg, "lm-logits-step0-uncond", u_log.data(), K, T, V);
        }

        // CFG + log_softmax pipeline. Result shape [K, T, V].
        // log_probs[v] = log_softmax(c_log_softmax + g * (c_log_softmax - u_log_softmax)).
        std::vector<float> log_probs((size_t) K * (size_t) T * (size_t) V);
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < T; t++) {
                size_t  off = ((size_t) k * T + t) * V;
                float * c   = c_log.data() + off;
                float * u   = u_log.data() + off;
                float * lp  = log_probs.data() + off;

                if (cfg.guidance_scale != 0.0f) {
                    maskgit_log_softmax_inplace(c, V);
                    maskgit_log_softmax_inplace(u, V);
                    for (int v = 0; v < V; v++) {
                        lp[v] = c[v] + cfg.guidance_scale * (c[v] - u[v]);
                    }
                    maskgit_log_softmax_inplace(lp, V);
                } else {
                    for (int v = 0; v < V; v++) {
                        lp[v] = c[v];
                    }
                    maskgit_log_softmax_inplace(lp, V);
                }
                lp[mask_id] = -INFINITY;
            }
        }

        // Predict tokens + confidence per (k, t).
        std::vector<int32_t> pred_tokens((size_t) K * (size_t) T);
        std::vector<float>   confidence((size_t) K * (size_t) T);
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < T; t++) {
                size_t  off = ((size_t) k * T + t) * V;
                float * lp  = log_probs.data() + off;

                std::vector<float> work;
                float *            sample_src = lp;
                if (cfg.class_temperature > 0.0f) {
                    work.assign(lp, lp + V);
                    maskgit_top_k_filter_inplace(work.data(), V, 0.1f);
                    maskgit_gumbel_inplace(work.data(), V, cfg.class_temperature, (int64_t) cfg.seed, ctr_lo);
                    sample_src = work.data();
                }
                int   best_v = 0;
                float best   = sample_src[0];
                for (int v = 1; v < V; v++) {
                    if (sample_src[v] > best) {
                        best   = sample_src[v];
                        best_v = v;
                    }
                }
                pred_tokens[(size_t) k * T + t] = best_v;

                float max_lp = lp[0];
                for (int v = 1; v < V; v++) {
                    if (lp[v] > max_lp) {
                        max_lp = lp[v];
                    }
                }
                confidence[(size_t) k * T + t] = max_lp;
            }
        }

        // Dump pred_tokens and confidence at step 0 only, before the layer
        // penalty so the dump matches Python _predict_tokens_with_scoring.
        if (step == 0 && dump_dir) {
            DebugDumper dbg;
            debug_init(&dbg, dump_dir);
            int kt_shape[2]  = { K, T };
            int ktv_shape[3] = { K, T, V };
            debug_dump_i32_as_f32(&dbg, "mg-pred-tokens-step0", pred_tokens.data(), kt_shape, 2);
            debug_dump_2d(&dbg, "mg-scores-step0", confidence.data(), K, T);
            debug_dump(&dbg, "mg-log-probs-step0", log_probs.data(), ktv_shape, 3);
        }

        // Apply layer penalty: scores -= k * layer_penalty_factor.
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < T; t++) {
                confidence[(size_t) k * T + t] -= (float) k * cfg.layer_penalty_factor;
            }
        }

        // Optional position-noise.
        if (cfg.position_temperature > 0.0f) {
            maskgit_gumbel_inplace(confidence.data(), K * T, cfg.position_temperature, (int64_t) cfg.seed, ctr_lo);
        }

        // Mask scores of slots that are already decoded (token != mask_id).
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < T; t++) {
                if (tokens[(size_t) k * T + t] != mask_id) {
                    confidence[(size_t) k * T + t] = -INFINITY;
                }
            }
        }

        // Top-k indices on the flat scores (k slow, t fast). Indices are
        // i = k*T + t, mirroring Python's flatten on (K, T).
        const int        N = K * T;
        std::vector<int> idx(N);
        for (int i = 0; i < N; i++) {
            idx[i] = i;
        }
        std::partial_sort(idx.begin(), idx.begin() + k_demask, idx.end(),
                          [&](int a, int b) { return confidence[a] > confidence[b]; });

        // Apply pred_tokens at the selected slots in tokens and the cond /
        // uncond input_ids buffers.
        for (int n = 0; n < k_demask; n++) {
            int     i = idx[n];
            int     k = i / T;
            int     t = i % T;
            int32_t v = pred_tokens[(size_t) k * T + t];

            tokens[(size_t) k * T + t] = v;

            // cond row: input_ids[0, k, audio_start_cond + t]
            prompt->input_ids[((size_t) 0 * K + k) * S + (size_t) (audio_start_cond + t)] = v;
            // uncond row: input_ids[1, k, t]
            prompt->input_ids[((size_t) 1 * K + k) * S + (size_t) t]                      = v;
        }

        fprintf(stderr, "[MaskGIT-Step] %d/%d demask=%d remaining=%d\n", step + 1, cfg.num_step, k_demask,
                (int) std::count(tokens.begin(), tokens.end(), mask_id));
    }

    fprintf(stderr, "[MaskGIT] Total LM forward: %.2f ms across %d steps (avg %.2f ms/step)\n", fwd_total_ms,
            cfg.num_step, fwd_total_ms / (double) cfg.num_step);

    if (ctr_lo_inout != nullptr) {
        *ctr_lo_inout = ctr_lo;
    }

    pipeline_tts_llm_batched_ctx_free(pt, &batched_ctx);
    return tokens;
}
