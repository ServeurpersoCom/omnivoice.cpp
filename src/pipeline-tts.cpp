// pipeline-tts.cpp: TTS pipeline orchestration.
//
// Phase 1 scope: load OmniVoiceLM weights onto the backend. Subsequent phases
// append the custom embed graph, the audio_heads readout, the MaskGIT loop,
// the prompt builder, and the audio_tokenizer decode. Each compute path
// allocates its own ggml_gallocr at call time, mirroring pipeline-codec.

#include "pipeline-tts.h"

#include "audio-postproc.h"
#include "bpe.h"
#include "debug.h"
#include "duration-estimator.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "maskgit-tts.h"
#include "pipeline-codec.h"
#include "prompt-tts.h"
#include "text-chunker.h"
#include "voice-design.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

bool pipeline_tts_load(PipelineTTS * pt, const char * gguf_path, BackendPair bp, bool use_fa, bool clamp_fp16) {
    *pt                = {};
    pt->bp             = bp;
    pt->backend        = bp.backend;
    pt->use_flash_attn = use_fa && bp.has_gpu;
    pt->clamp_fp16     = clamp_fp16;

    // Echo effective flags. Flash attention only activates on a GPU backend,
    // so the disabled log fires both for explicit --no-fa and for CPU only
    // runs where the request cannot be honoured.
    if (!pt->use_flash_attn) {
        fprintf(stderr, "[Load] Flash attention disabled\n");
    }
    if (pt->clamp_fp16) {
        fprintf(stderr, "[Load] FP16 clamp enabled\n");
    }

    if (!gf_load(&pt->gguf, gguf_path)) {
        return false;
    }

    // 1 embed_tokens + 1 final_norm + 1 audio_embeddings + 1 audio_heads
    // + 28 layers * 11 tensors max = 312, headroom to 512.
    wctx_init(&pt->wctx, 512);

    if (!omnivoice_lm_load(&pt->lm, pt->gguf, &pt->wctx)) {
        wctx_free(&pt->wctx);
        gf_close(&pt->gguf);
        return false;
    }

    if (!wctx_alloc(&pt->wctx, bp.backend)) {
        wctx_free(&pt->wctx);
        gf_close(&pt->gguf);
        return false;
    }

    gf_close(&pt->gguf);

    // Scheduler : routes ops the GPU backend cannot run (e.g. K-quant
    // get_rows on CUDA) to the CPU backend. 8192 nodes covers the full
    // 28L Qwen3 graph with batched MaskGIT.
    pt->sched = backend_sched_new(bp, 8192);

    return true;
}

void pipeline_tts_free(PipelineTTS * pt) {
    if (pt->sched) {
        ggml_backend_sched_free(pt->sched);
    }
    wctx_free(&pt->wctx);
    *pt = {};
}

// Full LLM forward in a single graph. Composes the custom embed, the 28L
// Qwen3 stack, and the audio_heads reshape. attention_mask is an optional
// [S, S] int 0/1 buffer (1 = attended, 0 = blocked). NULL means
// bidirectional (no padding).
std::vector<float> pipeline_tts_llm_forward(PipelineTTS *   pt,
                                            const int32_t * input_ids,
                                            const int32_t * audio_mask,
                                            const int32_t * attention_mask,
                                            int             K,
                                            int             S,
                                            const char *    dump_hidden_dir,
                                            const char *    dump_hidden_name) {
    if (K <= 0 || S <= 0) {
        return {};
    }
    if (K > pt->lm.num_audio_codebook) {
        fprintf(stderr, "[LM-Forward] FATAL: K=%d exceeds num_audio_codebook=%d\n", K, pt->lm.num_audio_codebook);
        return {};
    }

    const Qwen3Config & cfg = pt->lm.cfg;
    const int           V   = pt->lm.audio_vocab_size;

    // CPU pre-compute, identical to the embed_test pre-compute.
    std::vector<int32_t> shifted((size_t) K * (size_t) S);
    std::vector<int32_t> text_ids_buf(S);
    std::vector<float>   mask_f(S), inv_mask_f(S);
    for (int s = 0; s < S; s++) {
        int m           = (audio_mask[s] != 0) ? 1 : 0;
        mask_f[s]       = (float) m;
        inv_mask_f[s]   = (float) (1 - m);
        text_ids_buf[s] = input_ids[0 * S + s];
        for (int k = 0; k < K; k++) {
            shifted[(size_t) k * (size_t) S + s] = input_ids[(size_t) k * (size_t) S + s] * m + k * V;
        }
    }

    // Convert int 0/1 attention mask to F16 additive bias matching the Python
    // reference. OmniVoice passes a boolean attention_mask to transformers,
    // which promotes True/False to 1.0/0.0 floats and adds it to the attention
    // scores : allowed positions get a +1.0 boost, blocked positions stay at
    // 0.0. This is not a hard mask : every position still contributes to the
    // softmax, the model was trained against this exact bias semantics.
    // F16 is the type expected by ggml_flash_attn_ext, and 1.0 / 0.0 are
    // representable exactly in F16 so there is no precision loss.
    std::vector<uint16_t> attn_f16;
    if (attention_mask) {
        attn_f16.resize((size_t) S * (size_t) S);
        for (int sq = 0; sq < S; sq++) {
            for (int skv = 0; skv < S; skv++) {
                float v = (attention_mask[(size_t) sq * (size_t) S + (size_t) skv] != 0) ? 1.0f : 0.0f;
                attn_f16[(size_t) sq * (size_t) S + (size_t) skv] = ggml_fp32_to_fp16(v);
            }
        }
    }

    // Node budget : custom embed ~30, 28L stack ~850, audio_heads ~5.
    // 8192 leaves room for longer sequences and future fusions.
    const int    n_max_nodes    = 8192;
    const size_t graph_ctx_size = ggml_tensor_overhead() * n_max_nodes + ggml_graph_overhead_custom(n_max_nodes, false);

    struct ggml_init_params gp   = { graph_ctx_size, NULL, /*no_alloc=*/true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        return {};
    }

    // Custom embed inputs.
    struct ggml_tensor * t_text_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
    ggml_set_name(t_text_ids, "text_ids");
    ggml_set_input(t_text_ids);

    struct ggml_tensor * t_shifted = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, K);
    ggml_set_name(t_shifted, "shifted_ids");
    ggml_set_input(t_shifted);

    struct ggml_tensor * t_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 1, S);
    ggml_set_name(t_mask, "mask");
    ggml_set_input(t_mask);

    struct ggml_tensor * t_inv_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 1, S);
    ggml_set_name(t_inv_mask, "inv_mask");
    ggml_set_input(t_inv_mask);

    // Stack input : positions 0..S-1.
    struct ggml_tensor * t_positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
    ggml_set_name(t_positions, "positions");
    ggml_set_input(t_positions);

    // Optional attention mask tensor.
    struct ggml_tensor * t_attn = NULL;
    if (attention_mask) {
        t_attn = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, S, S);
        ggml_set_name(t_attn, "attn_mask");
        ggml_set_input(t_attn);
    }

    // Custom embed graph.
    struct ggml_tensor * text_embeds  = ggml_get_rows(gctx, pt->lm.embed_tokens, t_text_ids);
    struct ggml_tensor * audio_embeds = NULL;
    for (int k = 0; k < K; k++) {
        struct ggml_tensor * idx_k = ggml_view_1d(gctx, t_shifted, S, (size_t) k * (size_t) S * sizeof(int32_t));
        struct ggml_tensor * emb_k = ggml_get_rows(gctx, pt->lm.audio_embeddings, idx_k);
        audio_embeds               = (k == 0) ? emb_k : ggml_add(gctx, audio_embeds, emb_k);
    }
    struct ggml_tensor * text_branch   = ggml_mul(gctx, text_embeds, t_inv_mask);
    struct ggml_tensor * audio_branch  = ggml_mul(gctx, audio_embeds, t_mask);
    struct ggml_tensor * inputs_embeds = ggml_add(gctx, text_branch, audio_branch);

    // 28L Qwen3 stack + final RMSNorm. Mask is forwarded through (NULL -> bidir).
    // When dumping is active we also expose the input embedding (pre layer 0)
    // and a few mid stack hidden states so a Python reference can bisect the
    // origin of any drift layer by layer.
    std::vector<int>                  dump_layer_indices;
    std::vector<struct ggml_tensor *> dump_intermediates;
    std::vector<struct ggml_tensor *> sub_outs;
    if (dump_hidden_dir && dump_hidden_name) {
        dump_layer_indices = { 0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20 };
        ggml_set_name(inputs_embeds, "lm_inputs_embeds");
        ggml_set_output(inputs_embeds);
    }
    struct ggml_tensor * hidden = qwen3_build_layers(
        gctx, cfg, pt->lm.layers, pt->lm.final_norm, inputs_embeds, t_positions, t_attn, S, pt->use_flash_attn,
        pt->clamp_fp16, dump_hidden_dir && dump_hidden_name ? &dump_layer_indices : nullptr,
        dump_hidden_dir && dump_hidden_name ? &dump_intermediates : nullptr,
        dump_hidden_dir && dump_hidden_name ? 1 : -1, dump_hidden_dir && dump_hidden_name ? &sub_outs : nullptr);
    if (dump_hidden_dir && dump_hidden_name) {
        for (struct ggml_tensor * t : dump_intermediates) {
            ggml_set_output(t);
        }
        for (struct ggml_tensor * t : sub_outs) {
            ggml_set_output(t);
        }
        ggml_set_name(hidden, "lm_last_hidden");
        ggml_set_output(hidden);
    }

    // audio_heads readout + reshape to (V, K, S).
    struct ggml_tensor * logits_flat = ggml_mul_mat(gctx, pt->lm.audio_heads, hidden);
    struct ggml_tensor * logits      = ggml_reshape_3d(gctx, logits_flat, V, K, S);
    ggml_set_name(logits, "audio_logits");
    ggml_set_output(logits);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, logits);

    if (!ggml_backend_sched_alloc_graph(pt->sched, graph)) {
        fprintf(stderr, "[LM-Forward] FATAL: sched_alloc_graph failed (K=%d S=%d)\n", K, S);
        ggml_free(gctx);
        return {};
    }

    ggml_backend_tensor_set(t_text_ids, text_ids_buf.data(), 0, (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(t_shifted, shifted.data(), 0, (size_t) K * (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(t_mask, mask_f.data(), 0, (size_t) S * sizeof(float));
    ggml_backend_tensor_set(t_inv_mask, inv_mask_f.data(), 0, (size_t) S * sizeof(float));

    std::vector<int32_t> pos_data(S);
    for (int i = 0; i < S; i++) {
        pos_data[i] = i;
    }
    ggml_backend_tensor_set(t_positions, pos_data.data(), 0, (size_t) S * sizeof(int32_t));

    if (t_attn) {
        ggml_backend_tensor_set(t_attn, attn_f16.data(), 0, (size_t) S * (size_t) S * sizeof(uint16_t));
    }

    enum ggml_status st = ggml_backend_sched_graph_compute(pt->sched, graph);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[LM-Forward] FATAL: graph_compute status=%d\n", (int) st);
        ggml_backend_sched_reset(pt->sched);
        ggml_free(gctx);
        return {};
    }

    const size_t       n = ggml_nelements(logits);
    std::vector<float> out(n);
    ggml_backend_tensor_get(logits, out.data(), 0, n * sizeof(float));

    if (dump_hidden_dir && dump_hidden_name) {
        DebugDumper dbg;
        debug_init(&dbg, dump_hidden_dir);

        auto dump_tensor_2d = [&](struct ggml_tensor * t, const std::string & full_name) {
            const int          dim0  = (int) t->ne[0];
            const int          dim1  = (int) t->ne[1];
            const size_t       numel = (size_t) dim0 * (size_t) dim1;
            std::vector<float> buf(numel);
            ggml_backend_tensor_get(t, buf.data(), 0, numel * sizeof(float));
            // GGML layout : fast axis is dim0, slow axis is dim1. Numpy reads
            // back as [dim1, dim0] row-major, identical to hidden_states[b]
            // and inputs_embeds[b] from the Python reference.
            debug_dump_2d(&dbg, full_name.c_str(), buf.data(), dim1, dim0);
        };

        // Pre layer 0 embedding.
        dump_tensor_2d(inputs_embeds, std::string(dump_hidden_name) + "-embed");

        // Mid stack hidden states, in the order set by dump_layer_indices.
        for (size_t i = 0; i < dump_intermediates.size(); i++) {
            char suffix[32];
            snprintf(suffix, sizeof(suffix), "-l%d", dump_layer_indices[i]);
            dump_tensor_2d(dump_intermediates[i], std::string(dump_hidden_name) + suffix);
        }

        // Layer 1 sub-module taps : norm1, attn (pre residual), norm2, mlp (pre residual).
        const char * sub_names[4] = { "-l1-norm1", "-l1-attn", "-l1-norm2", "-l1-mlp" };
        for (size_t i = 0; i < sub_outs.size() && i < 4; i++) {
            dump_tensor_2d(sub_outs[i], std::string(dump_hidden_name) + sub_names[i]);
        }

        // Final hidden, post output norm, pre lm_head.
        dump_tensor_2d(hidden, dump_hidden_name);
    }

    ggml_backend_sched_reset(pt->sched);
    ggml_free(gctx);
    return out;
}

// Pre-compute the constant inputs that stay identical across the 32 MaskGIT
// steps of one chunk. attn_f16 is the only really expensive piece (B' * S * S
// F16 conversions, ~7 M ops on the typical voice cloning shape) so the win
// from caching is mostly there. mask_f / inv_mask / positions are smaller
// but free to cache too.
void pipeline_tts_llm_batched_ctx_init(MaskgitBatchedCtx * ctx,
                                       const int32_t *     audio_mask,
                                       const int32_t *     attention_mask,
                                       int                 B_prime,
                                       int                 S) {
    ctx->B_prime        = B_prime;
    ctx->S              = S;
    ctx->audio_mask_raw = audio_mask;
    ctx->attn_mask_raw  = attention_mask;
    ctx->has_attn_mask  = (attention_mask != NULL);

    ctx->mask_f.resize((size_t) B_prime * (size_t) S);
    ctx->inv_mask_f.resize((size_t) B_prime * (size_t) S);
    for (int b = 0; b < B_prime; b++) {
        for (int s = 0; s < S; s++) {
            int m                               = (audio_mask[(size_t) b * S + s] != 0) ? 1 : 0;
            ctx->mask_f[(size_t) b * S + s]     = (float) m;
            ctx->inv_mask_f[(size_t) b * S + s] = (float) (1 - m);
        }
    }

    ctx->positions.resize(S);
    for (int i = 0; i < S; i++) {
        ctx->positions[i] = i;
    }

    // The mask only takes two values (1.0 and 0.0). Pre-convert them once
    // instead of calling ggml_fp32_to_fp16 in the inner loop, which dominates
    // the CPU pre-compute on large S.
    if (ctx->has_attn_mask) {
        const uint16_t f16_one  = ggml_fp32_to_fp16(1.0f);
        const uint16_t f16_zero = ggml_fp32_to_fp16(0.0f);
        const size_t   n        = (size_t) B_prime * (size_t) S * (size_t) S;
        ctx->attn_f16.resize(n);
        for (size_t i = 0; i < n; i++) {
            ctx->attn_f16[i] = (attention_mask[i] != 0) ? f16_one : f16_zero;
        }
    } else {
        ctx->attn_f16.clear();
    }
}

// Batched LLM forward : single graph that fuses B' independent forwards on the
// trailing batch dim. Used for the cond + uncond CFG batching where row 0 is
// the cond row and row 1 is the uncond row, both running on the same S window.
// Pre-computed buffers (mask_f, inv_mask_f, positions, attn_f16) come from
// the ctx, shared across the 32 MaskGIT steps of a chunk.
std::vector<float> pipeline_tts_llm_forward_batched(PipelineTTS *             pt,
                                                    const int32_t *           input_ids,
                                                    const MaskgitBatchedCtx * ctx,
                                                    int                       K,
                                                    int                       T_audio,
                                                    const char *              dump_hidden_dir) {
    if (!ctx) {
        fprintf(stderr, "[LM-Forward-Batched] FATAL: ctx is NULL\n");
        return {};
    }
    const int B_prime = ctx->B_prime;
    const int S       = ctx->S;
    if (B_prime <= 0 || K <= 0 || S <= 0) {
        return {};
    }
    if (K > pt->lm.num_audio_codebook) {
        fprintf(stderr, "[LM-Forward-Batched] FATAL: K=%d exceeds num_audio_codebook=%d\n", K,
                pt->lm.num_audio_codebook);
        return {};
    }
    if (T_audio > S) {
        fprintf(stderr, "[LM-Forward-Batched] FATAL: T_audio=%d exceeds S=%d\n", T_audio, S);
        return {};
    }
    // Hidden-state debug dumps still go through the single-forward path so the
    // cond and uncond rows land in their own bin files.
    const bool force_loop = getenv("OMNIVOICE_LOOP_FORWARD") != nullptr;
    if (dump_hidden_dir || force_loop) {
        const int          V        = pt->lm.audio_vocab_size;
        const size_t       per_item = (size_t) V * (size_t) K * (size_t) S;
        std::vector<float> out((size_t) B_prime * per_item);
        for (int b = 0; b < B_prime; b++) {
            const int32_t * ids_b  = input_ids + (size_t) b * (size_t) K * (size_t) S;
            const int32_t * mask_b = ctx->audio_mask_raw + (size_t) b * (size_t) S;
            const int32_t * attn_b =
                ctx->has_attn_mask ? ctx->attn_mask_raw + (size_t) b * (size_t) S * (size_t) S : NULL;

            const char * hidden_name = nullptr;
            char         hidden_buf[64];
            if (b == 0) {
                hidden_name = "lm-hidden-step0-cond";
            } else if (b == 1) {
                hidden_name = "lm-hidden-step0-uncond";
            } else {
                snprintf(hidden_buf, sizeof(hidden_buf), "lm-hidden-step0-b%d", b);
                hidden_name = hidden_buf;
            }
            std::vector<float> logits_b =
                pipeline_tts_llm_forward(pt, ids_b, mask_b, attn_b, K, S, dump_hidden_dir, hidden_name);
            if (logits_b.size() != per_item) {
                fprintf(stderr, "[LM-Forward-Batched] FATAL: dump-mode item %d returned %zu f32 (expected %zu)\n", b,
                        logits_b.size(), per_item);
                return {};
            }
            std::copy(logits_b.begin(), logits_b.end(), out.begin() + (size_t) b * per_item);
        }
        return out;
    }

    const Qwen3Config & cfg = pt->lm.cfg;
    const int           V   = pt->lm.audio_vocab_size;
    const int           H   = cfg.hidden_size;

    // CPU pre-compute that depends on input_ids (mutates between MaskGIT
    // steps via the demask injection). Layouts :
    //   text_ids_buf [B_prime, S]    matches t_text_ids [B_prime * S], b slow
    //   shifted      [K, B_prime, S] matches t_shifted [B_prime * S, K], k slow
    std::vector<int32_t> shifted((size_t) K * (size_t) B_prime * (size_t) S);
    std::vector<int32_t> text_ids_buf((size_t) B_prime * (size_t) S);
    for (int b = 0; b < B_prime; b++) {
        for (int s = 0; s < S; s++) {
            int m                            = (ctx->audio_mask_raw[(size_t) b * S + s] != 0) ? 1 : 0;
            text_ids_buf[(size_t) b * S + s] = input_ids[((size_t) b * (size_t) K + 0) * (size_t) S + s];
            for (int k = 0; k < K; k++) {
                shifted[((size_t) k * (size_t) B_prime + (size_t) b) * (size_t) S + s] =
                    input_ids[((size_t) b * (size_t) K + (size_t) k) * (size_t) S + s] * m + k * V;
            }
        }
    }

    // Node budget : custom embed ~30, 28L stack ~850 (4D adds a few reshape
    // nodes per layer), audio_heads + reshape ~5. 8192 stays comfortable.
    const int    n_max_nodes    = 8192;
    const size_t graph_ctx_size = ggml_tensor_overhead() * n_max_nodes + ggml_graph_overhead_custom(n_max_nodes, false);

    struct ggml_init_params gp   = { graph_ctx_size, NULL, /*no_alloc=*/true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        return {};
    }

    struct ggml_tensor * t_text_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, B_prime * S);
    ggml_set_name(t_text_ids, "text_ids");
    ggml_set_input(t_text_ids);

    struct ggml_tensor * t_shifted = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, B_prime * S, K);
    ggml_set_name(t_shifted, "shifted_ids");
    ggml_set_input(t_shifted);

    // [1, S, B_prime] so multiplying with hidden states [H, S, B_prime]
    // broadcasts on H (dim 0) and matches per (s, b).
    struct ggml_tensor * t_mask = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, S, B_prime);
    ggml_set_name(t_mask, "mask");
    ggml_set_input(t_mask);

    struct ggml_tensor * t_inv_mask = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, S, B_prime);
    ggml_set_name(t_inv_mask, "inv_mask");
    ggml_set_input(t_inv_mask);

    // RoPE positions are 0..S-1, identical for cond and uncond rows.
    struct ggml_tensor * t_positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
    ggml_set_name(t_positions, "positions");
    ggml_set_input(t_positions);

    // Per-row attention bias. flash_attn_ext expects mask [n_kv, n_batch, ne32, ne33]
    // with n_head broadcast through ne32 and the outer batch through ne33. Layout
    // [S, S, 1, B_prime] : skv fast, sq mid, head broadcast, batch on the slowest.
    struct ggml_tensor * t_attn = NULL;
    if (ctx->has_attn_mask) {
        t_attn = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, S, S, 1, B_prime);
        ggml_set_name(t_attn, "attn_mask");
        ggml_set_input(t_attn);
    }

    // Custom embed. The flat get_rows runs on a B_prime * S row index buffer
    // and produces [H, B_prime * S] which we promote to [H, S, B_prime] before
    // the multiply broadcast and the stack.
    struct ggml_tensor * text_embeds_flat  = ggml_get_rows(gctx, pt->lm.embed_tokens, t_text_ids);
    struct ggml_tensor * audio_embeds_flat = NULL;
    for (int k = 0; k < K; k++) {
        struct ggml_tensor * idx_k =
            ggml_view_1d(gctx, t_shifted, B_prime * S, (size_t) k * (size_t) (B_prime * S) * sizeof(int32_t));
        struct ggml_tensor * emb_k = ggml_get_rows(gctx, pt->lm.audio_embeddings, idx_k);
        audio_embeds_flat          = (k == 0) ? emb_k : ggml_add(gctx, audio_embeds_flat, emb_k);
    }
    struct ggml_tensor * text_embeds   = ggml_reshape_3d(gctx, text_embeds_flat, H, S, B_prime);
    struct ggml_tensor * audio_embeds  = ggml_reshape_3d(gctx, audio_embeds_flat, H, S, B_prime);
    struct ggml_tensor * text_branch   = ggml_mul(gctx, text_embeds, t_inv_mask);
    struct ggml_tensor * audio_branch  = ggml_mul(gctx, audio_embeds, t_mask);
    struct ggml_tensor * inputs_embeds = ggml_add(gctx, text_branch, audio_branch);

    // 28L Qwen3 stack with B = B_prime, mask carried per-row.
    struct ggml_tensor * hidden =
        qwen3_build_layers(gctx, cfg, pt->lm.layers, pt->lm.final_norm, inputs_embeds, t_positions, t_attn, S,
                           pt->use_flash_attn, pt->clamp_fp16, nullptr, nullptr, -1, nullptr, B_prime);

    // audio_heads readout. hidden is [H, S, B_prime], audio_heads is [H, V*K],
    // mul_mat returns [V*K, S, B_prime] which we reshape to [V, K, S, B_prime].
    // Linear memory order [B_prime, S, K, V] matches the per-item layout the
    // single forward returns (V*K*S floats per row, B' rows stacked), so the
    // MaskGIT decoder reads it without further reshuffle.
    struct ggml_tensor * logits_flat = ggml_mul_mat(gctx, pt->lm.audio_heads, hidden);
    struct ggml_tensor * logits      = ggml_reshape_4d(gctx, logits_flat, V, K, S, B_prime);
    ggml_set_name(logits, "audio_logits");

    // Audio truncation : when T_audio > 0, the MaskGIT decoder only reads the
    // audio positions on cond row 0 (S range [S - T_audio, S)) and on uncond
    // row 1 (S range [0, T_audio)). Cutting these views before set_output
    // shrinks the GPU->CPU transfer from B_prime * V * K * S floats down to
    // 2 * V * K * T_audio floats, ~5.6x less for the typical voice cloning
    // shape. Math is identical : we just keep less of the same elements.
    struct ggml_tensor * cond_audio   = nullptr;
    struct ggml_tensor * uncond_audio = nullptr;
    if (T_audio > 0) {
        size_t               cond_offset = (size_t) (S - T_audio) * logits->nb[2] + (size_t) 0 * logits->nb[3];
        struct ggml_tensor * cond_view =
            ggml_view_4d(gctx, logits, V, K, T_audio, 1, logits->nb[1], logits->nb[2], logits->nb[3], cond_offset);
        cond_audio = ggml_cont(gctx, cond_view);
        ggml_set_name(cond_audio, "cond_audio_logits");
        ggml_set_output(cond_audio);

        size_t               uncond_offset = (size_t) 0 * logits->nb[2] + (size_t) 1 * logits->nb[3];
        struct ggml_tensor * uncond_view =
            ggml_view_4d(gctx, logits, V, K, T_audio, 1, logits->nb[1], logits->nb[2], logits->nb[3], uncond_offset);
        uncond_audio = ggml_cont(gctx, uncond_view);
        ggml_set_name(uncond_audio, "uncond_audio_logits");
        ggml_set_output(uncond_audio);
    } else {
        ggml_set_output(logits);
    }

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    if (T_audio > 0) {
        ggml_build_forward_expand(graph, cond_audio);
        ggml_build_forward_expand(graph, uncond_audio);
    } else {
        ggml_build_forward_expand(graph, logits);
    }

    if (!ggml_backend_sched_alloc_graph(pt->sched, graph)) {
        fprintf(stderr, "[LM-Forward-Batched] FATAL: sched_alloc_graph failed (B'=%d K=%d S=%d)\n", B_prime, K, S);
        ggml_free(gctx);
        return {};
    }

    ggml_backend_tensor_set(t_text_ids, text_ids_buf.data(), 0, (size_t) B_prime * (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(t_shifted, shifted.data(), 0, (size_t) K * (size_t) B_prime * (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(t_mask, ctx->mask_f.data(), 0, (size_t) B_prime * (size_t) S * sizeof(float));
    ggml_backend_tensor_set(t_inv_mask, ctx->inv_mask_f.data(), 0, (size_t) B_prime * (size_t) S * sizeof(float));
    ggml_backend_tensor_set(t_positions, ctx->positions.data(), 0, (size_t) S * sizeof(int32_t));

    if (t_attn) {
        ggml_backend_tensor_set(t_attn, ctx->attn_f16.data(), 0,
                                (size_t) B_prime * (size_t) S * (size_t) S * sizeof(uint16_t));
    }

    enum ggml_status st = ggml_backend_sched_graph_compute(pt->sched, graph);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[LM-Forward-Batched] FATAL: graph_compute status=%d\n", (int) st);
        ggml_backend_sched_reset(pt->sched);
        ggml_free(gctx);
        return {};
    }

    std::vector<float> out;
    if (T_audio > 0) {
        const size_t per_audio = (size_t) V * (size_t) K * (size_t) T_audio;
        out.resize(2 * per_audio);
        ggml_backend_tensor_get(cond_audio, out.data(), 0, per_audio * sizeof(float));
        ggml_backend_tensor_get(uncond_audio, out.data() + per_audio, 0, per_audio * sizeof(float));
    } else {
        const size_t n = ggml_nelements(logits);
        out.resize(n);
        ggml_backend_tensor_get(logits, out.data(), 0, n * sizeof(float));
    }

    ggml_backend_sched_reset(pt->sched);
    ggml_free(gctx);
    return out;
}

// Public TTS entry. Tokenize text, build prompt + CFG batch via prompt_tts_build,
// run the MaskGIT iterative decoder, return audio_tokens [K, T] flat.
std::vector<int32_t> pipeline_tts_generate(PipelineTTS *         pt,
                                           const BPETokenizer *  tok,
                                           const std::string &   text,
                                           const std::string &   lang,
                                           const std::string &   instruct,
                                           int                   T,
                                           bool                  denoise,
                                           const MaskgitConfig & mg_cfg,
                                           const std::string &   ref_text,
                                           const int32_t *       ref_audio_tokens,
                                           int                   ref_T,
                                           const char *          dump_dir,
                                           uint32_t *            ctr_lo_inout) {
    if (T <= 0) {
        fprintf(stderr, "[TTS] FATAL: T=%d must be positive\n", T);
        return {};
    }

    PromptTTS prompt = {};
    if (!prompt_tts_build(&prompt, tok, &pt->lm, text, lang, instruct, T, denoise, ref_text, ref_audio_tokens, ref_T)) {
        return {};
    }

    // Dump cond and uncond input_ids row k=0 for prompt diagnostic. Style and
    // text tokens are duplicated across all K codebooks so k=0 is enough.
    {
        DebugDumper dbg;
        debug_init(&dbg, dump_dir);
        int             ids_shape[1] = { prompt.c_len };
        const int32_t * cond_row     = prompt.input_ids.data();
        const int32_t * uncond_row   = prompt.input_ids.data() + (size_t) prompt.K * (size_t) prompt.c_len;
        debug_dump_i32_as_f32(&dbg, "prompt-cond-ids", cond_row, ids_shape, 1);
        debug_dump_i32_as_f32(&dbg, "prompt-uncond-ids", uncond_row, ids_shape, 1);
    }

    fprintf(stderr, "[TTS] Prompt: B'=%d K=%d S=%d c_len=%d u_len=%d\n", prompt.B_prime, prompt.K, prompt.S_max,
            prompt.c_len, prompt.u_len);

    return maskgit_generate(pt, &prompt, mg_cfg, T, dump_dir, ctr_lo_inout);
}

// Full TTS synthesis : tokens via pipeline_tts_generate, waveform via
// pipeline_codec_decode. Refuses to decode partially decoded outputs.
std::vector<float> pipeline_tts_synthesize(PipelineTTS *         pt,
                                           PipelineCodec *       pc,
                                           const BPETokenizer *  tok,
                                           const std::string &   text,
                                           const std::string &   lang,
                                           const std::string &   instruct,
                                           int                   T,
                                           bool                  denoise,
                                           const MaskgitConfig & mg_cfg,
                                           const std::string &   ref_text,
                                           const int32_t *       ref_audio_tokens,
                                           int                   ref_T,
                                           const char *          dump_dir,
                                           uint32_t *            ctr_lo_inout) {
    std::vector<int32_t> tokens = pipeline_tts_generate(pt, tok, text, lang, instruct, T, denoise, mg_cfg, ref_text,
                                                        ref_audio_tokens, ref_T, dump_dir, ctr_lo_inout);
    if (tokens.empty()) {
        return {};
    }

    const int K       = pt->lm.num_audio_codebook;
    const int mask_id = pt->lm.audio_mask_id;
    if ((int) tokens.size() != K * T) {
        fprintf(stderr, "[TTS] FATAL: token vector size %zu does not match K*T=%d*%d\n", tokens.size(), K, T);
        return {};
    }
    int n_residual_mask = 0;
    for (int32_t v : tokens) {
        if (v == mask_id) {
            n_residual_mask++;
        }
    }
    if (n_residual_mask) {
        fprintf(stderr, "[TTS] FATAL: %d residual mask tokens left after MaskGIT, refusing to decode\n",
                n_residual_mask);
        return {};
    }

    DebugDumper dbg;
    debug_init(&dbg, dump_dir);
    int tokens_shape[2] = { K, T };
    debug_dump_i32_as_f32(&dbg, "mg-tokens", tokens.data(), tokens_shape, 2);

    fprintf(stderr, "[TTS] Decode: K=%d T=%d expected_samples=%d\n", K, T, T * pc->hop_length);
    std::vector<float> audio = pipeline_codec_decode(pc, tokens.data(), K, T);

    if (!audio.empty()) {
        debug_dump_1d(&dbg, "output-audio", audio.data(), (int) audio.size());
    }
    return audio;
}

// Long-form TTS with automatic chunking and post-processing.
// Strict orchestration of upstream omnivoice/models/omnivoice.py:
//   - estimate target tokens for the full text via duration_estimate_tokens
//   - if T_total fits below the threshold, run single-shot
//   - else split text on punctuation, generate chunk 0 (no ref) and reuse its
//     audio tokens as the voice prompt for the remaining chunks (auto-voice
//     coherence trick from _generate_chunked, no-ref branch)
//   - cross-fade audio chunks
//   - apply post-processing (remove_silence, peak/0.5 when no ext ref,
//     fade_and_pad) on the merged waveform.
std::vector<float> pipeline_tts_synthesize_long(PipelineTTS *         pt,
                                                PipelineCodec *       pc,
                                                const BPETokenizer *  tok,
                                                const std::string &   text,
                                                const std::string &   lang,
                                                const std::string &   instruct,
                                                int                   T_override,
                                                float                 chunk_duration_sec,
                                                float                 chunk_threshold_sec,
                                                bool                  denoise,
                                                const MaskgitConfig & mg_cfg,
                                                const std::string &   ref_text,
                                                const int32_t *       ext_ref_tokens,
                                                int                   ext_ref_T,
                                                float                 ref_rms,
                                                const char *          dump_dir) {
    const int sr         = pc->sample_rate;
    const int hop        = pc->hop_length;
    const int frame_rate = sr / hop;

    // Estimated tokens for the full text. Chunking trigger uses the same
    // estimator as the single-shot path for consistency with upstream.
    int T_total = (T_override > 0) ? T_override : duration_estimate_tokens(text, ref_text, ext_ref_T);

    int  threshold_frames = (int) (chunk_threshold_sec * (float) frame_rate);
    bool no_chunk         = (T_override > 0) || (chunk_duration_sec <= 0.0f) || (T_total <= threshold_frames);

    std::vector<float> audio;

    // Shared Philox counter across MaskGIT calls. Mirrors PyTorch's global RNG
    // state which advances continuously through chunked model.generate(),
    // rather than resetting at each call. Initial value 0 corresponds to
    // PyTorch's freshly seeded generator just after fix_random_seed().
    uint32_t shared_ctr_lo = 0;

    if (no_chunk) {
        fprintf(stderr, "[TTS-Long] Single-shot path: T=%d frames (%.2fs), threshold=%d frames\n", T_total,
                (float) T_total / (float) frame_rate, threshold_frames);

        audio = pipeline_tts_synthesize(pt, pc, tok, text, lang, instruct, T_total, denoise, mg_cfg, ref_text,
                                        ext_ref_tokens, ext_ref_T, dump_dir, &shared_ctr_lo);

        if (audio.empty()) {
            return audio;
        }
    } else {
        // Per-chunk character budget derived from the full-text average
        // tokens-per-character, matching _generate_chunked upstream.
        int n_chars = chunker_utf8_count(text);
        if (n_chars < 1) {
            n_chars = 1;
        }

        double avg_tokens_per_char = (double) T_total / (double) n_chars;
        int    chunk_len           = (int) ((double) chunk_duration_sec * (double) frame_rate / avg_tokens_per_char);
        if (chunk_len < 1) {
            chunk_len = 1;
        }

        std::vector<std::string> chunks = chunk_text_punctuation(text, chunk_len, 3);

        if (chunks.empty()) {
            fprintf(stderr, "[TTS-Long] FATAL: chunker produced no chunks for input of %d chars\n", n_chars);
            return {};
        }

        fprintf(stderr, "[TTS-Long] Chunked: %d chunks, T_total=%d frames, chunk_len=%d codepoints\n",
                (int) chunks.size(), T_total, chunk_len);

        std::vector<std::vector<float>> chunk_audios;
        chunk_audios.reserve(chunks.size());

        // Active voice prompt for chunks 1..N. Initialised from the external
        // reference if provided, otherwise promoted from chunk 0 outputs.
        const int32_t *      prompt_tokens = ext_ref_tokens;
        int                  prompt_T      = ext_ref_T;
        std::string          prompt_text   = ref_text;
        std::vector<int32_t> chunk0_tokens;

        for (size_t i = 0; i < chunks.size(); i++) {
            const std::string & ct = chunks[i];

            // Chunk 0 in pure auto-voice runs without any reference. Every
            // other case rides on the active prompt.
            bool first_no_ref = (i == 0 && ext_ref_tokens == NULL);

            const int32_t *     this_ref      = first_no_ref ? NULL : prompt_tokens;
            int                 this_T        = first_no_ref ? 0 : prompt_T;
            const std::string & this_ref_text = first_no_ref ? std::string() : prompt_text;

            int Ti = duration_estimate_tokens(ct, this_ref_text, this_T);

            // Dump intermediate tensors only for chunk 0 so cossim tests
            // compare matching chunks across Python and C++.
            const char * chunk_dump_dir = (i == 0) ? dump_dir : NULL;

            fprintf(stderr, "[TTS-Long] Chunk %zu/%zu: chars=%d T=%d ref_T=%d\n", i + 1, chunks.size(),
                    chunker_utf8_count(ct), Ti, this_T);

            if (first_no_ref) {
                // Capture audio tokens before decoding so they can become the
                // voice prompt for chunks 1..N.
                chunk0_tokens = pipeline_tts_generate(pt, tok, ct, lang, instruct, Ti, denoise, mg_cfg, this_ref_text,
                                                      this_ref, this_T, chunk_dump_dir, &shared_ctr_lo);

                if (chunk0_tokens.empty()) {
                    fprintf(stderr, "[TTS-Long] FATAL: chunk 0 generate failed\n");
                    return {};
                }

                const int K = pt->lm.num_audio_codebook;
                if ((int) chunk0_tokens.size() != K * Ti) {
                    fprintf(stderr, "[TTS-Long] FATAL: chunk 0 token shape mismatch %zu vs K*T=%d*%d\n",
                            chunk0_tokens.size(), K, Ti);
                    return {};
                }

                std::vector<float> a = pipeline_codec_decode(pc, chunk0_tokens.data(), K, Ti);
                if (a.empty()) {
                    fprintf(stderr, "[TTS-Long] FATAL: chunk 0 decode failed\n");
                    return {};
                }

                // Mirror the single-shot pipeline_tts_synthesize dumps for chunk 0
                // so cossim tests see mg-tokens and decoded audio under the chunked
                // path too. Higher chunks go through pipeline_tts_synthesize which
                // already dumps these artefacts when chunk_dump_dir is non-null.
                if (chunk_dump_dir) {
                    DebugDumper dbg;
                    debug_init(&dbg, chunk_dump_dir);
                    int tokens_shape[2] = { K, Ti };
                    debug_dump_i32_as_f32(&dbg, "mg-tokens", chunk0_tokens.data(), tokens_shape, 2);
                    debug_dump_1d(&dbg, "output-audio", a.data(), (int) a.size());
                }

                chunk_audios.push_back(std::move(a));

                prompt_tokens = chunk0_tokens.data();
                prompt_T      = Ti;
                prompt_text   = ct;
            } else {
                std::vector<float> a =
                    pipeline_tts_synthesize(pt, pc, tok, ct, lang, instruct, Ti, denoise, mg_cfg, this_ref_text,
                                            this_ref, this_T, chunk_dump_dir, &shared_ctr_lo);
                if (a.empty()) {
                    fprintf(stderr, "[TTS-Long] FATAL: chunk %zu synthesize failed\n", i);
                    return {};
                }

                chunk_audios.push_back(std::move(a));
            }
        }

        audio = cross_fade_chunks(chunk_audios, sr, 0.3);

        if (audio.empty()) {
            fprintf(stderr, "[TTS-Long] FATAL: cross-fade produced empty output\n");
            return {};
        }

        fprintf(stderr, "[TTS-Long] Cross-faded %d chunks -> %zu samples\n", (int) chunk_audios.size(), audio.size());
    }

    // Post-processing: matches _post_process_audio in omnivoice.py.
    // remove_silence and fade_and_pad always run. The volume branch picks
    // peak/0.5 when there is no reference (ref_rms < 0), or rescales by
    // ref_rms / 0.1 for a quiet reference, or stays no-op for a loud one.
    size_t before = audio.size();

    remove_silence(audio, sr, 500, 100, 100, -50.0);

    if (ref_rms < 0.0f) {
        peak_normalize_half(audio);
    } else if (ref_rms < 0.1f) {
        float k = ref_rms / 0.1f;
        for (auto & s : audio) {
            s *= k;
        }
    }

    fade_and_pad(audio, sr, 0.1, 0.1);

    fprintf(stderr, "[TTS-Long] Post-proc: %zu -> %zu samples (%.2fs at %d Hz, ref_rms=%.4f)\n", before, audio.size(),
            (float) audio.size() / (float) sr, sr, ref_rms);

    return audio;
}

// Validate and normalise the raw instruct string against the voice-design
// vocabulary. Picks the target language from the synthesis text : any CJK
// ideograph -> Chinese, otherwise English.
bool pipeline_tts_resolve_instruct(const VoiceDesign * vd,
                                   const std::string & text,
                                   const std::string & raw,
                                   std::string *       out) {
    bool        use_zh = voice_design_has_cjk(text);
    std::string err;
    if (!voice_design_normalize(vd, raw, use_zh, out, &err)) {
        fprintf(stderr, "[TTS] ERROR: %s\n", err.c_str());
        return false;
    }
    return true;
}

// Convert a duration in seconds to a frame count using the codec frame rate
// (sample_rate / hop_length). Clamps to a minimum of 1 frame.
int pipeline_tts_duration_sec_to_tokens(const PipelineCodec * pc, float duration_sec) {
    float frame_rate = (float) pc->sample_rate / (float) pc->hop_length;
    int   T          = (int) (duration_sec * frame_rate);
    if (T < 1) {
        T = 1;
    }
    return T;
}

std::vector<float> pipeline_tts_synthesize_long_with_ref(PipelineTTS *         pt,
                                                         PipelineCodec *       pc,
                                                         const BPETokenizer *  tok,
                                                         const std::string &   text,
                                                         const std::string &   lang,
                                                         const std::string &   instruct,
                                                         int                   T_override,
                                                         float                 chunk_duration_sec,
                                                         float                 chunk_threshold_sec,
                                                         bool                  denoise,
                                                         bool                  preprocess_prompt,
                                                         const MaskgitConfig & mg_cfg,
                                                         const float *         ref_audio_24k,
                                                         int                   ref_n_samples,
                                                         const std::string &   ref_text_in,
                                                         const char *          dump_dir) {
    // No reference : pure TTS path. ref_rms = -1 routes the post-proc volume
    // branch to peak / 0.5 normalisation.
    if (ref_audio_24k == NULL || ref_n_samples <= 0) {
        return pipeline_tts_synthesize_long(pt, pc, tok, text, lang, instruct, T_override, chunk_duration_sec,
                                            chunk_threshold_sec, denoise, mg_cfg, "", NULL, 0, -1.0f, dump_dir);
    }

    // Encode the optional reference WAV into ref_audio_tokens once, before
    // the synthesize call. The codec encodes 24 kHz mono into [K, T_ref]
    // i32 codes ; the caller is expected to have resampled and downmixed
    // already (audio_read_mono or equivalent).
    std::string ref_text = ref_text_in;

    // Mirror Python preprocess_prompt: append a terminal "." (or ideographic
    // full stop for CJK) when missing.
    if (preprocess_prompt) {
        ref_text = add_punctuation(ref_text);
    }

    std::vector<float> ref_audio(ref_audio_24k, ref_audio_24k + ref_n_samples);

    // Mirror Python OmniVoice : compute ref_rms once on the loaded waveform.
    // Auto loudness normalisation when ref RMS is in (0, 0.1). Scales the
    // buffer so the new RMS hits exactly 0.1 ; the ORIGINAL ref_rms is what
    // we plumb into the post-proc to rescale the generated output back to
    // the reference loudness.
    double sumsq = 0.0;
    for (float v : ref_audio) {
        sumsq += (double) v * (double) v;
    }

    double ref_rms              = std::sqrt(sumsq / (double) ref_audio.size());
    float  ref_rms_for_postproc = (float) ref_rms;

    if (ref_rms > 0.0 && ref_rms < 0.1) {
        float gain = (float) (0.1 / ref_rms);
        for (float & v : ref_audio) {
            v *= gain;
        }

        fprintf(stderr, "[TTS] Reference: RMS %.4f -> 0.1 gain %.4f\n", ref_rms, gain);
    }

    // Mirror Python preprocess_prompt: silence-trim the reference clip with
    // mid=200ms, lead=100ms, trail=200ms, threshold=-50 dBFS before encoding.
    if (preprocess_prompt) {
        size_t before = ref_audio.size();
        remove_silence(ref_audio, 24000, 200, 100, 200, -50.0);
        fprintf(stderr, "[TTS] Reference: silence-trim %zu -> %zu samples\n", before, ref_audio.size());
    }

    int n_in      = (int) ref_audio.size();
    int n_aligned = (n_in / pc->hop_length) * pc->hop_length;
    fprintf(stderr, "[TTS] Reference: %d samples @ 24 kHz mono (%.2f s), aligned to %d (clip %d)\n", n_in,
            (double) n_in / 24000.0, n_aligned, n_in - n_aligned);

    std::vector<int32_t> ref_codes = pipeline_codec_encode(pc, ref_audio.data(), n_aligned, dump_dir);
    if (ref_codes.empty()) {
        fprintf(stderr, "[TTS] ERROR: codec_encode failed on reference audio\n");
        return {};
    }

    const int K = pt->lm.num_audio_codebook;
    if ((int) ref_codes.size() % K != 0) {
        fprintf(stderr, "[TTS] ERROR: ref codes size %zu not divisible by K=%d\n", ref_codes.size(), K);
        return {};
    }

    int ref_T = (int) ref_codes.size() / K;
    fprintf(stderr, "[TTS] Reference: encoded to [K=%d, T=%d] codes\n", K, ref_T);
    if (dump_dir) {
        DebugDumper dbg;
        debug_init(&dbg, dump_dir);
        int ref_shape[2] = { K, ref_T };
        debug_dump_i32_as_f32(&dbg, "ref-audio-codes", ref_codes.data(), ref_shape, 2);
    }

    return pipeline_tts_synthesize_long(pt, pc, tok, text, lang, instruct, T_override, chunk_duration_sec,
                                        chunk_threshold_sec, denoise, mg_cfg, ref_text, ref_codes.data(), ref_T,
                                        ref_rms_for_postproc, dump_dir);
}
