#pragma once
// omnivoice-llm.h: OmniVoice TTS LLM weights and graph helpers
//
// Container for the OmniVoice language model: a Qwen3 backbone driving a
// MaskGIT-style non autoregressive head over 8 audio codebooks. Pure weights
// holder, the backend, the scheduler and graph allocation live in the higher
// level pipeline-tts module that owns this struct.
//
// Architecture (reference layout):
//   input_ids   [B, num_audio_codebook, S] i32, row 0 doubles as text tokens
//   audio_mask  [B, S] bool, true on audio positions
//   custom embed:
//     text_embeds  = embed_tokens(input_ids[:, 0, :])
//     shifted_ids  = input_ids * audio_mask + codebook_layer_offsets
//     audio_embeds = sum_k audio_embeddings(shifted_ids[:, k, :])
//     inputs_embeds = where(audio_mask, audio_embeds, text_embeds)  [B, S, H]
//   stack       28 Qwen3 layers, bidirectional (is_causal = false)
//   final_norm  RMSNorm
//   audio_logits = audio_heads @ hidden -> reshape [B, num_codebook, S, audio_vocab]
//
// MaskGIT decoding runs N stateless forwards, no KV cache.

#include "gguf-weights.h"
#include "qwen3-enc.h"

#include <cstdio>

#define OMNIVOICE_MAX_CODEBOOKS 16

struct OmniVoiceLM {
    Qwen3Config cfg;
    Qwen3Layer  layers[QWEN3_MAX_LAYERS];

    // Text path
    struct ggml_tensor * embed_tokens;  // [H, vocab_size]
    struct ggml_tensor * final_norm;    // [H]

    // Audio path: 8 codebooks share one flat vocab of (num_audio_codebook * audio_vocab_size)
    // entries. shifted_ids = id + k*audio_vocab_size lifts each codebook to its slice.
    struct ggml_tensor * audio_embeddings;  // [H, num_audio_codebook * audio_vocab_size]
    struct ggml_tensor * audio_heads;       // [H, num_audio_codebook * audio_vocab_size], not tied

    // Audio config (read from GGUF KV)
    int num_audio_codebook;                               // 8
    int audio_vocab_size;                                 // 1025
    int audio_mask_id;                                    // 1024
    int codebook_layer_offsets[OMNIVOICE_MAX_CODEBOOKS];  // k * audio_vocab_size
};

// Load OmniVoiceLM weights from an already opened GGUF.
// wctx is shared with the higher level pipeline. Caller calls wctx_alloc
// once all modules are loaded, then closes the GGUF.
static bool omnivoice_lm_load(OmniVoiceLM * m, const GGUFModel & gf, WeightCtx * wctx) {
    *m = {};

    // Backbone config from canonical KV namespace omnivoice-lm.*
    m->cfg.hidden_size       = (int) gf_get_u32(gf, "omnivoice-lm.embedding_length");
    m->cfg.intermediate_size = (int) gf_get_u32(gf, "omnivoice-lm.feed_forward_length");
    m->cfg.n_heads           = (int) gf_get_u32(gf, "omnivoice-lm.attention.head_count");
    m->cfg.n_kv_heads        = (int) gf_get_u32(gf, "omnivoice-lm.attention.head_count_kv");
    m->cfg.head_dim          = (int) gf_get_u32(gf, "omnivoice-lm.attention.key_length");
    m->cfg.n_layers          = (int) gf_get_u32(gf, "omnivoice-lm.block_count");
    m->cfg.rope_theta        = gf_get_f32(gf, "omnivoice-lm.rope.freq_base");
    m->cfg.rms_norm_eps      = gf_get_f32(gf, "omnivoice-lm.attention.layer_norm_rms_epsilon");
    m->cfg.is_causal         = false;  // MaskGIT, full bidirectional attention

    if (m->cfg.n_layers > QWEN3_MAX_LAYERS) {
        fprintf(stderr, "[LM-Load] FATAL: n_layers=%d exceeds QWEN3_MAX_LAYERS=%d\n", m->cfg.n_layers,
                QWEN3_MAX_LAYERS);
        return false;
    }

    // Audio config from omnivoice.* namespace
    m->num_audio_codebook = (int) gf_get_u32(gf, "omnivoice.num_audio_codebook");
    m->audio_vocab_size   = (int) gf_get_u32(gf, "omnivoice.audio_vocab_size");
    m->audio_mask_id      = (int) gf_get_u32(gf, "omnivoice.audio_mask_id");

    if (m->num_audio_codebook > OMNIVOICE_MAX_CODEBOOKS) {
        fprintf(stderr, "[LM-Load] FATAL: num_audio_codebook=%d exceeds OMNIVOICE_MAX_CODEBOOKS=%d\n",
                m->num_audio_codebook, OMNIVOICE_MAX_CODEBOOKS);
        return false;
    }

    // codebook_layer_offsets: k * audio_vocab_size, computed at load time.
    // Reference builds this as register_buffer(arange(K) * audio_vocab_size).
    for (int k = 0; k < m->num_audio_codebook; k++) {
        m->codebook_layer_offsets[k] = k * m->audio_vocab_size;
    }

    // Tensors. Backbone uses prefix llm.layers.{i}, embeddings/norm at llm.* root.
    m->embed_tokens = gf_load_tensor(wctx, gf, "llm.embed_tokens.weight");
    m->final_norm   = gf_load_tensor_f32(wctx, gf, "llm.norm.weight");

    m->audio_embeddings = gf_load_tensor(wctx, gf, "audio_embeddings.weight");
    m->audio_heads      = gf_load_tensor(wctx, gf, "audio_heads.weight");

    fprintf(stderr, "[LM-Load] Loaded: %dL H=%d FFN=%d Nh=%d Nkv=%d D=%d theta=%.0f eps=%.0e", m->cfg.n_layers,
            m->cfg.hidden_size, m->cfg.intermediate_size, m->cfg.n_heads, m->cfg.n_kv_heads, m->cfg.head_dim,
            (double) m->cfg.rope_theta, (double) m->cfg.rms_norm_eps);
    fprintf(stderr, " | audio: K=%d V=%d mask_id=%d\n", m->num_audio_codebook, m->audio_vocab_size, m->audio_mask_id);

    for (int i = 0; i < m->cfg.n_layers; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "llm.layers.%d", i);
        qwen3_load_layer(wctx, gf, &m->layers[i], prefix, i);
    }

    return true;
}
