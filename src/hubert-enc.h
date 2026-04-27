#pragma once
// hubert-enc.h : HuBERT encoder for OmniVoice (GGML)
//
// HuBERT base model used as the semantic feature extractor inside the
// HiggsAudioV2 audio_tokenizer. The full module produces 13 hidden states
// (post pos-conv embedding + 12 transformer layer outputs) which are stacked,
// averaged channel-wise and downsampled by 2 before being fed to the
// SemanticEncoder.
//
// Pipeline reference (from transformers HiggsAudioV2Tokenizer) :
//   wav 24k -> resample 16k -> F.pad(160, 160)
//   feature_extractor    : 7 conv layers, cumul stride 320, 1 -> 512 channels
//   feature_projection   : LayerNorm(512) + Linear(512, 768)
//   pos_conv_embed       : grouped Conv1d (groups=16, k=128) + GELU + LayerNorm
//   12 x HubertLayer     : Post-LN attention + FFN (gelu)
//   final layer_norm     : LayerNorm(768)
//   stack + mean + ::2   : [B, 13, T_h, 768] -> [B, T_h/2, 768]
//
// Stage 1 ports the feature_extractor only. Layer 0 carries an InstanceNorm
// equivalent (HF feat_extract_norm = "group", num_groups == num_channels,
// affine=True) followed by GELU. Layers 1 to 6 are conv + GELU only. All
// convs use valid padding (pad = 0) and conv_bias = False.
//
// I/O layout : audio in is [T_audio, 1] f32 (T fast, IC=1 slow), output is
// [T_out, 512] f32 in GGML layout matching dac_conv1d output convention.

#include "dac-decoder.h"  // dac_conv1d, dac_load_passthrough, dac_load_bias_f32
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define HUBERT_FEAT_NUM_LAYERS 7
#define HUBERT_FEAT_LN_EPS     1e-5f
#define HUBERT_FEAT_KERNELS    { 10, 3, 3, 3, 3, 2, 2 }
#define HUBERT_FEAT_STRIDES    { 5, 2, 2, 2, 2, 2, 2 }
#define HUBERT_FEAT_DIMS       { 512, 512, 512, 512, 512, 512, 512 }

#define HUBERT_PROJ_IN     512
#define HUBERT_PROJ_OUT    768
#define HUBERT_PROJ_LN_EPS 1e-5f

#define HUBERT_HIDDEN     768
#define HUBERT_NUM_HEADS  12
#define HUBERT_HEAD_DIM   64  // 768 / 12
#define HUBERT_FFN_INNER  3072
#define HUBERT_NUM_LAYERS 12
#define HUBERT_LAYER_EPS  1e-5f

#define HUBERT_POS_K      128
#define HUBERT_POS_GROUPS 16
#define HUBERT_POS_IC_PG  48  // 768 / 16
#define HUBERT_POS_OC_PG  48  // 768 / 16
#define HUBERT_POS_PAD    64  // k // 2
#define HUBERT_POS_LN_EPS 1e-5f

// One conv layer of the feature_extractor stack. gn_w / gn_b are NULL on every
// layer except layer 0 where feat_extract_norm = "group" places an affine
// GroupNorm right after the conv.
struct HubertFeatLayer {
    struct ggml_tensor * cw;
    struct ggml_tensor * gn_w;
    struct ggml_tensor * gn_b;
    int                  k;
    int                  stride;
    int                  ic;
    int                  oc;
    bool                 has_norm;
};

struct HubertFeatExtractor {
    HubertFeatLayer       layers[HUBERT_FEAT_NUM_LAYERS];
    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

// Three phase load (describe / alloc / fill), mirror of sem_enc_load.
static bool hubert_feat_load(HubertFeatExtractor * h, const GGUFModel & gf, ggml_backend_t backend) {
    static const int K[] = HUBERT_FEAT_KERNELS;
    static const int S[] = HUBERT_FEAT_STRIDES;
    static const int D[] = HUBERT_FEAT_DIMS;

    // Phase 1 : describe
    const int               n_tensors_max = 32;
    size_t                  ctx_size      = (size_t) n_tensors_max * ggml_tensor_overhead() + 1024;
    struct ggml_init_params p             = { ctx_size, NULL, true };
    h->weight_ctx                         = ggml_init(p);
    struct ggml_context * ctx             = h->weight_ctx;

    int prev_dim = 1;
    for (int i = 0; i < HUBERT_FEAT_NUM_LAYERS; i++) {
        HubertFeatLayer & L = h->layers[i];
        L.k                 = K[i];
        L.stride            = S[i];
        L.ic                = prev_dim;
        L.oc                = D[i];
        L.has_norm          = (i == 0);
        std::string pf      = "semantic_model.feature_extractor.conv_layers." + std::to_string(i);
        L.cw                = ggml_new_tensor_3d(ctx, gf_get_type(gf, pf + ".conv.weight"), K[i], prev_dim, D[i]);
        if (L.has_norm) {
            L.gn_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D[i]);
            L.gn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D[i]);
        } else {
            L.gn_w = NULL;
            L.gn_b = NULL;
        }
        prev_dim = D[i];
    }

    // Phase 2 : alloc backend
    h->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!h->weight_buf) {
        fprintf(stderr, "[HuBERT-Feat] FATAL: failed to allocate weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(h->weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Phase 3 : fill (bf16 passthrough for convs, F32 cast for affine GroupNorm params).
    for (int i = 0; i < HUBERT_FEAT_NUM_LAYERS; i++) {
        HubertFeatLayer & L  = h->layers[i];
        std::string       pf = "semantic_model.feature_extractor.conv_layers." + std::to_string(i);
        dac_load_passthrough(L.cw, gf, pf + ".conv.weight");
        if (L.has_norm) {
            dac_load_bias_f32(L.gn_w, gf, pf + ".layer_norm.weight");
            dac_load_bias_f32(L.gn_b, gf, pf + ".layer_norm.bias");
        }
    }

    fprintf(stderr, "[HuBERT-Feat] Loaded: %d conv layers, cumul stride 320, weights %.1f MB\n", HUBERT_FEAT_NUM_LAYERS,
            (float) ggml_backend_buffer_get_size(h->weight_buf) / (1024 * 1024));
    return true;
}

static void hubert_feat_free(HubertFeatExtractor * h) {
    if (h->weight_buf) {
        ggml_backend_buffer_free(h->weight_buf);
        h->weight_buf = NULL;
    }
    if (h->weight_ctx) {
        ggml_free(h->weight_ctx);
        h->weight_ctx = NULL;
    }
}

// Build feature_extractor graph. Input [T_audio, 1] f32, output [T_out, 512] f32.
// HF HuBERT uses valid padding (pad = 0) and dilation 1 on every conv.
// Layer 0 : conv -> GroupNorm(G == C, affine) -> GELU
// Layer 1..6 : conv -> GELU
static struct ggml_tensor * hubert_feat_build_graph(struct ggml_context *       ctx,
                                                    const HubertFeatExtractor * h,
                                                    struct ggml_tensor *        x  // [T_audio, 1] f32
) {
    for (int i = 0; i < HUBERT_FEAT_NUM_LAYERS; i++) {
        const HubertFeatLayer & L = h->layers[i];
        x                         = dac_conv1d(ctx, L.cw, NULL, x, L.stride, 0, 1);
        if (L.has_norm) {
            // ggml_group_norm reads channels from ne[2], so reshape (T, C)
            // into (T, 1, C) for the normalization, then collapse back to
            // (T, C) for the affine mul + add. With n_groups == oc each
            // group covers ne[0] * ne[1] * 1 = T elements, which matches
            // reference GroupNorm(G == C) statistics over the time axis.
            const int T             = (int) x->ne[0];
            x                       = ggml_reshape_3d(ctx, x, T, 1, L.oc);
            x                       = ggml_group_norm(ctx, x, L.oc, HUBERT_FEAT_LN_EPS);
            x                       = ggml_reshape_2d(ctx, x, T, L.oc);
            struct ggml_tensor * w2 = ggml_reshape_2d(ctx, L.gn_w, 1, L.oc);
            struct ggml_tensor * b2 = ggml_reshape_2d(ctx, L.gn_b, 1, L.oc);
            x                       = ggml_mul(ctx, x, w2);
            x                       = ggml_add(ctx, x, b2);
        }
        x = ggml_gelu_erf(ctx, x);
    }
    return x;
}

// feature_projection : LayerNorm(conv_dim_last=512) + Linear(512 -> 768).
// HuBERT applies it on the [B, T, 512] post-feature_extractor tensor with a
// channel-axis LayerNorm (last dim) and a Linear projection without dropout
// at inference time. We run the whole stage in C-first layout (C, T) so
// ggml_norm normalizes along ne[0] = C and the matmul output naturally fits
// the rest of the encoder (pos_conv + 12 transformer layers + final LN are
// all C-first).
struct HubertFeatProjection {
    struct ggml_tensor * ln_w;
    struct ggml_tensor * ln_b;
    struct ggml_tensor * proj_w;
    struct ggml_tensor * proj_b;

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

static bool hubert_proj_load(HubertFeatProjection * p, const GGUFModel & gf, ggml_backend_t backend) {
    // Phase 1 : describe
    const int               n_tensors_max = 8;
    size_t                  ctx_size      = (size_t) n_tensors_max * ggml_tensor_overhead() + 1024;
    struct ggml_init_params gp            = { ctx_size, NULL, true };
    p->weight_ctx                         = ggml_init(gp);
    struct ggml_context * ctx             = p->weight_ctx;

    p->ln_w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_PROJ_IN);
    p->ln_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_PROJ_IN);
    p->proj_w = ggml_new_tensor_2d(ctx, gf_get_type(gf, "semantic_model.feature_projection.projection.weight"),
                                   HUBERT_PROJ_IN, HUBERT_PROJ_OUT);
    p->proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_PROJ_OUT);

    // Phase 2 : alloc backend
    p->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!p->weight_buf) {
        fprintf(stderr, "[HuBERT-Proj] FATAL: failed to allocate weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(p->weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Phase 3 : fill (BF16 passthrough for the projection matrix, F32 cast for
    // the affine LN params and the projection bias so ggml_add stays on the
    // CUDA-supported F32/F16 src1 path).
    dac_load_bias_f32(p->ln_w, gf, "semantic_model.feature_projection.layer_norm.weight");
    dac_load_bias_f32(p->ln_b, gf, "semantic_model.feature_projection.layer_norm.bias");
    dac_load_passthrough(p->proj_w, gf, "semantic_model.feature_projection.projection.weight");
    dac_load_bias_f32(p->proj_b, gf, "semantic_model.feature_projection.projection.bias");

    fprintf(stderr, "[HuBERT-Proj] Loaded: LN(%d) + Linear(%d -> %d), weights %.1f MB\n", HUBERT_PROJ_IN,
            HUBERT_PROJ_IN, HUBERT_PROJ_OUT, (float) ggml_backend_buffer_get_size(p->weight_buf) / (1024 * 1024));
    return true;
}

static void hubert_proj_free(HubertFeatProjection * p) {
    if (p->weight_buf) {
        ggml_backend_buffer_free(p->weight_buf);
        p->weight_buf = NULL;
    }
    if (p->weight_ctx) {
        ggml_free(p->weight_ctx);
        p->weight_ctx = NULL;
    }
}

// Build feature_projection graph. Input [T, 512] f32 (T fast, conv1d-style),
// output [768, T] f32 (C fast, transformer-style). The transpose lives here
// once; downstream transformer blocks stay in C-first natively.
static struct ggml_tensor * hubert_proj_build_graph(struct ggml_context *        ctx,
                                                    const HubertFeatProjection * p,
                                                    struct ggml_tensor *         x  // [T, 512] f32
) {
    // (T, C) -> (C, T) so ggml_norm normalizes along ne[0] = C.
    x                       = ggml_cont(ctx, ggml_transpose(ctx, x));
    x                       = ggml_norm(ctx, x, HUBERT_PROJ_LN_EPS);
    struct ggml_tensor * w2 = ggml_reshape_2d(ctx, p->ln_w, HUBERT_PROJ_IN, 1);
    struct ggml_tensor * b2 = ggml_reshape_2d(ctx, p->ln_b, HUBERT_PROJ_IN, 1);
    x                       = ggml_mul(ctx, x, w2);
    x                       = ggml_add(ctx, x, b2);
    // Linear : (in=512, T) @ proj_w (in=512, out=768) -> (out=768, T)
    x                       = ggml_mul_mat(ctx, p->proj_w, x);
    struct ggml_tensor * pb = ggml_reshape_2d(ctx, p->proj_b, HUBERT_PROJ_OUT, 1);
    x                       = ggml_add(ctx, x, pb);
    return x;
}

// HuBERT transformer encoder layer (Post-LN flavor : do_stable_layer_norm = false).
// Standard multi-head self-attention + feed-forward, no causal mask, bidirectional.
// All operations stay in C-first layout : input and output [768, T] f32.
//
// Forward (HubertEncoderLayer.forward in transformers) :
//   r  = x
//   x  = self_attention(x)                  # MHA, scaling = 1 / sqrt(64) = 0.125
//   x  = r + x
//   x  = layer_norm(x)                      # LN(768) post attn add
//   x  = x + feed_forward(x)                # Linear(768->3072) + GELU + Linear(3072->768)
//   x  = final_layer_norm(x)                # LN(768) post FFN add

struct HubertAttention {
    struct ggml_tensor * qw;
    struct ggml_tensor * qb;
    struct ggml_tensor * kw;
    struct ggml_tensor * kb;
    struct ggml_tensor * vw;
    struct ggml_tensor * vb;
    struct ggml_tensor * ow;
    struct ggml_tensor * ob;
};

struct HubertFFN {
    struct ggml_tensor * fc1_w;  // intermediate_dense.weight
    struct ggml_tensor * fc1_b;
    struct ggml_tensor * fc2_w;  // output_dense.weight
    struct ggml_tensor * fc2_b;
};

struct HubertLayerNormPair {
    struct ggml_tensor * w;
    struct ggml_tensor * b;
};

// One transformer encoder layer. Holds all 16 tensors (4 attn proj * 2 + 2 ffn dense
// * 2 + 2 LN pairs). Each layer owns its own backend buffer for clean lifecycle.
struct HubertLayer {
    HubertAttention     attn;
    HubertFFN           ffn;
    HubertLayerNormPair ln_attn;   // layer_norm (post attn add)
    HubertLayerNormPair ln_final;  // final_layer_norm (post ffn add)

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

static bool hubert_layer_load(HubertLayer * L, const GGUFModel & gf, ggml_backend_t backend, int idx) {
    // Phase 1 : describe (16 tensors)
    const int               n_tensors_max = 32;
    size_t                  ctx_size      = (size_t) n_tensors_max * ggml_tensor_overhead() + 1024;
    struct ggml_init_params gp            = { ctx_size, NULL, true };
    L->weight_ctx                         = ggml_init(gp);
    struct ggml_context * ctx             = L->weight_ctx;

    std::string lp = "semantic_model.encoder.layers." + std::to_string(idx) + ".";

    // Attention : 4 Linear(768, 768) with bias
    L->attn.qw = ggml_new_tensor_2d(ctx, gf_get_type(gf, lp + "attention.q_proj.weight"), HUBERT_HIDDEN, HUBERT_HIDDEN);
    L->attn.qb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);
    L->attn.kw = ggml_new_tensor_2d(ctx, gf_get_type(gf, lp + "attention.k_proj.weight"), HUBERT_HIDDEN, HUBERT_HIDDEN);
    L->attn.kb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);
    L->attn.vw = ggml_new_tensor_2d(ctx, gf_get_type(gf, lp + "attention.v_proj.weight"), HUBERT_HIDDEN, HUBERT_HIDDEN);
    L->attn.vb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);
    L->attn.ow =
        ggml_new_tensor_2d(ctx, gf_get_type(gf, lp + "attention.out_proj.weight"), HUBERT_HIDDEN, HUBERT_HIDDEN);
    L->attn.ob = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);

    // FFN : Linear(768, 3072) + Linear(3072, 768) with bias
    L->ffn.fc1_w = ggml_new_tensor_2d(ctx, gf_get_type(gf, lp + "feed_forward.intermediate_dense.weight"),
                                      HUBERT_HIDDEN, HUBERT_FFN_INNER);
    L->ffn.fc1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_FFN_INNER);
    L->ffn.fc2_w = ggml_new_tensor_2d(ctx, gf_get_type(gf, lp + "feed_forward.output_dense.weight"), HUBERT_FFN_INNER,
                                      HUBERT_HIDDEN);
    L->ffn.fc2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);

    // 2 LayerNorms with affine
    L->ln_attn.w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);
    L->ln_attn.b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);
    L->ln_final.w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);
    L->ln_final.b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);

    // Phase 2 : alloc backend
    L->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!L->weight_buf) {
        fprintf(stderr, "[HuBERT-Layer%d] FATAL: failed to allocate weight buffer\n", idx);
        return false;
    }
    ggml_backend_buffer_set_usage(L->weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Phase 3 : fill (BF16 passthrough on Linear weights, F32 cast on biases and LN params).
    std::string p = "semantic_model.encoder.layers." + std::to_string(idx) + ".";
    dac_load_passthrough(L->attn.qw, gf, p + "attention.q_proj.weight");
    dac_load_bias_f32(L->attn.qb, gf, p + "attention.q_proj.bias");
    dac_load_passthrough(L->attn.kw, gf, p + "attention.k_proj.weight");
    dac_load_bias_f32(L->attn.kb, gf, p + "attention.k_proj.bias");
    dac_load_passthrough(L->attn.vw, gf, p + "attention.v_proj.weight");
    dac_load_bias_f32(L->attn.vb, gf, p + "attention.v_proj.bias");
    dac_load_passthrough(L->attn.ow, gf, p + "attention.out_proj.weight");
    dac_load_bias_f32(L->attn.ob, gf, p + "attention.out_proj.bias");

    dac_load_passthrough(L->ffn.fc1_w, gf, p + "feed_forward.intermediate_dense.weight");
    dac_load_bias_f32(L->ffn.fc1_b, gf, p + "feed_forward.intermediate_dense.bias");
    dac_load_passthrough(L->ffn.fc2_w, gf, p + "feed_forward.output_dense.weight");
    dac_load_bias_f32(L->ffn.fc2_b, gf, p + "feed_forward.output_dense.bias");

    dac_load_bias_f32(L->ln_attn.w, gf, p + "layer_norm.weight");
    dac_load_bias_f32(L->ln_attn.b, gf, p + "layer_norm.bias");
    dac_load_bias_f32(L->ln_final.w, gf, p + "final_layer_norm.weight");
    dac_load_bias_f32(L->ln_final.b, gf, p + "final_layer_norm.bias");

    return true;
}

static void hubert_layer_free(HubertLayer * L) {
    if (L->weight_buf) {
        ggml_backend_buffer_free(L->weight_buf);
        L->weight_buf = NULL;
    }
    if (L->weight_ctx) {
        ggml_free(L->weight_ctx);
        L->weight_ctx = NULL;
    }
}

// LayerNorm helper : in/out [C, T] f32 with C on ne[0] (fast). ggml_norm
// normalizes along ne[0] = C, then we scale + shift via broadcast over T.
static struct ggml_tensor * hubert_apply_ln(struct ggml_context *       ctx,
                                            struct ggml_tensor *        x,
                                            const HubertLayerNormPair & ln,
                                            float                       eps) {
    x                       = ggml_norm(ctx, x, eps);
    struct ggml_tensor * w2 = ggml_reshape_2d(ctx, ln.w, HUBERT_HIDDEN, 1);
    struct ggml_tensor * b2 = ggml_reshape_2d(ctx, ln.b, HUBERT_HIDDEN, 1);
    x                       = ggml_mul(ctx, x, w2);
    x                       = ggml_add(ctx, x, b2);
    return x;
}

// Linear with bias on a (C_in, T) input, returning (C_out, T). Bias is F32
// [C_out] reshaped to (C_out, 1) so ggml_add broadcasts over the T axis.
static struct ggml_tensor * hubert_linear(struct ggml_context * ctx,
                                          struct ggml_tensor *  w,  // bf16 (C_in, C_out)
                                          struct ggml_tensor *  b,  // f32  (C_out)
                                          struct ggml_tensor *  x,  // f32  (C_in, T)
                                          int                   c_out) {
    x                       = ggml_mul_mat(ctx, w, x);              // (C_out, T)
    struct ggml_tensor * b2 = ggml_reshape_2d(ctx, b, c_out, 1);
    return ggml_add(ctx, x, b2);
}

// Multi-head self-attention. Input/output [768, T] f32 C-first. No mask, no
// causal (bidirectional). Heads layout follows the llama.cpp convention :
//   q : (D, T, H) for Q
//   k : (D, T, H) for K -> ggml_mul_mat(K, Q) returns scores (T_k, T_q, H)
//   v : (T, D, H) so ggml_mul_mat(V, scores) returns out (D, T_q, H)
static struct ggml_tensor * hubert_attention(struct ggml_context *   ctx,
                                             const HubertAttention & a,
                                             struct ggml_tensor *    x,  // [768, T]
                                             int                     T) {
    const int D = HUBERT_HEAD_DIM;
    const int H = HUBERT_NUM_HEADS;

    struct ggml_tensor * q = hubert_linear(ctx, a.qw, a.qb, x, HUBERT_HIDDEN);
    struct ggml_tensor * k = hubert_linear(ctx, a.kw, a.kb, x, HUBERT_HIDDEN);
    struct ggml_tensor * v = hubert_linear(ctx, a.vw, a.vb, x, HUBERT_HIDDEN);

    // Reshape (D*H, T) -> (D, H, T) then permute to per-head layouts.
    q = ggml_reshape_3d(ctx, q, D, H, T);
    k = ggml_reshape_3d(ctx, k, D, H, T);
    v = ggml_reshape_3d(ctx, v, D, H, T);

    // ggml_permute convention : new_ne[axis_i] = old_ne[i].
    // Q, K : (D, H, T) -> (D, T, H). new_ne[0]=D=old_ne[0], new_ne[1]=T=old_ne[2],
    // new_ne[2]=H=old_ne[1] -> axis args (0, 2, 1, 3).
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

    // V : (D, H, T) -> (T, D, H). new_ne[0]=T=old_ne[2], new_ne[1]=D=old_ne[0],
    // new_ne[2]=H=old_ne[1] -> axis args (1, 2, 0, 3).
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

    // Scaled dot-product attention.
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);  // (T_k, T_q, H)
    scores                      = ggml_scale(ctx, scores, 1.0f / sqrtf((float) D));
    scores                      = ggml_soft_max(ctx, scores);

    struct ggml_tensor * out = ggml_mul_mat(ctx, v, scores);  // (D, T_q, H)

    // (D, T, H) -> (D, H, T) and collapse heads into the channel axis.
    // new_ne[0]=D=old_ne[0], new_ne[1]=H=old_ne[2], new_ne[2]=T=old_ne[1]
    // -> axis args (0, 2, 1, 3).
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    out = ggml_reshape_2d(ctx, out, HUBERT_HIDDEN, T);

    out = hubert_linear(ctx, a.ow, a.ob, out, HUBERT_HIDDEN);
    return out;
}

// FFN : Linear(768 -> 3072) + GELU(exact erf) + Linear(3072 -> 768).
// hidden_act = "gelu" maps to torch.nn.functional.gelu(approximate='none').
static struct ggml_tensor * hubert_ffn(struct ggml_context * ctx,
                                       const HubertFFN &     f,
                                       struct ggml_tensor *  x,  // [768, T]
                                       int                   T) {
    (void) T;  // shape is implicit, kept for symmetry with hubert_attention
    x = hubert_linear(ctx, f.fc1_w, f.fc1_b, x, HUBERT_FFN_INNER);
    x = ggml_gelu_erf(ctx, x);
    x = hubert_linear(ctx, f.fc2_w, f.fc2_b, x, HUBERT_HIDDEN);
    return x;
}

// Full Post-LN encoder layer. Input/output [768, T] f32 C-first.
static struct ggml_tensor * hubert_layer_build_graph(struct ggml_context * ctx,
                                                     const HubertLayer *   L,
                                                     struct ggml_tensor *  x  // [768, T]
) {
    const int            T        = (int) x->ne[1];
    struct ggml_tensor * residual = x;

    x = hubert_attention(ctx, L->attn, x, T);
    x = ggml_add(ctx, residual, x);
    x = hubert_apply_ln(ctx, x, L->ln_attn, HUBERT_LAYER_EPS);

    struct ggml_tensor * ffn_out = hubert_ffn(ctx, L->ffn, x, T);
    x                            = ggml_add(ctx, x, ffn_out);
    x                            = hubert_apply_ln(ctx, x, L->ln_final, HUBERT_LAYER_EPS);
    return x;
}

// HuBERT positional convolutional embedding.
//
// HubertPositionalConvEmbedding wraps a single grouped Conv1d with weight_norm,
// followed by a SamePadLayer that removes the trailing sample when the kernel
// is even, then a GELU. Hyperparameters from config :
//   in = out = hidden_size = 768
//   kernel = num_conv_pos_embeddings = 128 (even)
//   padding = kernel // 2 = 64
//   groups = num_conv_pos_embedding_groups = 16  -> ic_per_group = oc_per_group = 48
//
// weight_norm reparametrizes the conv weight as (g, v) at training time, but
// at export the .bin reflects the materialized weight (g * v / ||v||) baked
// in. So we treat it as a plain grouped conv1d at load.
//
// ggml_conv_1d has no native groups support. We decompose into 16 sub-conv1d
// calls, each operating on its 48 ic / 48 oc slice. The weight tensor laid out
// as (k=128, ic_per_group=48, out_total=768) in GGML is sliced on ne[2] in
// chunks of 48 output channels, matching the source-side grouping convention where
// group g uses input channels [g*48..(g+1)*48] and produces output channels
// [g*48..(g+1)*48].

struct HubertPosConv {
    struct ggml_tensor * w;  // bf16 (k=128, ic_per_group=48, out_total=768)
    struct ggml_tensor * b;  // f32  (768)

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

static bool hubert_pos_conv_load(HubertPosConv * h, const GGUFModel & gf, ggml_backend_t backend) {
    // Phase 1 : describe
    const int               n_tensors_max = 4;
    size_t                  ctx_size      = (size_t) n_tensors_max * ggml_tensor_overhead() + 1024;
    struct ggml_init_params gp            = { ctx_size, NULL, true };
    h->weight_ctx                         = ggml_init(gp);
    struct ggml_context * ctx             = h->weight_ctx;

    h->w = ggml_new_tensor_3d(ctx, gf_get_type(gf, "semantic_model.encoder.pos_conv_embed.conv.weight"), HUBERT_POS_K,
                              HUBERT_POS_IC_PG, HUBERT_HIDDEN);
    h->b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);

    // Phase 2 : alloc backend
    h->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!h->weight_buf) {
        fprintf(stderr, "[HuBERT-PosConv] FATAL: failed to allocate weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(h->weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Phase 3 : fill (BF16 passthrough on the conv kernel, F32 cast on bias).
    dac_load_passthrough(h->w, gf, "semantic_model.encoder.pos_conv_embed.conv.weight");
    dac_load_bias_f32(h->b, gf, "semantic_model.encoder.pos_conv_embed.conv.bias");

    return true;
}

static void hubert_pos_conv_free(HubertPosConv * h) {
    if (h->weight_buf) {
        ggml_backend_buffer_free(h->weight_buf);
        h->weight_buf = NULL;
    }
    if (h->weight_ctx) {
        ggml_free(h->weight_ctx);
        h->weight_ctx = NULL;
    }
}

// Build pos_conv_embed graph. Input/output [768, T] f32 C-first.
// Steps mirror the reference HubertPositionalConvEmbedding.forward :
//   transpose(C, T) -> (T, C)
//   16 grouped conv1d slices  -> concat -> (T+1, 768)   (k even, padding=64)
//   SamePad : drop trailing sample -> (T, 768)
//   GELU exact
//   transpose back -> (768, T)
static struct ggml_tensor * hubert_pos_conv_build_graph(struct ggml_context * ctx,
                                                        const HubertPosConv * h,
                                                        struct ggml_tensor *  x  // [768, T]
) {
    const int T = (int) x->ne[1];

    // Switch to T-first layout to match dac_conv1d expectations (T, IC).
    x = ggml_cont(ctx, ggml_transpose(ctx, x));  // (T, 768)

    // Group input channels : (T, 768) -> (T, 48, 16) so each group of 48
    // input channels lives on ne[2]. The view stride on ne[1] is the
    // contiguous channel stride from the source layout.
    struct ggml_tensor * x3 = ggml_reshape_3d(ctx, x, T, HUBERT_POS_IC_PG, HUBERT_POS_GROUPS);

    // Per-group sub-conv1d. Each group is a plain 48 -> 48 conv with k=128, pad=64.
    struct ggml_tensor * outs[HUBERT_POS_GROUPS];
    for (int g = 0; g < HUBERT_POS_GROUPS; g++) {
        // Slice (T, 48) of the input at group offset g on ne[2].
        size_t               off_x = (size_t) g * x3->nb[2];
        struct ggml_tensor * xg    = ggml_view_2d(ctx, x3, T, HUBERT_POS_IC_PG, x3->nb[1], off_x);

        // Slice (k=128, ic=48, oc=48) of the weight. Weight ne=(128, 48, 768),
        // step on ne[2] is OC_PG output channels.
        size_t               off_w = (size_t) g * HUBERT_POS_OC_PG * h->w->nb[2];
        struct ggml_tensor * wg =
            ggml_view_3d(ctx, h->w, HUBERT_POS_K, HUBERT_POS_IC_PG, HUBERT_POS_OC_PG, h->w->nb[1], h->w->nb[2], off_w);

        // Slice (48) of the bias starting at g * 48.
        size_t               off_b = (size_t) g * HUBERT_POS_OC_PG * h->b->nb[0];
        struct ggml_tensor * bg    = ggml_view_1d(ctx, h->b, HUBERT_POS_OC_PG, off_b);

        outs[g] = dac_conv1d(ctx, wg, bg, xg, 1, HUBERT_POS_PAD, 1);  // (T_out, 48)
    }

    // Concat 16 (T_out, 48) tensors on the channel axis ne[1] -> (T_out, 768).
    struct ggml_tensor * y = outs[0];
    for (int g = 1; g < HUBERT_POS_GROUPS; g++) {
        y = ggml_concat(ctx, y, outs[g], 1);
    }
    // y : (T+1, 768). T_out = T - K + 2*PAD + 1 = T + 1 (k even).

    // SamePad : drop the trailing sample on the time axis ne[0]. We need a
    // contiguous tensor for ggml_gelu_erf and the next transpose, so cont.
    y = ggml_cont(ctx, ggml_view_2d(ctx, y, T, HUBERT_HIDDEN, y->nb[1], 0));

    y = ggml_gelu_erf(ctx, y);

    // Transpose back to C-first (768, T) for residual add and downstream blocks.
    return ggml_cont(ctx, ggml_transpose(ctx, y));
}

// Initial encoder block : pos_conv_embed (residual add) + first LayerNorm.
//   x = x + pos_conv_embed(x)
//   x = layer_norm(x)
// Holds the pos_conv weights and the LN params side by side. The 12 transformer
// layers come after this block.
struct HubertEncInit {
    HubertPosConv       pos;
    HubertLayerNormPair ln;

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

static bool hubert_enc_init_load(HubertEncInit * e, const GGUFModel & gf, ggml_backend_t backend) {
    if (!hubert_pos_conv_load(&e->pos, gf, backend)) {
        return false;
    }

    // LN params live in their own tiny buffer for clean lifecycle.
    const int               n_tensors_max = 4;
    size_t                  ctx_size      = (size_t) n_tensors_max * ggml_tensor_overhead() + 1024;
    struct ggml_init_params gp            = { ctx_size, NULL, true };
    e->weight_ctx                         = ggml_init(gp);
    struct ggml_context * ctx             = e->weight_ctx;

    e->ln.w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);
    e->ln.b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HUBERT_HIDDEN);

    e->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!e->weight_buf) {
        fprintf(stderr, "[HuBERT-EncInit] FATAL: failed to allocate LN buffer\n");
        hubert_pos_conv_free(&e->pos);
        return false;
    }
    ggml_backend_buffer_set_usage(e->weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    dac_load_bias_f32(e->ln.w, gf, "semantic_model.encoder.layer_norm.weight");
    dac_load_bias_f32(e->ln.b, gf, "semantic_model.encoder.layer_norm.bias");

    const float total_mb =
        (float) (ggml_backend_buffer_get_size(e->pos.weight_buf) + ggml_backend_buffer_get_size(e->weight_buf)) /
        (1024 * 1024);
    fprintf(stderr, "[HuBERT-EncInit] Loaded: pos_conv k=%d g=%d + LN(%d), weights %.1f MB\n", HUBERT_POS_K,
            HUBERT_POS_GROUPS, HUBERT_HIDDEN, total_mb);
    return true;
}

static void hubert_enc_init_free(HubertEncInit * e) {
    if (e->weight_buf) {
        ggml_backend_buffer_free(e->weight_buf);
        e->weight_buf = NULL;
    }
    if (e->weight_ctx) {
        ggml_free(e->weight_ctx);
        e->weight_ctx = NULL;
    }
    hubert_pos_conv_free(&e->pos);
}

// Forward of the initial encoder block. Input/output [768, T] f32 C-first.
static struct ggml_tensor * hubert_enc_init_build_graph(struct ggml_context * ctx,
                                                        const HubertEncInit * e,
                                                        struct ggml_tensor *  x  // [768, T]
) {
    struct ggml_tensor * pos = hubert_pos_conv_build_graph(ctx, &e->pos, x);
    x                        = ggml_add(ctx, x, pos);
    x                        = hubert_apply_ln(ctx, x, e->ln, HUBERT_POS_LN_EPS);
    return x;
}
