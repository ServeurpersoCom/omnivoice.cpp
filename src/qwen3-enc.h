#pragma once
// qwen3-enc.h: Qwen3 transformer building blocks via ggml
//
// Generic Qwen3 backbone, configurable for any layer count, hidden size,
// causal vs bidirectional attention. Provides config, per-layer weights,
// graph builders, and a layer loader. Higher-level modules own the backend,
// the scheduler, the embeddings, and the output head.
//
// Architecture per layer:
//   RMSNorm -> Q/K/V proj -> QK-Norm -> RoPE -> GQA -> O proj -> +residual
//   RMSNorm -> gate/up proj -> SwiGLU -> down proj -> +residual
// Stack final: RMSNorm

#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define QWEN3_MAX_LAYERS 32

// Config
struct Qwen3Config {
    int   hidden_size;        // H
    int   intermediate_size;  // FFN inner dim
    int   n_heads;            // Nh (query heads)
    int   n_kv_heads;         // Nkv (key/value heads, for GQA)
    int   head_dim;           // D = H / Nh
    int   n_layers;
    float rope_theta;
    float rms_norm_eps;
    bool  is_causal;  // true for causal LM stacks, false for bidirectional encoders
};

// Per-layer weights
struct Qwen3Layer {
    struct ggml_tensor * input_layernorm;      // [H]
    struct ggml_tensor * post_attn_layernorm;  // [H]

    // Attention (fused or separate, same pattern as DiT)
    struct ggml_tensor * qkv;     // [H, (Nh+2*Nkv)*D] full fused (or NULL)
    struct ggml_tensor * qk;      // [H, (Nh+Nkv)*D] Q+K fused (or NULL)
    struct ggml_tensor * q_proj;  // [H, Nh*D]  (NULL when fused)
    struct ggml_tensor * k_proj;  // [H, Nkv*D] (NULL when fused)
    struct ggml_tensor * v_proj;  // [H, Nkv*D] (NULL when QKV fused)
    struct ggml_tensor * o_proj;  // [Nh*D, H]
    struct ggml_tensor * q_norm;  // [D]
    struct ggml_tensor * k_norm;  // [D]

    // MLP (fused or separate)
    struct ggml_tensor * gate_up;    // [H, 2*FFN] fused (or NULL)
    struct ggml_tensor * gate_proj;  // [H, FFN] (NULL when fused)
    struct ggml_tensor * up_proj;    // [H, FFN] (NULL when fused)
    struct ggml_tensor * down_proj;  // [FFN, H]
};

// Helpers (pure graph ops, no side effects)
static struct ggml_tensor * qwen3_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        return t;
    }
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

static struct ggml_tensor * qwen3_linear(struct ggml_context * ctx, struct ggml_tensor * w, struct ggml_tensor * x) {
    return ggml_mul_mat(ctx, w, x);
}

static struct ggml_tensor * qwen3_linear_bias(struct ggml_context * ctx,
                                              struct ggml_tensor *  w,
                                              struct ggml_tensor *  b,
                                              struct ggml_tensor *  x) {
    struct ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    return ggml_add(ctx, out, qwen3_f32(ctx, b));
}

// F32 manual attention (fallback when flash_attn_ext is disabled).
// Works for 3D [D, S, X] and 4D [D, S, X, N] inputs.
// Returns same layout as flash_attn_ext: dims 1 and 2 swapped vs input.
static struct ggml_tensor * qwen3_attn_f32(struct ggml_context * ctx,
                                           struct ggml_tensor *  q,
                                           struct ggml_tensor *  k,
                                           struct ggml_tensor *  v,
                                           struct ggml_tensor *  mask,
                                           float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static struct ggml_tensor * qwen3_rms_norm(struct ggml_context * ctx,
                                           struct ggml_tensor *  x,
                                           struct ggml_tensor *  w,
                                           float                 eps) {
    struct ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, qwen3_f32(ctx, w));
}

// Graph builders
// These build sub-graphs and return output tensors.
// They operate on ggml layout: [H, S] for hidden states.

// Self-attention: norm_in [H, S, B] -> attn_out [H, S, B]
//
// B is the batch size on the trailing dim. B=1 keeps the historical 3D path
// pixel-identical: the reshape and view ops collapse the trailing dim and
// every kernel runs the exact same shapes as before. B>1 activates a parallel
// 4D path that broadcasts the same projections, RoPE, and attention over the
// extra batch dim, used by pipeline_tts_llm_forward_batched to fuse the
// cond + uncond CFG rows in a single graph.
static struct ggml_tensor * qwen3_build_self_attn(struct ggml_context * ctx,
                                                  const Qwen3Config &   c,
                                                  Qwen3Layer *          ly,
                                                  struct ggml_tensor *  x,          // [H, S] or [H, S, B]
                                                  struct ggml_tensor *  positions,  // [S] int32
                                                  struct ggml_tensor *  mask,
                                                  int                   S,
                                                  bool                  use_flash_attn = true,
                                                  bool                  clamp_fp16     = false,
                                                  int                   B              = 1) {
    int D   = c.head_dim;
    int Nh  = c.n_heads;
    int Nkv = c.n_kv_heads;

    // 1) Q/K/V projections (fused, partial, or separate). Output keeps the
    // batch dim trailing: [(q|kv)_dim, S] in 3D mode, [(q|kv)_dim, S, B] in 4D.
    struct ggml_tensor *q, *k, *v;
    int                 q_dim  = Nh * D;
    int                 kv_dim = Nkv * D;
    if (ly->qkv) {
        struct ggml_tensor * qkv = qwen3_linear(ctx, ly->qkv, x);
        if (B == 1) {
            q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, q_dim, S, qkv->nb[1], 0));
            k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, S, qkv->nb[1], (size_t) q_dim * qkv->nb[0]));
            v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, S, qkv->nb[1], (size_t) (q_dim + kv_dim) * qkv->nb[0]));
        } else {
            q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, q_dim, S, B, qkv->nb[1], qkv->nb[2], 0));
            k = ggml_cont(ctx,
                          ggml_view_3d(ctx, qkv, kv_dim, S, B, qkv->nb[1], qkv->nb[2], (size_t) q_dim * qkv->nb[0]));
            v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, kv_dim, S, B, qkv->nb[1], qkv->nb[2],
                                            (size_t) (q_dim + kv_dim) * qkv->nb[0]));
        }
    } else if (ly->qk) {
        struct ggml_tensor * qk = qwen3_linear(ctx, ly->qk, x);
        if (B == 1) {
            q = ggml_cont(ctx, ggml_view_2d(ctx, qk, q_dim, S, qk->nb[1], 0));
            k = ggml_cont(ctx, ggml_view_2d(ctx, qk, kv_dim, S, qk->nb[1], (size_t) q_dim * qk->nb[0]));
        } else {
            q = ggml_cont(ctx, ggml_view_3d(ctx, qk, q_dim, S, B, qk->nb[1], qk->nb[2], 0));
            k = ggml_cont(ctx, ggml_view_3d(ctx, qk, kv_dim, S, B, qk->nb[1], qk->nb[2], (size_t) q_dim * qk->nb[0]));
        }
        v = qwen3_linear(ctx, ly->v_proj, x);
    } else {
        q = qwen3_linear(ctx, ly->q_proj, x);
        k = qwen3_linear(ctx, ly->k_proj, x);
        v = qwen3_linear(ctx, ly->v_proj, x);
    }

    // 2) Reshape to heads. In 3D mode keep the existing [D, X, S] layout. In
    // 4D mode promote to [D, X, S, B] so the trailing batch dim flows through
    // every downstream op.
    if (B == 1) {
        q = ggml_reshape_3d(ctx, q, D, Nh, S);
        k = ggml_reshape_3d(ctx, k, D, Nkv, S);
        v = ggml_reshape_3d(ctx, v, D, Nkv, S);
    } else {
        q = ggml_reshape_4d(ctx, q, D, Nh, S, B);
        k = ggml_reshape_4d(ctx, k, D, Nkv, S, B);
        v = ggml_reshape_4d(ctx, v, D, Nkv, S, B);
    }

    // 3) QK-Norm: per-head RMSNorm on D, identical in 3D and 4D.
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, qwen3_f32(ctx, ly->q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, qwen3_f32(ctx, ly->k_norm));

    // 4) RoPE
    // ggml pitfall: mode=2 (NEOX half-split [i, i+D/2]), NOT mode=0 (consecutive [2i, 2i+1])
    // Python ref: rope_batch_kernel pairs ptr[d] with ptr[d+half] = NEOX
    q = ggml_rope_ext(ctx, q, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // 5) Permute for flash_attn_ext: [D, X, S(, B)] -> [D, S, X(, B)]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // Clamp V before attention: sub-Ampere tensor cores accumulate in FP16,
    // V projection can overflow to inf which corrupts all subsequent attention.
    if (clamp_fp16) {
        v = ggml_clamp(ctx, v, -65504.0f, 65504.0f);
    }

    // 6) Attention (flash or F32 manual fallback)
    float                scale = 1.0f / sqrtf((float) D);
    struct ggml_tensor * attn  = use_flash_attn ? ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f) :
                                                  qwen3_attn_f32(ctx, q, k, v, mask, scale);
    if (use_flash_attn) {
        ggml_prec_set_acc(attn, GGML_PREC_F32);
    }

    // 7) Reshape back: [D, Nh, S(, B)] -> [Nh*D, S(, B)]
    if (B == 1) {
        attn = ggml_reshape_2d(ctx, attn, Nh * D, S);
    } else {
        attn = ggml_reshape_3d(ctx, attn, Nh * D, S, B);
    }

    // 8) O projection
    return qwen3_linear(ctx, ly->o_proj, attn);
}

// MLP: SwiGLU (fused gate+up or separate)
static struct ggml_tensor * qwen3_build_mlp(struct ggml_context * ctx,
                                            Qwen3Layer *          ly,
                                            struct ggml_tensor *  x,  // [H, S]
                                            int                   S) {
    (void) S;
    struct ggml_tensor * ff;
    if (ly->gate_up) {
        struct ggml_tensor * gu = qwen3_linear(ctx, ly->gate_up, x);
        ff                      = ggml_swiglu(ctx, gu);
    } else {
        struct ggml_tensor * gate = qwen3_linear(ctx, ly->gate_proj, x);
        struct ggml_tensor * up   = qwen3_linear(ctx, ly->up_proj, x);
        ff                        = ggml_swiglu_split(ctx, gate, up);
    }
    return qwen3_linear(ctx, ly->down_proj, ff);
}

// Single layer: input [H, S(, B)] -> output [H, S(, B)]
// When sub_outs is non null, the layer pushes (norm1, attn, norm2, mlp) for
// post inspection. Sub tensors are pre residual outputs of each sub-module.
// B is forwarded to qwen3_build_self_attn for batched MaskGIT decoding.
static struct ggml_tensor * qwen3_build_layer(struct ggml_context *               ctx,
                                              const Qwen3Config &                 c,
                                              Qwen3Layer *                        ly,
                                              struct ggml_tensor *                hidden,
                                              struct ggml_tensor *                positions,
                                              struct ggml_tensor *                mask,
                                              int                                 S,
                                              bool                                use_flash_attn = true,
                                              bool                                clamp_fp16     = false,
                                              std::vector<struct ggml_tensor *> * sub_outs       = nullptr,
                                              int                                 B              = 1) {
    // Self-attention block
    struct ggml_tensor * norm = qwen3_rms_norm(ctx, hidden, ly->input_layernorm, c.rms_norm_eps);
    if (sub_outs) {
        sub_outs->push_back(norm);
    }
    struct ggml_tensor * attn =
        qwen3_build_self_attn(ctx, c, ly, norm, positions, mask, S, use_flash_attn, clamp_fp16, B);
    if (sub_outs) {
        sub_outs->push_back(attn);
    }
    hidden = ggml_add(ctx, hidden, attn);
    if (clamp_fp16) {
        hidden = ggml_clamp(ctx, hidden, -65504.0f, 65504.0f);
    }

    // MLP block
    norm = qwen3_rms_norm(ctx, hidden, ly->post_attn_layernorm, c.rms_norm_eps);
    if (sub_outs) {
        sub_outs->push_back(norm);
    }
    struct ggml_tensor * mlp = qwen3_build_mlp(ctx, ly, norm, S);
    if (sub_outs) {
        sub_outs->push_back(mlp);
    }
    hidden = ggml_add(ctx, hidden, mlp);
    if (clamp_fp16) {
        hidden = ggml_clamp(ctx, hidden, -65504.0f, 65504.0f);
    }

    return hidden;
}

// Full N-layer stack: input [H, S(, B)] -> output [H, S(, B)] (post final-norm)
// intermediates: optional out-param. When non-null, the hidden tensor right
// after each layer index listed in intermediate_indices is appended (in the
// same order as those indices). Caller must ggml_set_output on them before
// the graph build.
// B is the optional trailing batch dim, see qwen3_build_self_attn.
static struct ggml_tensor * qwen3_build_layers(struct ggml_context *               ctx,
                                               const Qwen3Config &                 c,
                                               Qwen3Layer *                        layers,
                                               struct ggml_tensor *                final_norm_w,
                                               struct ggml_tensor *                hidden,
                                               struct ggml_tensor *                positions,
                                               struct ggml_tensor *                mask,
                                               int                                 S,
                                               bool                                use_flash_attn       = true,
                                               bool                                clamp_fp16           = false,
                                               const std::vector<int> *            intermediate_indices = nullptr,
                                               std::vector<struct ggml_tensor *> * intermediates        = nullptr,
                                               int                                 dump_sub_layer       = -1,
                                               std::vector<struct ggml_tensor *> * sub_outs             = nullptr,
                                               int                                 B                    = 1) {
    for (int i = 0; i < c.n_layers; i++) {
        std::vector<struct ggml_tensor *> * subs_for_this = (i == dump_sub_layer) ? sub_outs : nullptr;
        hidden = qwen3_build_layer(ctx, c, &layers[i], hidden, positions, mask, S, use_flash_attn, clamp_fp16,
                                   subs_for_this, B);
        if (intermediate_indices && intermediates) {
            for (int idx : *intermediate_indices) {
                if (idx == i) {
                    intermediates->push_back(hidden);
                    break;
                }
            }
        }
    }
    return qwen3_rms_norm(ctx, hidden, final_norm_w, c.rms_norm_eps);
}

// Loading
static void qwen3_load_layer(WeightCtx *         wctx,
                             const GGUFModel &   gf,
                             Qwen3Layer *        ly,
                             const std::string & prefix,
                             int                 layer_idx = -1) {
    ly->input_layernorm     = gf_load_tensor_f32(wctx, gf, prefix + ".input_layernorm.weight");
    ly->post_attn_layernorm = gf_load_tensor_f32(wctx, gf, prefix + ".post_attention_layernorm.weight");

    // Attention: try Q+K+V fused, then Q+K partial, then separate
    ly->qkv = gf_load_qkv_fused(wctx, gf, prefix + ".self_attn.q_proj.weight", prefix + ".self_attn.k_proj.weight",
                                prefix + ".self_attn.v_proj.weight");
    if (!ly->qkv) {
        ly->qk = gf_load_pair_fused(wctx, gf, prefix + ".self_attn.q_proj.weight", prefix + ".self_attn.k_proj.weight");
        if (ly->qk) {
            ly->v_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.v_proj.weight");
            if (layer_idx == 0) {
                fprintf(stderr, "[Qwen3] Attn: Q+K fused, V separate\n");
            }
        } else {
            ly->q_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.q_proj.weight");
            ly->k_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.k_proj.weight");
            ly->v_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.v_proj.weight");
            if (layer_idx == 0) {
                fprintf(stderr, "[Qwen3] Attn: all separate\n");
            }
        }
    } else {
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] Attn: Q+K+V fused\n");
        }
    }

    ly->o_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.o_proj.weight");
    ly->q_norm = gf_load_tensor_f32(wctx, gf, prefix + ".self_attn.q_norm.weight");
    ly->k_norm = gf_load_tensor_f32(wctx, gf, prefix + ".self_attn.k_norm.weight");

    // MLP: try gate+up fused, then separate
    ly->gate_up = gf_load_pair_fused(wctx, gf, prefix + ".mlp.gate_proj.weight", prefix + ".mlp.up_proj.weight");
    if (ly->gate_up) {
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] MLP: gate+up fused\n");
        }
    } else {
        ly->gate_proj = gf_load_tensor(wctx, gf, prefix + ".mlp.gate_proj.weight");
        ly->up_proj   = gf_load_tensor(wctx, gf, prefix + ".mlp.up_proj.weight");
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] MLP: gate+up separate\n");
        }
    }
    ly->down_proj = gf_load_tensor(wctx, gf, prefix + ".mlp.down_proj.weight");
}
