#pragma once
// rvq-codec.h : Residual Vector Quantization codec (encode + decode, GGML)
// Reads 8 codebooks of 1024 entries (codebook_dim=64, hidden=1024).
//
// Decode side : codes [B, 8, T] i32 -> latent [B, 1024, T] f32 by summing
// project_out(embed[k][idx_k]) across all 8 codebooks.
// Encode side : embeddings [B, 1024, T] f32 -> codes [B, 8, T] i32 via the
// residual loop: at each step project_in to 64-d, find the nearest codebook
// entry by Euclidean distance, subtract the reconstructed quantized vector,
// continue with the residual.

#include "ggml.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cstdio>
#include <cstdlib>

#define RVQ_NUM_CODEBOOKS 8

// Per-codebook weights for RVQ. project_in is encode-only, project_out
// decode-only, embed is shared (used for both lookup at decode and nearest
// neighbor search at encode). embed_sq is a precomputed CPU side quantity
// that holds ||embed[:, j]||^2 for j in [0..codebook_size), reused at every
// encode step.
struct RVQCodebook {
    struct ggml_tensor * embed;          // [codebook_dim, codebook_size] bf16  (64, 1024)
    struct ggml_tensor * embed_sq;       // [codebook_size]               f32   (1024) precomputed
    struct ggml_tensor * project_in_w;   // [hidden, codebook_dim]        bf16  (1024, 64)
    struct ggml_tensor * project_in_b;   // [codebook_dim]                f32   (64) cast at load
    struct ggml_tensor * project_out_w;  // [codebook_dim, hidden]        bf16  (64, 1024)
    struct ggml_tensor * project_out_b;  // [hidden]                      f32   (1024) cast at load
};

struct RVQCodec {
    int         num_codebooks;  // 8
    int         codebook_size;  // 1024
    int         codebook_dim;   // 64
    int         hidden;         // 1024 (matches DAC decoder input via fc2)
    RVQCodebook cb[RVQ_NUM_CODEBOOKS];
};

// Load all 8 codebooks from a GGUF model. `gf` is the audio-tokenizer GGUF.
// Tensor names follow the upstream HiggsAudioV2Tokenizer convention :
//     quantizer.quantizers.{k}.codebook.embed         (1024, 64)
//     quantizer.quantizers.{k}.project_in.weight      (64, 1024)
//     quantizer.quantizers.{k}.project_in.bias        (64)
//     quantizer.quantizers.{k}.project_out.weight     (1024, 64)
//     quantizer.quantizers.{k}.project_out.bias       (1024)
static bool rvq_load(RVQCodec * rvq, WeightCtx * wctx, const GGUFModel & gf) {
    rvq->num_codebooks = RVQ_NUM_CODEBOOKS;
    rvq->codebook_size = (int) gf_get_u32(gf, "omnivoice.codebook_size");
    rvq->codebook_dim  = (int) gf_get_u32(gf, "omnivoice.codebook_dim");
    if (rvq->codebook_size == 0 || rvq->codebook_dim == 0) {
        fprintf(stderr, "[RVQ] missing metadata: omnivoice.codebook_size or codebook_dim\n");
        return false;
    }
    // hidden dim is read from project_out shape after loading the first weight
    rvq->hidden = 0;

    char name[128];
    for (int k = 0; k < RVQ_NUM_CODEBOOKS; k++) {
        snprintf(name, sizeof(name), "quantizer.quantizers.%d.codebook.embed", k);
        rvq->cb[k].embed = gf_load_tensor(wctx, gf, name);
        if (!rvq->cb[k].embed) {
            fprintf(stderr, "[RVQ] missing tensor: %s\n", name);
            return false;
        }

        // Precompute ||embed[:, j]||^2 per codebook entry j on CPU side.
        // The codebook is stored either F32 (preserved native precision for
        // accurate residual chain) or BF16 (downcast). We branch on the source
        // dtype to decode each value to f32 once, then accumulate squared sums
        // per entry.
        // Memory layout : embed ne=(codebook_dim, codebook_size) row-major
        //   so flat[j * codebook_dim + i] is the i-th component of entry j.
        const enum ggml_type embed_type = rvq->cb[k].embed->type;
        const void *         raw_embed  = gf_get_data(gf, name);
        if (!raw_embed) {
            fprintf(stderr, "[RVQ] cannot read raw bytes of %s\n", name);
            return false;
        }
        auto e_sq_buf = std::make_unique<float[]>(rvq->codebook_size);
        for (int j = 0; j < rvq->codebook_size; j++) {
            float s = 0.0f;
            for (int i = 0; i < rvq->codebook_dim; i++) {
                float v;
                if (embed_type == GGML_TYPE_F32) {
                    v = ((const float *) raw_embed)[j * rvq->codebook_dim + i];
                } else {
                    v = ggml_bf16_to_fp32(((const ggml_bf16_t *) raw_embed)[j * rvq->codebook_dim + i]);
                }
                s += v * v;
            }
            e_sq_buf[j] = s;
        }
        struct ggml_tensor * e_sq = ggml_new_tensor_1d(wctx->ctx, GGML_TYPE_F32, rvq->codebook_size);
        char                 esq_name[160];
        snprintf(esq_name, sizeof(esq_name), "quantizer.quantizers.%d.codebook.embed_sq", k);
        ggml_set_name(e_sq, esq_name);
        wctx->pending.push_back({ e_sq, e_sq_buf.get(), (size_t) rvq->codebook_size * sizeof(float), 0 });
        wctx->staging.push_back(std::move(e_sq_buf));
        rvq->cb[k].embed_sq = e_sq;

        snprintf(name, sizeof(name), "quantizer.quantizers.%d.project_in.weight", k);
        rvq->cb[k].project_in_w = gf_load_tensor(wctx, gf, name);
        if (!rvq->cb[k].project_in_w) {
            fprintf(stderr, "[RVQ] missing tensor: %s\n", name);
            return false;
        }

        snprintf(name, sizeof(name), "quantizer.quantizers.%d.project_in.bias", k);
        rvq->cb[k].project_in_b = gf_load_tensor_f32(wctx, gf, name);
        if (!rvq->cb[k].project_in_b) {
            fprintf(stderr, "[RVQ] missing tensor: %s\n", name);
            return false;
        }

        snprintf(name, sizeof(name), "quantizer.quantizers.%d.project_out.weight", k);
        rvq->cb[k].project_out_w = gf_load_tensor(wctx, gf, name);
        if (!rvq->cb[k].project_out_w) {
            fprintf(stderr, "[RVQ] missing tensor: %s\n", name);
            return false;
        }

        snprintf(name, sizeof(name), "quantizer.quantizers.%d.project_out.bias", k);
        // F32 cast at load : ggml_add CUDA only accepts src1 in F32 or F16.
        // Cost is 2 KB extra per codebook (8 of them), negligible.
        rvq->cb[k].project_out_b = gf_load_tensor_f32(wctx, gf, name);
        if (!rvq->cb[k].project_out_b) {
            fprintf(stderr, "[RVQ] missing tensor: %s\n", name);
            return false;
        }

        if (k == 0) {
            // project_out.weight shape is (hidden, codebook_dim) in the source layout,
            // stored as (codebook_dim, hidden) in GGML row-major.
            // ne[0] = codebook_dim, ne[1] = hidden.
            rvq->hidden = (int) rvq->cb[0].project_out_w->ne[1];
        }
    }
    return true;
}

// Build the decode graph: codes [T, 8] i32 -> latent [hidden, T] f32.
// The output is in GGML layout, ready to feed fc2 + DAC decoder.
//
// Codes layout in GGML: ne[0]=T (fast axis), ne[1]=num_codebooks (slow).
// Source numpy array is row-major (num_codebooks, T) C-order, so T is the
// contiguous dimension and maps to ne[0] without any transpose.
//
// Math:
//     for k in 0..7:
//         e_k = embed[k][codes[k, :]]                  # (codebook_dim, T)
//         p_k = project_out_w[k] @ e_k + bias[k]       # (hidden, T)
//         out += p_k
static struct ggml_tensor * rvq_decode_graph(struct ggml_context * ctx,
                                             const RVQCodec *      rvq,
                                             struct ggml_tensor *  codes  // [T, 8] i32, ne[0]=T fast
) {
    const int            T         = (int) codes->ne[0];
    const size_t         stride_cb = codes->nb[1];  // bytes from one codebook row to the next
    struct ggml_tensor * acc       = NULL;

    for (int k = 0; k < rvq->num_codebooks; k++) {
        // codes_k = codes[k, :]  shape [T] i32, view at byte offset k*nb[1]
        struct ggml_tensor * codes_k = ggml_view_1d(ctx, codes, T, (size_t) k * stride_cb);

        // e_k = embed_lookup -> [codebook_dim, T]
        struct ggml_tensor * e_k = ggml_get_rows(ctx, rvq->cb[k].embed, codes_k);

        // p_k = project_out_w @ e_k -> [hidden, T]
        struct ggml_tensor * p_k = ggml_mul_mat(ctx, rvq->cb[k].project_out_w, e_k);

        // p_k += bias broadcast across T
        p_k = ggml_add(ctx, p_k, rvq->cb[k].project_out_b);

        acc = (k == 0) ? p_k : ggml_add(ctx, acc, p_k);
    }
    return acc;  // [hidden, T]
}

// Build the encode graph. embeddings [hidden, T] f32 is the post-fc input.
// Fills `out_idx_per_k[0..7]` with one i32 [T] tensor per codebook (the
// argmax indices). Returns the last index tensor as a convenient pin point
// for ggml_build_forward_expand ; the caller must expand each of the 8
// outputs and copy them back to host separately. We avoid packing them into
// a single (T, 8) tensor because ggml_concat is F32-only on CUDA, and any
// chain of ggml_cpy into a shared destination would not survive the gallocr
// without ugly per-step ggml_set_output juggling.
//
// Math (per upstream HiggsAudioV2TokenizerEuclideanCodebook.quantize) :
//   residual = embeddings
//   for k in 0..7 :
//       h64    = project_in_w[k] @ residual + bias[k]      # (codebook_dim, T)
//       e_sq   = ||embed[k][:, j]||^2  for each j          # precomputed at load
//       score  = 2 * embed[k] @ h64 - e_sq                 # (codebook_size, T)
//       idx_k  = argmax(score)                             # (T,) i32
//       e_k    = embed[k][:, idx_k]                        # (codebook_dim, T)
//       quant  = project_out_w[k] @ e_k + project_out_b[k] # (hidden, T)
//       residual = residual - quant
//
// Note : ||h||^2 is dropped from the score because it is constant across all
// codebook entries for a given frame and so does not affect the argmax.
// e_sq is precomputed on CPU at rvq_load time to avoid a bf16 sqr per
// inference.
static struct ggml_tensor * rvq_encode_graph(struct ggml_context * ctx,
                                             const RVQCodec *      rvq,
                                             struct ggml_tensor *  embeddings,    // [hidden, T] f32
                                             struct ggml_tensor ** out_idx_per_k  // [num_codebooks] -> [T] i32 each
) {
    struct ggml_tensor * residual = embeddings;

    for (int k = 0; k < rvq->num_codebooks; k++) {
        const RVQCodebook & cb = rvq->cb[k];

        // 1. project_in : (hidden, T) -> (codebook_dim, T)
        struct ggml_tensor * h64 = ggml_mul_mat(ctx, cb.project_in_w, residual);
        h64                      = ggml_add(ctx, h64, cb.project_in_b);

        // 2. embed_sq is precomputed at load (||embed[:, j]||^2 per entry j).
        //    Shape (codebook_size,), broadcasts over T on the next subtraction.
        struct ggml_tensor * e_sq = cb.embed_sq;

        // 3. score = 2 * embed @ h64 - e_sq
        //    mul_mat(embed[64,1024], h64[64,T]) gives ne=(M=embed.ne[1]=1024,
        //    N=h64.ne[1]=T), so ne[0]=1024 fast which is what argmax expects.
        struct ggml_tensor * dot   = ggml_mul_mat(ctx, cb.embed, h64);
        struct ggml_tensor * score = ggml_scale(ctx, dot, 2.0f);
        score                      = ggml_sub(ctx, score, e_sq);

        // 4. argmax along ne[0] (codebook_size) -> (T,) i32
        struct ggml_tensor * idx_k = ggml_argmax(ctx, score);
        char                 nm[32];
        snprintf(nm, sizeof(nm), "rvq_idx_%d", k);
        ggml_set_name(idx_k, nm);
        out_idx_per_k[k] = idx_k;

        // 5. Reconstruct quant_k = project_out(embed[idx_k])
        struct ggml_tensor * e_k     = ggml_get_rows(ctx, cb.embed, idx_k);
        struct ggml_tensor * quant_k = ggml_mul_mat(ctx, cb.project_out_w, e_k);
        quant_k                      = ggml_add(ctx, quant_k, cb.project_out_b);

        // 6. residual <- residual - quant_k. The last subtraction is wasted
        //    work (no further iteration consumes residual) but keeping the
        //    branch-free loop is clearer than a special case.
        residual = ggml_sub(ctx, residual, quant_k);
    }

    return out_idx_per_k[rvq->num_codebooks - 1];
}
