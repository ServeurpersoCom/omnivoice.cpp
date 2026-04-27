#pragma once
// semantic-enc.h : SemanticEncoder for OmniVoice (GGML)
//
// Refines the per-frame HuBERT semantic features [768, T] into the same
// shape with a chain of residual ELU + Conv1d blocks. The output is
// concatenated channel-wise with the DAC acoustic encoder's output to feed
// the joint fc Linear before RVQ quantization.
//
// Architecture (config.strides=[1,1], config.channel_ratios=[1,1] in OmniVoice):
//   conv (768 -> 768, k=3, p=1, no bias)
//   2x SemanticBlock :
//       2x SemanticResUnit :
//           ELU -> conv1(768 -> 768, k=3, dil=1, no bias)
//           ELU -> conv2(768 -> 768, k=1, no bias)
//           skip add
//       conv(768 -> 768, k=3, stride=1, p=1, with bias)
//
// All channel counts stay at 768. Stride 1 means there is no temporal
// downsample : T_out == T_in == T_sem. The SemanticDecoder mirror is not
// needed at inference time and is excluded from the GGUF entirely.
//
// Input / output : [T, 768] f32 in GGML layout (T fast, 768 slow), matching
// what dac_conv1d expects so the DAC helpers can be reused as-is.

#include "dac-decoder.h"  // for dac_conv1d, dac_load_passthrough, dac_load_bias_f32
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define SEM_HIDDEN          768
#define SEM_NUM_BLOCKS      2
#define SEM_RES_UNITS       2
#define SEM_BLOCK_DILATIONS { 1, 1 }

// Two convs per residual unit, no bias on either (config.bias=False upstream).
struct SemanticResUnit {
    struct ggml_tensor * c1w;  // [3, 768, 768] bf16, dilated k=3
    struct ggml_tensor * c2w;  // [1, 768, 768] bf16, k=1
    int                  dilation;
};

// One block : SEM_RES_UNITS residual units then a final stride-1 conv with bias.
struct SemanticEncBlock {
    SemanticResUnit      ru[SEM_RES_UNITS];
    struct ggml_tensor * cw;  // [3, 768, 768] bf16
    struct ggml_tensor * cb;  // [768] f32, cast at load
};

struct SemanticEncoder {
    // initial 768 -> 768 conv, k=3, p=1, no bias
    struct ggml_tensor * c1w;

    SemanticEncBlock blk[SEM_NUM_BLOCKS];

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

// Full load : describe -> alloc -> fill, mirror of dac_load with no per-tensor
// transform other than the F32 cast on the per-block conv biases (so the
// CUDA add op accepts them as src1).
static bool sem_enc_load(SemanticEncoder * d, const GGUFModel & gf, ggml_backend_t backend) {
    static const int dilations[] = SEM_BLOCK_DILATIONS;

    // Phase 1 : describe
    const int               n_tensors_max = 64;
    size_t                  ctx_size      = (size_t) n_tensors_max * ggml_tensor_overhead() + 1024;
    struct ggml_init_params p             = { ctx_size, NULL, true };
    d->weight_ctx                         = ggml_init(p);
    struct ggml_context * ctx             = d->weight_ctx;

    // Initial 768 -> 768 conv, dtype mirrors the GGUF source (F32 or BF16)
    d->c1w = ggml_new_tensor_3d(ctx, gf_get_type(gf, "encoder_semantic.conv.weight"), 3, SEM_HIDDEN, SEM_HIDDEN);

    for (int i = 0; i < SEM_NUM_BLOCKS; i++) {
        SemanticEncBlock & b   = d->blk[i];
        std::string        pfx = "encoder_semantic.conv_blocks." + std::to_string(i);
        for (int r = 0; r < SEM_RES_UNITS; r++) {
            SemanticResUnit & ru = b.ru[r];
            ru.dilation          = dilations[r];
            std::string rp       = pfx + ".res_units." + std::to_string(r);
            ru.c1w = ggml_new_tensor_3d(ctx, gf_get_type(gf, rp + ".conv1.weight"), 3, SEM_HIDDEN, SEM_HIDDEN);
            ru.c2w = ggml_new_tensor_3d(ctx, gf_get_type(gf, rp + ".conv2.weight"), 1, SEM_HIDDEN, SEM_HIDDEN);
        }
        b.cw = ggml_new_tensor_3d(ctx, gf_get_type(gf, pfx + ".conv.weight"), 3, SEM_HIDDEN, SEM_HIDDEN);
        b.cb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, SEM_HIDDEN);
    }

    // Phase 2 : alloc backend
    d->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!d->weight_buf) {
        fprintf(stderr, "[Sem-Enc] FATAL: failed to allocate weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(d->weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Phase 3 : fill (passthrough bf16 + F32 cast for biases)
    dac_load_passthrough(d->c1w, gf, "encoder_semantic.conv.weight");

    for (int i = 0; i < SEM_NUM_BLOCKS; i++) {
        SemanticEncBlock & b   = d->blk[i];
        std::string        pfx = "encoder_semantic.conv_blocks." + std::to_string(i);
        for (int r = 0; r < SEM_RES_UNITS; r++) {
            SemanticResUnit & ru = b.ru[r];
            std::string       rp = pfx + ".res_units." + std::to_string(r);
            dac_load_passthrough(ru.c1w, gf, rp + ".conv1.weight");
            dac_load_passthrough(ru.c2w, gf, rp + ".conv2.weight");
        }
        dac_load_passthrough(b.cw, gf, pfx + ".conv.weight");
        dac_load_bias_f32(b.cb, gf, pfx + ".conv.bias");
    }

    fprintf(stderr, "[Sem-Enc] Loaded: %d blocks, hidden=%d, no temporal downsample, weights %.1f MB\n", SEM_NUM_BLOCKS,
            SEM_HIDDEN, (float) ggml_backend_buffer_get_size(d->weight_buf) / (1024 * 1024));
    return true;
}

static void sem_enc_free(SemanticEncoder * d) {
    if (d->weight_buf) {
        ggml_backend_buffer_free(d->weight_buf);
        d->weight_buf = NULL;
    }
    if (d->weight_ctx) {
        ggml_free(d->weight_ctx);
        d->weight_ctx = NULL;
    }
}

// One residual unit : ELU -> conv1 -> ELU -> conv2 + skip
// Same-padding for both convs (pad = (k-1)*dil/2). conv1 k=3 dil=1 -> pad=1 ;
// conv2 k=1 -> pad=0. No bias on either conv (passed as NULL to dac_conv1d).
static struct ggml_tensor * sem_res_unit(struct ggml_context *   ctx,
                                         const SemanticResUnit * ru,
                                         struct ggml_tensor *    x) {
    struct ggml_tensor * skip = x;
    x                         = ggml_elu(ctx, x);
    int pad1                  = ru->dilation;  // (k-1)*dil/2 with k=3 -> dil
    x                         = dac_conv1d(ctx, ru->c1w, NULL, x, 1, pad1, ru->dilation);
    x                         = ggml_elu(ctx, x);
    x                         = dac_conv1d(ctx, ru->c2w, NULL, x, 1, 0, 1);
    return ggml_add(ctx, skip, x);
}

// Build the SemanticEncoder graph. Input / output are both [T, 768] f32 in
// GGML layout. T_in == T_out (no stride other than 1, same padding).
static struct ggml_tensor * sem_enc_build_graph(struct ggml_context *   ctx,
                                                const SemanticEncoder * d,
                                                struct ggml_tensor *    x  // [T, 768] f32
) {
    // initial conv : 768 -> 768, k=3, p=1, no bias
    x = dac_conv1d(ctx, d->c1w, NULL, x, 1, 1, 1);

    for (int i = 0; i < SEM_NUM_BLOCKS; i++) {
        const SemanticEncBlock & b = d->blk[i];
        for (int r = 0; r < SEM_RES_UNITS; r++) {
            x = sem_res_unit(ctx, &b.ru[r], x);
        }
        // post-block conv : same channels, stride 1, k=3, p=1, with bias
        x = dac_conv1d(ctx, b.cw, b.cb, x, 1, 1, 1);
    }
    return x;  // [T, 768]
}
