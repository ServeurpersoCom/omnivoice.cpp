#pragma once
// dac-decoder.h : DAC acoustic decoder for OmniVoice (GGML)
// Mirrors acestep VAE structure : Snake + ConvTranspose1d (mul_mat + col2im_1d)
// + 3 dilated residual units per block. 5 blocks upsample by 8 5 4 2 3 (= 960x).
// Input  : latent [T_in, 256] f32  (output of fc2 fed with RVQ-decoded vectors)
// Output : audio  [T_out, 1]  f32  with T_out = 960 * T_in

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define DAC_NUM_BLOCKS    5
#define DAC_RES_UNITS     3
#define DAC_RU_DILATIONS  { 1, 3, 9 }
#define DAC_BLOCK_STRIDES { 8, 5, 4, 2, 3 }
#define DAC_BLOCK_IN_CH   { 1024, 512, 256, 128, 64 }
#define DAC_BLOCK_OUT_CH  { 512, 256, 128, 64, 32 }
#define DAC_FINAL_CH      32

// Snake activation : y = x + (1 / (alpha + 1e-9)) * sin(alpha * x)^2
// Stored as (a = alpha, inv_b = 1 / (alpha + 1e-9)) so the runtime kernel
// stays a single fused op : ggml_snake(x, a, inv_b).
struct DACSnake {
    struct ggml_tensor * a;      // [1, C] f32, alpha direct
    struct ggml_tensor * inv_b;  // [1, C] f32, 1 / (alpha + 1e-9)
};

// One residual unit : snake1 -> conv1(k=7, dil=d) -> snake2 -> conv2(k=1) + skip
struct DACResUnit {
    DACSnake             s1, s2;
    struct ggml_tensor * c1w;  // [7, C, C] bf16, source layout (C, C, 7)
    struct ggml_tensor * c1b;  // [C] f32
    struct ggml_tensor * c2w;  // [1, C, C] bf16, source layout (C, C, 1)
    struct ggml_tensor * c2b;  // [C] f32
    int                  dilation;
};

// Decoder block : snake1 -> conv_t1 (upsample) -> 3 res_units
struct DACBlock {
    DACSnake             s1;
    // ConvTranspose1d weight pre-permuted to ggml [IC, K*OC] for mul_mat.
    // source layout (IC, OC, K) is repacked at load time so col2im_1d gets the
    // expected [K*OC, T_in] column matrix where k varies faster than oc.
    struct ggml_tensor * ctw;  // [IC, K*OC] bf16
    struct ggml_tensor * ctb;  // [OC] f32
    DACResUnit           ru[DAC_RES_UNITS];
    int                  in_ch;
    int                  out_ch;
    int                  stride;
    int                  kernel;      // 2 * stride
    int                  pad;         // ceil(stride / 2)
    int                  output_pad;  // stride % 2 (right-pad applied after col2im_1d)
};

struct DACDecoder {
    // initial 256 -> 1024 conv1, k=7, pad=3
    struct ggml_tensor * c1w;  // [7, 256, 1024] bf16
    struct ggml_tensor * c1b;  // [1024] f32

    DACBlock blk[DAC_NUM_BLOCKS];

    DACSnake s_final;  // 32 channels

    // final 32 -> 1 conv2, k=7, pad=3
    struct ggml_tensor * c2w;  // [7, 32, 1] bf16
    struct ggml_tensor * c2b;  // [1] f32

    // Storage for the weight tensors (separate from the inference graph ctx).
    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

// Load a 1D-along-channel alpha tensor (stored shape (1, C, 1)) as f32 [C],
// with an optional reciprocal transform to produce inv_b = 1 / (alpha + 1e-9).
// The source can be F32 or BF16 ; we widen on the fly when needed.
static void dac_load_alpha(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name, bool reciprocal) {
    struct ggml_tensor * mt = ggml_get_tensor(gf.meta, name.c_str());
    if (!mt) {
        fprintf(stderr, "[DAC] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    // Stored shape is (1, C, 1). C lives on ne[1] in ggml row-major.
    int                C   = (int) mt->ne[1];
    const void *       raw = gf_get_data(gf, name.c_str());
    std::vector<float> d(C);
    for (int i = 0; i < C; i++) {
        float a;
        if (mt->type == GGML_TYPE_F32) {
            a = ((const float *) raw)[i];
        } else {
            a = ggml_bf16_to_fp32(((const ggml_bf16_t *) raw)[i]);
        }
        d[i] = reciprocal ? (1.0f / (a + 1e-9f)) : a;
    }
    ggml_backend_tensor_set(dst, d.data(), 0, (size_t) C * sizeof(float));
}

// Cast a 1D bias (or any 1D F32-target tensor) to F32 on the backend, widening
// from BF16 if needed. F32 sources copy through directly.
static void dac_load_bias_f32(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * mt = ggml_get_tensor(gf.meta, name.c_str());
    if (!mt) {
        fprintf(stderr, "[DAC] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    int          C   = (int) mt->ne[0];
    const void * raw = gf_get_data(gf, name.c_str());
    if (mt->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(dst, raw, 0, (size_t) C * sizeof(float));
        return;
    }
    std::vector<float> d(C);
    for (int i = 0; i < C; i++) {
        d[i] = ggml_bf16_to_fp32(((const ggml_bf16_t *) raw)[i]);
    }
    ggml_backend_tensor_set(dst, d.data(), 0, (size_t) C * sizeof(float));
}

// Raw memcpy from the GGUF mmap to the backend tensor, dtype-agnostic.
// Works for any storage type (F32, BF16, F16) as long as the destination
// tensor was allocated with the same type as the GGUF source. Pair with
// gf_get_type to mirror the native dtype on the backend.
static void dac_load_passthrough(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    const void * src = gf_get_data(gf, name.c_str());
    if (!src) {
        fprintf(stderr, "[DAC] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    ggml_backend_tensor_set(dst, src, 0, ggml_nbytes(dst));
}

// Permute ConvTranspose1d weight from source layout (IC, OC, K) row-major to
// the [IC, K*OC] layout col2im_1d expects (k varies faster than oc inside
// the K*OC axis). The destination keeps the source dtype : we move whole
// elements of size ggml_type_size(dst->type) regardless of whether they are
// F32 or BF16.
//
//   src flat[ic*OC*K + oc*K + k] = w[ic][oc][k]   source-side row-major
//   dst flat[(oc*K + k)*IC + ic] = w[ic][oc][k]   ggml row-major, ne=(IC, K*OC)
static void dac_load_ctw(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * mt = ggml_get_tensor(gf.meta, name.c_str());
    if (!mt) {
        fprintf(stderr, "[DAC] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    GGML_ASSERT(dst->type == mt->type);

    // ggml ne = (K, OC, IC) since source-side shape is (IC, OC, K) row-major.
    const int    K   = (int) mt->ne[0];
    const int    OC  = (int) mt->ne[1];
    const int    IC  = (int) mt->ne[2];
    const int    KOC = K * OC;
    const size_t es  = ggml_type_size(dst->type);

    GGML_ASSERT(dst->ne[0] == IC && dst->ne[1] == KOC);

    const char *      raw = (const char *) gf_get_data(gf, name.c_str());
    std::vector<char> packed((size_t) IC * KOC * es);
    for (int ic = 0; ic < IC; ic++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int k = 0; k < K; k++) {
                size_t src_off = ((size_t) ic * OC * K + (size_t) oc * K + k) * es;
                size_t dst_off = ((size_t) (oc * K + k) * IC + ic) * es;
                std::memcpy(packed.data() + dst_off, raw + src_off, es);
            }
        }
    }
    ggml_backend_tensor_set(dst, packed.data(), 0, packed.size());
}

// Allocate one snake pair (alpha + inv_b) in f32
static void dac_alloc_snake(struct ggml_context * ctx, DACSnake * s, int C) {
    s->a     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
    s->inv_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
}

static void dac_load_snake(DACSnake * s, const GGUFModel & gf, const std::string & name) {
    dac_load_alpha(s->a, gf, name, false);
    dac_load_alpha(s->inv_b, gf, name, true);
}

// Full load : creates tensors in a private ctx, allocates them on the backend,
// then copies the data with the appropriate transforms.
static bool dac_load(DACDecoder * d, const GGUFModel & gf, ggml_backend_t backend) {
    static const int strides[]   = DAC_BLOCK_STRIDES;
    static const int in_chs[]    = DAC_BLOCK_IN_CH;
    static const int out_chs[]   = DAC_BLOCK_OUT_CH;
    static const int dilations[] = DAC_RU_DILATIONS;

    // Phase 1 : describe all tensors in a no_alloc context
    const int               n_tensors_max = 256;
    size_t                  ctx_size      = (size_t) n_tensors_max * ggml_tensor_overhead() + 1024;
    struct ggml_init_params p             = { ctx_size, NULL, true };
    d->weight_ctx                         = ggml_init(p);
    struct ggml_context * ctx             = d->weight_ctx;

    // 256 -> 1024 conv1, dtype mirrors the GGUF source (F32 or BF16)
    d->c1w = ggml_new_tensor_3d(ctx, gf_get_type(gf, "acoustic_decoder.conv1.weight"), 7, 256, 1024);
    d->c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);

    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        DACBlock & b = d->blk[i];
        b.in_ch      = in_chs[i];
        b.out_ch     = out_chs[i];
        b.stride     = strides[i];
        b.kernel     = 2 * b.stride;
        b.pad        = (b.stride + 1) / 2;  // ceil(stride / 2)
        b.output_pad = b.stride % 2;

        std::string pfx = "acoustic_decoder.block." + std::to_string(i);
        dac_alloc_snake(ctx, &b.s1, b.in_ch);
        b.ctw = ggml_new_tensor_2d(ctx, gf_get_type(gf, pfx + ".conv_t1.weight"), b.in_ch, b.kernel * b.out_ch);
        b.ctb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b.out_ch);

        for (int r = 0; r < DAC_RES_UNITS; r++) {
            DACResUnit & ru = b.ru[r];
            ru.dilation     = dilations[r];
            std::string rp  = pfx + ".res_unit" + std::to_string(r + 1);
            dac_alloc_snake(ctx, &ru.s1, b.out_ch);
            ru.c1w = ggml_new_tensor_3d(ctx, gf_get_type(gf, rp + ".conv1.weight"), 7, b.out_ch, b.out_ch);
            ru.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b.out_ch);
            dac_alloc_snake(ctx, &ru.s2, b.out_ch);
            ru.c2w = ggml_new_tensor_3d(ctx, gf_get_type(gf, rp + ".conv2.weight"), 1, b.out_ch, b.out_ch);
            ru.c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b.out_ch);
        }
    }

    dac_alloc_snake(ctx, &d->s_final, DAC_FINAL_CH);
    d->c2w = ggml_new_tensor_3d(ctx, gf_get_type(gf, "acoustic_decoder.conv2.weight"), 7, DAC_FINAL_CH, 1);
    d->c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    // Phase 2 : allocate backend buffer for all tensors at once
    d->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!d->weight_buf) {
        fprintf(stderr, "[DAC] FATAL: failed to allocate weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(d->weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Phase 3 : copy data from the GGUF mapping into the freshly-allocated
    // tensors, with per-tensor transforms (snake reciprocal, ctw permutation).
    dac_load_passthrough(d->c1w, gf, "acoustic_decoder.conv1.weight");
    dac_load_bias_f32(d->c1b, gf, "acoustic_decoder.conv1.bias");

    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        DACBlock &  b   = d->blk[i];
        std::string pfx = "acoustic_decoder.block." + std::to_string(i);
        dac_load_snake(&b.s1, gf, pfx + ".snake1.alpha");
        dac_load_ctw(b.ctw, gf, pfx + ".conv_t1.weight");
        dac_load_bias_f32(b.ctb, gf, pfx + ".conv_t1.bias");
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            DACResUnit & ru = b.ru[r];
            std::string  rp = pfx + ".res_unit" + std::to_string(r + 1);
            dac_load_snake(&ru.s1, gf, rp + ".snake1.alpha");
            dac_load_passthrough(ru.c1w, gf, rp + ".conv1.weight");
            dac_load_bias_f32(ru.c1b, gf, rp + ".conv1.bias");
            dac_load_snake(&ru.s2, gf, rp + ".snake2.alpha");
            dac_load_passthrough(ru.c2w, gf, rp + ".conv2.weight");
            dac_load_bias_f32(ru.c2b, gf, rp + ".conv2.bias");
        }
    }

    dac_load_snake(&d->s_final, gf, "acoustic_decoder.snake1.alpha");
    dac_load_passthrough(d->c2w, gf, "acoustic_decoder.conv2.weight");
    dac_load_bias_f32(d->c2b, gf, "acoustic_decoder.conv2.bias");

    fprintf(stderr, "[DAC] Loaded: 5 blocks, upsample %dx (8*5*4*2*3), 24 kHz mono out, weights %.1f MB\n",
            8 * 5 * 4 * 2 * 3, (float) ggml_backend_buffer_get_size(d->weight_buf) / (1024 * 1024));
    return true;
}

static void dac_free(DACDecoder * d) {
    if (d->weight_buf) {
        ggml_backend_buffer_free(d->weight_buf);
        d->weight_buf = NULL;
    }
    if (d->weight_ctx) {
        ggml_free(d->weight_ctx);
        d->weight_ctx = NULL;
    }
}

// Snake fused op : y = x + sin^2(a * x) * inv_b
static struct ggml_tensor * dac_snake(struct ggml_context * ctx, struct ggml_tensor * x, const DACSnake & s) {
    return ggml_snake(ctx, x, s.a, s.inv_b);
}

// Conv1d : x [T_in, IC] -> [T_out, OC] with bias
static struct ggml_tensor * dac_conv1d(struct ggml_context * ctx,
                                       struct ggml_tensor *  w,  // bf16 [K, IC, OC]
                                       struct ggml_tensor *  b,  // f32  [OC] or NULL
                                       struct ggml_tensor *  x,  // [T_in, IC]
                                       int                   stride,
                                       int                   pad,
                                       int                   dilation) {
    // ggml_conv_1d expects 3D input [T, IC, N], so we add the batch dim.
    struct ggml_tensor * x3 = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1);
    struct ggml_tensor * y  = ggml_conv_1d(ctx, w, x3, stride, pad, dilation);
    // ggml_conv_1d returns [OL, OC, N=1], squeeze to 2D
    y                       = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y                        = ggml_add(ctx, y, b2d);
    }
    return y;
}

// ConvTranspose1d via GEMM + col2im_1d
// w   : pre-permuted [IC, K*OC] bf16
// b   : [OC] f32 or NULL
// x   : [T_in, IC]
// Returns [T_in * stride, OC] when output_pad = stride % 2 is honored via right-pad.
static struct ggml_tensor * dac_conv_t1d(struct ggml_context * ctx,
                                         struct ggml_tensor *  w,
                                         struct ggml_tensor *  b,
                                         struct ggml_tensor *  x,
                                         int                   stride,
                                         int                   pad,
                                         int                   oc,
                                         int                   output_pad) {
    // 1. transpose x : [T_in, IC] -> [IC, T_in] (contiguous copy)
    struct ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));

    // 2. mul_mat contracts over IC : col [K*OC, T_in]
    struct ggml_tensor * col = ggml_mul_mat(ctx, w, xt);

    // 3. col2im_1d scatters into [T_out_no_op, OC] with T_out_no_op = (T_in - 1)*stride + K - 2*pad
    struct ggml_tensor * y = ggml_col2im_1d(ctx, col, stride, oc, pad);

    // 4. reference ConvTranspose1d output_padding = right-pad with zeros along T axis
    if (output_pad > 0) {
        y = ggml_pad(ctx, y, output_pad, 0, 0, 0);
    }

    // 5. add bias
    if (b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y                        = ggml_add(ctx, y, b2d);
    }
    return y;
}

// Residual unit forward : skip + conv2(snake2(conv1(snake1(x))))
// kernel=7 same-padding via pad = (k-1)*dilation/2 = 3*dilation
static struct ggml_tensor * dac_res_unit(struct ggml_context * ctx,
                                         const DACResUnit *    ru,
                                         struct ggml_tensor *  x) {  // [T, C]
    struct ggml_tensor * skip = x;

    int pad1 = 3 * ru->dilation;
    x        = dac_snake(ctx, x, ru->s1);
    x        = dac_conv1d(ctx, ru->c1w, ru->c1b, x, 1, pad1, ru->dilation);

    x = dac_snake(ctx, x, ru->s2);
    x = dac_conv1d(ctx, ru->c2w, ru->c2b, x, 1, 0, 1);

    return ggml_add(ctx, skip, x);
}

// Build the full DAC decode graph
// latent : [T_in, 256] f32  ->  audio : [T_out, 1] f32  with T_out = 960 * T_in
//
// `dump_stages`, when non-NULL, is filled with up to 7 named intermediate
// tensors (initial conv1, post block 0..4, post final snake) so callers can
// pin them as graph outputs for stage-by-stage cossim diffs.
static struct ggml_tensor * dac_build_graph(struct ggml_context *               ctx,
                                            const DACDecoder *                  d,
                                            struct ggml_tensor *                latent,
                                            std::vector<struct ggml_tensor *> * dump_stages = nullptr) {
    // initial conv1 : 256 -> 1024
    struct ggml_tensor * x = dac_conv1d(ctx, d->c1w, d->c1b, latent, 1, 3, 1);
    if (dump_stages) {
        ggml_set_name(x, "dac_after_conv1");
        dump_stages->push_back(x);
    }

    // 5 upsampling blocks
    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        const DACBlock & b = d->blk[i];
        x                  = dac_snake(ctx, x, b.s1);
        x                  = dac_conv_t1d(ctx, b.ctw, b.ctb, x, b.stride, b.pad, b.out_ch, b.output_pad);
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            x = dac_res_unit(ctx, &b.ru[r], x);
        }
        if (dump_stages) {
            char nm[32];
            snprintf(nm, sizeof(nm), "dac_after_blk%d", i);
            ggml_set_name(x, nm);
            dump_stages->push_back(x);
        }
    }

    // final snake -> conv2 (32 -> 1, k=7, pad=3) + bias
    x                          = dac_snake(ctx, x, d->s_final);
    struct ggml_tensor * audio = dac_conv1d(ctx, d->c2w, d->c2b, x, 1, 3, 1);

    return audio;  // [T_out, 1]
}
