#pragma once
// dac-encoder.h : DAC acoustic encoder for OmniVoice (GGML)
//
// Mirror of dac-decoder.h. Reuses DACSnake, DACResUnit, load helpers, graph
// ops via #include "dac-decoder.h". Architecture :
//   conv1(1 -> 64, k=7) -> 5x ( 3 res_units -> snake -> strided Conv1d )
//                       -> snake -> conv2(2048 -> 256, k=3)
// Downsample : 8 * 5 * 4 * 2 * 3 = 960x (matches decoder upsample for round-trip).
// Block layout differs from decoder : encoder runs the residuals first then
// the snake + downsampling conv at the end, which is the inverse order of
// the decoder where snake + transposed conv come first.
//
// Input  : audio  [T_in,  1]    f32  (24 kHz mono waveform)
// Output : latent [T_out, 256]  f32  with T_out = T_in / 960

#include "dac-decoder.h"

// Encoder block : 3x ResUnit(in_ch) -> Snake(in_ch) -> strided Conv1d(in -> out)
// The res_units keep in_ch ; the post-residual snake works on in_ch ; the
// strided conv at the end of the block is what brings the channel count up
// from in_ch to out_ch alongside the temporal downsampling.
struct DACEncBlock {
    DACResUnit           ru[DAC_RES_UNITS];
    DACSnake             s_post;  // snake before the downsampling conv (in_ch)
    struct ggml_tensor * cw;      // strided conv [k, in_ch, out_ch] bf16
    struct ggml_tensor * cb;      // strided conv bias [out_ch] f32
    int                  in_ch;
    int                  out_ch;
    int                  stride;
    int                  kernel;  // per-block : 16, 10, 8, 4, 6
    int                  pad;     // per-block : 4, 3, 2, 1, 2
};

struct DACEncoder {
    // initial 1 -> 64 conv1, k=7, pad=3
    struct ggml_tensor * c1w;  // [7, 1, 64] bf16
    struct ggml_tensor * c1b;  // [64] f32

    DACEncBlock blk[DAC_NUM_BLOCKS];

    DACSnake s_final;  // 2048 channels

    // final 2048 -> 256 conv2, k=3, pad=1
    struct ggml_tensor * c2w;  // [3, 2048, 256] bf16
    struct ggml_tensor * c2b;  // [256] f32

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

#define DAC_ENC_BLOCK_IN_CH   { 64, 128, 256, 512, 1024 }
#define DAC_ENC_BLOCK_OUT_CH  { 128, 256, 512, 1024, 2048 }
#define DAC_ENC_BLOCK_STRIDES { 8, 5, 4, 2, 3 }
#define DAC_ENC_BLOCK_KERNELS { 16, 10, 8, 4, 6 }
#define DAC_ENC_BLOCK_PADS    { 4, 3, 2, 1, 2 }

// Full load : creates tensors in a private ctx, allocates them on the backend,
// then copies the data with the appropriate transforms. Same 3-phase pattern
// as dac_load (describe -> alloc -> fill).
static bool dac_enc_load(DACEncoder * d, const GGUFModel & gf, ggml_backend_t backend) {
    static const int in_chs[]    = DAC_ENC_BLOCK_IN_CH;
    static const int out_chs[]   = DAC_ENC_BLOCK_OUT_CH;
    static const int strides[]   = DAC_ENC_BLOCK_STRIDES;
    static const int kernels[]   = DAC_ENC_BLOCK_KERNELS;
    static const int pads[]      = DAC_ENC_BLOCK_PADS;
    static const int dilations[] = DAC_RU_DILATIONS;

    // Phase 1 : describe all tensors in a no_alloc context.
    const int               n_tensors_max = 256;
    size_t                  ctx_size      = (size_t) n_tensors_max * ggml_tensor_overhead() + 1024;
    struct ggml_init_params p             = { ctx_size, NULL, true };
    d->weight_ctx                         = ggml_init(p);
    struct ggml_context * ctx             = d->weight_ctx;

    // 1 -> 64 conv1, dtype mirrors the GGUF source (F32 or BF16)
    d->c1w = ggml_new_tensor_3d(ctx, gf_get_type(gf, "acoustic_encoder.conv1.weight"), 7, 1, 64);
    d->c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);

    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        DACEncBlock & b = d->blk[i];
        b.in_ch         = in_chs[i];
        b.out_ch        = out_chs[i];
        b.stride        = strides[i];
        b.kernel        = kernels[i];
        b.pad           = pads[i];

        std::string pfx = "acoustic_encoder.block." + std::to_string(i);

        // Three res_units operate on in_ch (residual loop on the same channel
        // count, kernel 7 dilated for conv1 and kernel 1 for conv2).
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            DACResUnit & ru = b.ru[r];
            ru.dilation     = dilations[r];
            std::string rp  = pfx + ".res_unit" + std::to_string(r + 1);
            dac_alloc_snake(ctx, &ru.s1, b.in_ch);
            ru.c1w = ggml_new_tensor_3d(ctx, gf_get_type(gf, rp + ".conv1.weight"), 7, b.in_ch, b.in_ch);
            ru.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b.in_ch);
            dac_alloc_snake(ctx, &ru.s2, b.in_ch);
            ru.c2w = ggml_new_tensor_3d(ctx, gf_get_type(gf, rp + ".conv2.weight"), 1, b.in_ch, b.in_ch);
            ru.c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b.in_ch);
        }

        // Post-residual snake then downsampling conv (in_ch -> out_ch).
        dac_alloc_snake(ctx, &b.s_post, b.in_ch);
        b.cw = ggml_new_tensor_3d(ctx, gf_get_type(gf, pfx + ".conv1.weight"), b.kernel, b.in_ch, b.out_ch);
        b.cb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b.out_ch);
    }

    // Final snake on 2048 channels then conv2 (2048 -> 256, k=3).
    dac_alloc_snake(ctx, &d->s_final, 2048);
    d->c2w = ggml_new_tensor_3d(ctx, gf_get_type(gf, "acoustic_encoder.conv2.weight"), 3, 2048, 256);
    d->c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);

    // Phase 2 : allocate backend buffer for all tensors at once.
    d->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!d->weight_buf) {
        fprintf(stderr, "[DAC-Enc] FATAL: failed to allocate weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(d->weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Phase 3 : copy data with per-tensor transforms (snake reciprocal, F32 cast).
    dac_load_passthrough(d->c1w, gf, "acoustic_encoder.conv1.weight");
    dac_load_bias_f32(d->c1b, gf, "acoustic_encoder.conv1.bias");

    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        DACEncBlock & b   = d->blk[i];
        std::string   pfx = "acoustic_encoder.block." + std::to_string(i);
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
        dac_load_snake(&b.s_post, gf, pfx + ".snake1.alpha");
        dac_load_passthrough(b.cw, gf, pfx + ".conv1.weight");
        dac_load_bias_f32(b.cb, gf, pfx + ".conv1.bias");
    }

    dac_load_snake(&d->s_final, gf, "acoustic_encoder.snake1.alpha");
    dac_load_passthrough(d->c2w, gf, "acoustic_encoder.conv2.weight");
    dac_load_bias_f32(d->c2b, gf, "acoustic_encoder.conv2.bias");

    fprintf(stderr, "[DAC-Enc] Loaded: 5 blocks, downsample %dx (8*5*4*2*3), 24 kHz mono in, weights %.1f MB\n",
            8 * 5 * 4 * 2 * 3, (float) ggml_backend_buffer_get_size(d->weight_buf) / (1024 * 1024));
    return true;
}

static void dac_enc_free(DACEncoder * d) {
    if (d->weight_buf) {
        ggml_backend_buffer_free(d->weight_buf);
        d->weight_buf = NULL;
    }
    if (d->weight_ctx) {
        ggml_free(d->weight_ctx);
        d->weight_ctx = NULL;
    }
}

// Build the DAC encode graph. Input audio [T_in, 1] -> latent [T_out, 256].
//
// Block forward (encoder, mirror inverse of the decoder block) :
//   x = res_unit1(x)
//   x = res_unit2(x)
//   x = snake_post(res_unit3(x))    # NOTE : snake applied AFTER ru3, not in ru3
//   x = strided_conv(x)             # downsample by stride, in_ch -> out_ch
static struct ggml_tensor * dac_enc_build_graph(struct ggml_context * ctx,
                                                const DACEncoder *    d,
                                                struct ggml_tensor *  audio  // [T_in, 1] f32
) {
    // initial conv1 : 1 -> 64, k=7, pad=3
    struct ggml_tensor * x = dac_conv1d(ctx, d->c1w, d->c1b, audio, 1, 3, 1);

    // 5 down-sampling blocks
    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        const DACEncBlock & b = d->blk[i];
        x                     = dac_res_unit(ctx, &b.ru[0], x);
        x                     = dac_res_unit(ctx, &b.ru[1], x);
        x                     = dac_res_unit(ctx, &b.ru[2], x);
        x                     = dac_snake(ctx, x, b.s_post);
        x                     = dac_conv1d(ctx, b.cw, b.cb, x, b.stride, b.pad, 1);
    }

    // final snake -> conv2 (2048 -> 256, k=3, pad=1)
    x                           = dac_snake(ctx, x, d->s_final);
    struct ggml_tensor * latent = dac_conv1d(ctx, d->c2w, d->c2b, x, 1, 1, 1);
    return latent;  // [T_out, 256] with T_out = T_in / 960
}
