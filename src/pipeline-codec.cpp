// pipeline-codec.cpp: decode-side audio tokenizer pipeline.
//
// Graph (fast axis ne[0] explicit) :
//   codes      [T, 8]      i32   ->  rvq_decode_graph
//   latent     [1024, T]   f32   <- rvq output, channels-first natively
//   h          [256, T]    f32   <- mul_mat(fc2_w, latent), ne=(M, N)
//                                       M = fc2_w.ne[1] = 256 (out)
//                                       N = latent.ne[1] = T
//   h          [256, T]    f32   <- + fc2_b (1D ne=256, broadcasts on ne[1]=T)
//   h          [T, 256]    f32   <- cont(transpose) for DAC conv input
//   audio      [T*960, 1]  f32   <- dac_build_graph
//   audio_1d   [T*960]     f32   <- view_1d
//
// ggml_mul_mat(A, B) requires A.ne[0]==B.ne[0]==K and produces ne=(M=A.ne[1],
// N=B.ne[1]) with M fast. So fc2 produces channels-first like RVQ, the bias
// add uses the 1D broadcast on ne[1] just like RVQ's project_out_b, and we
// transpose once before DAC because DAC's conv1d expects T fast.

#include "pipeline-codec.h"

#include "audio-resample.h"
#include "debug.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ov-error.h"
#include "utf8.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

bool pipeline_codec_load(PipelineCodec * pc, const char * gguf_path, BackendPair bp) {
    *pc                    = {};
    pc->bp                 = bp;
    pc->backend            = bp.backend;
    ggml_backend_t backend = bp.backend;

    if (!gf_load(&pc->gguf, gguf_path)) {
        return false;
    }

    pc->sample_rate = (int) gf_get_u32(pc->gguf, "omnivoice.sample_rate");
    pc->hop_length  = (int) gf_get_u32(pc->gguf, "omnivoice.acoustic.hop_length");
    if (pc->sample_rate == 0 || pc->hop_length == 0) {
        ov_log(OV_LOG_ERROR, "[PipelineCodec] missing sample_rate or hop_length in GGUF");
        gf_close(&pc->gguf);
        return false;
    }

    // RVQ: 8 codebooks * 5 tensors (embed + project_in w/b + project_out w/b)
    // = 40, plus fc2 weight + bias and fc weight + bias = 44. We pad to 64 for headroom.
    wctx_init(&pc->wctx, 64);

    if (!rvq_load(&pc->rvq, &pc->wctx, pc->gguf)) {
        wctx_free(&pc->wctx);
        gf_close(&pc->gguf);
        return false;
    }

    pc->fc2_w = gf_load_tensor(&pc->wctx, pc->gguf, "fc2.weight");
    // F32 cast at load: ggml_add CUDA only accepts src1 in F32 or F16.
    pc->fc2_b = gf_load_tensor_f32(&pc->wctx, pc->gguf, "fc2.bias");
    if (!pc->fc2_w || !pc->fc2_b) {
        ov_log(OV_LOG_ERROR, "[PipelineCodec] missing fc2.weight or fc2.bias");
        wctx_free(&pc->wctx);
        gf_close(&pc->gguf);
        return false;
    }

    pc->fc_w = gf_load_tensor(&pc->wctx, pc->gguf, "fc.weight");
    pc->fc_b = gf_load_tensor_f32(&pc->wctx, pc->gguf, "fc.bias");
    if (!pc->fc_w || !pc->fc_b) {
        ov_log(OV_LOG_ERROR, "[PipelineCodec] missing fc.weight or fc.bias");
        wctx_free(&pc->wctx);
        gf_close(&pc->gguf);
        return false;
    }

    if (!wctx_alloc(&pc->wctx, backend)) {
        wctx_free(&pc->wctx);
        gf_close(&pc->gguf);
        return false;
    }

    // DAC has its own private weight ctx and applies per-tensor transforms
    // at load time (snake reciprocal, conv_t permutation) using gf_get_data
    // raw mmap pointers, so the GGUF mmap must remain open across this call.
    if (!dac_load(&pc->dac, pc->gguf, backend)) {
        wctx_free(&pc->wctx);
        gf_close(&pc->gguf);
        return false;
    }

    // DAC encoder mirrors the same private weight ctx pattern. It is the
    // counterpart of the decoder used during the encode side of the codec
    // pipeline (waveform -> latent), and reuses the same load helpers.
    if (!dac_enc_load(&pc->dac_enc, pc->gguf, backend)) {
        dac_free(&pc->dac);
        wctx_free(&pc->wctx);
        gf_close(&pc->gguf);
        return false;
    }

    // SemanticEncoder refines the HuBERT semantic features before the joint
    // fc + RVQ encode stage. Its private weight ctx mirrors the DAC pattern
    // for consistency, even though it has no per-tensor transforms.
    if (!sem_enc_load(&pc->sem_enc, pc->gguf, backend)) {
        dac_enc_free(&pc->dac_enc);
        dac_free(&pc->dac);
        wctx_free(&pc->wctx);
        gf_close(&pc->gguf);
        return false;
    }

    // HuBERT feature_extractor: 7 conv stack with GroupNorm on layer 0 and
    // GELU on every layer. Cumulative stride 320 over 16 kHz audio.
    if (!hubert_feat_load(&pc->hubert_feat, pc->gguf, backend)) {
        sem_enc_free(&pc->sem_enc);
        dac_enc_free(&pc->dac_enc);
        dac_free(&pc->dac);
        wctx_free(&pc->wctx);
        gf_close(&pc->gguf);
        return false;
    }

    // HuBERT feature_projection: LN(512) + Linear(512, 768). Bridges the
    // feature_extractor (T fast layout) to the transformer encoder which
    // operates in C-first layout (C fast, T slow).
    if (!hubert_proj_load(&pc->hubert_proj, pc->gguf, backend)) {
        hubert_feat_free(&pc->hubert_feat);
        sem_enc_free(&pc->sem_enc);
        dac_enc_free(&pc->dac_enc);
        dac_free(&pc->dac);
        wctx_free(&pc->wctx);
        gf_close(&pc->gguf);
        return false;
    }

    // HuBERT encoder init block: grouped pos_conv_embed (residual add) +
    // first LayerNorm. Sits between feature_projection and the 12-layer stack.
    if (!hubert_enc_init_load(&pc->hubert_enc_init, pc->gguf, backend)) {
        hubert_proj_free(&pc->hubert_proj);
        hubert_feat_free(&pc->hubert_feat);
        sem_enc_free(&pc->sem_enc);
        dac_enc_free(&pc->dac_enc);
        dac_free(&pc->dac);
        wctx_free(&pc->wctx);
        gf_close(&pc->gguf);
        return false;
    }

    // HuBERT transformer encoder: 12 Post-LN layers loaded in sequence. Each
    // layer owns its own backend buffer for clean lifecycle. On failure at
    // index i, we unwind layers [0, i) before the rest of the chain.
    for (int i = 0; i < HUBERT_NUM_LAYERS; i++) {
        if (!hubert_layer_load(&pc->hubert_layers[i], pc->gguf, backend, i)) {
            for (int j = 0; j < i; j++) {
                hubert_layer_free(&pc->hubert_layers[j]);
            }
            hubert_enc_init_free(&pc->hubert_enc_init);
            hubert_proj_free(&pc->hubert_proj);
            hubert_feat_free(&pc->hubert_feat);
            sem_enc_free(&pc->sem_enc);
            dac_enc_free(&pc->dac_enc);
            dac_free(&pc->dac);
            wctx_free(&pc->wctx);
            gf_close(&pc->gguf);
            return false;
        }
    }
    size_t stack_bytes = 0;
    for (int i = 0; i < HUBERT_NUM_LAYERS; i++) {
        stack_bytes += ggml_backend_buffer_get_size(pc->hubert_layers[i].weight_buf);
    }
    ov_log(OV_LOG_INFO, "[HuBERT-Stack] Loaded: %d layers, hidden=%d heads=%d ffn=%d, weights %.1f MB",
           HUBERT_NUM_LAYERS, HUBERT_HIDDEN, HUBERT_NUM_HEADS, HUBERT_FFN_INNER, (float) stack_bytes / (1024 * 1024));

    // All weights are now on the backend. The mmap is no longer needed.
    gf_close(&pc->gguf);

    ov_log(OV_LOG_INFO, "[PipelineCodec] Loaded codec: sr=%d hop=%d backend=%s", pc->sample_rate, pc->hop_length,
           ggml_backend_name(backend));
    // Scheduler: routes ops the GPU backend cannot run (e.g. K-quant
    // get_rows on CUDA) to the CPU backend. 4096 nodes covers HuBERT 12L
    // + DAC encoder + DAC decoder graphs.
    pc->sched = backend_sched_new(bp, 4096);
    if (!pc->sched) {
        pipeline_codec_free(pc);
        return false;
    }

    return true;
}

std::vector<float> pipeline_codec_decode(PipelineCodec * pc, const int32_t * codes, int num_codebooks, int n_frames) {
    if (num_codebooks != RVQ_NUM_CODEBOOKS) {
        ov_log(OV_LOG_ERROR, "[PipelineCodec] codes have %d codebooks, expected %d", num_codebooks, RVQ_NUM_CODEBOOKS);
        return {};
    }
    if (n_frames <= 0) {
        ov_log(OV_LOG_ERROR, "[PipelineCodec] n_frames must be > 0");
        return {};
    }

    // Compute graph context: holds tensor descriptors for the forward pass.
    // 4096 nodes leaves ample headroom (DAC alone has ~150 ops).
    const int    n_max_nodes    = 4096;
    const size_t graph_ctx_size = ggml_tensor_overhead() * n_max_nodes + ggml_graph_overhead_custom(n_max_nodes, false);

    struct ggml_init_params gp   = { graph_ctx_size, NULL, /*no_alloc=*/true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        ov_log(OV_LOG_ERROR, "[PipelineCodec] ggml_init failed for graph ctx");
        return {};
    }

    // Input: codes [T, 8] i32, ne[0]=T fast
    struct ggml_tensor * codes_in = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, n_frames, num_codebooks);
    ggml_set_name(codes_in, "codes_in");
    ggml_set_input(codes_in);

    // RVQ: codes [T, 8] -> latent [1024, T]
    struct ggml_tensor * latent = rvq_decode_graph(gctx, &pc->rvq, codes_in);
    ggml_set_name(latent, "rvq_out");

    // fc2 Linear via ggml_mul_mat. GGML convention :
    //   A = fc2_w  ne=(K=1024, M=256)
    //   B = latent ne=(K=1024, N=T)
    //   result     ne=(M=256, N=T)        ne[0]=256=channels fast, ne[1]=T slow
    struct ggml_tensor * h = ggml_mul_mat(gctx, pc->fc2_w, latent);

    // 1D bias broadcast: fc2_b ne=(256) lines up with h.ne[0]=256, broadcasts
    // across ne[1]=T. Same pattern as the RVQ project_out_b add.
    h = ggml_add(gctx, h, pc->fc2_b);
    ggml_set_name(h, "fc2_out");

    // Stable reference for the optional dump: `h` is about to be reassigned
    // by cont(transpose), and only the original fc2_out tensor will be marked
    // as output. The post-transpose tensor would be reused by the gallocr
    // once DAC consumes it.
    struct ggml_tensor * fc2_out = h;

    // Optional debug: keep RVQ and fc2 outputs alive across the graph compute
    // so we can copy them back. Without this the gallocr is free to reuse
    // their backing memory once consumed downstream.
    const bool dump = std::getenv("OMNIVOICE_DUMP_STAGES") != NULL;
    if (dump) {
        ggml_set_output(latent);
        ggml_set_output(fc2_out);
    }

    // DAC expects [T, IC=256] T fast (ggml_conv_1d input layout). Current shape
    // is (256, T) so transpose and make contiguous before feeding the decoder.
    h = ggml_cont(gctx, ggml_transpose(gctx, h));

    // DAC decode -> [T*960, 1] f32
    std::vector<struct ggml_tensor *> dac_stages;
    struct ggml_tensor *              audio = dac_build_graph(gctx, &pc->dac, h, dump ? &dac_stages : nullptr);
    if (dump) {
        for (auto * t : dac_stages) {
            ggml_set_output(t);
        }
    }
    ggml_set_name(audio, "audio_out");
    ggml_set_output(audio);

    // Build forward graph
    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, audio);

    // Allocate intermediates + input/output buffers in one shot on the backend.
    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, graph)) {
        ov_log(OV_LOG_ERROR, "[PipelineCodec] gallocr_alloc_graph failed");
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    // Upload codes to the backend tensor (allocated by gallocr above).
    ggml_backend_tensor_set(codes_in, codes, 0, (size_t) n_frames * num_codebooks * sizeof(int32_t));

    // Compute
    enum ggml_status st = ggml_backend_sched_graph_compute(pc->sched, graph);
    if (st != GGML_STATUS_SUCCESS) {
        ov_log(OV_LOG_ERROR, "[PipelineCodec] graph_compute status=%d", (int) st);
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    // Optional debug dump of intermediate stages.
    if (dump) {
        auto dump_tensor = [](const char * path, struct ggml_tensor * t) {
            const size_t       n = ggml_nelements(t);
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(float));
            FILE * f = utf8_fopen(path, "wb");
            fwrite(tmp.data(), sizeof(float), n, f);
            fclose(f);
            ov_log(OV_LOG_INFO, "[PipelineCodec] Dumped %s: %zu f32 values, ne=(%lld, %lld, %lld, %lld)", path, n,
                   (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        };
        dump_tensor("cpp_rvq_out.raw", latent);
        dump_tensor("cpp_fc2_out.raw", fc2_out);
        for (auto * t : dac_stages) {
            char path[64];
            snprintf(path, sizeof(path), "cpp_%s.raw", ggml_get_name(t));
            dump_tensor(path, t);
        }
    }

    // Copy output back to host. audio->ne = (T_out, 1), total = T_out floats.
    const int          T_out = (int) audio->ne[0];
    std::vector<float> out((size_t) T_out);
    ggml_backend_tensor_get(audio, out.data(), 0, (size_t) T_out * sizeof(float));

    ggml_backend_sched_reset(pc->sched);
    ggml_free(gctx);
    return out;
}

// Compute T_a = output time length of the DAC encoder for a given audio
// sample count. Matches the upstream HiggsAudioV2 acoustic_encoder topology :
//   conv1 k=7 s=1 p=3                    -> T unchanged
//   blk0 k=16 s=8 p=4 (in=64  out=128)   -> T_out = (T + 8 - 16) / 8 + 1
//   blk1 k=10 s=5 p=3 (in=128 out=256)   -> T_out = (T + 6 - 10) / 5 + 1
//   blk2 k=8  s=4 p=2 (in=256 out=512)   -> T_out = (T + 4 - 8)  / 4 + 1
//   blk3 k=4  s=2 p=1 (in=512 out=1024)  -> T_out = (T + 2 - 4)  / 2 + 1
//   blk4 k=6  s=3 p=2 (in=1024 out=2048) -> T_out = (T + 4 - 6)  / 3 + 1
//   conv2 k=3 s=1 p=1                    -> T unchanged
// Used to detect the T_a != T_s case before deciding whether to pad audio.
static int compute_dac_output_length(int n_samples) {
    static const int K[5] = { 16, 10, 8, 4, 6 };
    static const int S[5] = { 8, 5, 4, 2, 3 };
    static const int P[5] = { 4, 3, 2, 1, 2 };
    int              T    = n_samples;
    for (int i = 0; i < 5; i++) {
        T = (T + 2 * P[i] - K[i]) / S[i] + 1;
    }
    return T;
}

// Forward declarations for the encode pipeline helpers: isolated graphs
// chained together by pipeline_codec_encode.
static std::vector<float> pipeline_codec_hubert_features_test(PipelineCodec * pc,
                                                              const float *   audio_f32,
                                                              int             n_samples,
                                                              const char *    dump_dir);
static std::vector<float> pipeline_codec_sem_enc_test(PipelineCodec * pc, const float * features_f32, int n_frames);
static std::vector<float> pipeline_codec_dac_enc_test(PipelineCodec * pc, const float * audio_f32, int n_samples);

std::vector<int32_t> pipeline_codec_encode(PipelineCodec * pc,
                                           const float *   audio_24k,
                                           int             n_samples,
                                           const char *    dump_dir) {
    if (n_samples <= 0) {
        return {};
    }

    // Step 1: resample 24 kHz -> 16 kHz mono and pad with 160 zeros on each
    // side, matching HiggsAudioV2 _extract_semantic_features.
    int     n_16k     = 0;
    float * resampled = audio_resample(audio_24k, n_samples, pc->sample_rate, 16000, 1, &n_16k);
    if (!resampled || n_16k <= 0) {
        free(resampled);
        return {};
    }
    if (dump_dir) {
        DebugDumper dbg;
        debug_init(&dbg, dump_dir);
        debug_dump_1d(&dbg, "ref-audio-16k", resampled, n_16k);
    }
    const int          n_padded = n_16k + 320;
    std::vector<float> audio_16k_padded((size_t) n_padded, 0.0f);
    std::memcpy(audio_16k_padded.data() + 160, resampled, (size_t) n_16k * sizeof(float));
    free(resampled);

    // Step 2: full HuBERT features pipeline -> e_semantic_input (768, T_s).
    std::vector<float> features = pipeline_codec_hubert_features_test(pc, audio_16k_padded.data(), n_padded, dump_dir);
    if (features.empty()) {
        return {};
    }
    const int T_s = (int) (features.size() / 768);

    if (dump_dir) {
        DebugDumper dbg;
        debug_init(&dbg, dump_dir);
        debug_dump_2d(&dbg, "ref-hubert-features", features.data(), T_s, 768);
    }

    // hubert_features returns ne=(768, T_s) with K fast (768 axis). sem_enc
    // graph input expects ne=(T_s, 768) with T fast (dac_conv1d layout, mirror
    // of the reference encoder_semantic which receives sem_in.transpose(1,2)
    // before the conv stack). Transpose the flat buffer: K-fast to T-fast.
    std::vector<float> features_t(features.size());
    for (int t = 0; t < T_s; t++) {
        for (int k = 0; k < SEM_HIDDEN; k++) {
            features_t[t + (size_t) k * T_s] = features[k + (size_t) t * SEM_HIDDEN];
        }
    }

    // Step 3: SemanticEncoder refines the features in-place (no temporal
    // downsample), output stays T-first (T_s, 768).
    std::vector<float> e_semantic = pipeline_codec_sem_enc_test(pc, features_t.data(), T_s);
    if (e_semantic.empty()) {
        return {};
    }

    // Step 4: DAC encoder. Decide pad branch from analytical T_a vs measured
    // T_s. Upstream uses self.pad = hop_length // 2 = 480 zeros on each side.
    const int          T_a_no_pad = compute_dac_output_length(n_samples);
    std::vector<float> e_acoustic;
    if (T_a_no_pad != T_s) {
        const int          p = pc->hop_length / 2;
        std::vector<float> padded((size_t) (n_samples + 2 * p), 0.0f);
        std::memcpy(padded.data() + p, audio_24k, (size_t) n_samples * sizeof(float));
        e_acoustic = pipeline_codec_dac_enc_test(pc, padded.data(), (int) padded.size());
    } else {
        e_acoustic = pipeline_codec_dac_enc_test(pc, audio_24k, n_samples);
    }
    if (e_acoustic.empty()) {
        return {};
    }
    const int T_a = (int) (e_acoustic.size() / 256);
    if (T_a != T_s) {
        ov_log(OV_LOG_ERROR, "[Encode] post-DAC T_a=%d does not match HuBERT T_s=%d", T_a, T_s);
        return {};
    }

    // Step 5: fused graph for concat + fc + RVQ encode. ac and sem land on
    // the backend as inputs, the rest stays on the GPU until the codes copy.
    const int    T              = T_s;
    const int    n_max_nodes    = 4096;
    const size_t graph_ctx_size = ggml_tensor_overhead() * n_max_nodes + ggml_graph_overhead_custom(n_max_nodes, false);

    struct ggml_init_params gp   = { graph_ctx_size, NULL, /*no_alloc=*/true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        return {};
    }

    // dac_enc and sem_enc both return tensors in T-first layout (ne=(T, C)
    // with T on the fast axis). We mirror that on the input side and transpose
    // to C-first inside the graph, so the cat-on-channel axis + fc Linear stay
    // simple mul_mat operations downstream.
    struct ggml_tensor * ac_t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, T, 256);
    ggml_set_name(ac_t, "acoustic");
    ggml_set_input(ac_t);

    struct ggml_tensor * sem_t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, T, 768);
    ggml_set_name(sem_t, "semantic");
    ggml_set_input(sem_t);

    // (T, C) -> (C, T) so the channel axis lives on ne[0] (fast).
    struct ggml_tensor * ac_c  = ggml_cont(gctx, ggml_transpose(gctx, ac_t));
    struct ggml_tensor * sem_c = ggml_cont(gctx, ggml_transpose(gctx, sem_t));

    // Concat [acoustic, semantic] on the channel axis ne[0]: (256+768=1024, T).
    // Upstream order in encode() is torch.cat([e_acoustic, e_semantic], dim=1).
    struct ggml_tensor * pre_fc = ggml_concat(gctx, ac_c, sem_c, 0);
    ggml_set_name(pre_fc, "encode_pre_fc");

    // fc Linear (1024 -> 1024). C-first layout makes mul_mat direct, no transpose.
    struct ggml_tensor * embed = ggml_mul_mat(gctx, pc->fc_w, pre_fc);
    struct ggml_tensor * fcb2d = ggml_reshape_2d(gctx, pc->fc_b, 1024, 1);
    embed                      = ggml_add(gctx, embed, fcb2d);
    ggml_set_name(embed, "encode_embed");

    // Optional debug: keep pre_fc and embed alive across the graph so we can
    // copy them back post compute, to compare the RVQ input bit-for-bit and
    // isolate where the divergence is born (concat vs fc Linear).
    const bool dump = std::getenv("OMNIVOICE_DUMP_STAGES") != NULL;
    if (dump) {
        ggml_set_output(pre_fc);
        ggml_set_output(embed);
    }

    // RVQ encode loop: 8 codebooks, each producing a (T,) i32 tensor with the
    // selected indices. The graph itself emits 8 separate output tensors.
    struct ggml_tensor * idx_per_k[RVQ_NUM_CODEBOOKS] = { 0 };
    rvq_encode_graph(gctx, &pc->rvq, embed, idx_per_k);
    for (int k = 0; k < RVQ_NUM_CODEBOOKS; k++) {
        ggml_set_output(idx_per_k[k]);
    }

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    for (int k = 0; k < RVQ_NUM_CODEBOOKS; k++) {
        ggml_build_forward_expand(graph, idx_per_k[k]);
    }

    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, graph)) {
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    ggml_backend_tensor_set(ac_t, e_acoustic.data(), 0, e_acoustic.size() * sizeof(float));
    ggml_backend_tensor_set(sem_t, e_semantic.data(), 0, e_semantic.size() * sizeof(float));

    enum ggml_status st = ggml_backend_sched_graph_compute(pc->sched, graph);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    // Optional debug dump of the fc input (post concat) and output (RVQ input).
    if (dump) {
        auto dump_tensor = [](const char * path, struct ggml_tensor * t) {
            const size_t       n = ggml_nelements(t);
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(float));
            FILE * f = utf8_fopen(path, "wb");
            fwrite(tmp.data(), sizeof(float), n, f);
            fclose(f);
            ov_log(OV_LOG_INFO, "[Encode] Dumped %s: %zu f32 values, ne=(%lld, %lld)", path, n, (long long) t->ne[0],
                   (long long) t->ne[1]);
        };
        dump_tensor("cpp_encode_pre_fc.raw", pre_fc);
        dump_tensor("cpp_encode_embed.raw", embed);
    }

    // Pack codes row-major: (CB, T) with codebook leading.
    std::vector<int32_t> codes((size_t) RVQ_NUM_CODEBOOKS * (size_t) T);
    for (int k = 0; k < RVQ_NUM_CODEBOOKS; k++) {
        ggml_backend_tensor_get(idx_per_k[k], &codes[(size_t) k * (size_t) T], 0, (size_t) T * sizeof(int32_t));
    }

    ggml_backend_sched_reset(pc->sched);
    ggml_free(gctx);
    return codes;
}

void pipeline_codec_free(PipelineCodec * pc) {
    if (pc->sched) {
        ggml_backend_sched_free(pc->sched);
    }
    for (int i = 0; i < HUBERT_NUM_LAYERS; i++) {
        hubert_layer_free(&pc->hubert_layers[i]);
    }
    hubert_enc_init_free(&pc->hubert_enc_init);
    hubert_proj_free(&pc->hubert_proj);
    hubert_feat_free(&pc->hubert_feat);
    sem_enc_free(&pc->sem_enc);
    dac_enc_free(&pc->dac_enc);
    dac_free(&pc->dac);
    wctx_free(&pc->wctx);
    // Idempotent: releases the GGUF mmap when a throw left it open mid-load,
    // no-op on the success path where it was closed after wctx_alloc.
    gf_close(&pc->gguf);
    *pc = {};
}

// Debug: run only the DAC encoder on a precomputed audio waveform. Used to
// validate dac-encoder.h in isolation against reference acoustic_encoder before
// the full encode pipeline (HuBERT, SemanticEncoder, fc, RVQ) is wired up.
//
// `audio_f32` is [T_in] mono 24 kHz. Returns the latent flat in GGML layout
// (T_out fast, 256 channels slow) so the caller can write it directly to a
// raw f32 file and compare against the reference.
static std::vector<float> pipeline_codec_dac_enc_test(PipelineCodec * pc, const float * audio_f32, int n_samples) {
    if (n_samples <= 0) {
        return {};
    }

    const int    n_max_nodes    = 4096;
    const size_t graph_ctx_size = ggml_tensor_overhead() * n_max_nodes + ggml_graph_overhead_custom(n_max_nodes, false);

    struct ggml_init_params gp   = { graph_ctx_size, NULL, /*no_alloc=*/true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        return {};
    }

    // Audio input [T_in, 1] f32 to match the dac_conv1d input layout
    // (ne[0]=T fast, ne[1]=channels slow).
    struct ggml_tensor * audio_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(audio_in, "audio_in");
    ggml_set_input(audio_in);

    struct ggml_tensor * latent = dac_enc_build_graph(gctx, &pc->dac_enc, audio_in);
    ggml_set_output(latent);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, latent);

    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, graph)) {
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    ggml_backend_tensor_set(audio_in, audio_f32, 0, (size_t) n_samples * sizeof(float));

    enum ggml_status st = ggml_backend_sched_graph_compute(pc->sched, graph);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    const size_t       n = ggml_nelements(latent);
    std::vector<float> latent_out(n);
    ggml_backend_tensor_get(latent, latent_out.data(), 0, n * sizeof(float));

    ggml_backend_sched_reset(pc->sched);
    ggml_free(gctx);
    return latent_out;
}

// Debug: run only the SemanticEncoder on a precomputed [T, 768] f32 input
// (typically the post-HuBERT downsampled features dumped from the reference). The
// output has the same shape, returned flat in GGML layout (T fast, 768 slow).
static std::vector<float> pipeline_codec_sem_enc_test(PipelineCodec * pc, const float * features_f32, int n_frames) {
    if (n_frames <= 0) {
        return {};
    }

    const int    n_max_nodes    = 4096;
    const size_t graph_ctx_size = ggml_tensor_overhead() * n_max_nodes + ggml_graph_overhead_custom(n_max_nodes, false);

    struct ggml_init_params gp   = { graph_ctx_size, NULL, /*no_alloc=*/true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        return {};
    }

    // Input [T, 768] f32, GGML layout T fast (matches dac_conv1d input).
    struct ggml_tensor * sem_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_frames, SEM_HIDDEN);
    ggml_set_name(sem_in, "sem_in");
    ggml_set_input(sem_in);

    struct ggml_tensor * sem_out = sem_enc_build_graph(gctx, &pc->sem_enc, sem_in);
    ggml_set_output(sem_out);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, sem_out);

    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, graph)) {
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    ggml_backend_tensor_set(sem_in, features_f32, 0, (size_t) n_frames * SEM_HIDDEN * sizeof(float));

    enum ggml_status st = ggml_backend_sched_graph_compute(pc->sched, graph);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    const size_t       n = ggml_nelements(sem_out);
    std::vector<float> out(n);
    ggml_backend_tensor_get(sem_out, out.data(), 0, n * sizeof(float));

    ggml_backend_sched_reset(pc->sched);
    ggml_free(gctx);
    return out;
}

// Debug: full _extract_semantic_features pipeline.
//   audio (16 kHz pre-padded) -> feat_extractor -> feat_projection
//   -> enc_init -> 12 transformer layers (capture 13 hidden states)
//   -> mean over the 13 states -> downsample time axis by 2
// All of this lives inside one graph for a single backend roundtrip.
static std::vector<float> pipeline_codec_hubert_features_test(PipelineCodec * pc,
                                                              const float *   audio_f32,
                                                              int             n_samples,
                                                              const char *    dump_dir) {
    if (n_samples <= 0) {
        return {};
    }

    // 12 layers + projection + pos_conv decomposition -> ~1500 nodes total.
    // Bumping to 16384 leaves headroom for the mean reduction on top.
    const int    n_max_nodes    = 16384;
    const size_t graph_ctx_size = ggml_tensor_overhead() * n_max_nodes + ggml_graph_overhead_custom(n_max_nodes, false);

    struct ggml_init_params gp   = { graph_ctx_size, NULL, /*no_alloc=*/true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        return {};
    }

    struct ggml_tensor * audio = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(audio, "audio");
    ggml_set_input(audio);

    // Feature extraction stack: audio -> conv -> projection -> enc_init.
    // The extra out_post_ln tap on the projection lets us tell the LayerNorm
    // and the 512 -> 768 Linear apart inside hubert-feat-proj.
    struct ggml_tensor * feat         = hubert_feat_build_graph(gctx, &pc->hubert_feat, audio);
    struct ggml_tensor * proj_post_ln = NULL;
    struct ggml_tensor * proj         = hubert_proj_build_graph(gctx, &pc->hubert_proj, feat, &proj_post_ln);
    struct ggml_tensor * h            = hubert_enc_init_build_graph(gctx, &pc->hubert_enc_init, proj);

    // Capture the 13 hidden states fed to the stack mean: (post enc_init,
    // post layer 0, ..., post layer 11). The reference encoder loop snapshots the input
    // before each layer call, then appends the final output once after the
    // loop, which is mathematically the same sequence.
    const int            n_states = HUBERT_NUM_LAYERS + 1;
    struct ggml_tensor * states[n_states];
    states[0] = h;
    for (int i = 0; i < HUBERT_NUM_LAYERS; i++) {
        h             = hubert_layer_build_graph(gctx, &pc->hubert_layers[i], h);
        states[i + 1] = h;
    }

    // Mean-pool the 13 states: sum then scale by 1/13. ggml_add chains nicely
    // and stays within the F32/F16 src1 contract on CUDA.
    struct ggml_tensor * sum = states[0];
    for (int i = 1; i < n_states; i++) {
        sum = ggml_add(gctx, sum, states[i]);
    }
    struct ggml_tensor * mean = ggml_scale(gctx, sum, 1.0f / (float) n_states);

    // Downsample by semantic_downsample_factor = 2 along the time axis.
    // mean has ne=(768, T_h). We pick every other column starting at 0 via a
    // strided view, then make it contiguous so the host copy stays simple.
    const int            T_h      = (int) mean->ne[1];
    const int            T_out    = T_h / 2;
    struct ggml_tensor * strided  = ggml_view_2d(gctx, mean, HUBERT_HIDDEN, T_out, 2 * mean->nb[1], 0);
    struct ggml_tensor * features = ggml_cont(gctx, strided);
    ggml_set_output(features);

    // Bisect taps: keep the buffers of the intermediate stages alive past the
    // memory planner so they can be copied back after compute. Cheap, only 8
    // [768, T_h] f32 tensors plus feat at [512, T_feat] and proj-ln at
    // [512, T_feat], total ~13 MB.
    if (dump_dir) {
        ggml_set_output(feat);
        ggml_set_output(proj_post_ln);
        ggml_set_output(proj);
        ggml_set_output(states[0]);
        ggml_set_output(states[1]);
        ggml_set_output(states[6]);
        ggml_set_output(states[8]);
        ggml_set_output(states[10]);
        ggml_set_output(states[12]);
    }

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, features);

    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, graph)) {
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    ggml_backend_tensor_set(audio, audio_f32, 0, (size_t) n_samples * sizeof(float));

    enum ggml_status st = ggml_backend_sched_graph_compute(pc->sched, graph);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    // Bisect dumps. ne layout is (C, T) on every tap, so we dump as (T, C) which
    // matches the natural numpy reshape on the linear buffer. Names mirror the
    // Python hooks installed in the test script.
    if (dump_dir) {
        DebugDumper dbg;
        debug_init(&dbg, dump_dir);
        auto dump_tap = [&](struct ggml_tensor * t, const char * name) {
            const int          T   = (int) t->ne[1];
            const int          C   = (int) t->ne[0];
            const size_t       nel = (size_t) T * (size_t) C;
            std::vector<float> buf(nel);
            ggml_backend_tensor_get(t, buf.data(), 0, nel * sizeof(float));
            debug_dump_2d(&dbg, name, buf.data(), T, C);
        };
        dump_tap(feat, "hubert-feat-extract");
        dump_tap(proj_post_ln, "hubert-feat-proj-ln");
        dump_tap(proj, "hubert-feat-proj");
        dump_tap(states[0], "hubert-enc-init");
        dump_tap(states[1], "hubert-l0");
        dump_tap(states[6], "hubert-l5");
        dump_tap(states[8], "hubert-l7");
        dump_tap(states[10], "hubert-l9");
        dump_tap(states[12], "hubert-l11");
    }

    const size_t       n = ggml_nelements(features);
    std::vector<float> out(n);
    ggml_backend_tensor_get(features, out.data(), 0, n * sizeof(float));

    ggml_backend_sched_reset(pc->sched);
    ggml_free(gctx);
    return out;
}
