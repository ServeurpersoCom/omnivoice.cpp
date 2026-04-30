#pragma once
// pipeline-codec.h : audio-tokenizer decode pipeline for OmniVoice.
// Loads the audio-tokenizer GGUF, holds RVQ + fc2 + DAC weights on the
// backend, and exposes a one-shot decode call :
//
//   codes [num_codebooks, T] i32  ->  audio [T * 960] f32 mono 24 kHz

#include "backend.h"
#include "dac-decoder.h"
#include "dac-encoder.h"
#include "ggml-backend.h"
#include "gguf-weights.h"
#include "hubert-enc.h"
#include "rvq-codec.h"
#include "semantic-enc.h"
#include "weight-ctx.h"

#include <cstdint>
#include <vector>

struct PipelineCodec {
    // GGUF source. Mmap is closed inside pipeline_codec_load once the data
    // has been copied to the backend.
    GGUFModel gguf;

    // Audio tokenizer modules
    RVQCodec             rvq;
    DACDecoder           dac;
    DACEncoder           dac_enc;
    SemanticEncoder      sem_enc;
    HubertFeatExtractor  hubert_feat;
    HubertFeatProjection hubert_proj;
    HubertEncInit        hubert_enc_init;
    HubertLayer          hubert_layers[HUBERT_NUM_LAYERS];

    // fc Linear post-concat acoustic+semantic in encode path. Loaded into the
    // shared wctx alongside RVQ and fc2.
    //   fc_w : bf16 ne=(1024, 1024)
    //   fc_b : f32  ne=(1024)
    struct ggml_tensor * fc_w;
    struct ggml_tensor * fc_b;

    // fc2 Linear : RVQ output 1024 -> DAC input 256
    // Bias loaded as F32 (ggml_add CUDA needs F32/F16 on src1, same as DAC)
    struct ggml_tensor * fc2_w;  // bf16 ne=(1024, 256)
    struct ggml_tensor * fc2_b;  // f32  ne=(256)

    // RVQ + fc2 share one WeightCtx. DACDecoder owns its own buffer because
    // it applies per-tensor transforms at load (snake reciprocal, conv_t
    // permutation) that don't fit the gf_load_tensor passthrough scheme.
    WeightCtx wctx;

    // Backend pair (GPU + CPU fallback) and scheduler. The scheduler routes
    // ops the GPU backend cannot run (e.g. K-quant get_rows on CUDA) to the
    // CPU backend. Compute uses ggml_backend_sched_graph_compute.
    BackendPair          bp;
    ggml_backend_t       backend;
    ggml_backend_sched_t sched;

    // Audio metadata read from GGUF KV
    int sample_rate;  // 24000
    int hop_length;   // 960 (cumulative DAC upsample stride)
};

// Open the GGUF, load all weights to the backend, close the GGUF mapping.
// Returns true on success. On failure the struct is left in a clean state.
bool pipeline_codec_load(PipelineCodec * pc, const char * gguf_path, BackendPair bp);

// Decode RVQ codes into a mono 24 kHz waveform.
//   codes : row-major i32, num_codebooks rows of n_frames each (T fast).
// Returns audio of length n_frames * pc->hop_length, empty on failure.
std::vector<float> pipeline_codec_decode(PipelineCodec * pc, const int32_t * codes, int num_codebooks, int n_frames);

// Encode a 24 kHz mono waveform into RVQ codes. Chains the full audio
// tokenizer pipeline :
//   resample 24k->16k + pad 160 (CPU)
//   HuBERT features                   -> e_semantic_input (768, T_s)
//   SemanticEncoder                   -> e_semantic       (768, T_s)
//   DAC encoder (with conditional pad) -> e_acoustic       (256, T_a)
//   concat [acoustic, semantic]        -> embeddings       (1024, T)
//   fc Linear (1024 -> 1024)           -> projected        (1024, T)
//   RVQ encode loop                    -> codes            (CB=8, T) i32
// Returns codes flat in (CB, T) row-major, empty on failure.
// dump_dir : when non null, writes ref-audio-16k and ref-hubert-features
// debug binaries into that directory. Pass NULL to skip.
std::vector<int32_t> pipeline_codec_encode(PipelineCodec * pc,
                                           const float *   audio_24k,
                                           int             n_samples,
                                           const char *    dump_dir = NULL);

// Free all backend buffers and GGML contexts. Safe to call on a zeroed struct.
void pipeline_codec_free(PipelineCodec * pc);
